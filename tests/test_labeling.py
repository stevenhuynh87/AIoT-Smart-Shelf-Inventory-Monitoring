"""Unit tests for the Label Studio annotation workflow helpers (S02-C2)."""

import csv
import json

import pytest

from vision.labeling.__main__ import main as labeling_main
from vision.labeling.labeling_utils import (
    build_label_config_xml,
    canonical_image_id,
    convert_label_studio_export,
    discover_images,
    load_classes,
    prepare_batch,
    select_batch,
    validate_yolo_labels,
)

CLASSES = ["product", "foreign_object"]


def _touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _box(x, y, w, h, label="product", rotation=0):
    return {
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "value": {"x": x, "y": y, "width": w, "height": h, "rotation": rotation, "rectanglelabels": [label]},
    }


def _choice(name, value):
    return {"type": "choices", "from_name": name, "to_name": "image", "value": {"choices": [value]}}


def test_default_classes_file_loads():
    from vision.labeling.__main__ import DEFAULT_CLASSES

    names = load_classes(DEFAULT_CLASSES)
    assert names[0] == "product" and len(names) == len(set(names))


def test_load_classes_rejects_duplicates_and_empty(tmp_path):
    (tmp_path / "dup.txt").write_text("a\na\n", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("# only comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_classes(tmp_path / "dup.txt")
    with pytest.raises(ValueError, match="No classes"):
        load_classes(tmp_path / "empty.txt")


def test_label_config_contains_classes_and_schema_enums():
    xml = build_label_config_xml(CLASSES)
    for token in ('value="product"', 'value="foreign_object"', 'name="occlusion"', 'name="lighting"',
                  'value="high"', 'value="uneven"', 'value="harsh"'):
        assert token in xml
    assert xml.startswith("<View>") and xml.rstrip().endswith("</View>")


def test_canonical_image_id_handles_double_extension():
    assert canonical_image_id("shelf_01.jpg.jpg") == "shelf_01"
    assert canonical_image_id("Shelf_12.JPG") == "shelf_12"
    assert canonical_image_id("photo.jpg.jpg") == "photo"


def test_select_batch_is_round_robin_over_groups_and_deterministic(tmp_path):
    imgs = []
    for g, count in (("g1", 6), ("g2", 10), ("g3", 6)):
        for i in range(count):
            imgs.append(_touch(tmp_path / g / f"{g}_{i}.jpg"))
    picked = select_batch(imgs, 12)
    assert len(picked) == 12
    assert {p.parent.name for p in picked} == {"g1", "g2", "g3"}
    assert picked == select_batch(imgs, 12)
    assert len(select_batch(imgs, 100)) == len(imgs)  # n larger than available
    with pytest.raises(ValueError):
        select_batch(imgs, 0)


def test_prepare_batch_copies_with_canonical_names_and_manifest(tmp_path):
    src = tmp_path / "Công việc"
    _touch(src / "nhóm 1" / "shelf_01.jpg.jpg", b"A")
    _touch(src / "nhóm 1" / "shelf_02.jpg.jpg", b"B")
    _touch(src / "Nhóm 2" / "shelf_7.jpg.jpg", b"C")
    gt = tmp_path / "gt.csv"
    gt.write_text("image_id,group_name,file_name,gt_count,scenario\nshelf_01,g,f,1,s\nshelf_7,g,f,4,s\n",
                  encoding="utf-8")

    rows = prepare_batch(src, tmp_path / "work", n=3, gt_csv=gt)

    assert {r["image_id"] for r in rows} == {"shelf_01", "shelf_02", "shelf_7"}
    assert (tmp_path / "work" / "images" / "shelf_01.jpg").read_bytes() == b"A"
    assert (src / "nhóm 1" / "shelf_01.jpg.jpg").read_bytes() == b"A"  # original untouched
    with open(tmp_path / "work" / "manifest.csv", newline="", encoding="utf-8-sig") as f:
        manifest = {r["image_id"]: r for r in csv.DictReader(f)}
    assert manifest["shelf_01"]["gt_count"] == "1"
    assert manifest["shelf_02"]["gt_count"] == ""  # no ground truth known -> blank
    assert manifest["shelf_7"]["source_group"] == "Nhóm 2"


def test_prepare_batch_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        prepare_batch(tmp_path / "missing", tmp_path / "out")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No images"):
        prepare_batch(tmp_path / "empty", tmp_path / "out")
    assert discover_images(tmp_path / "empty") == []


def _export(tmp_path, tasks):
    p = tmp_path / "export.json"
    p.write_text(json.dumps(tasks), encoding="utf-8")
    return p


def test_convert_percent_boxes_to_yolo_and_attributes(tmp_path):
    tasks = [
        {  # hash prefix added by Label Studio uploads must be stripped
            "id": 1,
            "data": {"image": "/data/upload/1/ab12cd34-shelf_01.jpg"},
            "annotations": [{"was_cancelled": False, "result": [
                _box(10, 20, 30, 40), _box(50, 50, 20, 20, label="foreign_object"),
                _choice("occlusion", "low"), _choice("lighting", "normal")]}],
        },
        {  # empty shelf: annotated with zero boxes -> empty (negative) label file
            "id": 2,
            "data": {"image": "/data/upload/1/00ff00ff-shelf_24.jpg"},
            "annotations": [{"result": [_choice("occlusion", "none"), _choice("lighting", "low")]}],
        },
        {"id": 3, "data": {"image": "/data/upload/1/11111111-shelf_03.jpg"}, "annotations": []},
        {  # cancelled (skipped) annotation must not produce a label
            "id": 4,
            "data": {"image": "/data/upload/1/22222222-shelf_04.jpg"},
            "annotations": [{"was_cancelled": True, "result": [_box(0, 0, 10, 10)]}],
        },
    ]
    labels = tmp_path / "labels"
    rep = convert_label_studio_export(_export(tmp_path, tasks), labels, CLASSES, tmp_path / "attrs.csv")

    assert rep.converted == 2 and rep.boxes == 2
    assert rep.skipped_no_annotation == ["shelf_03.jpg", "shelf_04.jpg"]
    lines = (labels / "shelf_01.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0 0.250000 0.400000 0.300000 0.400000"
    assert lines[1] == "1 0.600000 0.600000 0.200000 0.200000"
    assert (labels / "shelf_24.txt").read_text(encoding="utf-8") == ""
    assert not (labels / "shelf_03.txt").exists()
    with open(tmp_path / "attrs.csv", newline="", encoding="utf-8-sig") as f:
        attrs = {r["image_id"]: r for r in csv.DictReader(f)}
    assert attrs["shelf_01"]["occlusion_level"] == "low" and attrs["shelf_01"]["num_boxes"] == "2"
    assert attrs["shelf_24"]["lighting_level"] == "low"


def test_convert_reports_unknown_labels_and_rotated_boxes(tmp_path):
    tasks = [{
        "id": 1, "data": {"image": "/data/upload/1/aaaaaaaa-shelf_01.jpg"},
        "annotations": [{"result": [_box(0, 0, 10, 10, label="mystery"), _box(0, 0, 10, 10, rotation=15),
                                    _box(5, 5, 10, 10)]}],
    }]
    rep = convert_label_studio_export(_export(tmp_path, tasks), tmp_path / "labels", CLASSES)
    assert rep.unknown_labels == ["mystery"] and rep.skipped_rotated_boxes == 1 and rep.boxes == 1


def _make_dataset(tmp_path, label_text):
    images, labels = tmp_path / "images", tmp_path / "labels"
    _touch(images / "shelf_01.jpg")
    labels.mkdir()
    if label_text is not None:
        (labels / "shelf_01.txt").write_text(label_text, encoding="utf-8")
    return images, labels


def test_validate_ok_and_count_check_ignores_foreign_objects(tmp_path):
    images, labels = _make_dataset(tmp_path, "0 0.25 0.5 0.2 0.4\n0 0.75 0.5 0.2 0.4\n1 0.5 0.1 0.1 0.1\n")
    rep = validate_yolo_labels(images, labels, CLASSES, {"shelf_01": 2})
    assert rep.ok and rep.total_boxes == 3 and rep.count_mismatches == []
    rep = validate_yolo_labels(images, labels, CLASSES, {"shelf_01": 3})
    assert rep.ok and rep.count_mismatches == [("shelf_01", 2, 3)]


@pytest.mark.parametrize("text,fragment", [
    (None, "missing label file"),
    ("0 0.5 0.5 0.2\n", "malformed"),
    ("5 0.5 0.5 0.2 0.2\n", "out of range"),
    ("0 1.2 0.5 0.2 0.2\n", "outside"),
    ("0 0.95 0.5 0.2 0.2\n", "exceeds image bounds"),
    ("0 0.5 0.5 0 0.2\n", "empty box"),
])
def test_validate_detects_errors(tmp_path, text, fragment):
    images, labels = _make_dataset(tmp_path, text)
    rep = validate_yolo_labels(images, labels, CLASSES)
    assert not rep.ok and any(fragment in e for e in rep.errors)


def test_cli_end_to_end(tmp_path, capsys):
    src = tmp_path / "src"
    _touch(src / "g1" / "shelf_01.jpg.jpg")
    work = tmp_path / "work"
    assert labeling_main(["prepare", "--source", str(src), "--out", str(work), "-n", "1",
                          "--gt", str(tmp_path / "none.csv")]) == 0
    cfg = tmp_path / "cfg.xml"
    assert labeling_main(["config", "--out", str(cfg)]) == 0 and "<RectangleLabels" in cfg.read_text("utf-8")

    export = _export(tmp_path, [{"id": 1, "data": {"image": "/data/upload/1/abcdef12-shelf_01.jpg"},
                                 "annotations": [{"result": [_box(10, 10, 20, 20)]}]}])
    assert labeling_main(["convert", "--export", str(export), "--out", str(work)]) == 0
    assert labeling_main(["validate", "--workdir", str(work), "--gt", str(tmp_path / "none.csv")]) == 0
    assert "valid" in capsys.readouterr().out

