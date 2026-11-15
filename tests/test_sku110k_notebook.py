"""
Unit tests for `vision.sku110k_subset` and `notebooks/S02_C3_yolov8n_sku110k_colab.ipynb` (S02-C3).

Covers:
1. SKU-110K root validation and split list parsing.
2. Deterministic, seed-based subset selection and YOLO folder layout building.
3. Coordinate conversions (`yolo_to_xyxy`, `detections_to_array`) and thresholded counting
   integrated with `fusion.metrics`.
4. Notebook JSON structure, code cell syntax compilation, and embedded fallback synchronization.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from fusion import metrics as fusion_metrics
from notebooks import build_S02_C3_notebook as nb_builder
from vision import sku110k_subset as subset


def _make_fake_sku110k(root: Path, n_train: int = 12, n_val: int = 6) -> Path:
    """Create a synthetic SKU-110K root directory with train.txt, val.txt, images/, and labels/."""
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for split, count in (("train", n_train), ("val", n_val)):
        lines = []
        for idx in range(count):
            name = f"{split}_{idx}.jpg"
            (img_dir / name).write_bytes(b"FAKE_JPEG_BYTES")
            # Give each image idx+1 boxes
            boxes = [f"0 0.500000 0.500000 0.100000 0.100000" for _ in range(idx + 1)]
            (lbl_dir / f"{split}_{idx}.txt").write_text("\n".join(boxes) + "\n", encoding="utf-8")
            lines.append(f"./images/{name}")
        (root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class TestSku110kSubsetHelpers:
    """Unit tests for dataset validation, subset sampling, and label conversions."""

    def test_validate_sku110k_root_and_errors(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            subset.validate_sku110k_root(tmp_path / "nonexistent")

        root = _make_fake_sku110k(tmp_path / "sku", n_train=10, n_val=5)
        counts = subset.validate_sku110k_root(root, min_images={"train": 8, "val": 4})
        assert counts == {"train": 10, "val": 5}

        with pytest.raises(ValueError, match="expected at least"):
            subset.validate_sku110k_root(root, min_images={"train": 100, "val": 5})

    def test_label_path_for_image_replaces_last_images_component(self):
        img = Path("/content/datasets/images/sub/images/train_01.jpg")
        lbl = subset.label_path_for_image(img)
        assert lbl.as_posix().endswith("/datasets/images/sub/labels/train_01.txt")

        with pytest.raises(ValueError, match="no 'images' directory"):
            subset.label_path_for_image(Path("/content/datasets/photos/train_01.jpg"))

    def test_read_yolo_labels_and_count(self, tmp_path: Path):
        lbl = tmp_path / "sample.txt"
        lbl.write_text("0 0.5 0.5 0.2 0.4\n0 0.25 0.25 0.1 0.1\n", encoding="utf-8")
        arr = subset.read_yolo_labels(lbl)
        assert arr.shape == (2, 5)
        assert subset.count_label_boxes(lbl) == 2
        assert subset.count_label_boxes(tmp_path / "missing.txt") == 0

        bad = tmp_path / "bad.txt"
        bad.write_text("0 0.5 0.5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected >= 5"):
            subset.read_yolo_labels(bad)

    def test_select_subset_is_deterministic_and_order_independent(self):
        items_a = [Path(f"img_{i}.jpg") for i in range(20)]
        items_b = list(reversed(items_a))
        picked_a = subset.select_subset(items_a, 7, seed=42)
        picked_b = subset.select_subset(items_b, 7, seed=42)
        picked_diff_seed = subset.select_subset(items_a, 7, seed=99)

        assert len(picked_a) == 7
        assert picked_a == picked_b
        assert picked_a != picked_diff_seed

        with pytest.raises(ValueError, match="Cannot select"):
            subset.select_subset(items_a, 25, seed=0)

    def test_build_verify_and_write_data_yaml(self, tmp_path: Path):
        root = _make_fake_sku110k(tmp_path / "sku", n_train=15, n_val=8)
        out_dir = tmp_path / "subset"

        chosen_1 = subset.build_subset(root, out_dir, n_train=6, n_val=3, seed=0)
        verified = subset.verify_subset(out_dir, n_train=6, n_val=3)
        assert verified == {"train": 6, "val": 3}

        yaml_path = subset.write_data_yaml(out_dir)
        text = yaml_path.read_text(encoding="utf-8")
        assert "train: images/train" in text
        assert "val: images/val" in text
        assert "0: object" in text

        # Re-running with same seed yields identical image lists
        chosen_2 = subset.build_subset(root, out_dir, n_train=6, n_val=3, seed=0, overwrite=True)
        assert chosen_1 == chosen_2

        with pytest.raises(FileExistsError):
            subset.build_subset(root, out_dir, n_train=6, n_val=3, seed=0, overwrite=False)

    def test_conversions_and_integration_with_fusion_metrics(self):
        # Normalized YOLO box [cls=0, xc=0.5, yc=0.5, w=0.2, h=0.4] on 100x200 image
        # -> x1=40, y1=60, x2=60, y2=140
        yolo_lbl = np.array([[0.0, 0.5, 0.5, 0.2, 0.4]])
        xyxy = subset.yolo_to_xyxy(yolo_lbl, width=100, height=200)
        np.testing.assert_allclose(xyxy, [[40.0, 60.0, 60.0, 140.0]])

        dets = subset.detections_to_array(
            boxes_xyxy=[[40.0, 60.0, 60.0, 140.0], [0.0, 0.0, 10.0, 10.0]],
            confs=[0.90, 0.15],
            cls_ids=[0, 0],
        )
        assert dets.shape == (2, 6)
        assert subset.count_detections(dets, conf_threshold=0.25) == 1
        assert subset.count_detections(dets, conf_threshold=0.10) == 2

        by_thr = subset.counts_for_thresholds([dets], [0.10, 0.25, 0.95])
        assert by_thr[0.10] == (2,)
        assert by_thr[0.25] == (1,)
        assert by_thr[0.95] == (0,)

        # Evaluate with fusion.metrics
        gt_counts = [1]
        assert fusion_metrics.counting_mae(gt_counts, by_thr[0.25]) == pytest.approx(0.0)
        assert fusion_metrics.exact_accuracy(gt_counts, by_thr[0.25]) == pytest.approx(1.0)
        map_res = fusion_metrics.map_wrapper(
            predictions=[subset.filter_by_conf(dets, 0.25)],
            ground_truths=[xyxy],
        )
        assert map_res["map50"] == pytest.approx(1.0)


class TestColabNotebookIntegrity:
    """Verify the generated Colab notebook is valid, compiles cleanly, and stays in sync."""

    def test_notebook_file_exists_and_matches_builder(self):
        assert nb_builder.NOTEBOOK_PATH.is_file()
        on_disk = json.loads(nb_builder.NOTEBOOK_PATH.read_text(encoding="utf-8"))
        expected = nb_builder.build_notebook()
        assert on_disk == expected
        assert on_disk["nbformat"] == 4
        assert on_disk["metadata"]["accelerator"] == "GPU"

    def test_all_code_cells_compile_valid_python(self):
        nb = json.loads(nb_builder.NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 10
        for idx, cell in enumerate(code_cells):
            source = "".join(cell["source"])
            compile(source, f"<notebook_cell_{idx}>", "exec")

    def test_embedded_sources_decompress_to_exact_source_files(self):
        for key, path in nb_builder.EMBEDDED_SOURCES.items():
            blob = nb_builder.encode_blob(path)
            restored = zlib.decompress(base64.b64decode(blob))
            expected = nb_builder.normalise_source_bytes(path.read_bytes())
            assert restored == expected, f"Embedded blob '{key}' out of sync with {path}"
