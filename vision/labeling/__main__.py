"""
CLI for the annotation workflow.   python -m vision.labeling <command> [options]

  config    write label_config.xml (paste into Label Studio > Settings > Labeling Interface > Code)
  prepare   copy N images into data/labeling/images + manifest.csv
  convert   Label Studio JSON export -> YOLO txt + image_attributes.csv
  validate  check YOLO labels (format, ranges, box count vs ground_truth_yolo.csv)
"""

import argparse
import sys
from pathlib import Path

from vision.labeling.labeling_utils import (
    build_label_config_xml,
    convert_label_studio_export,
    load_classes,
    load_gt_counts,
    prepare_batch,
    validate_yolo_labels,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CLASSES = HERE / "classes.txt"
DEFAULT_WORKDIR = ROOT / "data" / "labeling"
DEFAULT_GT = ROOT / "vision" / "ground_truth_yolo.csv"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m vision.labeling", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("config")
    c.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    c.add_argument("--out", type=Path, default=HERE / "label_config.xml")

    pr = sub.add_parser("prepare")
    pr.add_argument("--source", type=Path, required=True, help="Folder with the original shelf photos")
    pr.add_argument("--out", type=Path, default=DEFAULT_WORKDIR)
    pr.add_argument("-n", type=int, default=30)
    pr.add_argument("--gt", type=Path, default=DEFAULT_GT)

    cv = sub.add_parser("convert")
    cv.add_argument("--export", type=Path, required=True, help="Label Studio 'JSON' export file")
    cv.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    cv.add_argument("--out", type=Path, default=DEFAULT_WORKDIR)

    va = sub.add_parser("validate")
    va.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    va.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    va.add_argument("--gt", type=Path, default=DEFAULT_GT)

    a = p.parse_args(argv)

    if a.cmd == "config":
        a.out.write_text(build_label_config_xml(load_classes(a.classes)), encoding="utf-8")
        print(f"[OK] Wrote {a.out}")
        return 0

    if a.cmd == "prepare":
        rows = prepare_batch(a.source, a.out, n=a.n, gt_csv=a.gt)
        groups = sorted({r["source_group"] for r in rows})
        print(f"[OK] Copied {len(rows)} images to {a.out / 'images'} (groups: {', '.join(groups)})")
        print(f"[OK] Manifest: {a.out / 'manifest.csv'}")
        return 0

    if a.cmd == "convert":
        rep = convert_label_studio_export(
            a.export,
            a.out / "labels",
            load_classes(a.classes),
            a.out / "image_attributes.csv",
            images_dir=a.out / "images",
        )
        print(f"[OK] Converted {rep.converted} images, {rep.boxes} boxes -> {a.out / 'labels'}")
        if rep.skipped_no_annotation:
            print(f"[WARN] No annotation yet: {', '.join(rep.skipped_no_annotation)}")
        if rep.unknown_labels:
            print(f"[WARN] Labels not in classes.txt were dropped: {sorted(set(rep.unknown_labels))}")
        if rep.skipped_rotated_boxes:
            print(f"[WARN] {rep.skipped_rotated_boxes} rotated boxes skipped (YOLO needs axis-aligned boxes)")
        return 0

    rep = validate_yolo_labels(
        a.workdir / "images", a.workdir / "labels", load_classes(a.classes), load_gt_counts(a.gt)
    )
    print(f"Images: {rep.images} | labeled: {rep.labeled} | boxes: {rep.total_boxes}")
    for err in rep.errors:
        print(f"[ERROR] {err}")
    for image_id, boxes, gt in rep.count_mismatches:
        print(f"[CHECK] {image_id}: {boxes} product boxes vs gt_count={gt}")
    if rep.ok and not rep.count_mismatches:
        print("[OK] Labels are valid and match ground_truth_yolo.csv")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())

