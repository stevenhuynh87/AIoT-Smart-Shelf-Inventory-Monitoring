"""
Helpers for the Label Studio annotation workflow (S02-C2).

Pipeline:  prepare_batch -> (annotate in Label Studio) -> convert_label_studio_export -> validate_yolo_labels

* `classes.txt` is the single source of truth: line order == YOLO class id.
* Image-level attributes (occlusion / lighting) use the enums from docs/dataset_spec.md.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OCCLUSION_LEVELS = ("none", "low", "medium", "high")
LIGHTING_LEVELS = ("normal", "low", "harsh", "uneven")
# Boxes of these classes are not counted as inventory when comparing with gt_count.
NON_PRODUCT_CLASSES = frozenset({"foreign_object"})

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_PALETTE = ("#22c55e", "#ef4444", "#3b82f6", "#f59e0b", "#a855f7", "#14b8a6", "#ec4899", "#84cc16")
_LS_UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


# --------------------------------------------------------------------------- classes / config
def load_classes(path: Path) -> List[str]:
    """Read class names (one per line, '#' comments allowed). Line order defines the YOLO id."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    if not names:
        raise ValueError(f"No classes found in {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class names in {path}")
    return names


def build_label_config_xml(classes: Sequence[str]) -> str:
    """Build the Label Studio labeling interface (boxes + per-image occlusion/lighting)."""
    labels = "\n".join(
        f'    <Label value="{c}" background="{_PALETTE[i % len(_PALETTE)]}"/>' for i, c in enumerate(classes)
    )
    occ = "\n".join(f'    <Choice value="{v}"/>' for v in OCCLUSION_LEVELS)
    lit = "\n".join(f'    <Choice value="{v}"/>' for v in LIGHTING_LEVELS)
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <RectangleLabels name="label" toName="image" strokeWidth="2">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        '  <Header value="Occlusion level (occlusion_level)"/>\n'
        '  <Choices name="occlusion" toName="image" choice="single" showInline="true" required="true">\n'
        f"{occ}\n"
        "  </Choices>\n"
        '  <Header value="Lighting level (lighting_level)"/>\n'
        '  <Choices name="lighting" toName="image" choice="single" showInline="true" required="true">\n'
        f"{lit}\n"
        "  </Choices>\n"
        "</View>\n"
    )


# --------------------------------------------------------------------------- prepare batch
def natural_key(text: str) -> List:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def canonical_image_id(file_name: str) -> str:
    """'shelf_01.jpg.jpg' -> 'shelf_01'; other names -> stem without image extensions."""
    m = re.search(r"(shelf_\d+)", file_name, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    name = file_name
    while Path(name).suffix.lower() in IMAGE_SUFFIXES:
        name = Path(name).stem
    return name.lower()


def discover_images(source_dir: Path) -> List[Path]:
    """Recursively find images (also names such as *.jpg.jpg), naturally sorted by name."""
    found = [
        p
        for p in Path(source_dir).rglob("*")
        if p.is_file() and any(s in p.name.lower() for s in IMAGE_SUFFIXES)
    ]
    return sorted(found, key=lambda p: natural_key(p.name))


def select_batch(images: Sequence[Path], n: int) -> List[Path]:
    """Deterministic round-robin pick over parent folders so every scenario group is represented."""
    if n <= 0:
        raise ValueError("n must be positive")
    groups: Dict[str, List[Path]] = {}
    for p in images:
        groups.setdefault(p.parent.name, []).append(p)
    ordered = [groups[k] for k in sorted(groups, key=natural_key)]
    picked: List[Path] = []
    idx = 0
    while len(picked) < n and any(idx < len(g) for g in ordered):
        for g in ordered:
            if idx < len(g) and len(picked) < n:
                picked.append(g[idx])
        idx += 1
    return sorted(picked, key=lambda p: natural_key(p.name))


def load_gt_counts(gt_csv: Optional[Path]) -> Dict[str, int]:
    """image_id -> gt_count from vision/ground_truth_yolo.csv (missing file -> {})."""
    if gt_csv is None or not Path(gt_csv).exists():
        return {}
    out: Dict[str, int] = {}
    with open(gt_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            val = (row.get("gt_count") or "").strip()
            if val != "":
                out[canonical_image_id(row["image_id"])] = int(float(val))
    return out


def prepare_batch(
    source_dir: Path,
    out_dir: Path,
    n: int = 30,
    gt_csv: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Copy n images to out_dir/images (canonical names) and write out_dir/manifest.csv."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {source_dir}")
    images = discover_images(source_dir)
    if not images:
        raise FileNotFoundError(f"No images found under: {source_dir}")

    gt = load_gt_counts(gt_csv)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for src in select_batch(images, n):
        image_id = canonical_image_id(src.name)
        dst = img_dir / f"{image_id}{'.png' if '.png' in src.name.lower() else '.jpg'}"
        shutil.copy2(src, dst)  # originals are never modified
        rows.append(
            {
                "image_id": image_id,
                "file_name": dst.name,
                "source_group": src.parent.name,
                "gt_count": gt.get(image_id, ""),
            }
        )
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "file_name", "source_group", "gt_count"])
        w.writeheader()
        w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- convert LS export -> YOLO
@dataclass
class ConversionReport:
    converted: int = 0
    boxes: int = 0
    skipped_no_annotation: List[str] = field(default_factory=list)
    skipped_rotated_boxes: int = 0
    unknown_labels: List[str] = field(default_factory=list)


def _task_image_name(task: dict) -> str:
    raw = str(task.get("data", {}).get("image", ""))
    name = Path(raw.replace("\\", "/")).name
    return _LS_UPLOAD_PREFIX.sub("", name)  # Label Studio prefixes uploads with an 8-hex hash


def convert_label_studio_export(
    export_json: Path,
    labels_dir: Path,
    classes: Sequence[str],
    attributes_csv: Optional[Path] = None,
    images_dir: Optional[Path] = None,
) -> ConversionReport:
    """
    Convert a Label Studio 'JSON' export to YOLO txt files (class id from `classes`, normalized xywh)
    and an image-level attribute CSV (occlusion_level, lighting_level).
    If `images_dir` is provided, matches the task's image name against real files in `images_dir`
    to resolve Label Studio's random upload truncation/suffixes.
    """
    tasks = json.loads(Path(export_json).read_text(encoding="utf-8"))
    class_id = {c: i for i, c in enumerate(classes)}
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    report = ConversionReport()
    attr_rows: List[Dict[str, str]] = []

    # Map candidate stems on disk if images_dir is available
    disk_stems: List[str] = []
    if images_dir and Path(images_dir).is_dir():
        disk_stems = [
            canonical_image_id(p.name)
            for p in Path(images_dir).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]

    for task in tasks:
        name = _task_image_name(task)
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns or not name:
            report.skipped_no_annotation.append(name or f"task_{task.get('id')}")
            continue

        lines: List[str] = []
        occlusion = lighting = ""
        for res in anns[0].get("result", []):
            kind, value = res.get("type"), res.get("value", {})
            if kind == "rectanglelabels":
                if abs(float(value.get("rotation", 0) or 0)) > 1e-6:
                    report.skipped_rotated_boxes += 1
                    continue
                label = (value.get("rectanglelabels") or [""])[0]
                if label not in class_id:
                    report.unknown_labels.append(label)
                    continue
                x, y, w, h = (float(value[k]) for k in ("x", "y", "width", "height"))
                xc, yc = (x + w / 2) / 100.0, (y + h / 2) / 100.0
                lines.append(f"{class_id[label]} {xc:.6f} {yc:.6f} {w / 100.0:.6f} {h / 100.0:.6f}")
            elif kind == "choices":
                choice = (value.get("choices") or [""])[0]
                if res.get("from_name") == "occlusion":
                    occlusion = choice
                elif res.get("from_name") == "lighting":
                    lighting = choice

        stem = canonical_image_id(name)
        # If Label Studio appended random characters or truncated, match with disk_stems
        if disk_stems and stem not in disk_stems:
            matched = [s for s in disk_stems if s.startswith(stem[:35]) or stem.startswith(s[:35])]
            if len(matched) == 1:
                stem = matched[0]

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        attr_rows.append(
            {"image_id": stem, "occlusion_level": occlusion, "lighting_level": lighting, "num_boxes": str(len(lines))}
        )
        report.converted += 1
        report.boxes += len(lines)

    if attributes_csv is not None:
        with open(attributes_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "occlusion_level", "lighting_level", "num_boxes"])
            w.writeheader()
            w.writerows(sorted(attr_rows, key=lambda r: natural_key(r["image_id"])))
    return report


# --------------------------------------------------------------------------- validate YOLO labels
@dataclass
class ValidationReport:
    images: int = 0
    labeled: int = 0
    total_boxes: int = 0
    errors: List[str] = field(default_factory=list)
    count_mismatches: List[Tuple[str, int, int]] = field(default_factory=list)  # (image_id, boxes, gt_count)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_yolo_labels(
    images_dir: Path,
    labels_dir: Path,
    classes: Sequence[str],
    gt_counts: Optional[Dict[str, int]] = None,
) -> ValidationReport:
    """Check every image has a well-formed label file; optionally compare product boxes to gt_count."""
    rep = ValidationReport()
    n_cls = len(classes)
    for img in sorted(Path(images_dir).iterdir(), key=lambda p: natural_key(p.name)):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rep.images += 1
        image_id = canonical_image_id(img.name)
        txt = Path(labels_dir) / f"{image_id}.txt"
        if not txt.exists():
            rep.errors.append(f"{image_id}: missing label file")
            continue
        rep.labeled += 1
        product_boxes = 0
        for ln, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError("expected 5 fields")
                cid = int(parts[0])
                xc, yc, w, h = (float(p) for p in parts[1:])
            except ValueError as e:
                rep.errors.append(f"{image_id}:{ln}: malformed line ({e})")
                continue
            if not 0 <= cid < n_cls:
                rep.errors.append(f"{image_id}:{ln}: class id {cid} out of range 0..{n_cls - 1}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
                rep.errors.append(f"{image_id}:{ln}: coordinates outside (0,1] or empty box")
                continue
            if xc - w / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc - h / 2 < -1e-6 or yc + h / 2 > 1 + 1e-6:
                rep.errors.append(f"{image_id}:{ln}: box exceeds image bounds")
                continue
            rep.total_boxes += 1
            if classes[cid] not in NON_PRODUCT_CLASSES:
                product_boxes += 1
        if gt_counts and image_id in gt_counts and product_boxes != gt_counts[image_id]:
            rep.count_mismatches.append((image_id, product_boxes, gt_counts[image_id]))
    return rep

