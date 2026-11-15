"""
Helpers for the SKU-110K YOLOv8n trial-training notebook (S02-C3).

Everything here is deliberately free of GPU / Ultralytics dependencies (only the standard
library and NumPy) so that the logic can be unit-tested locally with fake data and then
reused unchanged inside the Colab notebook ``notebooks/S02_C3_yolov8n_sku110k_colab.ipynb``.

Dataset layout assumed (produced by the official Ultralytics ``SKU-110K.yaml`` loader)::

    <root>/
        train.txt  val.txt  test.txt      # one image path per line, e.g. ./images/train_0.jpg
        images/train_0.jpg ...
        labels/train_0.txt ...            # YOLO format: ``cls x_center y_center w h`` (normalised)

Main entry points:

- :func:`validate_sku110k_root`  - sanity-check the downloaded dataset.
- :func:`select_subset`          - deterministic, seed-based choice of N images.
- :func:`build_subset`           - copy the chosen images/labels into a YOLO training layout.
- :func:`write_data_yaml`        - write the ``data.yaml`` used by ``model.train``.
- :func:`yolo_to_xyxy`, :func:`detections_to_array`, :func:`filter_by_conf`,
  :func:`count_detections`, :func:`counts_for_thresholds`, :func:`count_label_boxes`
  - convert labels / predictions into what ``fusion.metrics`` expects.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

DEFAULT_SEED = 0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_LIST_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

__all__ = [
    "DEFAULT_SEED",
    "validate_sku110k_root",
    "read_image_list",
    "label_path_for_image",
    "read_yolo_labels",
    "count_label_boxes",
    "select_subset",
    "build_subset",
    "write_data_yaml",
    "verify_subset",
    "yolo_to_xyxy",
    "detections_to_array",
    "filter_by_conf",
    "count_detections",
    "counts_for_thresholds",
]


# =============================================================================
# 1. DATASET DISCOVERY / VALIDATION
# =============================================================================


def read_image_list(list_file: PathLike, root: PathLike) -> List[Path]:
    """
    Read an Ultralytics split list (``train.txt`` / ``val.txt``) into absolute image paths.

    Lines are interpreted relative to ``root`` (e.g. ``./images/train_0.jpg``); blank lines are
    skipped. Duplicates are removed while keeping the original order.
    """
    root_path = Path(root)
    list_path = Path(list_file)
    if not list_path.is_file():
        raise FileNotFoundError(f"Split list not found: {list_path}")

    seen = set()
    images: List[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = root_path / image_path
        if image_path not in seen:
            seen.add(image_path)
            images.append(image_path)
    return images


def label_path_for_image(image_path: PathLike) -> Path:
    """
    Return the YOLO label path of an image using the Ultralytics convention
    (``.../images/<name>.jpg`` -> ``.../labels/<name>.txt``).
    """
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path has no 'images' directory component: {image_path}")
    # Replace the LAST 'images' component, like ultralytics.data.utils.img2label_paths.
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_sku110k_root(root: PathLike, min_images: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Check that a SKU-110K dataset root looks complete and return ``{split: n_images}``.

    Raises
    ------
    FileNotFoundError
        If the root, ``images/``, ``labels/`` or the ``train.txt`` / ``val.txt`` lists are missing.
    ValueError
        If a split has fewer listed images than ``min_images[split]`` or the first listed image /
        label files are not on disk (typical symptom of an interrupted download).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"SKU-110K root does not exist: {root_path}")
    for sub in ("images", "labels"):
        if not (root_path / sub).is_dir():
            raise FileNotFoundError(f"Missing directory '{sub}/' in {root_path} (incomplete download?).")

    required = {"train": 1, "val": 1}
    if min_images:
        required.update(min_images)

    counts: Dict[str, int] = {}
    for split, minimum in required.items():
        list_file = root_path / SPLIT_LIST_FILES[split]
        images = read_image_list(list_file, root_path)
        counts[split] = len(images)
        if len(images) < minimum:
            raise ValueError(f"Split '{split}' lists {len(images)} images, expected at least {minimum}.")
        probe = images[0]
        if not probe.is_file():
            raise ValueError(f"Image listed but missing on disk: {probe} (incomplete extraction?).")
        if not label_path_for_image(probe).is_file():
            raise ValueError(f"Label missing for {probe} (label conversion did not finish?).")
    return counts


# =============================================================================
# 2. LABEL READING / COUNTING
# =============================================================================


def read_yolo_labels(label_path: PathLike) -> np.ndarray:
    """
    Read a YOLO label file into an ``(N, 5)`` float array ``[cls, xc, yc, w, h]`` (normalised).

    A missing or empty file yields an empty ``(0, 5)`` array. Malformed lines (fewer than five
    columns) raise ``ValueError`` so corrupted labels are not silently counted.
    """
    path = Path(label_path)
    if not path.is_file():
        return np.zeros((0, 5), dtype=np.float64)

    rows: List[List[float]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}:{line_no}: expected >= 5 columns, got {len(parts)}.")
        rows.append([float(v) for v in parts[:5]])
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def count_label_boxes(label_path: PathLike) -> int:
    """Number of ground-truth boxes (= ground-truth item count) in one YOLO label file."""
    return int(read_yolo_labels(label_path).shape[0])


# =============================================================================
# 3. DETERMINISTIC SUBSET SELECTION
# =============================================================================


def select_subset(candidates: Iterable[PathLike], n: int, seed: int = DEFAULT_SEED) -> List[Path]:
    """
    Choose ``n`` items from ``candidates`` reproducibly.

    The candidates are de-duplicated and sorted first, so the result depends only on the *set*
    of candidates, ``n`` and ``seed`` - not on the order in which the files were listed.
    The returned list is sorted.

    Raises
    ------
    ValueError
        If ``n`` is negative or larger than the number of candidates.
    """
    unique = sorted({Path(c) for c in candidates})
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")
    if n > len(unique):
        raise ValueError(f"Cannot select {n} items from only {len(unique)} candidates.")
    rng = random.Random(seed)
    return sorted(rng.sample(unique, n))


def _usable_images(images: Sequence[Path]) -> List[Path]:
    """Keep images that exist on disk and have a non-empty label file."""
    usable = []
    for image_path in images:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            continue
        if count_label_boxes(label_path_for_image(image_path)) > 0:
            usable.append(image_path)
    return usable


def build_subset(
    root: PathLike,
    out_dir: PathLike,
    n_train: int = 400,
    n_val: int = 100,
    seed: int = DEFAULT_SEED,
    overwrite: bool = True,
) -> Dict[str, List[Path]]:
    """
    Copy a fixed random subset of SKU-110K into ``out_dir`` in standard YOLO layout::

        out_dir/images/{train,val}/<name>.jpg
        out_dir/labels/{train,val}/<name>.txt

    The subset is sampled from the *official* ``train.txt`` / ``val.txt`` splits (so train and val
    stay disjoint) using ``seed``. Only images that exist and have at least one labelled box are
    eligible. Re-running with identical arguments reproduces the identical subset.

    Returns
    -------
    Dict[str, List[Path]]
        ``{"train": [...source image paths...], "val": [...]}`` (sorted, source locations).
    """
    root_path = Path(root)
    out_path = Path(out_dir)
    sizes = {"train": n_train, "val": n_val}

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_path}")
        shutil.rmtree(out_path)

    chosen: Dict[str, List[Path]] = {}
    for split, n in sizes.items():
        images = read_image_list(root_path / SPLIT_LIST_FILES[split], root_path)
        usable = _usable_images(images)
        if len(usable) < n:
            raise ValueError(
                f"Split '{split}' has only {len(usable)} usable images (requested {n}). "
                "Is the dataset download complete?"
            )
        # Use a split-specific stream so changing n_val never alters the train subset.
        split_seed = seed if split == "train" else seed + 1
        chosen[split] = select_subset(usable, n, seed=split_seed)

    for split, images in chosen.items():
        img_dir = out_path / "images" / split
        lbl_dir = out_path / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            shutil.copy2(image_path, img_dir / image_path.name)
            shutil.copy2(label_path_for_image(image_path), lbl_dir / f"{image_path.stem}.txt")
    return chosen


def write_data_yaml(out_dir: PathLike, names: Optional[Dict[int, str]] = None) -> Path:
    """Write ``out_dir/data.yaml`` (absolute ``path``) for Ultralytics training and return it."""
    out_path = Path(out_dir).resolve()
    class_names = names or {0: "object"}
    lines = [
        f"path: {out_path.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {int(idx)}: {name}" for idx, name in sorted(class_names.items())]
    yaml_path = out_path / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def verify_subset(out_dir: PathLike, n_train: int, n_val: int) -> Dict[str, int]:
    """
    Verify a built subset: expected file counts, one label per image, train/val disjoint.

    Returns ``{"train": n, "val": n}``; raises ``ValueError`` on any inconsistency.
    """
    out_path = Path(out_dir)
    stems: Dict[str, set] = {}
    for split, expected in (("train", n_train), ("val", n_val)):
        images = sorted(p for p in (out_path / "images" / split).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted((out_path / "labels" / split).glob("*.txt"))
        if len(images) != expected:
            raise ValueError(f"Split '{split}': {len(images)} images on disk, expected {expected}.")
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f"Split '{split}': image and label file names do not match one-to-one.")
        stems[split] = {p.stem for p in images}
    overlap = stems["train"] & stems["val"]
    if overlap:
        raise ValueError(f"{len(overlap)} images appear in both train and val (e.g. {sorted(overlap)[0]}).")
    return {"train": n_train, "val": n_val}


# =============================================================================
# 4. CONVERSIONS FOR fusion.metrics
# =============================================================================


def yolo_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert normalised YOLO boxes ``[cls, xc, yc, w, h]`` to pixel ``[x1, y1, x2, y2]``.

    Returns an ``(N, 4)`` array (``(0, 4)`` for empty input).
    """
    arr = np.asarray(labels, dtype=np.float64).reshape(-1, 5)
    if arr.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float64)
    xc, yc = arr[:, 1] * width, arr[:, 2] * height
    half_w, half_h = arr[:, 3] * width / 2.0, arr[:, 4] * height / 2.0
    return np.stack([xc - half_w, yc - half_h, xc + half_w, yc + half_h], axis=1)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor / array-like to NumPy without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def detections_to_array(boxes_xyxy: Any, confs: Any, cls_ids: Optional[Any] = None) -> np.ndarray:
    """
    Stack prediction tensors into the ``(N, 6)`` array ``[x1, y1, x2, y2, conf, cls]`` accepted by
    ``fusion.metrics.map_wrapper``. Accepts torch tensors or array-likes; empty input -> ``(0, 6)``.
    """
    boxes = _to_numpy(boxes_xyxy).reshape(-1, 4)
    scores = _to_numpy(confs).reshape(-1)
    classes = np.zeros_like(scores) if cls_ids is None else _to_numpy(cls_ids).reshape(-1)
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError(
            f"Inconsistent detection lengths: boxes={len(boxes)}, confs={len(scores)}, cls={len(classes)}."
        )
    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float64)
    return np.column_stack([boxes, scores, classes])


def filter_by_conf(detections: np.ndarray, conf_threshold: float) -> np.ndarray:
    """Keep detections (``(N, 6)`` array) whose confidence is >= ``conf_threshold``."""
    arr = np.asarray(detections, dtype=np.float64).reshape(-1, 6)
    return arr[arr[:, 4] >= conf_threshold]


def count_detections(detections: np.ndarray, conf_threshold: float) -> int:
    """Predicted item count of one image = number of detections with conf >= ``conf_threshold``."""
    return int(filter_by_conf(detections, conf_threshold).shape[0])


def counts_for_thresholds(
    per_image_detections: Sequence[np.ndarray],
    conf_thresholds: Sequence[float],
) -> Dict[float, Tuple[int, ...]]:
    """Predicted counts of every image for each confidence threshold."""
    return {
        float(thr): tuple(count_detections(det, thr) for det in per_image_detections)
        for thr in conf_thresholds
    }
