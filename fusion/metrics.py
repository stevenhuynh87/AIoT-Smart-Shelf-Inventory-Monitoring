"""
Evaluation Metrics Module for AIoT Smart Shelf Inventory Monitoring (S02-C1).

Provides standardized metrics across three evaluation tiers:
1. Inventory Counting Metrics:
   - Counting MAE (Mean Absolute Error)
   - Counting RMSE (Root Mean Square Error)
   - Exact Count Accuracy, Undercount Rate, Overcount Rate
2. Object Detection Metrics (mAP Wrapper):
   - IoU computation (xyxy and xywh formats)
   - Average Precision (COCO 101-point and continuous interpolation)
   - Standalone NumPy mAP@0.5 / mAP@0.5:0.95 calculator & Ultralytics results extractor
3. Event-Level Classification Metrics:
   - Precision, Recall, F1-Score, False Alarm Rate (FAR), TP/FP/FN/TN
   - Supports both binary shelf alerts (e.g., Disagreement, Low-Stock, Stock-Out)
     and multi-class shelf events (e.g., TAKE, RESTOCK, IDLE or conflict modes)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

Number = Union[int, float, np.number]
ArrayLike = Union[Sequence[Any], np.ndarray, pd.Series]

__all__ = [
    "counting_mae",
    "counting_rmse",
    "exact_accuracy",
    "undercount_rate",
    "overcount_rate",
    "evaluate_counting",
    "compute_iou",
    "compute_iou_matrix",
    "compute_ap",
    "map_wrapper",
    "event_precision_recall_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "derive_inventory_events",
]


# =============================================================================
# 1. INVENTORY COUNTING METRICS
# =============================================================================


def _prepare_count_arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert ground-truth and predicted count sequences into 1D float arrays."""
    true_list = list(y_true) if not isinstance(y_true, np.ndarray) else y_true.tolist()
    pred_list = list(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred.tolist()

    if len(true_list) != len(pred_list):
        raise ValueError(
            f"Length mismatch between y_true ({len(true_list)}) and y_pred ({len(pred_list)})."
        )

    if len(true_list) == 0:
        raise ValueError("Input arrays y_true and y_pred cannot be empty.")

    valid_true: List[float] = []
    valid_pred: List[float] = []

    for t, p in zip(true_list, pred_list):
        t_is_na = t is None or (isinstance(t, float) and math.isnan(t)) or pd.isna(t)
        p_is_na = p is None or (isinstance(p, float) and math.isnan(p)) or pd.isna(p)

        if t_is_na or p_is_na:
            if dropna:
                continue
            raise ValueError("Input contains None/NaN values while dropna=False.")

        valid_true.append(float(t))
        valid_pred.append(float(p))

    if len(valid_true) == 0:
        raise ValueError("No valid (non-NaN) pairs remaining after filtering.")

    return np.asarray(valid_true, dtype=np.float64), np.asarray(valid_pred, dtype=np.float64)


def counting_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    dropna: bool = True,
) -> float:
    """
    Calculate Mean Absolute Error (MAE) for item counting:
        MAE = (1 / N) * sum(|y_pred_i - y_true_i|)

    Parameters
    ----------
    y_true : ArrayLike
        Ground-truth item counts on the shelf.
    y_pred : ArrayLike
        Predicted item counts (from Vision, Weight, or Sensor Fusion).
    dropna : bool, default=True
        If True, pairs containing None or NaN are ignored.

    Returns
    -------
    float
        Mean Absolute Error in item units (>= 0.0).
    """
    arr_true, arr_pred = _prepare_count_arrays(y_true, y_pred, dropna=dropna)
    return float(np.mean(np.abs(arr_pred - arr_true)))


def counting_rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    dropna: bool = True,
) -> float:
    """
    Calculate Root Mean Square Error (RMSE) for item counting:
        RMSE = sqrt((1 / N) * sum((y_pred_i - y_true_i)^2))
    """
    arr_true, arr_pred = _prepare_count_arrays(y_true, y_pred, dropna=dropna)
    return float(np.sqrt(np.mean(np.square(arr_pred - arr_true))))


def exact_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    as_percentage: bool = False,
    dropna: bool = True,
    tolerance: float = 1e-6,
) -> float:
    """
    Calculate Exact Count Match Accuracy:
        Accuracy = (1 / N) * sum(I(|y_pred_i - y_true_i| <= tolerance))

    Parameters
    ----------
    y_true : ArrayLike
        Ground-truth item counts.
    y_pred : ArrayLike
        Predicted item counts.
    as_percentage : bool, default=False
        If False, returns ratio in [0.0, 1.0]. If True, returns percentage in [0.0, 100.0].
    dropna : bool, default=True
        If True, pairs containing None or NaN are ignored.
    tolerance : float, default=1e-6
        Numerical tolerance for equality check.

    Returns
    -------
    float
        Exact count accuracy.
    """
    arr_true, arr_pred = _prepare_count_arrays(y_true, y_pred, dropna=dropna)
    matches = np.abs(arr_pred - arr_true) <= tolerance
    acc = float(np.mean(matches))
    return acc * 100.0 if as_percentage else acc


def undercount_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    as_percentage: bool = False,
    dropna: bool = True,
    tolerance: float = 1e-6,
) -> float:
    """
    Calculate the fraction of samples where predicted count is strictly less than ground truth
    (y_pred < y_true - tolerance), typically caused by visual occlusion or missed detections.
    """
    arr_true, arr_pred = _prepare_count_arrays(y_true, y_pred, dropna=dropna)
    rate = float(np.mean((arr_pred - arr_true) < -tolerance))
    return rate * 100.0 if as_percentage else rate


def overcount_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    as_percentage: bool = False,
    dropna: bool = True,
    tolerance: float = 1e-6,
) -> float:
    """
    Calculate the fraction of samples where predicted count is strictly greater than ground truth
    (y_pred > y_true + tolerance), typically caused by false positive detections or external forces.
    """
    arr_true, arr_pred = _prepare_count_arrays(y_true, y_pred, dropna=dropna)
    rate = float(np.mean((arr_pred - arr_true) > tolerance))
    return rate * 100.0 if as_percentage else rate


def evaluate_counting(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    as_percentage: bool = False,
    dropna: bool = True,
    tolerance: float = 1e-6,
) -> Dict[str, Union[float, int]]:
    """
    Compute all inventory counting metrics in a single pass.

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary with keys: 'mae', 'rmse', 'exact_accuracy', 'undercount_rate',
        'overcount_rate', 'total_samples', 'exact_matches', 'undercounts', 'overcounts'.
    """
    arr_true, arr_pred = _prepare_count_arrays(y_true, y_pred, dropna=dropna)
    diffs = arr_pred - arr_true
    abs_diffs = np.abs(diffs)

    total = int(len(arr_true))
    exact_mask = abs_diffs <= tolerance
    under_mask = diffs < -tolerance
    over_mask = diffs > tolerance

    exact_cnt = int(np.sum(exact_mask))
    under_cnt = int(np.sum(under_mask))
    over_cnt = int(np.sum(over_mask))

    scale = 100.0 if as_percentage else 1.0

    return {
        "mae": float(np.mean(abs_diffs)),
        "rmse": float(np.sqrt(np.mean(np.square(diffs)))),
        "exact_accuracy": float(exact_cnt / total) * scale,
        "undercount_rate": float(under_cnt / total) * scale,
        "overcount_rate": float(over_cnt / total) * scale,
        "total_samples": total,
        "exact_matches": exact_cnt,
        "undercounts": under_cnt,
        "overcounts": over_cnt,
    }


# =============================================================================
# 2. OBJECT DETECTION METRICS (IoU, AP, & mAP WRAPPER)
# =============================================================================


def _to_xyxy(boxes: np.ndarray, box_format: str = "xyxy") -> np.ndarray:
    """Convert bounding boxes of shape (..., 4) to [x1, y1, x2, y2] format."""
    arr = np.asarray(boxes, dtype=np.float64)
    fmt = box_format.strip().lower()
    if fmt == "xyxy":
        return arr
    if fmt == "xywh":
        out = np.empty_like(arr, dtype=np.float64)
        half_w = arr[..., 2] / 2.0
        half_h = arr[..., 3] / 2.0
        out[..., 0] = arr[..., 0] - half_w
        out[..., 1] = arr[..., 1] - half_h
        out[..., 2] = arr[..., 0] + half_w
        out[..., 3] = arr[..., 1] + half_h
        return out
    raise ValueError(f"Unsupported box_format '{box_format}'. Expected 'xyxy' or 'xywh'.")


def compute_iou(
    box1: Sequence[Number],
    box2: Sequence[Number],
    box_format: str = "xyxy",
) -> float:
    """
    Compute Intersection over Union (IoU) between two 2D bounding boxes.

    Parameters
    ----------
    box1, box2 : Sequence[Number]
        4-element bounding box coordinates.
    box_format : str, default='xyxy'
        Coordinate format: 'xyxy' ([x1, y1, x2, y2]) or 'xywh' ([x_center, y_center, w, h]).

    Returns
    -------
    float
        IoU value in [0.0, 1.0].
    """
    b1 = _to_xyxy(np.asarray(box1, dtype=np.float64), box_format=box_format)
    b2 = _to_xyxy(np.asarray(box2, dtype=np.float64), box_format=box_format)
    if b1.shape != (4,) or b2.shape != (4,):
        raise ValueError("Each bounding box must contain exactly 4 coordinates.")

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(
    boxes1: Union[Sequence[Sequence[Number]], np.ndarray],
    boxes2: Union[Sequence[Sequence[Number]], np.ndarray],
    box_format: str = "xyxy",
) -> np.ndarray:
    """
    Compute pairwise IoU matrix of shape (N, M) between two sets of bounding boxes.
    """
    arr1 = np.asarray(boxes1, dtype=np.float64)
    arr2 = np.asarray(boxes2, dtype=np.float64)

    if arr1.size == 0 or arr2.size == 0:
        n = 0 if arr1.size == 0 else arr1.shape[0]
        m = 0 if arr2.size == 0 else arr2.shape[0]
        return np.zeros((n, m), dtype=np.float64)

    b1 = _to_xyxy(arr1.reshape(-1, 4), box_format=box_format)
    b2 = _to_xyxy(arr2.reshape(-1, 4), box_format=box_format)

    inter_x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area

    return np.where(union_area > 0.0, inter_area / union_area, 0.0)


def compute_ap(
    recalls: ArrayLike,
    precisions: ArrayLike,
    method: str = "coco101",
) -> float:
    """
    Compute Average Precision (AP) from recall and precision curves.

    Parameters
    ----------
    recalls : ArrayLike
        Monotonically non-decreasing recall values in [0, 1].
    precisions : ArrayLike
        Corresponding precision values in [0, 1].
    method : str, default='coco101'
        Interpolation method:
        - 'coco101': 101-point interpolation used by MS-COCO and Ultralytics YOLO.
        - 'continuous': Exact area under the precision-recall envelope (VOC2010+).
    """
    rec = np.asarray(recalls, dtype=np.float64).ravel()
    prec = np.asarray(precisions, dtype=np.float64).ravel()

    if rec.size == 0 or prec.size == 0:
        return 0.0

    # Append sentinel values at ends: (0.0, 1.0) and (1.0, 0.0)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))

    # Compute upper precision envelope (monotonically decreasing from left to right)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    mode = method.strip().lower()
    if mode == "coco101":
        recall_thresholds = np.linspace(0.0, 1.0, 101)
        # For each threshold r, find first index where mrec >= r
        indices = np.searchsorted(mrec, recall_thresholds, side="left")
        ap_vals = np.zeros_like(recall_thresholds)
        valid = indices < len(mpre)
        ap_vals[valid] = mpre[indices[valid]]
        return float(np.mean(ap_vals))
    elif mode == "continuous":
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    else:
        raise ValueError(f"Unsupported AP interpolation method '{method}'.")


def _normalize_image_detections(
    item: Any,
    is_prediction: bool = True,
) -> List[Dict[str, Any]]:
    """
    Normalize per-image detection records into a uniform list of dicts:
        [{"bbox": [x1, y1, x2, y2], "conf": float, "cls_id": int}, ...]
    """
    if item is None:
        return []

    # Case 1: List of dicts [{"bbox": ..., "conf": ..., "cls_id": ...}, ...]
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return []
        if isinstance(item[0], dict):
            normalized = []
            for d in item:
                bbox = d.get("bbox", d.get("box"))
                if bbox is None:
                    raise ValueError("Detection dict must contain 'bbox' or 'box' key.")
                cls_id = int(d.get("cls_id", d.get("label", d.get("cls", 0))))
                conf = float(d.get("conf", d.get("score", 1.0))) if is_prediction else 1.0
                normalized.append({"bbox": list(bbox), "conf": conf, "cls_id": cls_id})
            return normalized

    # Case 2: Dict of arrays {"boxes": ..., "scores": ..., "labels": ...}
    if isinstance(item, dict):
        boxes = item.get("boxes", item.get("bboxes", []))
        labels = item.get("labels", item.get("cls_ids", [0] * len(boxes)))
        if is_prediction:
            scores = item.get("scores", item.get("confs", [1.0] * len(boxes)))
        else:
            scores = [1.0] * len(boxes)

        return [
            {"bbox": list(b), "conf": float(s), "cls_id": int(c)}
            for b, s, c in zip(boxes, scores, labels)
        ]

    # Case 3: NumPy 2D array
    arr = np.asarray(item, dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.ndim != 2:
        raise ValueError("Detection array must be 2D.")

    normalized = []
    for row in arr:
        if is_prediction:
            if len(row) >= 6:
                normalized.append(
                    {"bbox": row[:4].tolist(), "conf": float(row[4]), "cls_id": int(row[5])}
                )
            elif len(row) == 5:
                normalized.append(
                    {"bbox": row[:4].tolist(), "conf": float(row[4]), "cls_id": 0}
                )
            else:
                raise ValueError("Prediction array rows must have 5 ([x1,y1,x2,y2,conf]) or 6 columns.")
        else:
            if len(row) >= 5:
                normalized.append(
                    {"bbox": row[:4].tolist(), "conf": 1.0, "cls_id": int(row[4])}
                )
            elif len(row) == 4:
                normalized.append(
                    {"bbox": row[:4].tolist(), "conf": 1.0, "cls_id": 0}
                )
            else:
                raise ValueError("Ground-truth array rows must have 4 or 5 columns.")
    return normalized


def _extract_from_ultralytics(ultralytics_obj: Any) -> Dict[str, Any]:
    """Extract standardized mAP metrics from an Ultralytics DetMetrics or validation result object."""
    # Support object with .box attribute (standard ultralytics DetMetrics)
    if hasattr(ultralytics_obj, "box"):
        box_metrics = ultralytics_obj.box
        map50 = float(getattr(box_metrics, "map50", 0.0))
        map75 = float(getattr(box_metrics, "map75", 0.0))
        map50_95 = float(getattr(box_metrics, "map", 0.0))
        mp = float(getattr(box_metrics, "mp", 0.0))
        mr = float(getattr(box_metrics, "mr", 0.0))
        ap50_arr = getattr(box_metrics, "ap50", [])
        ap_arr = getattr(box_metrics, "ap", [])
        per_class_ap50 = {int(i): float(v) for i, v in enumerate(ap50_arr)} if ap50_arr is not None else {}
        per_class_ap50_95 = {int(i): float(v) for i, v in enumerate(ap_arr)} if ap_arr is not None else {}
        return {
            "map50": map50,
            "map75": map75,
            "map50_95": map50_95,
            "precision": mp,
            "recall": mr,
            "per_class_ap50": per_class_ap50,
            "per_class_ap50_95": per_class_ap50_95,
            "source": "ultralytics",
        }

    # Support dict or results_dict from Ultralytics
    res_dict = (
        ultralytics_obj.results_dict
        if hasattr(ultralytics_obj, "results_dict")
        else (ultralytics_obj if isinstance(ultralytics_obj, dict) else None)
    )
    if isinstance(res_dict, dict):
        map50 = float(
            res_dict.get("metrics/mAP50(B)", res_dict.get("map50", res_dict.get("mAP50", 0.0)))
        )
        map50_95 = float(
            res_dict.get(
                "metrics/mAP50-95(B)", res_dict.get("map50_95", res_dict.get("mAP50-95", 0.0))
            )
        )
        mp = float(res_dict.get("metrics/precision(B)", res_dict.get("precision", 0.0)))
        mr = float(res_dict.get("metrics/recall(B)", res_dict.get("recall", 0.0)))
        return {
            "map50": map50,
            "map75": float(res_dict.get("map75", 0.0)),
            "map50_95": map50_95,
            "precision": mp,
            "recall": mr,
            "per_class_ap50": {},
            "per_class_ap50_95": {},
            "source": "ultralytics_dict",
        }

    raise TypeError(
        "Unsupported ultralytics_results object. Expected Ultralytics DetMetrics (.box) "
        "or a dictionary containing 'metrics/mAP50(B)' / 'map50'."
    )


def map_wrapper(
    predictions: Optional[Sequence[Any]] = None,
    ground_truths: Optional[Sequence[Any]] = None,
    ultralytics_results: Optional[Any] = None,
    iou_thresholds: Optional[Sequence[float]] = None,
    box_format: str = "xyxy",
    interpolation: str = "coco101",
) -> Dict[str, Any]:
    """
    Unified mAP Wrapper supporting both Ultralytics validation objects and standalone
    NumPy bounding-box evaluation.

    Parameters
    ----------
    predictions : Sequence[Any], optional
        Per-image predicted detections, or an Ultralytics validation result object if
        `ground_truths` is omitted.
    ground_truths : Sequence[Any], optional
        Per-image ground-truth annotations matching the length of `predictions`.
    ultralytics_results : Any, optional
        Ultralytics `model.val()` result object (`DetMetrics`) or `results_dict`.
    iou_thresholds : Sequence[float], optional
        IoU thresholds to evaluate. Defaults to `np.linspace(0.50, 0.95, 10)`.
    box_format : str, default='xyxy'
        Bounding box coordinate format ('xyxy' or 'xywh').
    interpolation : str, default='coco101'
        AP curve interpolation method ('coco101' or 'continuous').

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'map50': float (mAP at IoU=0.50)
        - 'map75': float (mAP at IoU=0.75)
        - 'map50_95': float (mean AP across IoU=0.50:0.05:0.95)
        - 'per_class_ap50': Dict[int, float]
        - 'per_class_ap50_95': Dict[int, float]
        - 'iou_thresholds': List[float]
        - 'num_classes': int
    """
    if ultralytics_results is not None:
        return _extract_from_ultralytics(ultralytics_results)

    if predictions is not None and ground_truths is None and (
        hasattr(predictions, "box") or hasattr(predictions, "results_dict")
    ):
        return _extract_from_ultralytics(predictions)

    if predictions is None or ground_truths is None:
        raise ValueError(
            "Provide either `ultralytics_results` OR both `predictions` and `ground_truths`."
        )

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch between predictions ({len(predictions)}) and ground_truths ({len(ground_truths)})."
        )

    if iou_thresholds is None:
        thresholds = [round(float(t), 2) for t in np.linspace(0.50, 0.95, 10)]
    else:
        thresholds = [round(float(t), 4) for t in iou_thresholds]
        if len(thresholds) == 0:
            raise ValueError("iou_thresholds cannot be empty.")

    norm_preds = [_normalize_image_detections(p, is_prediction=True) for p in predictions]
    norm_gts = [_normalize_image_detections(g, is_prediction=False) for g in ground_truths]

    # Collect all classes present in ground truths (and predictions if no GT exists)
    gt_classes = {d["cls_id"] for img_gts in norm_gts for d in img_gts}
    pred_classes = {d["cls_id"] for img_preds in norm_preds for d in img_preds}
    eval_classes = sorted(gt_classes) if gt_classes else sorted(pred_classes)

    if not eval_classes:
        return {
            "map50": 0.0,
            "map75": 0.0,
            "map50_95": 0.0,
            "per_class_ap50": {},
            "per_class_ap50_95": {},
            "iou_thresholds": thresholds,
            "num_classes": 0,
            "source": "standalone_numpy",
        }

    # Evaluate AP(cls, thr)
    ap_matrix = np.zeros((len(eval_classes), len(thresholds)), dtype=np.float64)

    for c_idx, cls_id in enumerate(eval_classes):
        # Gather all GT boxes per image for this class
        gt_by_img: Dict[int, np.ndarray] = {}
        total_gt = 0
        for img_idx, img_gts in enumerate(norm_gts):
            cls_boxes = [d["bbox"] for d in img_gts if d["cls_id"] == cls_id]
            if cls_boxes:
                gt_by_img[img_idx] = np.asarray(cls_boxes, dtype=np.float64)
                total_gt += len(cls_boxes)
            else:
                gt_by_img[img_idx] = np.zeros((0, 4), dtype=np.float64)

        # Gather all predictions for this class across all images
        cls_preds: List[Tuple[int, float, List[float]]] = []
        for img_idx, img_preds in enumerate(norm_preds):
            for d in img_preds:
                if d["cls_id"] == cls_id:
                    cls_preds.append((img_idx, float(d["conf"]), d["bbox"]))

        # Sort predictions by confidence descending
        cls_preds.sort(key=lambda x: x[1], reverse=True)

        if total_gt == 0 or len(cls_preds) == 0:
            ap_matrix[c_idx, :] = 0.0
            continue

        for t_idx, thr in enumerate(thresholds):
            tp = np.zeros(len(cls_preds), dtype=np.float64)
            fp = np.zeros(len(cls_preds), dtype=np.float64)
            matched_gt: Dict[int, np.ndarray] = {
                img_idx: np.zeros(len(boxes), dtype=bool)
                for img_idx, boxes in gt_by_img.items()
            }

            for p_idx, (img_idx, _conf, p_box) in enumerate(cls_preds):
                gt_boxes = gt_by_img[img_idx]
                if len(gt_boxes) == 0:
                    fp[p_idx] = 1.0
                    continue

                ious = compute_iou_matrix([p_box], gt_boxes, box_format=box_format)[0]
                # Find best unmatched GT box if any, or standard COCO greedy match (highest IoU overall, if >= thr and unmatched)
                best_gt_idx = int(np.argmax(ious))
                best_iou = float(ious[best_gt_idx])

                if best_iou >= thr and not matched_gt[img_idx][best_gt_idx]:
                    tp[p_idx] = 1.0
                    matched_gt[img_idx][best_gt_idx] = True
                else:
                    # Check if another unmatched GT also exceeds threshold with next highest IoU
                    unmatched_indices = np.where((ious >= thr) & (~matched_gt[img_idx]))[0]
                    if len(unmatched_indices) > 0:
                        best_unmatched = unmatched_indices[int(np.argmax(ious[unmatched_indices]))]
                        tp[p_idx] = 1.0
                        matched_gt[img_idx][best_unmatched] = True
                    else:
                        fp[p_idx] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / float(total_gt)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

            ap_matrix[c_idx, t_idx] = compute_ap(recalls, precisions, method=interpolation)

    # Identify index for 0.50 and 0.75 if present
    idx_50 = int(np.argmin(np.abs(np.asarray(thresholds) - 0.50)))
    idx_75 = int(np.argmin(np.abs(np.asarray(thresholds) - 0.75)))

    per_class_ap50 = {int(c): float(ap_matrix[i, idx_50]) for i, c in enumerate(eval_classes)}
    per_class_ap50_95 = {int(c): float(np.mean(ap_matrix[i, :])) for i, c in enumerate(eval_classes)}

    return {
        "map50": float(np.mean(ap_matrix[:, idx_50])),
        "map75": float(np.mean(ap_matrix[:, idx_75])),
        "map50_95": float(np.mean(ap_matrix)),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap50_95,
        "iou_thresholds": thresholds,
        "num_classes": len(eval_classes),
        "source": "standalone_numpy",
    }


# =============================================================================
# 3. EVENT-LEVEL PRECISION / RECALL / F1 METRICS
# =============================================================================


def _safe_div(num: float, den: float, zero_division: float = 0.0) -> float:
    """Safely divide two numbers, returning `zero_division` when denominator is 0."""
    return float(num / den) if den > 0.0 else float(zero_division)


def _compute_binary_counts(
    y_true_list: List[Any],
    y_pred_list: List[Any],
    positive_label: Any,
) -> Tuple[int, int, int, int]:
    """Compute (TP, FP, FN, TN) for a target positive_label."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true_list, y_pred_list):
        t_pos = t == positive_label
        p_pos = p == positive_label
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def event_precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label: Any = 1,
    labels: Optional[Sequence[Any]] = None,
    zero_division: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute Precision, Recall, F1-Score, False Alarm Rate (FAR), and Confusion Counts
    for smart shelf events (e.g., Disagreement alerts, Low-Stock/Stock-Out alerts, or
    TAKE/RESTOCK/IDLE transitions).

    Parameters
    ----------
    y_true : ArrayLike
        Ground-truth event labels (e.g., [0, 1, 1, 0] or ['IDLE', 'TAKE', 'RESTOCK']).
    y_pred : ArrayLike
        Predicted event labels matching the length of `y_true`.
    average : str, default='binary'
        Averaging mode:
        - 'binary': Report metrics for `positive_label`.
        - 'macro': Unweighted mean of per-class Precision, Recall, and F1.
        - 'micro': Global calculation by summing total TP, FP, FN across all classes.
        - 'weighted': Support-weighted mean of per-class Precision, Recall, and F1.
    positive_label : Any, default=1
        Target positive class when `average='binary'` (e.g., 1, True, 'DISAGREEMENT', 'TAKE').
    labels : Sequence[Any], optional
        Explicit list of class labels to include when `average` is 'macro', 'micro', or 'weighted'.
    zero_division : float, default=0.0
        Value to return when a denominator (TP+FP or TP+FN) is zero.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'precision', 'recall', 'f1', 'accuracy', 'support', and
        either binary counts ('tp', 'fp', 'fn', 'tn', 'false_alarm_rate') or
        'per_class' breakdown.
    """
    true_list = list(y_true) if not isinstance(y_true, np.ndarray) else y_true.tolist()
    pred_list = list(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred.tolist()

    if len(true_list) != len(pred_list):
        raise ValueError(
            f"Length mismatch between y_true ({len(true_list)}) and y_pred ({len(pred_list)})."
        )
    if len(true_list) == 0:
        raise ValueError("Input event sequences y_true and y_pred cannot be empty.")

    mode = average.strip().lower()
    total_samples = len(true_list)
    overall_acc = float(sum(1 for t, p in zip(true_list, pred_list) if t == p) / total_samples)

    if mode == "binary":
        # Auto-adapt boolean True when positive_label=1 is used with bool inputs
        pos_lbl = positive_label
        tp, fp, fn, tn = _compute_binary_counts(true_list, pred_list, pos_lbl)
        prec = _safe_div(tp, tp + fp, zero_division=zero_division)
        rec = _safe_div(tp, tp + fn, zero_division=zero_division)
        f1 = _safe_div(2.0 * prec * rec, prec + rec, zero_division=zero_division)
        far = _safe_div(fp, fp + tn, zero_division=zero_division)

        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": overall_acc,
            "false_alarm_rate": far,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": tp + fn,
            "total_samples": total_samples,
            "average": "binary",
            "positive_label": pos_lbl,
        }

    if mode not in ("macro", "micro", "weighted"):
        raise ValueError(
            f"Unsupported average='{average}'. Expected 'binary', 'macro', 'micro', or 'weighted'."
        )

    if labels is not None:
        class_labels = list(dict.fromkeys(labels))
    else:
        # Preserve deterministic order of appearance across y_true then y_pred
        seen = list(dict.fromkeys(true_list + pred_list))
        try:
            class_labels = sorted(seen)
        except TypeError:
            class_labels = seen

    per_class: Dict[Any, Dict[str, Union[float, int]]] = {}
    total_tp = total_fp = total_fn = 0

    for lbl in class_labels:
        tp, fp, fn, tn = _compute_binary_counts(true_list, pred_list, lbl)
        prec = _safe_div(tp, tp + fp, zero_division=zero_division)
        rec = _safe_div(tp, tp + fn, zero_division=zero_division)
        f1 = _safe_div(2.0 * prec * rec, prec + rec, zero_division=zero_division)
        far = _safe_div(fp, fp + tn, zero_division=zero_division)
        support = tp + fn

        per_class[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_alarm_rate": far,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if mode == "micro":
        agg_prec = _safe_div(total_tp, total_tp + total_fp, zero_division=zero_division)
        agg_rec = _safe_div(total_tp, total_tp + total_fn, zero_division=zero_division)
        agg_f1 = _safe_div(2.0 * agg_prec * agg_rec, agg_prec + agg_rec, zero_division=zero_division)
    elif mode == "macro":
        n_cls = len(class_labels)
        agg_prec = float(sum(per_class[c]["precision"] for c in class_labels) / n_cls) if n_cls else 0.0
        agg_rec = float(sum(per_class[c]["recall"] for c in class_labels) / n_cls) if n_cls else 0.0
        agg_f1 = float(sum(per_class[c]["f1"] for c in class_labels) / n_cls) if n_cls else 0.0
    else:  # weighted
        total_sup = sum(int(per_class[c]["support"]) for c in class_labels)
        if total_sup == 0:
            agg_prec = agg_rec = agg_f1 = float(zero_division)
        else:
            agg_prec = float(
                sum(float(per_class[c]["precision"]) * int(per_class[c]["support"]) for c in class_labels)
                / total_sup
            )
            agg_rec = float(
                sum(float(per_class[c]["recall"]) * int(per_class[c]["support"]) for c in class_labels)
                / total_sup
            )
            agg_f1 = float(
                sum(float(per_class[c]["f1"]) * int(per_class[c]["support"]) for c in class_labels)
                / total_sup
            )

    return {
        "precision": agg_prec,
        "recall": agg_rec,
        "f1": agg_f1,
        "accuracy": overall_acc,
        "support": sum(int(per_class[c]["support"]) for c in class_labels),
        "total_samples": total_samples,
        "average": mode,
        "per_class": per_class,
    }


def event_precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label: Any = 1,
    labels: Optional[Sequence[Any]] = None,
    zero_division: float = 0.0,
) -> float:
    """Convenience function returning only the Precision score for shelf events."""
    return float(
        event_precision_recall_f1(
            y_true,
            y_pred,
            average=average,
            positive_label=positive_label,
            labels=labels,
            zero_division=zero_division,
        )["precision"]
    )


def event_recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label: Any = 1,
    labels: Optional[Sequence[Any]] = None,
    zero_division: float = 0.0,
) -> float:
    """Convenience function returning only the Recall score for shelf events."""
    return float(
        event_precision_recall_f1(
            y_true,
            y_pred,
            average=average,
            positive_label=positive_label,
            labels=labels,
            zero_division=zero_division,
        )["recall"]
    )


def event_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    positive_label: Any = 1,
    labels: Optional[Sequence[Any]] = None,
    zero_division: float = 0.0,
) -> float:
    """Convenience function returning only the F1-Score for shelf events."""
    return float(
        event_precision_recall_f1(
            y_true,
            y_pred,
            average=average,
            positive_label=positive_label,
            labels=labels,
            zero_division=zero_division,
        )["f1"]
    )


def derive_inventory_events(
    counts_prev: ArrayLike,
    counts_curr: ArrayLike,
    delta_threshold: int = 0,
) -> List[str]:
    """
    Convert pairs of sequential item counts (N_{t-1}, N_t) into shelf transition events:
      - 'TAKE':    (N_t - N_{t-1}) < -delta_threshold
      - 'RESTOCK': (N_t - N_{t-1}) > +delta_threshold
      - 'IDLE':    |N_t - N_{t-1}| <= delta_threshold
    """
    arr_prev, arr_curr = _prepare_count_arrays(counts_prev, counts_curr, dropna=False)
    deltas = arr_curr - arr_prev
    events: List[str] = []
    for d in deltas:
        if d < -delta_threshold:
            events.append("TAKE")
        elif d > delta_threshold:
            events.append("RESTOCK")
        else:
            events.append("IDLE")
    return events

