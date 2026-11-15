"""
Unit tests for `fusion.metrics` and `vision.metrics` (Task S02-C1).

Covers:
1. Inventory Counting Metrics (MAE, RMSE, Exact Accuracy, Undercount/Overcount Rate)
2. Object Detection mAP Wrapper (IoU, AP, Standalone NumPy mAP, Ultralytics Extractor)
3. Event-level Classification Metrics (Precision, Recall, F1, FAR, Binary & Multi-class)
4. Edge cases (empty inputs, NaN filtering, zero-division, format conversions, re-exports)
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fusion
import vision
from fusion.metrics import (
    compute_ap,
    compute_iou,
    compute_iou_matrix,
    counting_mae,
    counting_rmse,
    derive_inventory_events,
    evaluate_counting,
    event_f1,
    event_precision,
    event_precision_recall_f1,
    event_recall,
    exact_accuracy,
    map_wrapper,
    overcount_rate,
    undercount_rate,
)
from vision import metrics as vision_metrics


# =============================================================================
# 1. COUNTING METRICS TESTS
# =============================================================================


class TestCountingMetrics:
    """Tests for MAE, RMSE, Exact Accuracy, Undercount Rate, and Overcount Rate."""

    def test_perfect_counts(self):
        y_true = [0, 2, 5, 8, 10]
        y_pred = [0, 2, 5, 8, 10]

        assert counting_mae(y_true, y_pred) == pytest.approx(0.0)
        assert counting_rmse(y_true, y_pred) == pytest.approx(0.0)
        assert exact_accuracy(y_true, y_pred) == pytest.approx(1.0)
        assert exact_accuracy(y_true, y_pred, as_percentage=True) == pytest.approx(100.0)
        assert undercount_rate(y_true, y_pred) == pytest.approx(0.0)
        assert overcount_rate(y_true, y_pred) == pytest.approx(0.0)

    def test_mixed_counting_errors(self):
        # Sample 1: exact (2 -> 2, err=0)
        # Sample 2: undercount due to occlusion (4 -> 3, err=-1)
        # Sample 3: overcount due to foreign object (5 -> 7, err=+2)
        # Sample 4: exact (3 -> 3, err=0)
        y_true = [2, 4, 5, 3]
        y_pred = [2, 3, 7, 3]

        # MAE = (0 + 1 + 2 + 0) / 4 = 0.75
        assert counting_mae(y_true, y_pred) == pytest.approx(0.75)
        # RMSE = sqrt((0 + 1 + 4 + 0) / 4) = sqrt(1.25)
        assert counting_rmse(y_true, y_pred) == pytest.approx(math.sqrt(1.25))
        # Exact accuracy = 2 / 4 = 0.50 (50%)
        assert exact_accuracy(y_true, y_pred) == pytest.approx(0.50)
        assert exact_accuracy(y_true, y_pred, as_percentage=True) == pytest.approx(50.0)
        # Undercount = 1 / 4 = 0.25, Overcount = 1 / 4 = 0.25
        assert undercount_rate(y_true, y_pred) == pytest.approx(0.25)
        assert overcount_rate(y_true, y_pred) == pytest.approx(0.25)

        summary = evaluate_counting(y_true, y_pred, as_percentage=True)
        assert summary["mae"] == pytest.approx(0.75)
        assert summary["rmse"] == pytest.approx(math.sqrt(1.25))
        assert summary["exact_accuracy"] == pytest.approx(50.0)
        assert summary["undercount_rate"] == pytest.approx(25.0)
        assert summary["overcount_rate"] == pytest.approx(25.0)
        assert summary["total_samples"] == 4
        assert summary["exact_matches"] == 2
        assert summary["undercounts"] == 1
        assert summary["overcounts"] == 1

    def test_pandas_and_nan_handling(self):
        s_true = pd.Series([3, None, 5, np.nan])
        s_pred = pd.Series([3, 4, 4, 2])

        # Valid pairs: (3, 3) and (5, 4) -> MAE = 0.5, Exact = 0.5
        assert counting_mae(s_true, s_pred, dropna=True) == pytest.approx(0.5)
        assert exact_accuracy(s_true, s_pred, dropna=True) == pytest.approx(0.5)

        with pytest.raises(ValueError, match="None/NaN"):
            counting_mae(s_true, s_pred, dropna=False)

    def test_invalid_inputs_raise_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            counting_mae([], [])

        with pytest.raises(ValueError, match="Length mismatch"):
            exact_accuracy([1, 2, 3], [1, 2])

        with pytest.raises(ValueError, match="No valid"):
            counting_mae([None, np.nan], [1, 2], dropna=True)


# =============================================================================
# 2. OBJECT DETECTION & mAP WRAPPER TESTS
# =============================================================================


class TestObjectDetectionMapWrapper:
    """Tests for IoU, AP, and mAP wrapper (standalone NumPy + Ultralytics extractor)."""

    def test_compute_iou_xyxy_and_xywh(self):
        box_a = [10, 10, 50, 50]  # 40x40 = 1600
        box_b = [10, 10, 50, 50]
        box_c = [60, 60, 100, 100]  # Disjoint

        assert compute_iou(box_a, box_b, box_format="xyxy") == pytest.approx(1.0)
        assert compute_iou(box_a, box_c, box_format="xyxy") == pytest.approx(0.0)

        # Partial overlap: [0, 0, 10, 10] and [5, 0, 15, 10] -> inter = 5x10 = 50, union = 100+100-50 = 150
        assert compute_iou([0, 0, 10, 10], [5, 0, 15, 10], box_format="xyxy") == pytest.approx(
            50.0 / 150.0
        )

        # Equivalent in xywh: center (5, 5, 10, 10) and center (10, 5, 10, 10)
        assert compute_iou([5, 5, 10, 10], [10, 5, 10, 10], box_format="xywh") == pytest.approx(
            50.0 / 150.0
        )

    def test_compute_iou_matrix(self):
        boxes1 = [[0, 0, 10, 10], [20, 20, 30, 30]]
        boxes2 = [[0, 0, 10, 10], [5, 0, 15, 10]]
        mat = compute_iou_matrix(boxes1, boxes2)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(1.0 / 3.0)
        assert mat[1, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_map_wrapper_perfect_detections(self):
        ground_truths = [
            [
                {"bbox": [10, 10, 60, 80], "cls_id": 0},
                {"bbox": [100, 20, 150, 90], "cls_id": 1},
            ],
            [
                {"bbox": [30, 30, 90, 110], "cls_id": 0},
            ],
        ]
        predictions = [
            [
                {"bbox": [10, 10, 60, 80], "conf": 0.95, "cls_id": 0},
                {"bbox": [100, 20, 150, 90], "conf": 0.88, "cls_id": 1},
            ],
            [
                {"bbox": [30, 30, 90, 110], "conf": 0.91, "cls_id": 0},
            ],
        ]

        res = map_wrapper(predictions=predictions, ground_truths=ground_truths)
        assert res["map50"] == pytest.approx(1.0)
        assert res["map75"] == pytest.approx(1.0)
        assert res["map50_95"] == pytest.approx(1.0)
        assert res["num_classes"] == 2
        assert res["per_class_ap50"][0] == pytest.approx(1.0)
        assert res["per_class_ap50"][1] == pytest.approx(1.0)

    def test_map_wrapper_with_false_positive_and_missed_gt(self):
        # 1 class (cls_id=0), 2 GT boxes total across 1 image.
        # Prediction 1: high conf (0.95), matches GT 1 (TP) -> recall=0.5, prec=1.0
        # Prediction 2: low conf (0.40), false positive (FP) -> recall=0.5, prec=0.5
        # GT 2 is missed (FN).
        # Under continuous AP: area = 0.5 * 1.0 = 0.50.
        # Under COCO 101-point AP: 51 points (0.00..0.50) have precision 1.0, 50 points have 0.0 -> 51/101.
        ground_truths = [
            [
                {"bbox": [0, 0, 50, 50], "cls_id": 0},
                {"bbox": [100, 100, 150, 150], "cls_id": 0},
            ]
        ]
        predictions = [
            [
                {"bbox": [0, 0, 50, 50], "conf": 0.95, "cls_id": 0},
                {"bbox": [200, 200, 250, 250], "conf": 0.40, "cls_id": 0},
            ]
        ]

        res_cont = map_wrapper(
            predictions=predictions,
            ground_truths=ground_truths,
            interpolation="continuous",
        )
        assert res_cont["map50"] == pytest.approx(0.50)

        res_coco = map_wrapper(
            predictions=predictions,
            ground_truths=ground_truths,
            interpolation="coco101",
        )
        assert res_coco["map50"] == pytest.approx(51.0 / 101.0)

    def test_map_wrapper_duplicate_detection_penalized_as_fp(self):
        # Two predictions overlapping the same single GT box:
        # First is TP (rec=1.0, prec=1.0), second is FP (rec=1.0, prec=0.5).
        # Upper precision envelope stays 1.0 across [0, 1], so AP is still 1.0 if high-conf box was first;
        # If a false positive has HIGHER confidence than the true positive:
        # Pred 1 (conf 0.99, FP) -> rec=0.0, prec=0.0
        # Pred 2 (conf 0.80, TP) -> rec=1.0, prec=0.5
        # Envelope is 0.5 across [0, 1] -> AP = 0.5!
        ground_truths = [[{"bbox": [10, 10, 50, 50], "cls_id": 0}]]
        predictions = [
            [
                {"bbox": [200, 200, 240, 240], "conf": 0.99, "cls_id": 0},  # High-conf FP
                {"bbox": [10, 10, 50, 50], "conf": 0.80, "cls_id": 0},  # Lower-conf TP
            ]
        ]
        res = map_wrapper(
            predictions=predictions,
            ground_truths=ground_truths,
            interpolation="continuous",
        )
        assert res["map50"] == pytest.approx(0.50)

    def test_map_wrapper_ultralytics_extractor(self):
        mock_box = SimpleNamespace(
            map50=0.892,
            map75=0.745,
            map=0.681,
            mp=0.910,
            mr=0.865,
            ap50=np.array([0.92, 0.864]),
            ap=np.array([0.70, 0.662]),
        )
        mock_ultra = SimpleNamespace(box=mock_box)

        res = map_wrapper(ultralytics_results=mock_ultra)
        assert res["map50"] == pytest.approx(0.892)
        assert res["map75"] == pytest.approx(0.745)
        assert res["map50_95"] == pytest.approx(0.681)
        assert res["precision"] == pytest.approx(0.910)
        assert res["recall"] == pytest.approx(0.865)
        assert res["per_class_ap50"][0] == pytest.approx(0.92)
        assert res["per_class_ap50"][1] == pytest.approx(0.864)
        assert res["source"] == "ultralytics"

        # Also test dict-based results_dict extraction
        dict_ultra = {
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.62,
            "metrics/precision(B)": 0.88,
            "metrics/recall(B)": 0.81,
        }
        res_dict = map_wrapper(ultralytics_results=dict_ultra)
        assert res_dict["map50"] == pytest.approx(0.85)
        assert res_dict["map50_95"] == pytest.approx(0.62)
        assert res_dict["precision"] == pytest.approx(0.88)
        assert res_dict["recall"] == pytest.approx(0.81)


# =============================================================================
# 3. EVENT-LEVEL PRECISION / RECALL / F1 TESTS
# =============================================================================


class TestEventMetrics:
    """Tests for binary and multi-class shelf event Precision, Recall, F1, and FAR."""

    def test_binary_disagreement_event_metrics(self):
        # 6 timestamps:
        # TP = 2 (indices 1, 2)
        # FP = 1 (index 3)
        # FN = 1 (index 4)
        # TN = 2 (indices 0, 5)
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0, 0]

        res = event_precision_recall_f1(y_true, y_pred, average="binary", positive_label=1)
        # Precision = 2 / (2 + 1) = 2/3
        # Recall    = 2 / (2 + 1) = 2/3
        # F1        = 2/3
        # FAR       = FP / (FP + TN) = 1 / (1 + 2) = 1/3
        # Accuracy  = (2 + 2) / 6 = 4/6
        assert res["tp"] == 2
        assert res["fp"] == 1
        assert res["fn"] == 1
        assert res["tn"] == 2
        assert res["precision"] == pytest.approx(2.0 / 3.0)
        assert res["recall"] == pytest.approx(2.0 / 3.0)
        assert res["f1"] == pytest.approx(2.0 / 3.0)
        assert res["false_alarm_rate"] == pytest.approx(1.0 / 3.0)
        assert res["accuracy"] == pytest.approx(4.0 / 6.0)

        assert event_precision(y_true, y_pred) == pytest.approx(2.0 / 3.0)
        assert event_recall(y_true, y_pred) == pytest.approx(2.0 / 3.0)
        assert event_f1(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_multiclass_shelf_events_macro_micro_weighted(self):
        y_true = ["TAKE", "TAKE", "RESTOCK", "IDLE", "IDLE", "IDLE"]
        y_pred = ["TAKE", "IDLE", "RESTOCK", "IDLE", "IDLE", "TAKE"]

        # Class breakdown:
        # TAKE:    TP=1, FP=1 (idx 5), FN=1 (idx 1) -> P=0.5, R=0.5, F1=0.5, support=2
        # RESTOCK: TP=1, FP=0,         FN=0         -> P=1.0, R=1.0, F1=1.0, support=1
        # IDLE:    TP=2, FP=1 (idx 1), FN=1 (idx 5) -> P=2/3, R=2/3, F1=2/3, support=3
        res_macro = event_precision_recall_f1(
            y_true, y_pred, average="macro", labels=["TAKE", "RESTOCK", "IDLE"]
        )
        expected_macro_f1 = (0.5 + 1.0 + (2.0 / 3.0)) / 3.0
        assert res_macro["f1"] == pytest.approx(expected_macro_f1)
        assert res_macro["per_class"]["RESTOCK"]["f1"] == pytest.approx(1.0)
        assert res_macro["per_class"]["TAKE"]["f1"] == pytest.approx(0.5)

        res_weighted = event_precision_recall_f1(
            y_true, y_pred, average="weighted", labels=["TAKE", "RESTOCK", "IDLE"]
        )
        # Weighted F1 = (2*0.5 + 1*1.0 + 3*(2/3)) / 6 = (1 + 1 + 2) / 6 = 4/6
        assert res_weighted["f1"] == pytest.approx(4.0 / 6.0)

        res_micro = event_precision_recall_f1(
            y_true, y_pred, average="micro", labels=["TAKE", "RESTOCK", "IDLE"]
        )
        # Micro F1 equals overall accuracy for single-label multi-class = 4 / 6
        assert res_micro["f1"] == pytest.approx(4.0 / 6.0)

    def test_zero_division_protection(self):
        # No positive events in ground truth or predictions
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]

        res = event_precision_recall_f1(
            y_true, y_pred, average="binary", positive_label=1, zero_division=0.0
        )
        assert res["tp"] == 0
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert res["tn"] == 4
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["f1"] == 0.0
        assert res["false_alarm_rate"] == 0.0

    def test_derive_inventory_events(self):
        counts_prev = [5, 5, 3, 6]
        counts_curr = [5, 3, 6, 6]
        events = derive_inventory_events(counts_prev, counts_curr)
        assert events == ["IDLE", "TAKE", "RESTOCK", "IDLE"]


# =============================================================================
# 4. PACKAGE RE-EXPORT CONSISTENCY TESTS
# =============================================================================


def test_fusion_and_vision_reexports_identical():
    """Verify that `vision.metrics`, `fusion.metrics`, `vision`, and `fusion` expose the same functions."""
    assert vision_metrics.counting_mae is counting_mae
    assert vision_metrics.exact_accuracy is exact_accuracy
    assert vision_metrics.map_wrapper is map_wrapper
    assert vision_metrics.event_precision_recall_f1 is event_precision_recall_f1
    assert fusion.counting_mae is counting_mae
    assert vision.counting_mae is counting_mae

