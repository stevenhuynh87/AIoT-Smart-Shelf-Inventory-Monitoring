"""
Re-export of standardized evaluation metrics for the Vision subsystem.

All core implementations reside in `fusion.metrics` to ensure a single source of truth
across Vision-Only, Weight-Only, and Disagreement-Aware Sensor Fusion evaluations.
"""

from fusion.metrics import (
    __all__,
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

