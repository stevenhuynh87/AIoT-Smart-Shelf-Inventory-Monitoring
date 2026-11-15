"""Vision subsystem package for smart shelf object detection."""

from vision.metrics import (
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
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
