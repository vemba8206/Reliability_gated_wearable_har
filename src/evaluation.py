"""Evaluation metrics calculation: classification, fault resilience, reliability, and cost."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    canonical_classes: List[int],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    class_prec = precision_score(y_true, y_pred, labels=canonical_classes, average=None, zero_division=0)
    class_rec = recall_score(y_true, y_pred, labels=canonical_classes, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=canonical_classes, average=None, zero_division=0)

    per_class = {}
    for idx, c in enumerate(canonical_classes):
        name = class_names.get(c, f"Class_{c}") if class_names else f"Class_{c}"
        per_class[name] = {
            "class_id": int(c),
            "precision": float(class_prec[idx]),
            "recall": float(class_rec[idx]),
            "f1": float(class_f1[idx])
        }

    cm = confusion_matrix(y_true, y_pred, labels=canonical_classes)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def compute_reliability_detection_metrics(
    injected_fault_labels: np.ndarray,
    estimated_reliabilities: np.ndarray,
    fault_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Calculate detection performance of the SensorReliabilityEstimator against injected ground truth.
    Estimated fault decision: reliability < fault_threshold.
    """
    y_true_fault = np.asarray(injected_fault_labels, dtype=int)
    # Lower reliability predicts fault (1 - r is anomaly score)
    anomaly_scores = 1.0 - np.asarray(estimated_reliabilities, dtype=float)
    y_pred_fault = (np.asarray(estimated_reliabilities) < fault_threshold).astype(int)

    # AUROC
    try:
        if len(np.unique(y_true_fault)) > 1:
            auroc = float(roc_auc_score(y_true_fault, anomaly_scores))
        else:
            auroc = 1.0 if y_true_fault[0] == 0 else 0.5
    except Exception:
        auroc = 0.5

    prec = float(precision_score(y_true_fault, y_pred_fault, zero_division=0))
    rec = float(recall_score(y_true_fault, y_pred_fault, zero_division=0))
    f1 = float(f1_score(y_true_fault, y_pred_fault, zero_division=0))

    # False alarm rate (false positive rate)
    negatives = np.sum(y_true_fault == 0)
    fp = np.sum((y_true_fault == 0) & (y_pred_fault == 1))
    far = float(fp / max(1, negatives))

    return {
        "auroc": auroc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_alarm_rate": far
    }


def compute_cost_and_efficiency_metrics(
    timeline_active_masks: List[Dict[str, int]],
    total_switches: int,
    total_probes: int,
    cost_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate normalized sensing cost proxy, average active sensors, and switching frequency.
    """
    if cost_weights is None:
        cost_weights = {"wrist": 1.0, "chest": 1.0, "ankle": 1.0}
    total_w = sum(cost_weights.values())

    costs = []
    active_counts = []
    for mask in timeline_active_masks:
        active_w = sum(cost_weights[s] * mask[s] for s in mask)
        costs.append(active_w / total_w)
        active_counts.append(sum(mask.values()))

    n_windows = max(1, len(timeline_active_masks))
    avg_cost = float(np.mean(costs)) if costs else 1.0
    avg_active_sensors = float(np.mean(active_counts)) if active_counts else 3.0
    switching_freq = float(total_switches / n_windows)

    return {
        "mean_normalized_cost": avg_cost,
        "mean_active_sensors": avg_active_sensors,
        "active_sensor_ratio": float(avg_active_sensors / len(cost_weights)),
        "total_switches": total_switches,
        "switching_frequency": switching_freq,
        "total_probes": total_probes,
        "cost_timeline": costs
    }
