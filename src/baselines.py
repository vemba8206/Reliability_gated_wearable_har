"""Evaluation harness executing the 5 required baselines under identical test sequences."""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.reliability import SensorReliabilityEstimator
from src.fusion import fuse_sensor_probabilities
from src.adaptive_policy import AdaptiveSensingController
from src.evaluation import (
    compute_classification_metrics,
    compute_cost_and_efficiency_metrics,
    compute_reliability_detection_metrics
)

logger = logging.getLogger(__name__)


def run_baseline_wrist_only(
    test_windows: Dict[str, np.ndarray],
    y_test: np.ndarray,
    models: Dict[str, Any],
    canonical_classes: List[int],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Baseline 1: Single wrist sensor always active."""
    N = len(y_test)
    t0 = time.time()
    # Predict directly using wrist model
    probs_wrist = models["wrist"].predict_proba_aligned(test_windows["wrist"])
    preds = np.array([canonical_classes[idx] for idx in np.argmax(probs_wrist, axis=1)])
    total_time = time.time() - t0
    latency_per_win_ms = (total_time / max(1, N)) * 1000.0

    metrics = compute_classification_metrics(y_test, preds, canonical_classes, class_names)
    masks = [{"wrist": 1, "chest": 0, "ankle": 0} for _ in range(N)]
    cost_metrics = compute_cost_and_efficiency_metrics(masks, total_switches=0, total_probes=0)

    return {
        "name": "wrist_only",
        "description": "Wrist IMU only (Single-sensor Anchor)",
        "predictions": preds.tolist(),
        "metrics": metrics,
        "cost": cost_metrics,
        "latency_ms_per_window": latency_per_win_ms
    }


def run_baseline_wrist_chest_static(
    test_windows: Dict[str, np.ndarray],
    y_test: np.ndarray,
    models: Dict[str, Any],
    canonical_classes: List[int],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Baseline 2: Static Wrist + Chest configuration with uniform fusion."""
    N = len(y_test)
    t0 = time.time()
    pw = models["wrist"].predict_proba_aligned(test_windows["wrist"])
    pc = models["chest"].predict_proba_aligned(test_windows["chest"])
    fused_probs = (pw + pc) / 2.0
    preds = np.array([canonical_classes[idx] for idx in np.argmax(fused_probs, axis=1)])
    total_time = time.time() - t0
    latency_per_win_ms = (total_time / max(1, N)) * 1000.0

    metrics = compute_classification_metrics(y_test, preds, canonical_classes, class_names)
    masks = [{"wrist": 1, "chest": 1, "ankle": 0} for _ in range(N)]
    cost_metrics = compute_cost_and_efficiency_metrics(masks, total_switches=0, total_probes=0)

    return {
        "name": "wrist_chest_static",
        "description": "Static Wrist + Chest with Uniform Fusion",
        "predictions": preds.tolist(),
        "metrics": metrics,
        "cost": cost_metrics,
        "latency_ms_per_window": latency_per_win_ms
    }


def run_baseline_all_uniform(
    test_windows: Dict[str, np.ndarray],
    y_test: np.ndarray,
    models: Dict[str, Any],
    canonical_classes: List[int],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Baseline 3: All 3 sensors always active with uniform probability fusion."""
    N = len(y_test)
    t0 = time.time()
    pw = models["wrist"].predict_proba_aligned(test_windows["wrist"])
    pc = models["chest"].predict_proba_aligned(test_windows["chest"])
    pa = models["ankle"].predict_proba_aligned(test_windows["ankle"])
    fused_probs = (pw + pc + pa) / 3.0
    preds = np.array([canonical_classes[idx] for idx in np.argmax(fused_probs, axis=1)])
    total_time = time.time() - t0
    latency_per_win_ms = (total_time / max(1, N)) * 1000.0

    metrics = compute_classification_metrics(y_test, preds, canonical_classes, class_names)
    masks = [{"wrist": 1, "chest": 1, "ankle": 1} for _ in range(N)]
    cost_metrics = compute_cost_and_efficiency_metrics(masks, total_switches=0, total_probes=0)

    return {
        "name": "all_sensors_uniform",
        "description": "All Sensors Always Active (Uniform Fusion)",
        "predictions": preds.tolist(),
        "metrics": metrics,
        "cost": cost_metrics,
        "latency_ms_per_window": latency_per_win_ms
    }


def run_baseline_all_reliability(
    test_windows: Dict[str, np.ndarray],
    y_test: np.ndarray,
    models: Dict[str, Any],
    estimators: Dict[str, SensorReliabilityEstimator],
    canonical_classes: List[int],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Baseline 4: All 3 sensors always active with reliability-weighted fusion."""
    N = len(y_test)
    t0 = time.time()
    preds = []
    uncertainties = []
    reliabilities_log = {"wrist": [], "chest": [], "ankle": []}

    pw = models["wrist"].predict_proba_aligned(test_windows["wrist"])
    pc = models["chest"].predict_proba_aligned(test_windows["chest"])
    pa = models["ankle"].predict_proba_aligned(test_windows["ankle"])

    for i in range(N):
        rw = estimators["wrist"].estimate(test_windows["wrist"][i])["reliability"]
        rc = estimators["chest"].estimate(test_windows["chest"][i])["reliability"]
        ra = estimators["ankle"].estimate(test_windows["ankle"][i])["reliability"]

        reliabilities_log["wrist"].append(rw)
        reliabilities_log["chest"].append(rc)
        reliabilities_log["ankle"].append(ra)

        fusion_out = fuse_sensor_probabilities(
            active_mask={"wrist": 1, "chest": 1, "ankle": 1},
            reliabilities={"wrist": rw, "chest": rc, "ankle": ra},
            probabilities={"wrist": pw[i], "chest": pc[i], "ankle": pa[i]}
        )
        preds.append(canonical_classes[fusion_out["predicted_idx"]])
        uncertainties.append(fusion_out["uncertainty"])

    total_time = time.time() - t0
    latency_per_win_ms = (total_time / max(1, N)) * 1000.0

    preds_arr = np.array(preds)
    metrics = compute_classification_metrics(y_test, preds_arr, canonical_classes, class_names)
    masks = [{"wrist": 1, "chest": 1, "ankle": 1} for _ in range(N)]
    cost_metrics = compute_cost_and_efficiency_metrics(masks, total_switches=0, total_probes=0)

    return {
        "name": "all_sensors_reliability",
        "description": "All Sensors Always Active (Reliability-Weighted Fusion)",
        "predictions": preds_arr.tolist(),
        "metrics": metrics,
        "cost": cost_metrics,
        "reliabilities": reliabilities_log,
        "uncertainties": uncertainties,
        "latency_ms_per_window": latency_per_win_ms
    }


def run_proposed_adaptive(
    test_windows: Dict[str, np.ndarray],
    y_test: np.ndarray,
    models: Dict[str, Any],
    estimators: Dict[str, SensorReliabilityEstimator],
    controller: AdaptiveSensingController,
    canonical_classes: List[int],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Baseline 5 / Proposed Method:
    Reliability-gated closed-loop adaptive sensing with next-window selection.
    Strictly simulates physical realism: inactive sensor readings are never queried or predicted.
    """
    N = len(y_test)
    controller.reset()

    preds = []
    uncertainties = []
    actions_log = []
    masks_log = []
    reliabilities_log = {"wrist": [], "chest": [], "ankle": []}

    t0 = time.time()

    # Precompute per-sensor predictions and reliabilities in vectorized batches
    precomputed_probs = {s: models[s].predict_proba_aligned(test_windows[s]) for s in ["wrist", "chest", "ankle"]}
    precomputed_rels = {s: [estimators[s].estimate(test_windows[s][idx])["reliability"] for idx in range(N)] for s in ["wrist", "chest", "ankle"]}

    # Initial warm-up for window 0: observe currently active mask
    current_active = dict(controller.active_mask)

    for i in range(N):
        # 1. Acquire readings ONLY for active sensors in window i
        current_reliabilities = {}
        current_probabilities = {}

        for sensor in ["wrist", "chest", "ankle"]:
            if current_active[sensor] == 1:
                # Sensor is active: query observed reliability and model prediction
                r_val = precomputed_rels[sensor][i]
                current_reliabilities[sensor] = r_val
                reliabilities_log[sensor].append(r_val)

                p_val = precomputed_probs[sensor][i]
                current_probabilities[sensor] = p_val
            else:
                # Inactive sensor: signal is completely HIDDEN from controller and fusion!
                # Keep last known reliability logged for offline plotting
                reliabilities_log[sensor].append(controller.last_observed_reliability[sensor])

        # 2. Fuse active sensor probabilities
        fusion_out = fuse_sensor_probabilities(
            active_mask=current_active,
            reliabilities=current_reliabilities,
            probabilities=current_probabilities
        )
        pred_label = canonical_classes[fusion_out["predicted_idx"]]
        preds.append(pred_label)
        uncertainties.append(fusion_out["uncertainty"])
        masks_log.append(dict(current_active))

        # 3. Closed-loop controller decides active sensors for NEXT window i+1
        next_active, action, meta = controller.step(
            current_reliabilities=current_reliabilities,
            fused_uncertainty=fusion_out["uncertainty"]
        )
        actions_log.append(action)
        current_active = next_active

    total_time = time.time() - t0
    latency_per_win_ms = (total_time / max(1, N)) * 1000.0

    preds_arr = np.array(preds)
    metrics = compute_classification_metrics(y_test, preds_arr, canonical_classes, class_names)
    cost_metrics = compute_cost_and_efficiency_metrics(
        masks_log,
        total_switches=controller.total_switches,
        total_probes=controller.total_probes
    )

    return {
        "name": "proposed_adaptive",
        "description": "Proposed Reliability-Gated Adaptive Sensing",
        "predictions": preds_arr.tolist(),
        "metrics": metrics,
        "cost": cost_metrics,
        "actions": actions_log,
        "masks": masks_log,
        "reliabilities": reliabilities_log,
        "uncertainties": uncertainties,
        "latency_ms_per_window": latency_per_win_ms
    }
