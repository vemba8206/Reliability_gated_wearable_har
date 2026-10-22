"""Reliability-weighted multi-sensor probability fusion with safe fallback."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.uncertainty import compute_normalized_entropy, compute_sensor_disagreement

logger = logging.getLogger(__name__)


def fuse_sensor_probabilities(
    active_mask: Dict[str, int],
    reliabilities: Dict[str, float],
    probabilities: Dict[str, np.ndarray],
    epsilon: float = 1e-6,
    min_usable_weight: float = 0.05
) -> Dict[str, Any]:
    """
    Perform reliability-weighted fusion across active sensors:
        p_fused(y | t) = sum_i [ a_i(t) * r_i(t) * p_i(y) ] / ( sum_i [ a_i(t) * r_i(t) ] + epsilon )
        
    Handles cases:
    - Normal multi-sensor or single-sensor operation.
    - Sensor failure (low r_i downweights or eliminates sensor contribution).
    - All-active-sensors-unreliable case: triggers safe fallback (uniform probability + high uncertainty flag).
    """
    # Find dimension K
    sample_prob = next(iter(probabilities.values()))
    K = len(sample_prob)

    numerator = np.zeros(K, dtype=np.float64)
    total_weight = 0.0
    active_probs_list = []

    for sensor_name, is_active in active_mask.items():
        if is_active and sensor_name in probabilities and sensor_name in reliabilities:
            p_i = np.asarray(probabilities[sensor_name], dtype=np.float64)
            r_i = float(reliabilities[sensor_name])
            weight = float(is_active) * r_i

            numerator += weight * p_i
            total_weight += weight
            active_probs_list.append(p_i)

    # Check fallback condition
    fallback_triggered = False
    if total_weight < min_usable_weight:
        fallback_triggered = True
        logger.warning("Active sensors have insufficient usable reliability! Triggering safe fallback.")
        # Safe fallback: uniform prior distribution across classes
        fused_probs = np.ones(K, dtype=np.float64) / K
    else:
        fused_probs = numerator / (total_weight + epsilon)
        # Re-normalize to guard against precision drift
        fused_probs = fused_probs / np.sum(fused_probs)

    # Compute prediction uncertainty
    uncertainty = compute_normalized_entropy(fused_probs)
    disagreement = compute_sensor_disagreement(active_probs_list) if len(active_probs_list) > 1 else 0.0
    pred_idx = int(np.argmax(fused_probs))

    return {
        "fused_probs": fused_probs,
        "predicted_idx": pred_idx,
        "uncertainty": uncertainty,
        "disagreement": disagreement,
        "total_weight": float(total_weight),
        "fallback_triggered": fallback_triggered
    }
