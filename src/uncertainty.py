"""Uncertainty quantification via normalized Shannon entropy and sensor disagreement."""

import numpy as np
from typing import List, Optional


def compute_normalized_entropy(probs: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    Compute normalized Shannon entropy:
        u(t) = - sum_{k=1}^K p_k log(p_k) / log(K)
    Guaranteed to lie in [0, 1] for valid probability distributions.
    u = 0 indicates absolute certainty (one-hot).
    u = 1 indicates maximum uncertainty (uniform distribution).
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, epsilon, 1.0)
    # Ensure normalized
    probs = probs / np.sum(probs)
    K = len(probs)
    if K <= 1:
        return 0.0

    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(K))
    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return norm_entropy


def compute_sensor_disagreement(prob_list: List[np.ndarray]) -> float:
    """
    Compute pairwise Jensen-Shannon divergence or mean prediction disagreement
    across active sensor probability distributions.
    Returns disagreement score in [0, 1].
    """
    if len(prob_list) < 2:
        return 0.0

    preds = [np.argmax(p) for p in prob_list]
    # Fraction of pairs with differing class predictions
    n_pairs = 0
    diff_count = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            n_pairs += 1
            if preds[i] != preds[j]:
                diff_count += 1

    return float(diff_count / max(1, n_pairs))
