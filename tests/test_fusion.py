"""Unit tests for reliability-weighted probability fusion and uncertainty."""

import pytest
import numpy as np

from src.fusion import fuse_sensor_probabilities
from src.uncertainty import compute_normalized_entropy


def test_normalized_entropy_bounds():
    # One-hot: entropy = 0.0
    p_certain = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isclose(compute_normalized_entropy(p_certain), 0.0, atol=1e-4)

    # Uniform: entropy = 1.0
    p_uniform = np.ones(6) / 6.0
    assert np.isclose(compute_normalized_entropy(p_uniform), 1.0, atol=1e-4)


def test_fusion_normalization_and_weights():
    active_mask = {"wrist": 1, "chest": 1, "ankle": 0}
    reliabilities = {"wrist": 0.8, "chest": 0.2, "ankle": 0.0}
    probs = {
        "wrist": np.array([0.9, 0.1, 0.0]),
        "chest": np.array([0.1, 0.8, 0.1]),
        "ankle": np.array([0.33, 0.33, 0.34])
    }

    out = fuse_sensor_probabilities(active_mask, reliabilities, probs)
    fused = out["fused_probs"]

    assert len(fused) == 3
    assert np.isclose(np.sum(fused), 1.0)
    assert not out["fallback_triggered"]
    # Wrist has 4x the weight of chest, so class 0 should dominate
    assert fused[0] > fused[1]


def test_fusion_all_unreliable_fallback():
    active_mask = {"wrist": 1, "chest": 0, "ankle": 0}
    reliabilities = {"wrist": 0.01, "chest": 0.0, "ankle": 0.0}
    probs = {"wrist": np.array([0.9, 0.05, 0.05])}

    out = fuse_sensor_probabilities(active_mask, reliabilities, probs, min_usable_weight=0.05)
    assert out["fallback_triggered"] is True
    # Should fall back to uniform distribution
    assert np.allclose(out["fused_probs"], 1.0 / 3.0)
