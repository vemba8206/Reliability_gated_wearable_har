"""Unit tests for sensor reliability estimator and quality indicators."""

import pytest
import numpy as np

from src.reliability import (
    compute_missing_ratio,
    compute_flatline_indicator,
    compute_high_frequency_noise_indicator,
    compute_clipping_indicator,
    SensorReliabilityEstimator
)


def test_quality_indicators_ranges():
    w_clean = np.random.randn(256, 6)
    q_miss = compute_missing_ratio(w_clean)
    assert 0.0 <= q_miss <= 1.0

    q_flat = compute_flatline_indicator(w_clean)
    assert 0.0 <= q_flat <= 1.0

    q_noise = compute_high_frequency_noise_indicator(w_clean)
    assert 0.0 <= q_noise <= 1.0

    q_clip = compute_clipping_indicator(w_clean)
    assert 0.0 <= q_clip <= 1.0


def test_reliability_on_flatline():
    est = SensorReliabilityEstimator()
    w_flat = np.ones((256, 6)) * 9.81
    res = est.estimate(w_flat)
    assert 0.0 <= res["reliability"] <= 1.0
    # A completely frozen signal must produce a low reliability score
    assert res["reliability"] < 0.45
    assert res["is_faulty"] is True


def test_reliability_on_clean():
    est = SensorReliabilityEstimator()
    rng = np.random.RandomState(42)
    w_nominal = rng.randn(256, 6) * 1.5
    res = est.estimate(w_nominal)
    assert res["reliability"] >= 0.70
    assert res["is_faulty"] is False
