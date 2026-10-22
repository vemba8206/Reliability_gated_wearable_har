"""Unit tests for fault simulator: determinism, correct corruptions, and labels."""

import pytest
import numpy as np

from src.fault_injection import SensorFaultSimulator


def test_fault_injection_determinism():
    sim1 = SensorFaultSimulator(random_seed=42)
    sim2 = SensorFaultSimulator(random_seed=42)
    w = np.ones((256, 6))

    out1, meta1 = sim1.inject_fault(w, "noise", severity=0.5)
    out2, meta2 = sim2.inject_fault(w, "noise", severity=0.5)

    assert np.allclose(out1, out2)
    assert meta1["is_faulty"] is True


def test_dropout_fault():
    sim = SensorFaultSimulator(random_seed=42)
    w = np.ones((256, 6)) * 5.0
    out, meta = sim.inject_fault(w, "dropout", severity=1.0)
    assert np.isnan(out).all()


def test_flatline_fault():
    sim = SensorFaultSimulator(random_seed=42)
    w = np.random.randn(256, 6)
    out, meta = sim.inject_fault(w, "flatline", severity=1.0)
    # Variance should be 0 along time axis
    assert np.all(np.var(out, axis=0) < 1e-6)
