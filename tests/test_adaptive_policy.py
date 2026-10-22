"""Unit tests for the closed-loop AdaptiveSensingController."""

import pytest
from src.adaptive_policy import AdaptiveSensingController


def test_controller_initialization_and_cost():
    ctrl = AdaptiveSensingController(anchor_sensor="wrist")
    assert ctrl.active_mask["wrist"] == 1
    assert ctrl.active_mask["chest"] == 0
    assert ctrl.active_mask["ankle"] == 0
    # Cost should be 1/3 ~ 0.333
    cost = ctrl.compute_normalized_cost()
    assert abs(cost - 0.333) < 0.01


def test_controller_activates_on_fault():
    ctrl = AdaptiveSensingController(anchor_sensor="wrist", r_low=0.40)
    # Simulate degraded wrist reliability
    next_mask, action, meta = ctrl.step(
        current_reliabilities={"wrist": 0.20},
        fused_uncertainty=0.20
    )
    # Must activate secondary sensor
    assert action == "ACTIVATE"
    assert sum(next_mask.values()) >= 2


def test_controller_activates_on_uncertainty():
    ctrl = AdaptiveSensingController(anchor_sensor="wrist", u_high=0.65)
    # Simulate high uncertainty despite ok reliability
    next_mask, action, meta = ctrl.step(
        current_reliabilities={"wrist": 0.90},
        fused_uncertainty=0.85
    )
    assert action == "ACTIVATE"
    assert sum(next_mask.values()) >= 2


def test_at_least_one_sensor_active_invariant():
    ctrl = AdaptiveSensingController(anchor_sensor="wrist")
    # Even under extreme deactivation conditions, at least 1 sensor stays active
    for _ in range(10):
        next_mask, action, meta = ctrl.step(
            current_reliabilities={"wrist": 0.95, "chest": 0.95, "ankle": 0.95},
            fused_uncertainty=0.10
        )
        assert sum(next_mask.values()) >= 1
