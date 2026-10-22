"""Regression checks for research provenance and next-window policy constraints."""

import numpy as np
import pandas as pd
import pytest

from src.adaptive_policy import AdaptiveSensingController
from src.data_download import acquire_dataset, create_synthetic_pamap2_subset
from src.windowing import extract_windows_for_subject
from src.data_loader import SENSOR_CHANNELS
from src.fault_injection import SensorFaultSimulator
from src.reliability import SensorReliabilityEstimator


def test_budget_and_anchor_are_hard_constraints():
    controller = AdaptiveSensingController(cost_budget=2 / 3, min_dwell_windows=1)
    for _ in range(30):
        mask, _, _ = controller.step({s: 0.1 for s, active in controller.active_mask.items() if active}, 1.0)
        assert mask["wrist"] == 1
        assert controller.compute_normalized_cost(mask) <= 2 / 3


def test_synthetic_data_cannot_enter_research_run(tmp_path):
    create_synthetic_pamap2_subset(str(tmp_path), subjects=[])
    with pytest.raises(RuntimeError, match="Synthetic"):
        acquire_dataset(str(tmp_path))


def test_window_does_not_cross_filtered_timestamp_gap():
    columns = [name for sensor in SENSOR_CHANNELS.values() for name in sensor["names"]]
    frame = pd.DataFrame(np.zeros((8, len(columns))), columns=columns)
    frame["timestamp"] = [0, .01, .02, .03, 3, 3.01, 3.02, 3.03]
    frame["activity_id"] = 1
    frame["activity_name"] = "Lying"
    frame["subject_id"] = 101
    windows, labels, _ = extract_windows_for_subject(frame, window_size=8, stride=8)
    assert len(labels) == 0


def test_empty_selected_subject_has_no_windows():
    windows, labels, metadata = extract_windows_for_subject(pd.DataFrame(), window_size=256)
    assert len(labels) == 0 and metadata == []
    assert all(arr.shape == (0, 256, 6) for arr in windows.values())


def test_missing_fault_is_observable_without_ground_truth_label():
    simulator = SensorFaultSimulator(random_seed=7)
    healthy = np.random.default_rng(7).normal(size=(256, 6))
    broken, metadata = simulator.inject_fault(healthy, "dropout", severity=1.0)
    score = SensorReliabilityEstimator().estimate(broken)
    assert metadata["is_faulty"]
    assert score["indicators"]["missing"] == 1.0
    assert score["reliability"] < 0.4
