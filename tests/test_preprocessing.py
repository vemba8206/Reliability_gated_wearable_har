"""Unit tests for preprocessing, imputation, and sliding windowing."""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import impute_short_gaps, SubjectAwareScaler
from src.windowing import extract_windows_for_subject
from src.data_loader import SENSOR_CHANNELS


def test_impute_short_gaps():
    cols = ["wrist_acc_x", "wrist_acc_y"]
    df = pd.DataFrame({
        "wrist_acc_x": [1.0, np.nan, 3.0, 4.0],
        "wrist_acc_y": [10.0, 20.0, np.nan, 40.0]
    })
    imputed, nan_counts = impute_short_gaps(df, cols, max_gap_samples=5)
    assert nan_counts["wrist_acc_x"] == 1
    assert nan_counts["wrist_acc_y"] == 1
    assert not imputed["wrist_acc_x"].isna().any()
    assert imputed["wrist_acc_x"].iloc[1] == 2.0


def test_window_extraction_and_shapes():
    # Synthetic dataframe
    n_samples = 600
    rows = []
    for i in range(n_samples):
        r = {"timestamp": i * 0.01, "activity_id": 1, "activity_name": "Lying", "subject_id": 101}
        for s in ["wrist", "chest", "ankle"]:
            for name in SENSOR_CHANNELS[s]["names"]:
                r[name] = 1.0 + np.sin(i * 0.1)
        rows.append(r)
    df = pd.DataFrame(rows)

    windows, y, meta = extract_windows_for_subject(df, window_size=256, stride=128, purity_threshold=0.8)
    assert len(y) > 0
    assert windows["wrist"].shape == (len(y), 256, 6)
    assert windows["chest"].shape == (len(y), 256, 6)
    assert windows["ankle"].shape == (len(y), 256, 6)
    assert len(meta) == len(y)
    assert meta[0]["purity"] >= 0.8
