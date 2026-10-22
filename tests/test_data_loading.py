"""Unit tests for dataset loading, format validation, and activity filtering."""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import (
    load_single_subject_raw,
    filter_and_format_subject_data,
    inspect_raw_missing_stats,
    extract_subject_id_from_path,
    DEFAULT_ACTIVITIES,
    SENSOR_CHANNELS
)


def test_extract_subject_id():
    p = Path("data/raw/PAMAP2_Dataset/Protocol/subject105.dat")
    assert extract_subject_id_from_path(p) == 105


def test_load_single_subject_and_cols():
    p = Path("data/raw/PAMAP2_Dataset/Protocol/subject101.dat")
    df = load_single_subject_raw(p)
    assert df.shape[1] == 54
    assert len(df) > 0
    # Check timestamp monotonicity
    diffs = np.diff(df[0].to_numpy())
    assert np.all(diffs >= 0)


def test_filter_and_format_activity_selection():
    p = Path("data/raw/PAMAP2_Dataset/Protocol/subject101.dat")
    raw_df = load_single_subject_raw(p)
    clean_df, stats = filter_and_format_subject_data(raw_df, 101, DEFAULT_ACTIVITIES)

    # Activity ID 0 (transient) must not be in clean_df
    assert 0 not in clean_df["activity_id"].values
    # Check that all remaining activities belong to selected activities
    for act_id in clean_df["activity_id"].unique():
        assert act_id in DEFAULT_ACTIVITIES

    # Verify all 18 sensor channels are present
    for s in ["wrist", "chest", "ankle"]:
        for ch in SENSOR_CHANNELS[s]["names"]:
            assert ch in clean_df.columns
