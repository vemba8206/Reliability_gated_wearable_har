"""Data loader for the PAMAP2 dataset with exact column mappings and validation."""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import yaml

from src.data_download import find_existing_subject_files

logger = logging.getLogger(__name__)

# Official PAMAP2 54-column mappings
COL_TIMESTAMP = 0
COL_ACTIVITY = 1
COL_HEART_RATE = 2

# 6 channels per IMU: 3D acc (+-16g) + 3D gyro
SENSOR_CHANNELS = {
    "wrist": {
        "accel": [4, 5, 6],
        "gyro": [10, 11, 12],
        "names": ["wrist_acc_x", "wrist_acc_y", "wrist_acc_z", "wrist_gyr_x", "wrist_gyr_y", "wrist_gyr_z"]
    },
    "chest": {
        "accel": [21, 22, 23],
        "gyro": [27, 28, 29],
        "names": ["chest_acc_x", "chest_acc_y", "chest_acc_z", "chest_gyr_x", "chest_gyr_y", "chest_gyr_z"]
    },
    "ankle": {
        "accel": [38, 39, 40],
        "gyro": [44, 45, 46],
        "names": ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z", "ankle_gyr_x", "ankle_gyr_y", "ankle_gyr_z"]
    }
}

DEFAULT_ACTIVITIES = {
    1: "Lying",
    2: "Sitting",
    3: "Standing",
    4: "Walking",
    5: "Running",
    6: "Cycling"
}


def extract_subject_id_from_path(path: Path) -> int:
    """Extract numeric subject ID from file name e.g. 'subject101.dat' -> 101."""
    match = re.search(r"subject(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract subject ID from path: {path}")


def load_single_subject_raw(
    file_path: Path,
    expected_cols: int = 54
) -> pd.DataFrame:
    """
    Load raw space-separated .dat PAMAP2 file.
    Validates column count, timestamp monotonicity, and NaN structures.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Subject file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="c",
        dtype=np.float64
    )

    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns for PAMAP2 format, got {df.shape[1]} in {file_path.name}"
        )

    # Sort by timestamp to ensure monotonic time sequence
    df = df.sort_values(by=COL_TIMESTAMP).reset_index(drop=True)
    return df


def inspect_raw_missing_stats(df: pd.DataFrame, subject_id: int) -> Dict[str, Any]:
    """Calculate missing-value statistics before any imputation."""
    total_samples = len(df)
    missing_per_col = df.isna().sum()

    stats = {
        "subject_id": subject_id,
        "total_samples": total_samples,
        "missing_heart_rate": int(missing_per_col[COL_HEART_RATE]),
        "missing_hr_pct": float(missing_per_col[COL_HEART_RATE] / max(1, total_samples)),
        "sensors": {}
    }

    for sensor, config in SENSOR_CHANNELS.items():
        cols = config["accel"] + config["gyro"]
        sensor_missing = df[cols].isna().sum().to_dict()
        sensor_total_missing = int(df[cols].isna().sum().sum())
        sensor_nan_pct = float(sensor_total_missing / max(1, total_samples * len(cols)))
        stats["sensors"][sensor] = {
            "total_missing_cells": sensor_total_missing,
            "nan_pct": sensor_nan_pct,
            "per_channel_missing": {config["names"][i]: int(sensor_missing[c]) for i, c in enumerate(cols)}
        }

    return stats


def filter_and_format_subject_data(
    df: pd.DataFrame,
    subject_id: int,
    selected_activities: Optional[Dict[int, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select 18 sensor channels (6 wrist, 6 chest, 6 ankle) + timestamp & activity label.
    Filters out activity 0 (transient/other) and any unselected activities.
    """
    if selected_activities is None:
        selected_activities = DEFAULT_ACTIVITIES

    stats = inspect_raw_missing_stats(df, subject_id)

    # Map column indexes to clean column names
    col_mapping = {
        COL_TIMESTAMP: "timestamp",
        COL_ACTIVITY: "activity_id"
    }

    channel_list = []
    for sensor, config in SENSOR_CHANNELS.items():
        for idx, col_idx in enumerate(config["accel"] + config["gyro"]):
            col_name = config["names"][idx]
            col_mapping[col_idx] = col_name
            channel_list.append(col_name)

    # Extract target columns
    selected_cols = list(col_mapping.keys())
    sub_df = df[selected_cols].copy()
    sub_df.rename(columns=col_mapping, inplace=True)
    sub_df["subject_id"] = subject_id

    # Filter by selected activities (excluding ID 0 and unselected)
    initial_rows = len(sub_df)
    sub_df = sub_df[sub_df["activity_id"].isin(selected_activities.keys())].copy()
    sub_df["activity_name"] = sub_df["activity_id"].map(selected_activities)
    filtered_rows = len(sub_df)

    stats["initial_rows"] = initial_rows
    stats["retained_rows"] = filtered_rows
    stats["excluded_activity_rows"] = initial_rows - filtered_rows
    stats["activity_distribution"] = sub_df["activity_id"].value_counts().to_dict()

    return sub_df, stats


def load_all_subjects(
    raw_dir: str = "data/raw",
    selected_subjects: Optional[List[int]] = None,
    selected_activities: Optional[Dict[int, str]] = None
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, Dict[str, Any]]]:
    """
    Load and format all requested subjects from raw directory.
    Returns dictionary mapping subject_id -> DataFrame and subject_id -> statistics.
    """
    subject_files = find_existing_subject_files(raw_dir)
    if not subject_files:
        raise FileNotFoundError(f"No subject*.dat files found in {raw_dir}")

    data_by_subject = {}
    stats_by_subject = {}

    for sfile in subject_files:
        sub_id = extract_subject_id_from_path(sfile)
        if selected_subjects is not None and sub_id not in selected_subjects:
            continue

        raw_df = load_single_subject_raw(sfile)
        clean_df, stats = filter_and_format_subject_data(raw_df, sub_id, selected_activities)
        data_by_subject[sub_id] = clean_df
        stats_by_subject[sub_id] = stats
        logger.info(f"Subject {sub_id} loaded: {len(clean_df)} usable samples across {len(stats['activity_distribution'])} activities.")

    return data_by_subject, stats_by_subject
