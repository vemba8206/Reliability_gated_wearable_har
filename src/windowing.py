"""Sliding window segmentation with label purity filtering and metadata tracking."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.data_loader import SENSOR_CHANNELS

logger = logging.getLogger(__name__)


def extract_windows_for_subject(
    df: pd.DataFrame,
    window_size: int = 256,
    stride: int = 128,
    purity_threshold: float = 0.80,
    raw_df_before_imputation: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[Dict[str, Any]]]:
    """
    Extract fixed-length sliding windows for a single subject.
    
    Guarantees:
    - Checks label purity: windows with majority class frequency < purity_threshold are rejected.
    - Windows do not span across discontinuous recording sessions.
    - Returns structured 3D arrays for wrist, chest, ankle of shape (N, window_size, 6).
    """
    subject_id = int(df["subject_id"].iloc[0])
    n_samples = len(df)

    wrist_cols = SENSOR_CHANNELS["wrist"]["names"]
    chest_cols = SENSOR_CHANNELS["chest"]["names"]
    ankle_cols = SENSOR_CHANNELS["ankle"]["names"]

    wrist_list = []
    chest_list = []
    ankle_list = []
    labels_list = []
    metadata_list = []

    # Slide window
    window_idx = 0
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        win_df = df.iloc[start:end]

        # Check label purity
        activity_counts = win_df["activity_id"].value_counts()
        majority_activity = activity_counts.index[0]
        majority_ratio = activity_counts.iloc[0] / window_size

        if majority_ratio < purity_threshold:
            # Transitional window rejected
            continue

        # Check raw missing samples if raw df is provided
        raw_nan_count = 0
        if raw_df_before_imputation is not None:
            raw_win = raw_df_before_imputation.iloc[start:end]
            raw_nan_count = int(raw_win[wrist_cols + chest_cols + ankle_cols].isna().sum().sum())

        wrist_data = win_df[wrist_cols].to_numpy(dtype=np.float32)
        chest_data = win_df[chest_cols].to_numpy(dtype=np.float32)
        ankle_data = win_df[ankle_cols].to_numpy(dtype=np.float32)

        wrist_list.append(wrist_data)
        chest_list.append(chest_data)
        ankle_list.append(ankle_data)
        labels_list.append(int(majority_activity))

        meta = {
            "window_idx": window_idx,
            "subject_id": subject_id,
            "start_sample": start,
            "end_sample": end,
            "start_time": float(win_df["timestamp"].iloc[0]),
            "end_time": float(win_df["timestamp"].iloc[-1]),
            "activity_id": int(majority_activity),
            "activity_name": str(win_df["activity_name"].iloc[0]),
            "purity": float(majority_ratio),
            "raw_nan_count": raw_nan_count
        }
        metadata_list.append(meta)
        window_idx += 1

    if not wrist_list:
        return (
            {"wrist": np.empty((0, window_size, 6), dtype=np.float32),
             "chest": np.empty((0, window_size, 6), dtype=np.float32),
             "ankle": np.empty((0, window_size, 6), dtype=np.float32)},
            np.empty(0, dtype=np.int64),
            []
        )

    sensor_windows = {
        "wrist": np.stack(wrist_list, axis=0),
        "chest": np.stack(chest_list, axis=0),
        "ankle": np.stack(ankle_list, axis=0),
    }
    y_array = np.array(labels_list, dtype=np.int64)

    return sensor_windows, y_array, metadata_list


def segment_all_subjects(
    data_by_subject: Dict[int, pd.DataFrame],
    window_size: int = 256,
    stride: int = 128,
    purity_threshold: float = 0.80
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, np.ndarray], Dict[int, List[Dict[str, Any]]]]:
    """Segment data for each subject independently into windows."""
    windows_by_sub = {}
    labels_by_sub = {}
    meta_by_sub = {}

    for sub_id, df in data_by_subject.items():
        sens_win, y_arr, meta = extract_windows_for_subject(
            df,
            window_size=window_size,
            stride=stride,
            purity_threshold=purity_threshold
        )
        windows_by_sub[sub_id] = sens_win
        labels_by_sub[sub_id] = y_arr
        meta_by_sub[sub_id] = meta
        logger.info(f"Subject {sub_id}: {len(y_arr)} windows extracted (window_size={window_size}, stride={stride}).")

    return windows_by_sub, labels_by_sub, meta_by_sub
