"""Vectorized time-domain feature extraction for sensor-wise Random Forest classifiers."""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

CHANNEL_BASE_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "acc_mag", "gyr_mag"]
STAT_NAMES = ["mean", "std", "min", "max", "rms", "median", "iqr", "skew", "kurt"]


def get_feature_names(sensor_name: str) -> List[str]:
    """Generate standardized feature names for a given sensor."""
    names = []
    for ch in CHANNEL_BASE_NAMES:
        for st in STAT_NAMES:
            names.append(f"{sensor_name}_{ch}_{st}")
    return names


def extract_features_for_windows(windows: np.ndarray) -> np.ndarray:
    """
    Vectorized extraction of 72 time-domain features for an array of windows.
    windows shape: (N, 256, 6)
    Returns: (N, 72) feature array.
    """
    if len(windows) == 0:
        return np.empty((0, len(CHANNEL_BASE_NAMES) * len(STAT_NAMES)), dtype=np.float32)

    # Clean NaNs
    win_clean = np.nan_to_num(windows, nan=0.0)

    # Compute 3D norms for accel (cols 0..2) and gyro (cols 3..5)
    acc_mag = np.linalg.norm(win_clean[:, :, 0:3], axis=-1, keepdims=True)  # (N, 256, 1)
    gyr_mag = np.linalg.norm(win_clean[:, :, 3:6], axis=-1, keepdims=True)  # (N, 256, 1)
    expanded = np.concatenate([win_clean, acc_mag, gyr_mag], axis=-1)       # (N, 256, 8)

    # Fast vectorized moments along axis 1 (time steps)
    mean_v = np.mean(expanded, axis=1)                          # (N, 8)
    std_v = np.std(expanded, axis=1)                            # (N, 8)
    min_v = np.min(expanded, axis=1)                            # (N, 8)
    max_v = np.max(expanded, axis=1)                            # (N, 8)
    rms_v = np.sqrt(np.mean(expanded ** 2, axis=1))             # (N, 8)
    med_v = np.median(expanded, axis=1)                         # (N, 8)
    q75, q25 = np.percentile(expanded, [75, 25], axis=1)        # (N, 8)
    iqr_v = q75 - q25

    # Safe skewness and kurtosis
    diff = expanded - mean_v[:, np.newaxis, :]                  # (N, 256, 8)
    safe_std3 = np.maximum(std_v ** 3, 1e-5)
    safe_std4 = np.maximum(std_v ** 4, 1e-5)
    skew_v = np.mean(diff ** 3, axis=1) / safe_std3             # (N, 8)
    kurt_v = (np.mean(diff ** 4, axis=1) / safe_std4) - 3.0      # (N, 8)

    feature_blocks = []
    for ch_idx in range(8):
        ch_features = np.column_stack([
            mean_v[:, ch_idx],
            std_v[:, ch_idx],
            min_v[:, ch_idx],
            max_v[:, ch_idx],
            rms_v[:, ch_idx],
            med_v[:, ch_idx],
            iqr_v[:, ch_idx],
            skew_v[:, ch_idx],
            kurt_v[:, ch_idx]
        ])  # (N, 9)
        feature_blocks.append(ch_features)

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_features_single_window(window_data: np.ndarray) -> np.ndarray:
    """Extract features for a single window (1, 256, 6) -> (72,)."""
    return extract_features_for_windows(window_data[np.newaxis, :, :])[0]
