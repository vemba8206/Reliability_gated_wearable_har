"""Subject-aware preprocessing and normalization without data leakage."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def impute_short_gaps(
    df: pd.DataFrame,
    channel_cols: List[str],
    max_gap_samples: int = 20
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Impute small missing intervals within sensor channels using linear interpolation.
    Gaps larger than max_gap_samples are forward-filled up to limit, then filled with 0.0.
    Returns imputed DataFrame and a dictionary of missing sample counts per channel before imputation.
    """
    imputed_df = df.copy()
    nan_counts = {col: int(imputed_df[col].isna().sum()) for col in channel_cols}

    for col in channel_cols:
        if nan_counts[col] > 0:
            # Interpolate short gaps linearly
            imputed_df[col] = imputed_df[col].interpolate(
                method="linear",
                limit=max_gap_samples,
                limit_direction="both"
            )
            # Fill any remaining NaNs (at edges or long gaps) with median or 0.0
            imputed_df[col] = imputed_df[col].fillna(0.0)

    return imputed_df, nan_counts


class SubjectAwareScaler:
    """
    Standardizes sensor channels by fitting mean and std strictly on training subjects.
    Never fits on validation or test subjects to prevent information leakage.
    """
    def __init__(self, channel_cols: List[str]):
        self.channel_cols = channel_cols
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, training_dfs: List[pd.DataFrame]) -> "SubjectAwareScaler":
        """Fit mean and std across training subjects only."""
        if not training_dfs:
            raise ValueError("Cannot fit scaler on empty training set.")

        combined = pd.concat(training_dfs, ignore_index=True)
        for col in self.channel_cols:
            mean_val = float(combined[col].mean())
            std_val = float(combined[col].std())
            # Guard against constant signals or zero variance
            if np.isnan(std_val) or std_val < 1e-6:
                std_val = 1.0
            self.means[col] = mean_val
            self.stds[col] = std_val

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted standardization to a DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("SubjectAwareScaler must be fitted before transforming data.")

        out_df = df.copy()
        for col in self.channel_cols:
            out_df[col] = (out_df[col] - self.means[col]) / self.stds[col]
        return out_df

    def transform_array(self, X: np.ndarray, sensor_channels: List[str]) -> np.ndarray:
        """
        Transform a 3D array of shape (N_windows, time_steps, len(sensor_channels))
        using the fitted channel statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("SubjectAwareScaler must be fitted before transforming arrays.")

        X_norm = np.copy(X)
        for idx, col in enumerate(sensor_channels):
            mean = self.means[col]
            std = self.stds[col]
            X_norm[:, :, idx] = (X_norm[:, :, idx] - mean) / std
        return X_norm

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler statistics."""
        return {
            "channel_cols": self.channel_cols,
            "means": self.means,
            "stds": self.stds,
            "is_fitted": self.is_fitted
        }
