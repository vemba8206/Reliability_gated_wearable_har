"""Sensor reliability estimation based on observable physical signal quality indicators."""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_missing_ratio(window: np.ndarray, raw_nan_count: int = 0) -> float:
    """
    Calculate fraction of missing / NaN samples in the window.
    window shape: (time_steps, channels)
    """
    time_steps, channels = window.shape
    total_cells = time_steps * channels
    nan_cells = int(np.isnan(window).sum()) + raw_nan_count
    return float(np.clip(nan_cells / max(1, total_cells), 0.0, 1.0))


def compute_flatline_indicator(window: np.ndarray, var_threshold: float = 1e-4) -> float:
    """
    Detect stuck-at / flatline sensor behavior by measuring per-channel variance.
    If variance across channels is below var_threshold, the signal is stagnant/uninformative.
    Returns penalty in [0, 1].
    """
    clean_w = np.nan_to_num(window, nan=0.0)
    variances = np.var(clean_w, axis=0)  # Shape (channels,)
    # Fraction of channels that are flatlined
    flat_channels = np.sum(variances < var_threshold)
    ratio = flat_channels / float(window.shape[1])
    return float(np.clip(ratio, 0.0, 1.0))


def compute_high_frequency_noise_indicator(
    window: np.ndarray,
    diff_var_threshold: float = 50.0
) -> float:
    """
    Detect severe additive high-frequency noise using the variance of first differences.
    Physical human motion is band-limited; excessive first-difference variance signals noise.
    Returns penalty in [0, 1].
    """
    clean_w = np.nan_to_num(window, nan=0.0)
    diffs = np.diff(clean_w, axis=0)  # Shape (time_steps - 1, channels)
    diff_vars = np.var(diffs, axis=0)
    max_diff_var = float(np.max(diff_vars)) if len(diff_vars) > 0 else 0.0
    # Soft saturation curve
    penalty = max_diff_var / (max_diff_var + diff_var_threshold)
    return float(np.clip(penalty, 0.0, 1.0))


def compute_clipping_indicator(
    window: np.ndarray,
    acc_clip_threshold: float = 155.0,  # ~16g in m/s^2
    max_clip_ratio_tolerance: float = 0.05
) -> float:
    """
    Detect sensor saturation / clipping where accelerometer values hit maximum physical scale.
    Returns penalty in [0, 1].
    """
    clean_w = np.nan_to_num(window, nan=0.0)
    # Accelerometer channels are first 3
    acc_channels = clean_w[:, 0:3]
    clipped_samples = np.sum(np.abs(acc_channels) >= acc_clip_threshold)
    ratio = clipped_samples / float(acc_channels.size)
    penalty = ratio / max(1e-4, max_clip_ratio_tolerance)
    return float(np.clip(penalty, 0.0, 1.0))


def compute_distribution_shift_indicator(
    window: np.ndarray,
    baseline_mean: Optional[np.ndarray] = None,
    baseline_std: Optional[np.ndarray] = None,
    z_threshold: float = 4.0
) -> float:
    """
    Detect abnormal sensor-local bias or DC drift compared to baseline training statistics.
    Returns penalty in [0, 1].
    """
    if baseline_mean is None or baseline_std is None:
        return 0.0

    clean_w = np.nan_to_num(window, nan=0.0)
    win_mean = np.mean(clean_w, axis=0)
    z_scores = np.abs(win_mean - baseline_mean) / np.maximum(baseline_std, 1e-3)
    max_z = float(np.max(z_scores))
    if max_z > z_threshold:
        return float(np.clip((max_z - z_threshold) / z_threshold, 0.0, 1.0))
    return 0.0


class SensorReliabilityEstimator:
    """
    Lightweight runtime estimator that produces reliability r_i(t) in [0, 1].
    Uses weighted combination of observable quality indicators:
        r_i(t) = clip(1 - sum_j (w_j * q_ij(t)), 0.0, 1.0)
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        flatline_var_thresh: float = 1e-4,
        noise_diff_var_thresh: float = 50.0,
        clipping_acc_thresh: float = 155.0,
        fault_threshold: float = 0.45
    ):
        if weights is None:
            self.weights = {
                "missing": 0.35,
                "flatline": 0.30,
                "noise": 0.20,
                "clipping": 0.15
            }
        else:
            self.weights = weights

        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.flatline_var_thresh = flatline_var_thresh
        self.noise_diff_var_thresh = noise_diff_var_thresh
        self.clipping_acc_thresh = clipping_acc_thresh
        self.fault_threshold = fault_threshold

        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None

    def calibrate_baseline(self, training_windows: np.ndarray) -> None:
        """
        Calibrate expected nominal mean and std on training data only.
        Never calibrated on test data.
        """
        all_samples = training_windows.reshape(-1, training_windows.shape[-1])
        self.baseline_mean = np.mean(all_samples, axis=0)
        self.baseline_std = np.std(all_samples, axis=0)

    def estimate(
        self,
        window: np.ndarray,
        raw_nan_count: int = 0
    ) -> Dict[str, Any]:
        """
        Compute reliability score and detailed quality metrics for an observed window.
        window: (time_steps, 6)
        """
        q_missing = compute_missing_ratio(window, raw_nan_count=raw_nan_count)
        q_flatline = compute_flatline_indicator(window, var_threshold=self.flatline_var_thresh)
        q_noise = compute_high_frequency_noise_indicator(window, diff_var_threshold=self.noise_diff_var_thresh)
        q_clip = compute_clipping_indicator(window, acc_clip_threshold=self.clipping_acc_thresh)

        # Weighted penalty
        penalty = (
            self.weights.get("missing", 0.35) * q_missing +
            self.weights.get("flatline", 0.30) * q_flatline +
            self.weights.get("noise", 0.20) * q_noise +
            self.weights.get("clipping", 0.15) * q_clip
        )

        # Severe fault triggers immediate degradation
        if q_flatline >= 0.8 or q_missing >= 0.5:
            penalty = max(penalty, 0.85)

        reliability = float(np.clip(1.0 - penalty, 0.0, 1.0))
        is_faulty = reliability < self.fault_threshold

        return {
            "reliability": reliability,
            "is_faulty": is_faulty,
            "indicators": {
                "missing": q_missing,
                "flatline": q_flatline,
                "noise": q_noise,
                "clipping": q_clip,
                "penalty": float(penalty)
            }
        }
