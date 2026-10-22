"""Sensor fault simulator for controlled, reproducible stress testing."""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SensorFaultSimulator:
    """
    Simulates physical sensor faults on wearable IMU signals.
    Never alters training data; applied strictly to test sequences.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def apply_dropout(
        self,
        window: np.ndarray,
        severity: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete or partial sensor dropout (unavailable samples / dead node).
        severity represents fraction of window marked missing (up to 1.0 = entire window).
        """
        corrupted = np.copy(window)
        time_steps, _ = corrupted.shape
        drop_steps = int(time_steps * np.clip(severity, 0.1, 1.0))
        corrupted[:drop_steps, :] = np.nan

        meta = {
            "fault_type": "dropout",
            "severity": float(severity),
            "drop_samples": drop_steps,
            "is_faulty": True
        }
        return corrupted, meta

    def apply_intermittent_missing(
        self,
        window: np.ndarray,
        severity: float = 0.5,
        block_size: int = 16
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulate packet dropouts / missing communication blocks.
        Marks multiple discrete communication intervals missing.
        """
        corrupted = np.copy(window)
        time_steps, channels = corrupted.shape
        n_blocks = time_steps // block_size
        drop_count = int(n_blocks * np.clip(severity, 0.1, 0.9))

        if drop_count > 0:
            block_indices = self.rng.choice(n_blocks, size=drop_count, replace=False)
            for b in block_indices:
                start = b * block_size
                corrupted[start:start + block_size, :] = np.nan

        meta = {
            "fault_type": "intermittent",
            "severity": float(severity),
            "dropped_blocks": drop_count,
            "is_faulty": True
        }
        return corrupted, meta

    def apply_gaussian_noise(
        self,
        window: np.ndarray,
        severity: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Additive white Gaussian noise on sensor channels.
        Noise std scaled by severity * 12.0 in the supplied array's units.
        """
        corrupted = np.copy(window)
        noise_std = float(severity) * 12.0
        noise = self.rng.normal(0.0, noise_std, size=window.shape)
        corrupted = corrupted + noise

        meta = {
            "fault_type": "noise",
            "severity": float(severity),
            "noise_std": noise_std,
            "is_faulty": True
        }
        return corrupted, meta

    def apply_flatline(
        self,
        window: np.ndarray,
        severity: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Stuck-at / flatline fault holding sensor measurements constant.
        severity controls fraction of the window frozen at the initial sample.
        """
        corrupted = np.copy(window)
        time_steps, _ = corrupted.shape
        stuck_steps = int(time_steps * np.clip(severity, 0.2, 1.0))
        stuck_val = corrupted[0:1, :]
        corrupted[:stuck_steps, :] = stuck_val

        meta = {
            "fault_type": "flatline",
            "severity": float(severity),
            "stuck_samples": stuck_steps,
            "is_faulty": True
        }
        return corrupted, meta

    def apply_drift(
        self,
        window: np.ndarray,
        severity: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Linear sensor bias drift accumulating across the window.
        """
        corrupted = np.copy(window)
        time_steps, channels = corrupted.shape
        max_drift = float(severity) * 20.0
        drift_ramp = np.linspace(0.0, max_drift, time_steps)[:, np.newaxis]
        corrupted = corrupted + drift_ramp

        meta = {
            "fault_type": "drift",
            "severity": float(severity),
            "max_drift": max_drift,
            "is_faulty": True
        }
        return corrupted, meta

    def inject_fault(
        self,
        window: np.ndarray,
        fault_type: str,
        severity: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dispatch fault injection by type name."""
        if fault_type == "clean" or severity <= 0.0:
            return np.copy(window), {"fault_type": "clean", "severity": 0.0, "is_faulty": False}
        elif fault_type == "dropout":
            return self.apply_dropout(window, severity)
        elif fault_type == "intermittent":
            return self.apply_intermittent_missing(window, severity)
        elif fault_type == "noise":
            return self.apply_gaussian_noise(window, severity)
        elif fault_type == "flatline":
            return self.apply_flatline(window, severity)
        elif fault_type == "drift":
            return self.apply_drift(window, severity)
        elif fault_type == "combined":
            # Combined noise + intermittent dropout
            w_noisy, _ = self.apply_gaussian_noise(window, severity * 0.7)
            w_comb, _ = self.apply_intermittent_missing(w_noisy, severity * 0.7)
            return w_comb, {"fault_type": "combined", "severity": float(severity), "is_faulty": True}
        else:
            raise ValueError(f"Unknown fault scenario: {fault_type}")
