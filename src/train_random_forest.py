"""Sensor-wise Random Forest classifier training, probability alignment, and serialization."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.feature_extraction import extract_features_for_windows, get_feature_names

logger = logging.getLogger(__name__)


class SensorRandomForest:
    """Encapsulates a Random Forest trained on features from a single wearable sensor."""

    def __init__(
        self,
        sensor_name: str,
        class_list: List[int],
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.sensor_name = sensor_name
        self.canonical_classes = sorted(list(class_list))
        self.n_classes = len(self.canonical_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.canonical_classes)}

        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_names = get_feature_names(sensor_name)
        self.is_trained = False
        self.train_duration_sec: float = 0.0

    def fit(self, X_windows: np.ndarray, y: np.ndarray) -> "SensorRandomForest":
        """
        Extract features and fit Random Forest.
        X_windows: (N, time_steps, 6)
        y: (N,) activity IDs
        """
        logger.info(f"Training {self.sensor_name} Random Forest on {len(y)} windows...")
        t0 = time.time()
        X_feats = extract_features_for_windows(X_windows)

        self.clf.fit(X_feats, y)
        self.train_duration_sec = time.time() - t0
        self.is_trained = True
        logger.info(f"{self.sensor_name} RF trained in {self.train_duration_sec:.2f}s.")
        return self

    def predict_proba_aligned(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Predict activity class probabilities aligned to canonical_classes.
        Output shape: (N, len(canonical_classes)).
        Handles absent classes gracefully by zeroing their column.
        """
        if not self.is_trained:
            raise RuntimeError(f"Model for {self.sensor_name} is not trained.")

        N = len(X_windows)
        if N == 0:
            return np.empty((0, self.n_classes), dtype=np.float32)

        X_feats = extract_features_for_windows(X_windows)
        raw_probs = self.clf.predict_proba(X_feats)  # Shape (N, len(self.clf.classes_))

        aligned_probs = np.zeros((N, self.n_classes), dtype=np.float32)
        for model_col_idx, cls_label in enumerate(self.clf.classes_):
            if cls_label in self.class_to_idx:
                canonical_idx = self.class_to_idx[cls_label]
                aligned_probs[:, canonical_idx] = raw_probs[:, model_col_idx]

        # Re-normalize rows to guard against rounding errors
        row_sums = aligned_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        aligned_probs = aligned_probs / row_sums
        return aligned_probs

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        """Predict canonical class IDs directly from aligned probabilities."""
        probs = self.predict_proba_aligned(X_windows)
        best_indices = np.argmax(probs, axis=1)
        return np.array([self.canonical_classes[idx] for idx in best_indices])

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dictionary mapping feature names to their Gini importance."""
        if not self.is_trained:
            return {}
        importances = self.clf.feature_importances_
        return {name: float(val) for name, val in zip(self.feature_names, importances)}

    def save(self, filepath: Path) -> None:
        """Serialize trained model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "SensorRandomForest":
        """Load trained model from disk."""
        return joblib.load(filepath)
