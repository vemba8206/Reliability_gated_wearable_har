import numpy as np

from src.evaluation import compute_classification_metrics


def test_fixed_six_class_f1_includes_absent_classes():
    metrics = compute_classification_metrics(
        np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2]), list(range(1, 7))
    )
    assert metrics["macro_f1"] == 1.0
    assert np.isclose(metrics["macro_f1_all_classes"], 2 / 6)
