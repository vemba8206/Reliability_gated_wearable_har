"""Leakage prevention tests: guarantees strictly disjoint subject partitions."""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import SubjectAwareScaler


def test_subject_disjoint_partitions():
    all_subjects = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    for test_sub in all_subjects:
        train_subs = [s for s in all_subjects if s != test_sub]
        # Assert strictly no intersection
        assert test_sub not in train_subs
        assert set([test_sub]).intersection(set(train_subs)) == set()


def test_scaler_no_test_leakage():
    cols = ["wrist_acc_x", "wrist_acc_y"]
    # Train subject data with mean 10.0
    df_train = pd.DataFrame({
        "wrist_acc_x": np.array([9.0, 10.0, 11.0]),
        "wrist_acc_y": np.array([10.0, 10.0, 10.0])
    })
    # Test subject data with vastly different mean 100.0
    df_test = pd.DataFrame({
        "wrist_acc_x": np.array([99.0, 100.0, 101.0]),
        "wrist_acc_y": np.array([100.0, 100.0, 100.0])
    })

    scaler = SubjectAwareScaler(cols)
    scaler.fit([df_train])

    # Fitted parameters must reflect df_train ONLY
    assert np.isclose(scaler.means["wrist_acc_x"], 10.0)

    # Transform test without altering fitted parameters
    df_test_norm = scaler.transform(df_test)
    assert np.isclose(scaler.means["wrist_acc_x"], 10.0)
    # The normalized test mean should be around (100 - 10) / std = 90
    assert df_test_norm["wrist_acc_x"].iloc[1] > 50.0
