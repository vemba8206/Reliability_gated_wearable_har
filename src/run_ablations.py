"""Replay a saved official-data RF fold with one policy component disabled at a time."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.adaptive_policy import AdaptiveSensingController
from src.baselines import run_proposed_adaptive
from src.data_loader import DEFAULT_ACTIVITIES, SENSOR_CHANNELS, load_all_subjects
from src.fault_injection import SensorFaultSimulator
from src.preprocessing import SubjectAwareScaler, impute_short_gaps
from src.reliability import SensorReliabilityEstimator
from src.train_random_forest import SensorRandomForest
from src.windowing import segment_all_subjects


def run_ablations(run_dir: Path, train_subject: int, test_subject: int) -> Path:
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    activity_names = {int(k): str(v) for k, v in cfg["dataset"]["activities"].items()}
    channels = [name for sensor in SENSOR_CHANNELS.values() for name in sensor["names"]]
    raw, _ = load_all_subjects(cfg["dataset"]["raw_dir"], [train_subject, test_subject], activity_names)
    clean = {s: impute_short_gaps(frame, channels)[0] for s, frame in raw.items()}
    windows, labels, _ = segment_all_subjects(clean, cfg["dataset"]["window_size"],
        cfg["dataset"]["window_stride"], cfg["dataset"]["label_purity_threshold"])

    state = json.loads((run_dir / f"subject_{test_subject}_models" / "scaler.json").read_text(encoding="utf-8"))
    scaler = SubjectAwareScaler(channels)
    scaler.means, scaler.stds, scaler.is_fitted = state["means"], state["stds"], True
    sensors = ["wrist", "chest", "ankle"]
    x_train = {s: scaler.transform_array(windows[train_subject][s], SENSOR_CHANNELS[s]["names"]) for s in sensors}
    x_test = {s: scaler.transform_array(windows[test_subject][s], SENSOR_CHANNELS[s]["names"]) for s in sensors}
    y_test = labels[test_subject]
    models = {s: SensorRandomForest.load(run_dir / f"subject_{test_subject}_models" / f"{s}_rf.joblib") for s in sensors}
    estimators = {}
    for s in sensors:
        est = SensorReliabilityEstimator(weights=cfg["reliability"]["weights"],
            flatline_var_thresh=cfg["reliability"]["thresholds"]["flatline_var"],
            noise_diff_var_thresh=cfg["reliability"]["thresholds"]["noise_diff_var"],
            clipping_acc_thresh=cfg["reliability"]["thresholds"]["clipping_accel"])
        est.calibrate_baseline(x_train[s])
        estimators[s] = est

    conditions = {"clean": x_test}
    for fault in ("dropout", "noise"):
        simulator = SensorFaultSimulator(cfg["fault_injection"]["random_state"])
        corrupted = {s: np.copy(x_test[s]) for s in sensors}
        corrupted["wrist"] = np.stack([simulator.inject_fault(w, fault, 0.5)[0] for w in x_test["wrist"]])
        conditions[f"moderate_{fault}"] = corrupted

    variants = {
        "standard": {},
        "without_reliability_gate": {"r_low": -1.0},
        "without_uncertainty_control": {"u_high": 2.0, "u_low": -1.0},
        "without_budget_constraint": {"cost_budget": 1.0},
        "without_weighted_fusion": {"uniform_fusion": True},
        "without_dwell": {"min_dwell_windows": 0},
    }
    base = cfg["adaptive_policy"]
    results = {}
    for condition_name, test_windows in conditions.items():
        results[condition_name] = {}
        for name, override in variants.items():
            controller = AdaptiveSensingController(sensors=sensors, anchor_sensor=base["anchor_sensor"],
                r_low=override.get("r_low", base["r_low"]), r_good=base["r_good"],
                u_high=override.get("u_high", base["u_high"]), u_low=override.get("u_low", base["u_low"]),
                min_dwell_windows=override.get("min_dwell_windows", base["min_dwell_windows"]),
                probe_interval=base["probe_interval"], cost_budget=override.get("cost_budget", base["cost_budget"]),
                cost_weights={s: cfg["dataset"]["sensors"][s]["cost_weight"] for s in sensors})
            output = run_proposed_adaptive(test_windows, y_test, models, estimators, controller,
                sorted(activity_names), activity_names, offline_cache=True,
                uniform_fusion=override.get("uniform_fusion", False))
            results[condition_name][name] = {
                "macro_f1": output["metrics"]["macro_f1"],
                "accuracy": output["metrics"]["accuracy"],
                "mean_normalized_cost": output["cost"]["mean_normalized_cost"],
                "total_switches": output["cost"]["total_switches"],
                "total_probes": output["cost"]["total_probes"],
                "windows": len(y_test),
            }
    destination = run_dir / f"subject_{test_subject}_ablations.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-subject", type=int, required=True)
    parser.add_argument("--test-subject", type=int, required=True)
    arguments = parser.parse_args()
    print(run_ablations(arguments.run_dir, arguments.train_subject, arguments.test_subject))
