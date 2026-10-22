"""Nested, subject-wise RF/policy comparison using the verified PAMAP2 run.

The test subject is never used for fitting or selecting a configuration. Each
fold's original validation subject selects between two RF configurations and
three prespecified policies. Historical model/scaler files are reused intact.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.adaptive_policy import AdaptiveSensingController
from src.baselines import (
    run_baseline_all_reliability,
    run_baseline_all_uniform,
    run_baseline_wrist_chest_static,
    run_baseline_wrist_only,
    run_proposed_adaptive,
)
from src.data_loader import SENSOR_CHANNELS, load_all_subjects
from src.fault_injection import SensorFaultSimulator
from src.preprocessing import SubjectAwareScaler, impute_short_gaps
from src.reliability import SensorReliabilityEstimator
from src.train_random_forest import SensorRandomForest
from src.windowing import segment_all_subjects

SENSORS = ("wrist", "chest", "ankle")
POLICIES = {
    "original": {},
    "conservative_switching": {"u_high": 0.75, "u_low": 0.45, "min_dwell_windows": 8, "probe_interval": 40},
    "early_reinforcement": {"u_high": 0.50, "u_low": 0.25, "min_dwell_windows": 8, "probe_interval": 12},
}
TUNED_RF = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2, "class_weight": "balanced"}
VALIDATION_FAULTS = (("dropout", "moderate"), ("noise", "moderate"), ("combined", "moderate"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_controller(cfg, variant):
    p = {**cfg["adaptive_policy"], **POLICIES[variant]}
    return AdaptiveSensingController(
        sensors=list(SENSORS), anchor_sensor=p["anchor_sensor"], r_low=p["r_low"],
        r_good=p["r_good"], u_high=p["u_high"], u_low=p["u_low"],
        min_dwell_windows=p["min_dwell_windows"], probe_interval=p["probe_interval"],
        cost_budget=p["cost_budget"],
        cost_weights={s: cfg["dataset"]["sensors"][s]["cost_weight"] for s in SENSORS},
    )


def make_faults(x, scenario, severity, seed):
    simulator = SensorFaultSimulator(random_seed=seed)
    result = {s: x[s].copy() for s in SENSORS}
    result["wrist"] = np.stack([
        simulator.inject_fault(window, scenario, severity)[0] for window in x["wrist"]
    ])
    return result


def pack(result, keep_log=False):
    out = {
        "metrics": result["metrics"], "cost": result["cost"],
        "host_latency_ms_per_window": result["latency_ms_per_window"],
    }
    if keep_log:
        out.update({k: result[k] for k in ("predictions", "masks", "actions", "uncertainties")})
    return out


def f1(result):
    return result["metrics"]["macro_f1_all_classes"]


def prepare(cfg, subjects):
    raw, _ = load_all_subjects(cfg["dataset"]["raw_dir"], subjects,
                               {int(k): v for k, v in cfg["dataset"]["activities"].items()})
    names = [name for s in SENSORS for name in SENSOR_CHANNELS[s]["names"]]
    clean = {sub: impute_short_gaps(raw[sub], names)[0] for sub in subjects}
    windows, labels, _ = segment_all_subjects(
        clean, cfg["dataset"]["window_size"], cfg["dataset"]["window_stride"],
        cfg["dataset"]["label_purity_threshold"])
    return windows, labels


def fold(cfg, historical, output, subject, windows, labels, full_fault_grid):
    subjects = list(cfg["dataset"]["all_subjects"])
    train_subjects = [s for s in subjects if s != subject]
    val_subject = train_subjects[-1]
    fit_subjects = [s for s in train_subjects if s != val_subject]
    model_dir = historical / f"subject_{subject}_models"
    scaler_info = json.loads((model_dir / "scaler.json").read_text())
    scaler = SubjectAwareScaler(scaler_info["channel_cols"])
    scaler.means, scaler.stds, scaler.is_fitted = scaler_info["means"], scaler_info["stds"], True

    def transform(sub):
        return {s: scaler.transform_array(windows[sub][s], SENSOR_CHANNELS[s]["names"]) for s in SENSORS}

    x_val, x_test = transform(val_subject), transform(subject)
    y_val, y_test = labels[val_subject], labels[subject]
    x_train = {s: np.concatenate([transform(sub)[s] for sub in fit_subjects]) for s in SENSORS}
    y_train = np.concatenate([labels[sub] for sub in fit_subjects])
    classes = sorted(int(k) for k in cfg["dataset"]["activities"])
    class_names = {int(k): v for k, v in cfg["dataset"]["activities"].items()}

    original = {s: SensorRandomForest.load(model_dir / f"{s}_rf.joblib") for s in SENSORS}
    tuned = {}
    for s in SENSORS:
        tuned[s] = SensorRandomForest(s, classes, **TUNED_RF,
                                    random_state=cfg["experiment"]["random_seed"], n_jobs=-1)
        tuned[s].fit(x_train[s], y_train)
    estimators = {}
    for s in SENSORS:
        est = SensorReliabilityEstimator(
            weights=cfg["reliability"]["weights"],
            flatline_var_thresh=cfg["reliability"]["thresholds"]["flatline_var"],
            noise_diff_var_thresh=cfg["reliability"]["thresholds"]["noise_diff_var"],
        )
        est.calibrate_baseline(x_train[s])
        estimators[s] = est
    del x_train

    val_conditions = {"clean": x_val}
    for scenario, level in VALIDATION_FAULTS:
        val_conditions[f"{scenario}_{level}"] = make_faults(
            x_val, scenario, cfg["fault_injection"]["severities"][level],
            cfg["experiment"]["random_seed"] + val_subject * 1000 +
            list(cfg["fault_injection"]["scenarios"]).index(scenario) * 10 + 2,
        )

    model_validation = {}
    for key, models in (("original_50", original), ("tuned_100", tuned)):
        scores = {name: f1(run_baseline_all_uniform(x, y_val, models, classes, class_names))
                  for name, x in val_conditions.items()}
        objective = 0.35 * scores["clean"] + 0.25 * scores["dropout_moderate"] + 0.20 * scores["noise_moderate"] + 0.20 * scores["combined_moderate"]
        model_validation[key] = {"scores": scores, "objective": objective}
    model_key = max(("original_50", "tuned_100"), key=lambda k: (model_validation[k]["objective"], k == "original_50"))
    models = original if model_key == "original_50" else tuned

    policy_validation = {}
    for name in POLICIES:
        runs = {condition: run_proposed_adaptive(x, y_val, models, estimators,
                make_controller(cfg, name), classes, class_names, offline_cache=True)
                for condition, x in val_conditions.items()}
        scores = {condition: f1(r) for condition, r in runs.items()}
        clean_cost = runs["clean"]["cost"]["mean_normalized_cost"]
        switch_rate = runs["clean"]["cost"]["total_switches"] / len(y_val)
        objective = (0.35 * scores["clean"] + 0.25 * scores["dropout_moderate"] +
                     0.20 * scores["noise_moderate"] + 0.20 * scores["combined_moderate"] -
                     0.10 * clean_cost - 0.02 * switch_rate)
        policy_validation[name] = {"scores": scores, "clean_cost": clean_cost,
                                   "switches_per_window": switch_rate, "objective": objective}
    policy_key = max(POLICIES, key=lambda k: (policy_validation[k]["objective"], k == "original"))

    results = {"subject": subject, "validation_subject": val_subject,
               "fit_subjects": fit_subjects, "train_windows": len(y_train),
               "validation_windows": len(y_val), "test_windows": len(y_test),
               "model_validation": model_validation, "policy_validation": policy_validation,
               "selected_model": model_key, "selected_policy": policy_key, "test": {}}
    if model_key == "tuned_100":
        selected_dir = output / f"subject_{subject}_selected_models"
        selected_dir.mkdir(parents=True, exist_ok=True)
        for s in SENSORS:
            tuned[s].save(selected_dir / f"{s}_rf.joblib")

    # Every method receives identical held-out windows for each condition.
    conditions = [("clean", None, None)]
    faults = list(cfg["fault_injection"]["scenarios"])
    levels = list(cfg["fault_injection"]["severities"])
    if not full_fault_grid:
        levels = ["moderate"]
    conditions += [(f"{scenario}_{level}", scenario, level) for scenario in faults for level in levels]
    for key, scenario, level in conditions:
        if scenario is None:
            x = x_test
        else:
            x = make_faults(x_test, scenario, cfg["fault_injection"]["severities"][level],
                            cfg["experiment"]["random_seed"] + subject * 1000 +
                            faults.index(scenario) * 10 + list(cfg["fault_injection"]["severities"]).index(level))
        condition = {}
        if key == "clean":
            condition["wrist_only"] = pack(run_baseline_wrist_only(x, y_test, models, classes, class_names))
            condition["wrist_chest"] = pack(run_baseline_wrist_chest_static(x, y_test, models, classes, class_names))
        condition["all_uniform"] = pack(run_baseline_all_uniform(x, y_test, models, classes, class_names))
        condition["all_weighted"] = pack(run_baseline_all_reliability(x, y_test, models, estimators, classes, class_names))
        for variant in dict.fromkeys(("original", policy_key)):
            name = "adaptive_original" if variant == "original" else "adaptive_selected"
            keep = key in ("clean", "dropout_moderate") and variant == policy_key
            outcome = run_proposed_adaptive(x, y_test, models, estimators,
                       make_controller(cfg, variant), classes, class_names, offline_cache=True)
            condition[name] = pack(outcome, keep_log=False)
            if keep:
                log = {"subject": subject, "condition": key, "policy": policy_key,
                       "y_true": y_test.tolist(), **{k: outcome[k] for k in ("predictions", "masks", "actions", "uncertainties")}}
                (output / f"subject_{subject}_{key}_selected_log.json").write_text(json.dumps(log), encoding="utf-8")
        if policy_key == "original": condition["adaptive_selected"] = condition["adaptive_original"]
        if key in ("clean", "dropout_moderate", "noise_moderate", "combined_moderate"):
            condition["historical_model_and_policy"] = pack(run_proposed_adaptive(
                x, y_test, original, estimators, make_controller(cfg, "original"),
                classes, class_names, offline_cache=True))
        results["test"][key] = condition
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-run", type=Path, default=Path("results/runs/full_fast_1792678712_34ed82"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--full-fault-grid", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    cfg = yaml.safe_load((args.historical_run / "config_snapshot.yaml").read_text(encoding="utf-8"))
    subjects = list(cfg["dataset"]["all_subjects"])
    folds = args.folds or subjects
    if any(s not in subjects for s in folds):
        raise ValueError("Every held-out subject must be in the historical eight-subject run")
    old = json.loads((args.historical_run / "experiment_summary.json").read_text(encoding="utf-8"))
    raw_root = Path(cfg["dataset"]["raw_dir"])
    for filename, info in old["dataset_manifest"].items():
        candidates = list(raw_root.rglob(filename))
        if not candidates or candidates[0].stat().st_size != info["bytes"] or digest(candidates[0]) != info["sha256"]:
            raise RuntimeError(f"Official dataset mismatch: {filename}")
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(json.dumps({
        "source_run": str(args.historical_run), "subjects": subjects, "folds": folds,
        "rf_candidates": {"original_50": "verified historical checkpoint", "tuned_100": TUNED_RF},
        "policy_candidates": POLICIES, "validation_faults": VALIDATION_FAULTS,
        "objective": "0.35 clean + 0.25 dropout + 0.20 noise + 0.20 combined fixed-six F1; policy minus 0.10 clean cost and 0.02 switches/window",
        "test_fault_grid": "all configured severities" if args.full_fault_grid else "moderate only",
        "seed": cfg["experiment"]["random_seed"], "dataset_manifest": old["dataset_manifest"],
        "config_snapshot": cfg,
    }, indent=2), encoding="utf-8")
    started = time.time()
    windows, labels = prepare(cfg, subjects)
    for subject in folds:
        print(f"Evaluating held-out subject {subject}", flush=True)
        record = fold(cfg, args.historical_run, args.output, subject, windows, labels, args.full_fault_grid)
        (args.output / f"subject_{subject}_results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Subject {subject}: model={record['selected_model']} policy={record['selected_policy']} clean-F1={record['test']['clean']['adaptive_selected']['metrics']['macro_f1_all_classes']:.4f}", flush=True)
    (args.output / "completed.json").write_text(json.dumps({"folds": folds, "seconds": time.time()-started}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
