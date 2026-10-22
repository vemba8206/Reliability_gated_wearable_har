"""Master experiment runner: LOSO evaluation, fault injection grid, baseline comparison, and artifact generation."""

import os
import sys
import time
import uuid
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import yaml

from src.data_download import acquire_dataset, find_existing_subject_files
from src.data_loader import load_all_subjects, DEFAULT_ACTIVITIES
from src.preprocessing import impute_short_gaps, SubjectAwareScaler
from src.windowing import segment_all_subjects
from src.train_random_forest import SensorRandomForest
from src.train_cnn_lstm import SensorCNNLSTM
from src.reliability import SensorReliabilityEstimator
from src.adaptive_policy import AdaptiveSensingController
from src.fault_injection import SensorFaultSimulator
from src.baselines import (
    run_baseline_wrist_only,
    run_baseline_wrist_chest_static,
    run_baseline_all_uniform,
    run_baseline_all_reliability,
    run_proposed_adaptive
)
from src.evaluation import compute_reliability_detection_metrics
from src.plots import (
    plot_activity_distribution,
    plot_confusion_matrix,
    plot_macro_f1_vs_severity,
    plot_cost_vs_macro_f1,
    plot_reliability_and_activation_timeline,
    plot_cnn_training_curves,
    plot_rf_feature_importance
)
from src.report_exports import ReportArtifactExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment_pipeline(
    mode: str = "quick",
    config_path: str = "config/default.yaml",
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Execute complete end-to-end research experiment pipeline."""
    t_start = time.time()
    cfg = load_config(config_path)
    run_id = f"{mode}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    if save_dir is None:
        run_dir = Path(cfg["paths"]["results_dir"]) / run_id
    else:
        run_dir = Path(save_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Experiment Pipeline (Run ID: {run_id}, Mode: {mode.upper()}) ===")

    # 1. Dataset Acquisition
    raw_dir = cfg["dataset"]["raw_dir"]
    if (Path(raw_dir) / "SYNTHETIC_DATA_DO_NOT_USE_FOR_RESEARCH").exists():
        raise RuntimeError("Synthetic smoke-test data cannot be used for research experiments.")
    requested_subjects = cfg["dataset"].get("quick_subjects", [101, 102, 105]) if mode in ("quick", "three_subject_loso") else cfg["dataset"].get("all_subjects", list(range(101, 110)))
    present_subjects = {int(p.stem.replace("subject", "")) for p in find_existing_subject_files(raw_dir)}
    if set(requested_subjects) <= present_subjects:
        acquired, msg = True, f"Requested subject files already present in {raw_dir}."
    else:
        acquired, msg = acquire_dataset(raw_dir=raw_dir)
    if not acquired:
        raise FileNotFoundError(msg)
    dataset_manifest = {}
    for subject_file in find_existing_subject_files(raw_dir):
        digest = hashlib.sha256()
        with subject_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        dataset_manifest[subject_file.name] = {"sha256": digest.hexdigest(), "bytes": subject_file.stat().st_size}
    logger.info(f"Dataset status: {msg}")

    # 2. Select subjects and activities
    if mode in ("quick", "three_subject_loso"):
        subjects = cfg["dataset"].get("quick_subjects", [101, 102, 105])
    else:
        subjects = cfg["dataset"].get("all_subjects", [101, 102, 103, 104, 105, 106, 107, 108, 109])

    activity_cfg = cfg["dataset"].get("activities", DEFAULT_ACTIVITIES)
    # Ensure keys are integers
    activities = {int(k): str(v) for k, v in activity_cfg.items()}
    canonical_classes = sorted(list(activities.keys()))

    logger.info(f"Loading data for subjects: {subjects}")
    raw_by_sub, stats_by_sub = load_all_subjects(
        raw_dir=raw_dir,
        selected_subjects=subjects,
        selected_activities=activities
    )
    missing_subjects = set(subjects) - set(raw_by_sub)
    if missing_subjects:
        raise FileNotFoundError(f"Missing requested official PAMAP2 subject files: {sorted(missing_subjects)}")

    # 3. Preprocessing (Short-gap Imputation per subject)
    clean_by_sub = {}
    channel_names = []
    from src.data_loader import SENSOR_CHANNELS
    for s in ["wrist", "chest", "ankle"]:
        channel_names.extend(SENSOR_CHANNELS[s]["names"])

    for sub_id, df in raw_by_sub.items():
        imputed_df, nan_counts = impute_short_gaps(df, channel_names)
        clean_by_sub[sub_id] = imputed_df

    # 4. Windowing
    win_size = cfg["dataset"]["window_size"]
    stride = cfg["dataset"]["window_stride"]
    purity_thresh = cfg["dataset"]["label_purity_threshold"]

    windows_by_sub, labels_by_sub, meta_by_sub = segment_all_subjects(
        clean_by_sub,
        window_size=win_size,
        stride=stride,
        purity_threshold=purity_thresh
    )

    total_windows = sum(len(y) for y in labels_by_sub.values())
    empty_subjects = [subject for subject in subjects if len(labels_by_sub[subject]) == 0]
    if empty_subjects:
        raise ValueError(f"Selected subjects have no windows for configured activities: {empty_subjects}")
    logger.info(f"Windowing complete: {total_windows} total windows across {len(subjects)} subjects.")

    # 5. Outer Evaluation Protocol (LOSO Cross-Validation)
    # For quick mode: use subject 105 as test, {101, 102} as train
    # For full mode: iterate through all subjects as test folds
    test_subjects = [subjects[-1]] if mode == "quick" else subjects

    fold_results = []
    fault_simulator = SensorFaultSimulator(random_seed=cfg["experiment"]["random_seed"])

    # Collect timelines and feature importances for plots
    last_rf_models = {}
    last_cnn_models = {}
    last_estimators = {}
    last_test_windows = {}
    last_y_test = None
    last_cnn_history = {}
    last_adaptive_res = None
    last_all_rel_res = None

    for fold_idx, test_sub in enumerate(test_subjects):
        train_subs = [s for s in subjects if s != test_sub]
        val_sub = train_subs[-1] if len(train_subs) > 1 else train_subs[0]
        train_fit_subs = [s for s in train_subs if s != val_sub] or [train_subs[0]]

        logger.info(f"--- Fold {fold_idx + 1}/{len(test_subjects)}: Held-out Test Subject: {test_sub} (Train: {train_fit_subs}, Val: {val_sub}) ---")

        # Fit Scaler strictly on training subjects
        scaler = SubjectAwareScaler(channel_names)
        scaler.fit([clean_by_sub[s] for s in train_fit_subs])

        # Prepare Train & Val Arrays
        train_windows = {s: [] for s in ["wrist", "chest", "ankle"]}
        train_y_list = []
        for s in train_fit_subs:
            for sensor in ["wrist", "chest", "ankle"]:
                # Normalize windows using train scaler
                norm_w = scaler.transform_array(windows_by_sub[s][sensor], SENSOR_CHANNELS[sensor]["names"])
                train_windows[sensor].append(norm_w)
            train_y_list.append(labels_by_sub[s])

        X_train = {sensor: np.concatenate(train_windows[sensor], axis=0) for sensor in ["wrist", "chest", "ankle"]}
        y_train = np.concatenate(train_y_list, axis=0)

        # Prepare Val Arrays
        X_val = {sensor: scaler.transform_array(windows_by_sub[val_sub][sensor], SENSOR_CHANNELS[sensor]["names"]) for sensor in ["wrist", "chest", "ankle"]}
        y_val = labels_by_sub[val_sub]

        # Prepare Test Arrays (Held-out subject)
        X_test = {sensor: scaler.transform_array(windows_by_sub[test_sub][sensor], SENSOR_CHANNELS[sensor]["names"]) for sensor in ["wrist", "chest", "ankle"]}
        y_test = labels_by_sub[test_sub]

        # Train Sensor-Wise Random Forests
        rf_models = {}
        rf_cfg = cfg["models"]["random_forest"]
        for sensor in ["wrist", "chest", "ankle"]:
            rf = SensorRandomForest(
                sensor_name=sensor,
                class_list=canonical_classes,
                n_estimators=rf_cfg["n_estimators"] if mode == "full" else 50,
                max_depth=rf_cfg["max_depth"],
                min_samples_leaf=rf_cfg["min_samples_leaf"],
                class_weight=rf_cfg["class_weight"],
                random_state=rf_cfg["random_state"]
            )
            rf.fit(X_train[sensor], y_train)
            rf_models[sensor] = rf

        # Train Sensor-Wise CNN-LSTMs
        cnn_models = {}
        cnn_cfg = cfg["models"]["cnn_lstm"]
        cnn_epochs = cnn_cfg["epochs"] if mode == "full" else 5
        for sensor in ["wrist", "chest", "ankle"]:
            cnn = SensorCNNLSTM(
                sensor_name=sensor,
                class_list=canonical_classes,
                in_channels=6,
                conv1_filters=cnn_cfg["conv1_filters"],
                conv1_kernel=cnn_cfg["conv1_kernel"],
                conv2_filters=cnn_cfg["conv2_filters"],
                conv2_kernel=cnn_cfg["conv2_kernel"],
                lstm_hidden=cnn_cfg["lstm_hidden"],
                dropout=cnn_cfg["dropout"],
                learning_rate=cnn_cfg["learning_rate"],
                epochs=cnn_epochs,
                patience=cnn_cfg["patience"],
                random_state=cnn_cfg["random_state"]
            )
            cnn.fit(X_train[sensor], y_train, X_val[sensor], y_val)
            cnn_models[sensor] = cnn

        fold_model_dir = run_dir / f"subject_{test_sub}_models"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        for sensor in ["wrist", "chest", "ankle"]:
            rf_models[sensor].save(fold_model_dir / f"{sensor}_rf.joblib")
            cnn_models[sensor].save(fold_model_dir / f"{sensor}_cnn.pt")
        with (fold_model_dir / "scaler.json").open("w", encoding="utf-8") as stream:
            json.dump(scaler.to_dict(), stream, indent=2)

        # Calibrate Sensor Reliability Estimators on training data
        estimators = {}
        for sensor in ["wrist", "chest", "ankle"]:
            est = SensorReliabilityEstimator(
                weights=cfg["reliability"]["weights"],
                flatline_var_thresh=cfg["reliability"]["thresholds"]["flatline_var"],
                noise_diff_var_thresh=cfg["reliability"]["thresholds"]["noise_diff_var"],
                clipping_acc_thresh=cfg["reliability"]["thresholds"]["clipping_accel"]
            )
            est.calibrate_baseline(X_train[sensor])
            estimators[sensor] = est

        # Instantiate Adaptive Sensing Controller
        controller = AdaptiveSensingController(
            sensors=["wrist", "chest", "ankle"],
            anchor_sensor=cfg["adaptive_policy"]["anchor_sensor"],
            r_low=cfg["adaptive_policy"]["r_low"],
            r_good=cfg["adaptive_policy"]["r_good"],
            u_high=cfg["adaptive_policy"]["u_high"],
            u_low=cfg["adaptive_policy"]["u_low"],
            min_dwell_windows=cfg["adaptive_policy"]["min_dwell_windows"],
            probe_interval=cfg["adaptive_policy"]["probe_interval"],
            cost_budget=cfg["adaptive_policy"]["cost_budget"],
            cost_weights={s: cfg["dataset"]["sensors"][s]["cost_weight"] for s in ["wrist", "chest", "ankle"]}
        )

        # --- Evaluate Clean Condition across Baselines ---
        clean_res = {
            "wrist_only": run_baseline_wrist_only(X_test, y_test, rf_models, canonical_classes, activities),
            "wrist_chest": run_baseline_wrist_chest_static(X_test, y_test, rf_models, canonical_classes, activities),
            "all_uniform": run_baseline_all_uniform(X_test, y_test, rf_models, canonical_classes, activities),
            "all_reliability": run_baseline_all_reliability(X_test, y_test, rf_models, estimators, canonical_classes, activities),
            "proposed_adaptive": run_proposed_adaptive(X_test, y_test, rf_models, estimators, controller, canonical_classes, activities)
        }

        # Also evaluate CNN-LSTM on clean condition
        cnn_clean_res = {
            "all_reliability": run_baseline_all_reliability(X_test, y_test, cnn_models, estimators, canonical_classes, activities),
            "proposed_adaptive": run_proposed_adaptive(X_test, y_test, cnn_models, estimators, controller, canonical_classes, activities)
        }
        with (run_dir / f"subject_{test_sub}_clean_adaptive_log.json").open("w", encoding="utf-8") as stream:
            json.dump({"subject": test_sub, "y_true": y_test.tolist(),
                       "predictions": clean_res["proposed_adaptive"]["predictions"],
                       "masks": clean_res["proposed_adaptive"]["masks"],
                       "actions": clean_res["proposed_adaptive"]["actions"],
                       "reliabilities": clean_res["proposed_adaptive"]["reliabilities"],
                       "uncertainties": clean_res["proposed_adaptive"]["uncertainties"]}, stream)

        # --- Evaluate Faulted Conditions ---
        # Inject controlled fault onto target sensor (wrist anchor or chest)
        # Scenarios: dropout, intermittent, noise, flatline, drift
        fault_scenarios = cfg["fault_injection"]["scenarios"]
        fault_severities = cfg["fault_injection"]["severities"]
        fault_results = {}

        for scenario in fault_scenarios:
            fault_results[scenario] = {}
            for sev_name, sev_val in fault_severities.items():
                corrupted_test_w = {k: np.copy(v) for k, v in X_test.items()}
                # Corrupt wrist sensor
                target_sensor = "wrist"
                corrupted_stream = []
                injected_labels = []
                for i in range(len(y_test)):
                    corrupted_win, meta = fault_simulator.inject_fault(
                        corrupted_test_w[target_sensor][i],
                        fault_type=scenario,
                        severity=sev_val
                    )
                    corrupted_stream.append(corrupted_win)
                    injected_labels.append(1 if meta["is_faulty"] else 0)

                corrupted_test_w[target_sensor] = np.stack(corrupted_stream, axis=0)
                injected_labels_arr = np.array(injected_labels)

                # Re-run baselines on corrupted test sequence
                b_all_uniform = run_baseline_all_uniform(corrupted_test_w, y_test, rf_models, canonical_classes, activities)
                b_all_rel = run_baseline_all_reliability(corrupted_test_w, y_test, rf_models, estimators, canonical_classes, activities)
                b_adaptive = run_proposed_adaptive(corrupted_test_w, y_test, rf_models, estimators, controller, canonical_classes, activities, offline_cache=True)

                # Reliability fault-detection performance
                clean_rels = np.array(clean_res["all_reliability"]["reliabilities"][target_sensor])
                fault_rels = np.array(b_all_rel["reliabilities"][target_sensor])
                rel_detect = compute_reliability_detection_metrics(
                    np.concatenate([np.zeros(len(clean_rels), dtype=int), injected_labels_arr]),
                    np.concatenate([clean_rels, fault_rels])
                )

                fault_results[scenario][sev_name] = {
                    "uniform_f1": b_all_uniform["metrics"]["macro_f1"],
                    "reliability_f1": b_all_rel["metrics"]["macro_f1"],
                    "adaptive_f1": b_adaptive["metrics"]["macro_f1"],
                    "adaptive_cost": b_adaptive["cost"]["mean_normalized_cost"],
                    "reliability_detection": rel_detect
                }

                if scenario == "dropout" and sev_name == "moderate":
                    with (run_dir / f"subject_{test_sub}_dropout_moderate_adaptive_log.json").open("w", encoding="utf-8") as stream:
                        json.dump({"subject": test_sub, "y_true": y_test.tolist(),
                                   "predictions": b_adaptive["predictions"], "masks": b_adaptive["masks"],
                                   "actions": b_adaptive["actions"], "reliabilities": b_adaptive["reliabilities"],
                                   "uncertainties": b_adaptive["uncertainties"]}, stream)
                    last_adaptive_res = b_adaptive
                    last_all_rel_res = b_all_rel

        fold_record = {
            "test_subject": test_sub,
            "clean_results": clean_res,
            "cnn_clean_results": cnn_clean_res,
            "fault_results": fault_results
        }
        fold_results.append(fold_record)

        last_rf_models = rf_models
        last_cnn_models = cnn_models
        last_estimators = estimators
        last_test_windows = X_test
        last_y_test = y_test
        last_cnn_history = cnn_models["wrist"].history

    # 6. Aggregate Experiment Metrics
    logger.info("Aggregating multi-fold experimental outcomes...")

    # Clean comparison table data
    clean_table_rows = []
    baseline_keys = [
        ("wrist_only", "Wrist Only (Single Anchor)", "RF"),
        ("wrist_chest", "Wrist + Chest (Static Uniform)", "RF"),
        ("all_uniform", "All Sensors (Uniform Fusion)", "RF"),
        ("all_reliability", "All Sensors (Reliability-Weighted)", "RF"),
        ("proposed_adaptive", "Proposed Reliability-Gated Adaptive", "RF")
    ]
    for b_key, b_name, m_fam in baseline_keys:
        accs = [f["clean_results"][b_key]["metrics"]["accuracy"] for f in fold_results]
        f1s = [f["clean_results"][b_key]["metrics"]["macro_f1"] for f in fold_results]
        costs = [f["clean_results"][b_key]["cost"]["mean_normalized_cost"] for f in fold_results]
        lats = [f["clean_results"][b_key]["latency_ms_per_window"] for f in fold_results]

        clean_table_rows.append({
            "Baseline Architecture": b_name,
            "Classifier": m_fam,
            "Accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
            "Macro-F1": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
            "Normalized Cost C(t)": f"{np.mean(costs):.2f}",
            "Inference Latency (ms)": f"{np.mean(lats):.2f}"
        })

    # Add CNN-LSTM proposed adaptive
    cnn_accs = [f["cnn_clean_results"]["proposed_adaptive"]["metrics"]["accuracy"] for f in fold_results]
    cnn_f1s = [f["cnn_clean_results"]["proposed_adaptive"]["metrics"]["macro_f1"] for f in fold_results]
    cnn_costs = [f["cnn_clean_results"]["proposed_adaptive"]["cost"]["mean_normalized_cost"] for f in fold_results]
    cnn_lats = [f["cnn_clean_results"]["proposed_adaptive"]["latency_ms_per_window"] for f in fold_results]
    clean_table_rows.append({
        "Baseline Architecture": "Proposed Reliability-Gated Adaptive",
        "Classifier": "CNN-LSTM",
        "Accuracy": f"{np.mean(cnn_accs):.4f} +/- {np.std(cnn_accs):.4f}",
        "Macro-F1": f"{np.mean(cnn_f1s):.4f} +/- {np.std(cnn_f1s):.4f}",
        "Normalized Cost C(t)": f"{np.mean(cnn_costs):.2f}",
        "Inference Latency (ms)": f"{np.mean(cnn_lats):.2f}"
    })

    # Faulted comparison rows
    faulted_table_rows = []
    scenarios = fault_scenarios
    for sc in scenarios:
        unif_f1 = np.mean([f["fault_results"][sc]["moderate"]["uniform_f1"] for f in fold_results])
        rel_f1 = np.mean([f["fault_results"][sc]["moderate"]["reliability_f1"] for f in fold_results])
        adapt_f1 = np.mean([f["fault_results"][sc]["moderate"]["adaptive_f1"] for f in fold_results])
        adapt_cost = np.mean([f["fault_results"][sc]["moderate"]["adaptive_cost"] for f in fold_results])

        clean_ref_f1 = np.mean([f["clean_results"]["proposed_adaptive"]["metrics"]["macro_f1"] for f in fold_results])
        deg_pp = (clean_ref_f1 - adapt_f1) * 100.0

        faulted_table_rows.append({
            "Fault Scenario (50% Severity)": sc.capitalize(),
            "All Uniform F1": f"{unif_f1:.4f}",
            "All Rel-Weighted F1": f"{rel_f1:.4f}",
            "Proposed Adaptive F1": f"{adapt_f1:.4f}",
            "F1 Degradation (pp)": f"{deg_pp:.2f} pp",
            "Normalized Cost C(t)": f"{adapt_cost:.2f}"
        })

    # Cost comparison table rows
    cost_table_rows = []
    for b_key, b_name, _ in baseline_keys:
        costs = [f["clean_results"][b_key]["cost"]["mean_normalized_cost"] for f in fold_results]
        ratios = [f["clean_results"][b_key]["cost"]["active_sensor_ratio"] for f in fold_results]
        switches = [f["clean_results"][b_key]["cost"]["total_switches"] for f in fold_results]
        cost_table_rows.append({
            "Strategy": b_name,
            "Mean Normalized Cost": f"{np.mean(costs):.2f}",
            "Active Sensor Ratio": f"{np.mean(ratios):.2f}",
            "Switching Count": f"{int(np.mean(switches))}",
            "Cost Reduction Proxy": f"{(1.0 - np.mean(costs)) * 100.0:.1f}%"
        })

    # Latency comparison table rows
    latency_table_rows = [
        {"Model": f"{sensor.title()} RF", "Parameters / Trees": str(last_rf_models[sensor].clf.n_estimators),
         "Training Time (s)": f"{last_rf_models[sensor].train_duration_sec:.2f}", "Inference Latency (ms)": "not measured"}
        for sensor in ["wrist", "chest", "ankle"]
    ] + [
        {"Model": f"{sensor.title()} CNN-LSTM", "Parameters / Trees": "see checkpoint",
         "Training Time (s)": f"{last_cnn_models[sensor].train_duration_sec:.2f}", "Inference Latency (ms)": "not measured"}
        for sensor in ["wrist", "chest", "ankle"]
    ]

    # Per-subject results table rows
    per_subject_rows = []
    for f in fold_results:
        sub_id = f["test_subject"]
        sub_f1_clean = f["clean_results"]["proposed_adaptive"]["metrics"]["macro_f1"]
        sub_f1_fault = f["fault_results"]["dropout"]["moderate"]["adaptive_f1"]
        sub_clean_cost = f["clean_results"]["proposed_adaptive"]["cost"]["mean_normalized_cost"]
        sub_fault_cost = f["fault_results"]["dropout"]["moderate"]["adaptive_cost"]
        per_subject_rows.append({
            "Subject ID": f"Subject {sub_id}",
            "Clean Macro-F1": f"{sub_f1_clean:.4f}",
            "Faulted Macro-F1 (Dropout)": f"{sub_f1_fault:.4f}",
            "Clean Cost Proxy": f"{sub_clean_cost:.2f}",
            "Moderate-Dropout Cost Proxy": f"{sub_fault_cost:.2f}"
        })

    # Overall synthesis
    best_clean_val = float(np.mean([f["clean_results"]["proposed_adaptive"]["metrics"]["macro_f1"] for f in fold_results]))
    best_faulted_val = float(np.mean([f["fault_results"]["dropout"]["moderate"]["adaptive_f1"] for f in fold_results]))
    avg_proposed_cost = float(np.mean([f["clean_results"]["proposed_adaptive"]["cost"]["mean_normalized_cost"] for f in fold_results]))
    mean_fault_auroc = float(np.nanmean([
        f["fault_results"][scenario][severity]["reliability_detection"]["auroc"]
        for f in fold_results for scenario in fault_scenarios for severity in fault_severities
    ]))

    overall_synthesis_rows = [
        {"Dimension": "Clean Macro-F1", "All Uniform Baseline": f"{np.mean([f['clean_results']['all_uniform']['metrics']['macro_f1'] for f in fold_results]):.4f}", "Proposed Adaptive": f"{best_clean_val:.4f}", "Advantage": f"{(best_clean_val - np.mean([f['clean_results']['all_uniform']['metrics']['macro_f1'] for f in fold_results])) * 100:.1f} pp"},
        {"Dimension": "Faulted Macro-F1 (50% Dropout)", "All Uniform Baseline": f"{np.mean([f['fault_results']['dropout']['moderate']['uniform_f1'] for f in fold_results]):.4f}", "Proposed Adaptive": f"{best_faulted_val:.4f}", "Advantage": f"{(best_faulted_val - np.mean([f['fault_results']['dropout']['moderate']['uniform_f1'] for f in fold_results])) * 100.0:.1f} pp"},
        {"Dimension": "Normalized Sensing Cost", "All Uniform Baseline": "1.00", "Proposed Adaptive": f"{avg_proposed_cost:.2f}", "Advantage": f"{(1.0 - avg_proposed_cost) * 100.0:.1f}% reduction"},
        {"Dimension": "Fault Detection AUROC", "All Uniform Baseline": "N/A", "Proposed Adaptive": f"{mean_fault_auroc:.4f}", "Advantage": "across injected faults"},
        {"Dimension": "Switches per test window", "All Uniform Baseline": "0", "Proposed Adaptive": f"{np.mean([f['clean_results']['proposed_adaptive']['cost']['total_switches'] for f in fold_results]) / max(1, np.mean([len(labels_by_sub[f['test_subject']]) for f in fold_results])):.4f}", "Advantage": "observed mask toggles"}
    ]

    experiment_summary = {
        "run_id": run_id,
        "mode": mode,
        "subjects": subjects,
        "dataset_manifest": dataset_manifest,
        "duration_sec": time.time() - t_start,
        "best_clean_f1": best_clean_val,
        "best_faulted_f1": best_faulted_val,
        "proposed_avg_cost": avg_proposed_cost,
        "clean_comparison": clean_table_rows,
        "faulted_comparison": faulted_table_rows,
        "cost_comparison": cost_table_rows,
        "latency_comparison": latency_table_rows,
        "per_subject_results": per_subject_rows,
        "overall_synthesis": overall_synthesis_rows,
        "reliability_detection": {scenario: {severity: [f["fault_results"][scenario][severity]["reliability_detection"] for f in fold_results]
            for severity in fault_severities} for scenario in fault_scenarios}
    }

    # 7. Generate Figures and Tables
    logger.info("Generating report-ready figures and LaTeX tables...")
    exporter = ReportArtifactExporter(output_dir=cfg["paths"]["report_assets_dir"])

    # Figure 1: Activity distribution
    act_counts = {}
    for sub_id, df in clean_by_sub.items():
        for act_id, count in df["activity_id"].value_counts().items():
            name = activities.get(int(act_id), f"Act_{act_id}")
            act_counts[name] = act_counts.get(name, 0) + int(count)
    plot_activity_distribution(act_counts, exporter.figures_dir / "fig1_activity_distribution.png")

    # Figure 2: RF clean confusion matrix
    cm_clean_rf = np.array(fold_results[0]["clean_results"]["proposed_adaptive"]["metrics"]["confusion_matrix"])
    plot_confusion_matrix(cm_clean_rf, list(activities.values()), "RF Proposed Adaptive: Clean LOSO", exporter.figures_dir / "fig2_clean_confusion_matrix_rf.png")

    # Figure 3: CNN clean confusion matrix
    cm_clean_cnn = np.array(fold_results[0]["cnn_clean_results"]["proposed_adaptive"]["metrics"]["confusion_matrix"])
    plot_confusion_matrix(cm_clean_cnn, list(activities.values()), "CNN-LSTM Proposed Adaptive: Clean LOSO", exporter.figures_dir / "fig3_clean_confusion_matrix_cnn.png")

    # Figure 4: Faulted confusion matrix
    if last_adaptive_res:
        cm_fault = np.array(last_adaptive_res["metrics"]["confusion_matrix"])
        plot_confusion_matrix(cm_fault, list(activities.values()), "Proposed Adaptive Under Moderate Dropout", exporter.figures_dir / "fig4_faulted_confusion_matrix.png")

    # Figure 5 & 6: Macro-F1 vs Severity
    sev_levels = [0.0, 0.25, 0.50, 0.85]
    f0 = fold_results[0]
    rf_sev_scores = {key: [float(np.mean([f["clean_results"][clean_key]["metrics"]["macro_f1"] for f in fold_results]))] +
        [float(np.mean([f["fault_results"]["dropout"][severity][fault_key] for f in fold_results]))
         for severity in fault_severities]
        for key, clean_key, fault_key in [
            ("all_uniform", "all_uniform", "uniform_f1"),
            ("all_reliability", "all_reliability", "reliability_f1"),
            ("proposed_adaptive", "proposed_adaptive", "adaptive_f1")]}
    plot_macro_f1_vs_severity(sev_levels, rf_sev_scores, {}, exporter.figures_dir / "fig5_macro_f1_vs_severity.png")

    # Figure 7: Cost vs Macro-F1
    tradeoff_data = {name: {"cost": float(np.mean([f["clean_results"][key]["cost"]["mean_normalized_cost"] for f in fold_results])),
        "macro_f1": float(np.mean([f["clean_results"][key]["metrics"]["macro_f1"] for f in fold_results])), "label": name}
        for key, name in [("wrist_only", "Wrist Only"), ("wrist_chest", "Wrist + Chest"),
            ("all_uniform", "All Uniform"), ("all_reliability", "All Rel-Weighted"),
            ("proposed_adaptive", "Proposed Adaptive")]}
    plot_cost_vs_macro_f1(tradeoff_data, exporter.figures_dir / "fig7_cost_vs_macro_f1.png")

    # Figure 9 & 10: Timeline
    if last_adaptive_res:
        plot_reliability_and_activation_timeline(
            last_adaptive_res["reliabilities"],
            last_adaptive_res["masks"],
            exporter.figures_dir / "fig9_reliability_and_activation_timeline.png"
        )

    # Figure 11: CNN training curves
    plot_cnn_training_curves(last_cnn_history, exporter.figures_dir / "fig11_cnn_training_curves.png")

    # Figure 12: RF Feature Importances
    plot_rf_feature_importance(last_rf_models["wrist"].get_feature_importances(), "wrist", exporter.figures_dir / "fig12_rf_feature_importance.png")

    # Export All 10 Tables
    exporter.export_all_tables(experiment_summary)

    # Generate Markdown and LaTeX Narratives
    exporter.generate_narratives(experiment_summary)

    # Create Report Asset ZIP Bundle
    zip_path = exporter.create_report_bundle_zip()

    # Save run snapshot
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)

    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    logger.info(f"=== Experiment Pipeline Finished in {experiment_summary['duration_sec']:.2f}s ===")
    logger.info(f"Results stored in: {run_dir}")
    logger.info(f"Downloadable ZIP: {zip_path}")

    return experiment_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reliability-Gated HAR Experiments")
    parser.add_argument("--mode", type=str, choices=["quick", "three_subject_loso", "full_fast", "full"], default="quick", help="Execution mode")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()

    results = run_experiment_pipeline(mode=args.mode, config_path=args.config)
    print("\n--- Summary of Verified Results ---")
    print(f"Run ID: {results['run_id']}")
    print(f"Clean Macro-F1 (Proposed): {results['best_clean_f1']:.4f}")
    print(f"Faulted Macro-F1 (Proposed): {results['best_faulted_f1']:.4f}")
    print(f"Average Normalized Sensing Cost: {results['proposed_avg_cost']:.2f}")
    print(f"Total Duration: {results['duration_sec']:.1f} s")
