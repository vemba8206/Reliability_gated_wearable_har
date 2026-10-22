"""Aggregate saved nested RF/policy fold records without rerunning models."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    completed = json.loads((root / "completed.json").read_text(encoding="utf-8"))
    folds = []
    for sub in completed["folds"]:
        plain = root / f"subject_{sub}_results.json"
        packed = root / f"subject_{sub}_results.json.gz"
        if plain.exists():
            folds.append(json.loads(plain.read_text(encoding="utf-8")))
        else:
            with gzip.open(packed, "rt", encoding="utf-8") as stream:
                folds.append(json.load(stream))
    assert len(folds) == 8 and len({f["subject"] for f in folds}) == 8, "Expected eight distinct LOSO folds"
    methods = ("wrist_only", "wrist_chest", "all_uniform", "all_weighted",
               "historical_model_and_policy", "adaptive_original", "adaptive_selected")
    clean_rows = []
    for method in methods:
        entries = [f["test"]["clean"][method] for f in folds]
        f1 = np.array([v["metrics"]["macro_f1_all_classes"] for v in entries])
        observed = np.array([v["metrics"]["macro_f1"] for v in entries])
        cost = np.array([v["cost"]["mean_normalized_cost"] for v in entries])
        switches = sum(v["cost"]["total_switches"] for v in entries)
        clean_rows.append({"method": method, "mean_fixed_six_macro_f1": float(f1.mean()),
                           "subject_sd_fixed_six_macro_f1": float(f1.std()),
                           "mean_historical_observed_label_macro_f1": float(observed.mean()),
                           "mean_clean_cost_proxy": float(cost.mean()),
                           "total_switches": int(switches),
                           "switches_per_test_window": float(switches/sum(f["test_windows"] for f in folds))})
    pd.DataFrame(clean_rows).to_csv(root / "clean_comparison.csv", index=False)
    fault_rows = []
    conditions = [key for key in folds[0]["test"] if key != "clean"]
    for condition in conditions:
        for method in ("all_uniform", "all_weighted", "adaptive_original", "adaptive_selected"):
            entries = [f["test"][condition][method] for f in folds]
            scores = np.array([v["metrics"]["macro_f1_all_classes"] for v in entries])
            costs = np.array([v["cost"]["mean_normalized_cost"] for v in entries])
            fault_rows.append({"condition": condition, "method": method,
                               "mean_fixed_six_macro_f1": float(scores.mean()),
                               "subject_sd_fixed_six_macro_f1": float(scores.std()),
                               "mean_cost_proxy": float(costs.mean())})
    pd.DataFrame(fault_rows).to_csv(root / "fault_grid.csv", index=False)
    summary = {"subjects": completed["folds"], "total_test_windows": sum(f["test_windows"] for f in folds),
               "selected_models": dict(Counter(f["selected_model"] for f in folds)),
               "selected_policies": dict(Counter(f["selected_policy"] for f in folds)),
               "clean_comparison": clean_rows, "fault_grid": fault_rows,
               "score_definition": "fixed six-class macro-F1; subject means and population SD over eight LOSO folds",
               "cost_definition": "normalized active IMU acquisition/communication proxy; not measured energy"}
    (root / "aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for row in clean_rows:
        ax.scatter(row["mean_clean_cost_proxy"], row["mean_fixed_six_macro_f1"],
                   s=45, label=row["method"].replace("_", " "))
    ax.set(xlabel="Mean normalized sensing-cost proxy", ylabel="Fixed six-class macro-F1",
           title="Nested eight-subject clean evaluation", xlim=(0.25, 1.08))
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.subplots_adjust(right=0.66, left=0.10, bottom=0.17, top=0.87)
    fig.savefig(figures / "clean_cost_tradeoff.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for method in ("all_uniform", "all_weighted", "adaptive_selected"):
        vals = [next(r["mean_fixed_six_macro_f1"] for r in fault_rows
                     if r["condition"] == f"dropout_{level}" and r["method"] == method)
                for level in ("mild", "moderate", "severe")]
        ax.plot([0.25, 0.5, 0.85], vals, marker="o", label=method.replace("_", " "))
    ax.set(xlabel="Wrist dropout severity", ylabel="Fixed six-class macro-F1",
           title="Nested eight-subject wrist dropout")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "dropout_severity.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"clean": clean_rows, "selected_models": summary["selected_models"],
                      "selected_policies": summary["selected_policies"]}, indent=2))


if __name__ == "__main__":
    main()
