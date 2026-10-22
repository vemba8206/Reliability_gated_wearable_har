"""Publication-ready 300-DPI Matplotlib and Plotly figures for IEEE research report."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# IEEE two-column publication style guidelines
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_activity_distribution(
    activity_counts: Dict[str, int],
    save_path: Path
) -> None:
    """Figure 1: Activity-class distribution across protocol dataset."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    names = list(activity_counts.keys())
    counts = list(activity_counts.values())
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]

    bars = ax.bar(names, counts, color=colors[:len(names)], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Number of Usable Windows")
    ax.set_title("Dataset Activity Class Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path
) -> None:
    """Confusion matrix heatmap."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    cm_norm = cm.astype("float") / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Activity Label",
        xlabel="Predicted Activity Label",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_macro_f1_vs_severity(
    severities: List[float],
    rf_scores: Dict[str, List[float]],
    cnn_scores: Dict[str, List[float]],
    save_path: Path
) -> None:
    """Figure 6: Macro-F1 degradation across fault severities for baselines."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    styles = {
        "wrist_only": ("--", "gray", "Wrist Only"),
        "all_uniform": ("-.", "red", "All Sensors (Uniform)"),
        "all_reliability": (":", "blue", "All Sensors (Rel-Weighted)"),
        "proposed_adaptive": ("-", "green", "Proposed Adaptive Sensing")
    }

    for key, (ls, color, label) in styles.items():
        if key in rf_scores:
            ax.plot(severities, rf_scores[key], linestyle=ls, color=color, marker="o", label=f"RF: {label}")

    ax.set_xlabel("Fault Severity (0: Clean -> 1: Maximum Corruption)")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Recognition Robustness vs. Fault Severity")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cost_vs_macro_f1(
    baseline_results: Dict[str, Dict[str, float]],
    save_path: Path
) -> None:
    """Figure 8: Normalized Sensing Cost Proxy vs. Faulted Macro-F1 trade-off."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    markers = ["o", "s", "^", "D", "*"]
    colors = ["#7f7f7f", "#ff7f0e", "#d62728", "#1f77b4", "#2ca02c"]

    for idx, (name, data) in enumerate(baseline_results.items()):
        cost = data["cost"]
        f1 = data["macro_f1"]
        label = data.get("label", name)
        m = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        s = 180 if idx == len(baseline_results) - 1 else 100

        ax.scatter(cost, f1, s=s, marker=m, color=c, label=label, edgecolor="black", linewidth=1.0)
        ax.annotate(label, xy=(cost, f1), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Normalized Sensing Cost Proxy C(t)")
    ax.set_ylabel("Macro-F1 Score (Under Fault)")
    ax.set_title("Pareto Trade-off: Sensing Efficiency vs. Accuracy")
    ax.set_xlim([0.2, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_reliability_and_activation_timeline(
    timeline_reliabilities: Dict[str, List[float]],
    timeline_masks: List[Dict[str, int]],
    save_path: Path,
    max_windows: int = 100
) -> None:
    """Figure 10 & 11: Real-time sensor reliability trajectories and active configuration."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)

    n_pts = min(max_windows, len(timeline_masks))
    x = np.arange(n_pts)

    # Plot reliability
    colors = {"wrist": "#1f77b4", "chest": "#ff7f0e", "ankle": "#2ca02c"}
    for sensor, rels in timeline_reliabilities.items():
        ax1.plot(x, rels[:n_pts], label=f"{sensor.capitalize()} Reliability", color=colors.get(sensor, "black"), lw=1.5)

    ax1.axhline(0.40, color="red", linestyle="--", alpha=0.7, label="Low Reliability Threshold (R_LOW)")
    ax1.set_ylabel("Estimated Reliability")
    ax1.set_title("Dynamic Sensor Health & Adaptive Activation Timeline")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=7)
    ax1.set_ylim([-0.05, 1.05])

    # Plot active masks
    wrist_active = [timeline_masks[i]["wrist"] for i in range(n_pts)]
    chest_active = [timeline_masks[i]["chest"] for i in range(n_pts)]
    ankle_active = [timeline_masks[i]["ankle"] for i in range(n_pts)]

    ax2.step(x, np.array(wrist_active) * 0.9 + 2.0, where="post", label="Wrist Active", color="#1f77b4", lw=2)
    ax2.step(x, np.array(chest_active) * 0.9 + 1.0, where="post", label="Chest Active", color="#ff7f0e", lw=2)
    ax2.step(x, np.array(ankle_active) * 0.9 + 0.0, where="post", label="Ankle Active", color="#2ca02c", lw=2)

    ax2.set_yticks([0.45, 1.45, 2.45])
    ax2.set_yticklabels(["Ankle", "Chest", "Wrist"])
    ax2.set_ylabel("Active Sensor")
    ax2.set_xlabel("Time (Sensing Window index t)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim([-0.2, 3.2])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cnn_training_curves(
    history: Dict[str, List[float]],
    save_path: Path
) -> None:
    """Figure 12: CNN-LSTM training and validation loss curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    ax.plot(epochs, history.get("train_loss", []), "b-o", label="Training Loss", lw=1.5)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", lw=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("CNN-LSTM Convergence (LOSO Inner Split)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_rf_feature_importance(
    importances: Dict[str, float],
    sensor_name: str,
    save_path: Path,
    top_k: int = 12
) -> None:
    """Figure 13: Random Forest feature importances."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
    names = [x[0] for x in sorted_feats]
    vals = [x[1] for x in sorted_feats]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align="center", color="#3470a3", edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gini Feature Importance (Predictive Relevance)")
    ax.set_title(f"{sensor_name.capitalize()} RF Top Feature Importances")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
