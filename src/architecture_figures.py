"""Reproducible vector architecture figures for the IEEE manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, label, color="#e8f0f5"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                linewidth=0.9, edgecolor="#294455", facecolor=color))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)


def _arrow(ax, start, end, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color="#294455", connectionstyle=connectionstyle))


def _canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def make_architecture(path: Path):
    fig, ax = _canvas(7.1, 2.2)
    labels = ["Wearable\nIMUs", "Acquire and\npreprocess",
              "Quality and\nHAR models", "Fuse and\nentropy",
              "Budgeted\npolicy", "Gateway and\napplication"]
    xs = [0.01, 0.175, 0.34, 0.505, 0.67, 0.835]
    for x, label in zip(xs, labels):
        _box(ax, x, 0.55, 0.145, 0.27, label, "#e8f0f5" if x < 0.67 else "#e4f2e7")
    for x in xs[:-1]:
        _arrow(ax, (x + 0.145, 0.685), (x + 0.165, 0.685))
    _arrow(ax, (0.74, 0.53), (0.08, 0.44), "arc3,rad=-0.38")
    ax.text(0.50, 0.04, "Window t decision configures window t+1 acquisition", ha="center", fontsize=8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_policy(path: Path):
    fig, ax = _canvas(3.5, 3.8)
    labels = ["Window t: acquire active IMUs", "Estimate reliability of active IMUs",
              "Fuse predictions and compute entropy", "Check cost budget and dwell state",
              "KEEP / ACTIVATE / DEACTIVATE / PROBE", "Window t+1: acquire selected IMUs"]
    ys = [0.83, 0.69, 0.55, 0.41, 0.27, 0.13]
    for i, (y, label) in enumerate(zip(ys, labels)):
        _box(ax, 0.08, y, 0.84, 0.10, label, "#e4f2e7" if i >= 4 else "#e8f0f5")
        if i < len(labels) - 1:
            _arrow(ax, (0.5, y - 0.01), (0.5, ys[i + 1] + 0.11))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_fusion(path: Path):
    fig, ax = _canvas(3.5, 2.55)
    for x, name in [(0.04, "Wrist"), (0.36, "Chest"), (0.68, "Ankle")]:
        _box(ax, x, 0.75, 0.27, 0.15, f"{name} IMU\nclassifier")
        _box(ax, x, 0.48, 0.27, 0.13, "Quality r", "#fff1d6")
        _arrow(ax, (x + 0.135, 0.73), (x + 0.135, 0.62))
        _arrow(ax, (x + 0.135, 0.46), (0.5, 0.34))
    _box(ax, 0.18, 0.20, 0.64, 0.13, "Reliability-weighted fusion", "#e4f2e7")
    _arrow(ax, (0.5, 0.18), (0.5, 0.09))
    ax.text(0.5, 0.035, "Class probabilities and normalized entropy", ha="center", fontsize=8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    output = Path("overleaf_report/figures")
    output.mkdir(parents=True, exist_ok=True)
    make_architecture(output / "architecture.pdf")
    make_policy(output / "adaptive_policy.pdf")
    make_fusion(output / "model_workflow.pdf")
