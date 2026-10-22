"""Create report plots only from a saved experiment summary."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.plots import plot_cost_vs_macro_f1


def generate(summary_path: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rf_rows = [row for row in summary["clean_comparison"] if row["Classifier"] == "RF"]
    short = {"Wrist Only (Single Anchor)": "Wrist only",
             "Wrist + Chest (Static Uniform)": "Wrist + chest",
             "All Sensors (Uniform Fusion)": "All uniform",
             "All Sensors (Reliability-Weighted)": "All weighted",
             "Proposed Reliability-Gated Adaptive": "Adaptive"}
    names = [short[row["Baseline Architecture"]] for row in rf_rows]
    means, stds = zip(*(map(float, row["Macro-F1"].split(" +/- ")) for row in rf_rows))
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ys = range(len(names))
    colors = ["#778b9a"] * 4 + ["#29865b"]
    ax.barh(ys, means, xerr=stds, color=colors, capsize=2, height=0.62)
    ax.set_yticks(list(ys), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Clean macro-F1 (mean and subject SD)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "clean_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    data = {short[row["Baseline Architecture"]]: {
        "cost": float(row["Normalized Cost C(t)"]), "macro_f1": float(row["Macro-F1"].split(" +/- ")[0])}
        for row in rf_rows}
    plot_cost_vs_macro_f1(data, output_dir / "fig7_cost_vs_macro_f1.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.summary, args.output)
