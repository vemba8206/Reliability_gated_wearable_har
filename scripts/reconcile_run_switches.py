"""Recalculate switch counts from saved window masks for older runs."""

import argparse
import json
from pathlib import Path


def reconcile(run_dir: Path) -> None:
    summary_path = run_dir / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = []
    lengths = []
    for subject in summary["subjects"]:
        path = run_dir / f"subject_{subject}_clean_adaptive_log.json"
        if not path.exists():
            continue
        log = json.loads(path.read_text(encoding="utf-8"))
        masks = log["masks"]
        counts.append(sum(sum(old[s] != new[s] for s in old) for old, new in zip(masks, masks[1:])))
        lengths.append(len(masks))
    if not counts:
        raise FileNotFoundError("No clean adaptive window logs")
    for row in summary["cost_comparison"]:
        if row["Strategy"] == "Proposed Reliability-Gated Adaptive":
            row["Switching Count"] = str(int(sum(counts) / len(counts)))
    for row in summary["overall_synthesis"]:
        if row["Dimension"] == "Switches per test window":
            row["Proposed Adaptive"] = f"{sum(counts) / sum(lengths):.4f}"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{run_dir}: {counts} observed toggles over {lengths} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    reconcile(parser.parse_args().run_dir)
