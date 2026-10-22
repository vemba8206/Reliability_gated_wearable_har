# Reliability-Gated Adaptive Sensing for Fault-Resilient Human Activity Recognition in Wearable IoT

A reproducible research implementation of activity recognition from three wearable inertial measurement units (IMUs). Separate models classify wrist, chest, and ankle windows. An observable signal-quality score weights their predictions, and a rule-based controller chooses the sensors available in the **next** window. The experiments compare recognition under simulated faults with a normalized sensor-activation cost proxy. No battery energy or deployed radio system was measured.

## What is implemented

- Official PAMAP2 loading, exact IMU channel mapping, bounded missing-value imputation, timestamp-aware sliding windows, and training-subject-only scaling.
- Per-IMU random forests and optional CNN–LSTM classifiers, normalized prediction entropy, signal-quality estimation, reliability-weighted fusion, and an adaptive wrist-anchored controller with a budget, minimum dwell, and periodic probes.
- Wrist-only, fixed wrist-plus-chest, all-sensor uniform, all-sensor reliability-weighted, and adaptive comparisons.
- Seeded dropout, intermittent missingness, Gaussian noise, flatline, drift, and combined-fault simulation; clean and faulted metrics, sensing-cost and switching logs, plots, and subject-105 policy ablations.
- An exploratory nested eight-subject comparison of a 50-tree and 100-tree forest and three sensing policies. Its validation-selected configuration did **not** improve held-out recognition.

[System architecture diagram](overleaf_report/figures/architecture.pdf) · [IEEE LaTeX manuscript](overleaf_report/main.tex) · [Overleaf ZIP](overleaf_report.zip) · [Detailed handover](PROJECT_SUMMARY.md)

## Repository map

```text
config/default.yaml                  Dataset, model, fault, and policy settings
src/                                 Loading, preprocessing, models, fusion, policy, evaluation
scripts/nested_rf_policy_experiment.py  Inner-validation RF/policy comparison
scripts/summarize_nested_experiment.py  Aggregate saved nested-fold results
src/run_experiments.py              Original quick, three-subject, and eight-subject runs
src/run_ablations.py                Subject-105 one-factor replay
tests/                              Automated tests
results/runs/                       Saved summaries and local checkpoints/logs
report_assets/                      Historical eight-subject tables and plots
overleaf_report/                    IEEEtran source, bibliography, and figures
```

## Setup and dataset

Python 3.11 or newer is intended; the final tests here ran on Python 3.13.12 with the installed dependencies. The full run trains PyTorch CNN–LSTM models and may take substantial CPU time and disk space. A GPU is optional. To use the configured Git remote:

```bash
git clone https://github.com/vemba8206/Reliability_gated_wearable_har.git
cd Reliability_gated_wearable_har
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain [PAMAP2 from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring) (DOI: 10.24432/C5NW2H). With network access, run `python -m src.data_download --download`. Alternatively, extract the official archive so that `data/raw/PAMAP2_Dataset/Protocol/subject101.dat` through `subject109.dat` exist. The raw dataset and downloaded archives remain local and are not part of Git. The runner rejects synthetic smoke-test data; it does not silently substitute it for PAMAP2.

The study selects activity IDs **1–6** (lying, sitting, standing, walking, running, cycling) and 100 Hz accelerometer ±16g and gyroscope axes from wrist/hand, chest, and ankle. Windows contain 256 samples with stride 128. Subject 109 has no retained rows for those six activities, so the full evaluation holds out subjects **101–108**, one at a time. Subject 103 has only four of the selected activities. Training, validation, and held-out subjects are disjoint within each fold.

## Run and test

From the repository root, with the official files installed:

```bash
python -m pytest -q
python -m src.run_experiments --mode quick
python -m src.run_experiments --mode three_subject_loso
python -m src.run_experiments --mode full_fast
```

`quick` uses two fitting subjects and one held-out subject. `full_fast` performs the eight-subject reduced-compute LOSO run with 50 RF trees and up to five CNN epochs; `full` uses the larger configured model settings and was **not** completed for this report. Each command creates a new run under `results/runs/`. The completed full-fast run ID is `full_fast_1792678712_34ed82`; its [configuration snapshot](results/runs/full_fast_1792678712_34ed82/config_snapshot.yaml) and [summary](results/runs/full_fast_1792678712_34ed82/experiment_summary.json) identify the dataset hashes and metrics. The earlier tracked `quick_1792671218_26a836` result lacks sufficient provenance and must not be cited.

The nested follow-up reuses a completed full-fast run's 50-tree checkpoints and training-only scalers. Its output path must be new. It validates the official input hashes before running:

```bash
python -m scripts.nested_rf_policy_experiment --historical-run results/runs/full_fast_1792678712_34ed82 --output results/runs/my_nested_run --full-fault-grid
python -m scripts.summarize_nested_experiment results/runs/my_nested_run
```

The completed follow-up is in [nested_rf_policy_fullgrid_20261022](results/runs/nested_rf_policy_fullgrid_20261022/PHASE1_HANDOVER.md). Its [protocol](results/runs/nested_rf_policy_fullgrid_20261022/protocol.json), [clean comparison](results/runs/nested_rf_policy_fullgrid_20261022/clean_comparison.csv), [full fault grid](results/runs/nested_rf_policy_fullgrid_20261022/fault_grid.csv), and [aggregate JSON](results/runs/nested_rf_policy_fullgrid_20261022/aggregate.json) are the authoritative follow-up artifacts. The search used seed 42 and validated candidates on a non-test subject in each fold. It is exploratory because the same participants appeared in the historical evaluation.

The eight detailed fold records are tracked as `subject_XXX_results.json.gz`; the summarizer reads these compressed files directly. The uncompressed JSON, selected model checkpoints, and raw dataset remain local and can be regenerated with the commands above.

For the separate saved subject-105 ablation, run:

```bash
python -m src.run_ablations --run-dir results/runs/quick_1792677773_f689ab --train-subject 101 --test-subject 105
```

That command writes into the specified run directory, so use a copied run if its existing ablation JSON must remain unchanged. The [saved ablation file](results/runs/quick_1792677773_f689ab/subject_105_ablations.json) is a separate fault replay and should not be pooled with the nested grid.

## Verified findings

| Evaluation and metric | Adaptive RF | Comparator | Mean clean cost proxy |
|---|---:|---:|---:|
| Historical eight-subject **observed-label** macro-F1 | 0.8574 ± 0.0861 | Fixed wrist + chest: 0.9150 ± 0.0482 | 0.5281 vs 0.6667 |
| Nested eight-subject **fixed-six-class** macro-F1, validation-selected RF/policy | 0.8227 ± 0.1194 | Original RF/policy replayed: 0.8428 ± 0.1158 | 0.5310 vs 0.5281 |

The historical adaptive RF scored **0.7919** under moderate wrist dropout, versus **0.8501** for all-sensor uniform fusion, on the historical observed-label metric. In the nested follow-up's fixed-six metric and independently seeded moderate-dropout stream, selected adaptive scored **0.7559** versus **0.8217** for all-uniform. Under moderate Gaussian noise, the selected adaptive score was **0.6471**, versus **0.7922** for all-sensor reliability weighting. The selected method logged **858 sensor toggles over 8,273 windows** (0.1037/window), versus **930** (0.1124/window) for the replayed original. The smaller switching count did not offset the recognition loss.

Historical macro-F1 averages labels appearing in the true or predicted labels of a given fold; the nested follow-up also records a fixed six-class score, assigning zero to absent classes. These metrics must be compared within the same definition. All reported sensing costs count active IMUs with equal weights; 0.5281 means a **47.2% reduction in that proxy** relative to all three IMUs active. It does not establish a 47.2% battery-energy reduction. Faults were injected into prerecorded, standardized signals. Desktop timing, where saved, is host-side and is not embedded-device latency.

## Report and reproducibility

[overleaf_report](overleaf_report/README.md) contains `main.tex`, `references.bib`, and every referenced figure. Upload [overleaf_report.zip](overleaf_report.zip) to Overleaf, set `main.tex` as the root, and compile with pdfLaTeX/BibTeX. The updated source and separately extracted ZIP were checked with the Codex desktop LaTeX compiler; the repository does not contain an exported final PDF. The figures in the report are derived from the historical saved run, the nested saved run, or the architecture source. [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) records the provenance audit.

To reproduce the manuscript's historical figures from a saved summary:

```bash
python -m src.architecture_figures
python -m src.report_figures results/runs/full_fast_1792678712_34ed82/experiment_summary.json overleaf_report/figures
```

The nested plots are generated by `scripts.summarize_nested_experiment` into its run's `figures/` directory; the report copies those exact image bytes. Results depend on the official dataset, the saved configuration, and the seeded fault process. Exact host timings may differ.

## Limits, people, and citations

The controller was tested on prerecorded data, not deployed devices. Physical radio scheduling, energy, memory usage, natural fault detection, clinical generalization, and an external subject cohort remain unmeasured. Only a small RF/policy candidate grid and one random seed were evaluated in the nested search. The fixed wrist-plus-chest comparator performed better on clean recognition, and the tuned adaptive configuration did not improve the held-out trade-off.

The report lists Anusuya M, Vemba Kumar R, and Tailam Praneeth, School of Computer Engineering, Manipal Institute of Technology, MAHE, Manipal, India. The [bibliography](overleaf_report/references.bib) cites PAMAP2 and the six surveyed papers. This repository currently has **no license file**; repository publication does not itself grant reuse rights. The group presentation and institutional similarity report are separate submission items.
