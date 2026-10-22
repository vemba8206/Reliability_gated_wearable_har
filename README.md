# Reliability-Gated Adaptive Sensing for Fault-Resilient Human Activity Recognition in Wearable IoT

A research implementation of human activity recognition (HAR) from three wearable inertial measurement units (IMUs). It combines sensor-specific classifiers, observable signal-quality estimates, reliability-weighted prediction fusion, and a controller that selects the IMUs used in the next sensing window. The repository includes official PAMAP2 preprocessing, subject-held-out evaluation, simulated sensor faults, baselines, ablations, and saved experiment records.

## Research objective

Continuous sensing with every IMU offers redundancy but increases the amount of data acquired and communicated. A reduced sensor set lowers that load but can fail when an active device becomes unreliable. This project evaluates whether runtime signal quality and classification uncertainty can guide a stable, budgeted sensor choice while retaining useful recognition under faults. It reports both recognition and a **normalized active-IMU cost proxy**; battery energy, radio traffic, and on-device performance were not measured.

## System design

1. **Prepare subject-specific windows.** Load official PAMAP2 protocol files, select six activity classes and six inertial channels per IMU, reject invalid timestamp gaps, and fit scaling parameters on training subjects only.
2. **Classify active sensors.** A separate random forest for each wrist/hand, chest, and ankle IMU produces class probabilities. A compact CNN–LSTM is available as a comparison model.
3. **Estimate reliability and fuse predictions.** Missingness, flatline behavior, first-difference noise, and distribution shift form an observable quality score. Only currently active IMUs contribute to reliability-weighted probability fusion. Normalized prediction entropy summarizes uncertainty.
4. **Choose the next active set.** A wrist-anchored rule-based policy applies a sensing budget, reliability and uncertainty thresholds, minimum dwell, and periodic probes. Its decision at one window changes the active IMUs in the next.
5. **Evaluate the trade-off.** Compare with wrist-only, fixed wrist-plus-chest, all-sensor uniform, and all-sensor reliability-weighted strategies under clean and injected-fault conditions. Save predictions, masks, costs, switching counts, and subject-level metrics.

The policy is evaluated by replaying prerecorded streams. Inactive sensors cannot supply a current prediction or fresh quality score to the controller. The implementation does not control physical wearable hardware.

## Repository map

| Path | Purpose |
|---|---|
| `config/default.yaml` | Dataset, model, fault, and policy settings |
| `src/` | Data loading, preprocessing, models, reliability, fusion, policy, and evaluation |
| `src/run_experiments.py` | Quick, three-subject, and eight-subject experiment runner |
| `src/run_ablations.py` | Subject-105 one-factor ablation replay |
| `scripts/nested_rf_policy_experiment.py` | Inner-validation RF and policy comparison |
| `scripts/summarize_nested_experiment.py` | Aggregation from saved fold records |
| `tests/` | Automated checks |
| `results/runs/` | Experiment protocols, metrics, logs, and local checkpoints |
| `report_assets/` | Tables and figures from the historical run |

## Setup and dataset

Use Python 3.11 or newer; the verified test environment used Python 3.13.12. The reduced-compute eight-subject run trains PyTorch CNN–LSTM models as well as random forests and can take substantial CPU time and disk space. A GPU is optional.

From a checkout of this repository:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain [PAMAP2 from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring) (DOI: 10.24432/C5NW2H). With network access, run `python -m src.data_download --download`. Alternatively, extract the official archive so that `data/raw/PAMAP2_Dataset/Protocol/subject101.dat` through `subject109.dat` exist. The raw dataset and downloaded archives remain local and are not part of Git. The runner rejects synthetic smoke-test data; it does not silently substitute it for PAMAP2.

The study selects activity IDs **1–6** (lying, sitting, standing, walking, running, cycling) and 100 Hz accelerometer ±16g and gyroscope axes from wrist/hand, chest, and ankle. Windows contain 256 samples with stride 128, subject to label-purity and timestamp-gap checks. Subject 109 has no retained rows for these six activities, so the full evaluation holds out subjects **101–108**, one at a time. Subject 103 has only four of the selected activities. Training, validation, and held-out subjects are disjoint within each fold; the eight held-out subjects contribute **8,273 test windows**.

## Run and test

From the repository root, with the official files installed:

```bash
python -m pytest -q
python -m src.run_experiments --mode full_fast
```

`full_fast` performs the eight-subject reduced-compute leave-one-subject-out (LOSO) evaluation with 50 RF trees and up to five CNN epochs. It creates a new directory under `results/runs/`. The completed [full-fast configuration](results/runs/full_fast_1792678712_34ed82/config_snapshot.yaml) and [summary](results/runs/full_fast_1792678712_34ed82/experiment_summary.json) record input hashes and results. The optional `quick` and `three_subject_loso` modes run smaller diagnostics; the larger `full` configuration was not completed. An older `quick_1792671218_26a836` result lacks sufficient provenance and is excluded from the findings below.

The exploratory nested follow-up compares the original 50-tree forest with a 100-tree candidate and selects among three sensing policies on a non-test subject in each outer fold. It **requires a completed full-fast run's local model checkpoints and scalers**, which are excluded from Git because of their size. On a fresh clone, run `full_fast` first and replace the `--historical-run` path below with the directory it creates. Use a new output path for each nested run; the runner validates the official input hashes.

```bash
python -m scripts.nested_rf_policy_experiment --historical-run results/runs/full_fast_1792678712_34ed82 --output results/runs/my_nested_run --full-fault-grid
python -m scripts.summarize_nested_experiment results/runs/my_nested_run
```

The completed follow-up provides a [protocol](results/runs/nested_rf_policy_fullgrid_20261022/protocol.json), [clean comparison](results/runs/nested_rf_policy_fullgrid_20261022/clean_comparison.csv), [six-fault severity grid](results/runs/nested_rf_policy_fullgrid_20261022/fault_grid.csv), [aggregate JSON](results/runs/nested_rf_policy_fullgrid_20261022/aggregate.json), and [detailed run handover](results/runs/nested_rf_policy_fullgrid_20261022/PHASE1_HANDOVER.md). It used seed 42. The same participants appeared in the earlier evaluation, so this is an exploratory follow-up rather than an independent cohort.

The eight detailed fold records are tracked as `subject_XXX_results.json.gz`; the summarizer reads these compressed files directly. Uncompressed JSON, trained checkpoints, and raw data remain local. Aggregation can be repeated from the tracked compressed records without retraining.

For the separate saved subject-105 ablation, run:

```bash
python -m src.run_ablations --run-dir results/runs/quick_1792677773_f689ab --train-subject 101 --test-subject 105
```

That command writes into the specified run directory, so use a copied run if its existing ablation JSON must remain unchanged. The [saved ablation file](results/runs/quick_1792677773_f689ab/subject_105_ablations.json) is a separate fault replay and should not be pooled with the nested grid.

## Verified results

| Eight-subject evaluation | Configuration | Clean macro-F1, mean ± subject SD | Mean cost proxy |
|---|---|---:|---:|
| Historical, observed-label metric | Adaptive RF and original policy | 0.8574 ± 0.0861 | 0.5281 |
| Historical, observed-label metric | Fixed wrist + chest | 0.9150 ± 0.0482 | 0.6667 |
| Nested follow-up, fixed-six metric | Original RF and policy replayed | 0.8428 ± 0.1158 | 0.5281 |
| Nested follow-up, fixed-six metric | Validation-selected RF and policy | 0.8227 ± 0.1194 | 0.5310 |

The historical adaptive RF scored **0.7919** under moderate wrist dropout, versus **0.8501** for all-sensor uniform fusion. In the nested follow-up's independently seeded moderate-dropout stream, selected adaptive scored **0.7559** versus **0.8217** for all-uniform. Under moderate Gaussian noise, selected adaptive scored **0.6471** versus **0.7922** for all-sensor reliability weighting. The selected method logged **858 sensor toggles over 8,273 windows** (0.1037/window), versus **930** (0.1124/window) for the replayed original. Lower switching did not offset its recognition loss.

The historical macro-F1 averages labels appearing in a fold's truth or predictions. The nested follow-up also records a fixed six-class score that assigns zero F1 to absent classes. Compare configurations **within the same metric definition**. Costs count active IMUs with equal weights; the historical 0.5281 cost is a **47.2% reduction in that proxy** relative to using all three sensors. It does not establish a battery-energy reduction. Faults were injected into prerecorded signals, and saved desktop timing is not embedded-device latency.

## Saved outputs and provenance

- [Historical eight-subject summary](results/runs/full_fast_1792678712_34ed82/experiment_summary.json): configuration, per-subject scores, fault results, and dataset input hashes.
- [Nested aggregate](results/runs/nested_rf_policy_fullgrid_20261022/aggregate.json): clean comparisons, selected configurations, and the complete fault grid derived from saved fold records.
- [Nested result figures](results/runs/nested_rf_policy_fullgrid_20261022/figures/): cost trade-off and dropout-severity plots generated by the aggregation script.
- [Historical tables and figures](report_assets/): exported experiment comparisons and visualizations.
- [Project summary](PROJECT_SUMMARY.md) and [research audit](RESEARCH_AUDIT.md): implementation decisions, provenance checks, and remaining limitations.

The official dataset and trained checkpoints are intentionally excluded from Git. Saved summaries, protocols, plots, compressed nested fold records, and selected window-level logs are included to support result inspection. The earlier unverified quick result is excluded from research claims.

## Limitations and next steps

The controller has been tested on prerecorded data, not deployed devices. Its cost is an equal-weight active-sensor proxy; physical radio scheduling, energy, memory usage, and on-device latency remain unmeasured. Injected faults do not establish performance against natural hardware failures. The study uses six PAMAP2 activities and eight held-out subjects, and the nested search covers a small RF/policy grid with one seed. The fixed wrist-plus-chest comparator performs better on clean recognition, and the validation-selected adaptive configuration does not improve the held-out recognition–cost trade-off.

Useful next steps are hardware energy and latency measurements, evaluation of naturally occurring failures, broader activity and device coverage, and policy selection using more than one inner-validation subject. This repository currently has **no license file**; repository publication alone does not grant reuse rights.
