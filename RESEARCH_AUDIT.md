# Research provenance and implementation audit — 22 October 2026

## Initial repository state

- Clean Git `main` at `908c991` (`Adding ES reliability-gated wearable HAR research project`). No user changes were overwritten.
- Existing Python modules cover PAMAP2 loading, subject-wise windowing, random forest and CNN–LSTM classifiers, reliability scoring, probability fusion, an adaptive controller, fault injection, baseline evaluation, plots, and report export. Eight test files and one saved quick-run summary were present.
- No official PAMAP2 `.dat` files or model checkpoints were present in the repository or the inspected local course directory. The initial quick-run JSON lists subjects 101, 102, and 105 but does not include a dataset manifest or hashes.

## Critical findings

1. `acquire_dataset` silently generated synthetic PAMAP2-shaped files when official download failed. The saved quick-run summary therefore cannot be attributed to PAMAP2 without independent source verification.
2. `run_experiments.py` used hard-coded values for several plot series, trade-off coordinates, latency rows, reliability-detection metrics, and synthesis text. `report_exports.py` supplied plausible numeric defaults and made unconditional superiority claims. The original tracked `report_assets/` were excluded; newly regenerated assets are based on the verified run recorded below.
3. The adaptive baseline precomputed model outputs and quality values for all sensors, including inactive ones. Clean runtime evaluation now calls models only for active sensors. Offline fault-grid evaluation caches all model outputs for speed but reveals only active outputs to the policy; its timing is not treated as deployment latency.
4. The controller did not enforce the configured budget and could deactivate its wrist anchor.
5. Windowing filtered activity rows but did not enforce timestamp continuity, so some windows could bridge an excluded transition.
6. Fault-detection AUROC used only fault-positive windows, for which AUROC is undefined. Its old output was a fixed value.
7. The reliability score omitted its separately implemented bias/drift indicator; this is now included. The Gaussian-noise injection and quality threshold operate on scaled arrays. They require validation before claiming reliable fault detection; the physical-unit clipping penalty is disabled in the default weight configuration.
8. The runner originally orchestrated only five individual faults although the simulator supported combined corruption; this was added to the configured grid. Formal policy ablations have a standalone replay script; comprehensive per-scenario logs and physical energy measurements are outstanding.

## Changes made

- Official-data runs now fail visibly if acquisition fails, and synthetic smoke-test generation writes a marker rejected by the research runner.
- Adaptive inference evaluates only active sensors. The controller enforces a normalized budget and always-on wrist anchor.
- Windowing rejects timestamp discontinuities. Fault-detection evaluation compares clean with injected-fault windows.
- New result exports derive plot and summary values from run records; unavailable per-model latency is labeled unmeasured. New runs record input SHA-256 hashes and per-window clean and moderate-dropout adaptive logs.
- `overleaf_report/` is an IEEEtran project that reports only new official-data measurements and explicitly limits claims by the number of held-out subjects.

## Verified dataset and runs

The official legacy UCI `PAMAP2_Dataset.zip` downloaded into the ignored `data/raw/` directory. Its ZIP test found no bad member; SHA-256 is `6c2282b5b52775cd0588d450d10a315260c94fa3228b9fe453f48c47fda106f4`. All nine protocol files were extracted with entry CRC checks. Subject 109 contains no rows for the configured six activities, so the full task explicitly evaluates subjects 101--108. The official data and Python dependencies are now present.

The new single-held-out quick run is `results/runs/quick_1792677773_f689ab/`, the three-subject LOSO run is `results/runs/three_subject_loso_1792678177_dce0aa/`, and the main eight-subject reduced-compute LOSO run is `results/runs/full_fast_1792678712_34ed82/`. The main run gave adaptive RF clean macro-F1 `0.8574 +/- 0.0861` and normalized cost proxy `0.5281`; all-sensor uniform RF gave `0.8891 +/- 0.0726` at cost `1.00`, while the fixed wrist-plus-chest pair gave `0.9150 +/- 0.0482` at cost `0.67`. Under moderate wrist dropout, adaptive RF macro-F1 was `0.7919`. These are measured simulation outcomes, not battery measurements. The quick-run subject-105 ablations are saved in `subject_105_ablations.json`.

All 29 tests pass with the official files after the fixed-six-class metric test was added. A later policy fix revealed that probe activations had been omitted from the runner's switching counter; `scripts/reconcile_run_switches.py` recalculated switching from logged active masks. The main run has 930 observed sensor toggles over 8,273 held-out windows. The IEEE manuscript compiled successfully with the Codex desktop LaTeX compiler on 22 October 2026. Its citations and graphics paths were also checked against the source tree.

Final reconciliation found that the original per-subject cost column mixed clean F1 with moderate-dropout cost. The table now gives separately labeled clean and dropout cost proxies, recalculated from saved masks without repeating model training. The synthesis table now reports the computed mean fault-detection AUROC, 0.8882, instead of saying it was not aggregated. Supplemental figure captions were corrected to identify plots based on individual folds; four nonexistent figures were removed from the list. Figure 1 now labels selected protocol samples, not windows. The `overleaf_report.zip` archive passed CRC and source/asset checks, and a separate extraction compiled successfully with the Codex desktop LaTeX compiler. The compiler did not export a standalone PDF file to the repository.

## Reproduction

From this repository root, install `requirements.txt` in an environment with package access. Obtain the official PAMAP2 archive from the UCI dataset page and place `subject101.dat` through `subject109.dat` under `data/raw/PAMAP2_Dataset/Protocol/`. Ensure no `SYNTHETIC_DATA_DO_NOT_USE_FOR_RESEARCH` marker exists. Run `python -m pytest -q`, `python -m src.run_experiments --mode quick`, `python -m src.run_experiments --mode three_subject_loso`, and `python -m src.run_experiments --mode full_fast`. For the saved quick fold, run `python -m src.run_ablations --run-dir results/runs/quick_1792677773_f689ab --train-subject 101 --test-subject 105`. Regenerate paper plots with `python -m src.architecture_figures` and `python -m src.report_figures results/runs/full_fast_1792678712_34ed82/experiment_summary.json overleaf_report/figures`. The original legacy run must remain excluded from evidence.

The final institutional Turnitin report and team presentation remain separate deliverables.

## Phase 2 follow-up: nested optimization and documentation

The completed `results/runs/nested_rf_policy_fullgrid_20261022/` run compared the verified historical 50-tree RF against a 100-tree candidate and three policies using an inner validation subject for each of eight outer LOSO folds. It retained the same official input hashes, evaluated all six configured faults at three severities, and saved fold metrics, selected checkpoints, and clean/moderate-dropout logs. New `macro_f1_all_classes` fixes the six-class denominator across folds while preserving historical `macro_f1` for backward compatibility. Validation selected the larger RF in six folds and changed the policy in three, but the selected method's held-out fixed-six clean F1 was 0.8227 at cost 0.5310, below 0.8428 at cost 0.5281 for the replayed original model/policy. The selected policy made 858 toggles versus the original's 930. Thus the original remains the recommended configuration. The detailed protocol, caveats, and all fault outcomes are in the run's `PHASE1_HANDOVER.md`, `protocol.json`, `aggregate.json`, and CSVs.

The report and README now distinguish historical observed-label scores from the nested fixed-six scores. `PROJECT_SUMMARY.md` is the complete project handover. The Overleaf ZIP and independent extraction were rebuilt and compiled after those updates. Git ignore rules exclude official raw data, scratch runs, and large model checkpoints while allowing final source, metrics, and figures to be reviewed for tracking. No hardware energy or embedded latency was measured.
