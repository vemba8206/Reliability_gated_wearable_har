# Project Summary and Final Handover

**Project:** Reliability-Gated Adaptive Sensing for Fault-Resilient Human Activity Recognition in Wearable IoT
**Institution:** School of Computer Engineering, Manipal Institute of Technology, MAHE, Manipal, India
**Report authors:** Anusuya M, Vemba Kumar R, Tailam Praneeth
**Status:** Implementation, official-data experiments, nested follow-up, and IEEE source package completed; presentation and institutional similarity report remain separate tasks.

## 1. Overview, research problem, and objectives

Wearable human activity recognition (HAR) can fail when body-worn inertial measurements disappear, become noisy, freeze, or drift. Keeping all sensors active improves redundancy but increases acquisition and communication load. This project tests whether observable signal quality, classifier uncertainty, and a sensing budget can select the IMUs available in the **next** window while preserving useful recognition. The research contribution is an auditable reliability-gated control loop evaluated under simulated faults and compared with fixed and all-sensor strategies. All sensing-cost findings are equal-weight activation proxies, not measured joules.

The submitted Part 1 methodology proposed sensor reliability estimation, uncertainty-aware fusion, and adaptive sensing together. Its scoped gap was the conjunction of post-acquisition fault handling with pre-acquisition resource decisions in a reproducible HAR benchmark. The implemented controller is rule based and tested on prerecorded data; no physical radio or wearable firmware was built.

## 2. Architecture and implementation

Three IMUs provide six channels each: accelerometer ±16g and gyroscope axes from the hand/wrist, chest, and ankle. Each active IMU supplies one per-sensor class-probability vector. A signal-quality estimator penalizes missingness, flatline, first-difference noise, and distribution shift. Reliability-weighted probability fusion yields the current activity and normalized entropy. The wrist-anchored controller applies its budget, dwell, reliability and entropy thresholds, and periodic probes to the **next** active mask. Inactive sensors do not contribute a current prediction or fresh quality score. An offline cache accelerates fault-grid replay but withholds inactive outputs from the controller and fusion.

| File | Responsibility |
|---|---|
| `src/data_download.py`, `src/data_loader.py` | Official PAMAP2 acquisition, format validation, channel mapping; synthetic smoke data explicitly marked and rejected by research runs |
| `src/preprocessing.py`, `src/windowing.py` | Bounded imputation, train-subject-only scaling, timestamp-aware 256-sample windows |
| `src/feature_extraction.py`, `src/train_random_forest.py`, `src/train_cnn_lstm.py` | 72-feature per-IMU RF and temporal CNN–LSTM models |
| `src/reliability.py`, `src/uncertainty.py`, `src/fusion.py` | Observable quality, normalized entropy, active-sensor probability fusion |
| `src/adaptive_policy.py`, `src/baselines.py` | Next-window sensing policy and comparator execution |
| `src/fault_injection.py`, `src/evaluation.py` | Seeded injected faults, recognition, detection, cost, and switching metrics |
| `src/run_experiments.py`, `src/run_ablations.py` | Historical official-data runs and subject-105 one-factor replay |
| `scripts/nested_rf_policy_experiment.py`, `scripts/summarize_nested_experiment.py` | Inner-validation search, held-out fault grid, saved-record aggregation |
| `src/architecture_figures.py`, `src/report_figures.py`, `src/report_exports.py` | Paper diagrams, result plots, tables, narratives |

The original RF uses 50 trees, maximum depth 15, minimum leaf size two, balanced class weights, and seed 42 in reduced-compute mode. The CNN–LSTM uses two temporal convolution layers, a 64-unit LSTM, dropout 0.3, and up to five epochs in that mode. The original policy uses `r_low=0.40`, normalized entropy thresholds `u_high=0.65` and `u_low=0.35`, minimum dwell four windows, probe interval 20, and budget 0.667. The wrist anchor stays active. A policy action at window `t` affects acquisition at `t+1`.

## 3. Dataset and evaluation scope

The official UCI PAMAP2 archive and all nine protocol files were downloaded and CRC checked locally. The official archive SHA-256 is `6c2282b5b52775cd0588d450d10a315260c94fa3228b9fe453f48c47fda106f4`; individual input hashes are in each completed run's manifest. Raw files are under `data/raw/PAMAP2_Dataset/Protocol/` locally and are excluded from Git. The six-class task selects activity IDs 1–6. Subject 109 contains no retained rows for those classes and is excluded. Subjects 101–108 contribute **8,273** retained windows across eight LOSO test folds. Subject 103 has only four selected classes, which matters for macro-F1 interpretation. There was no completed nine-subject evaluation.

Windows are 256 samples (2.56 seconds at 100 Hz) with stride 128, minimum 80% label purity, and a timestamp-gap rejection check. Scaler statistics are fitted only on the fold's fitting subjects. The outer test subject is excluded from fitting and inner selection. Imputation of natural missing values limits what can be claimed about real-world fault detection; reported detection metrics refer to injected faults.

## 4. Experiments completed and provenance

- `results/runs/quick_1792677773_f689ab/`: official-data quick run with subject 105 held out; subject-105 ablation JSON is saved there.
- `results/runs/three_subject_loso_1792678177_dce0aa/`: separate three-subject LOSO diagnostic. Adaptive observed-label macro-F1 was **0.7713 ± 0.0238**; all-uniform was **0.8446 ± 0.0622**. These values are not pooled with eight-subject results.
- `results/runs/full_fast_1792678712_34ed82/`: main historical eight-subject reduced-compute LOSO run, all six faults and three severities, saved RF/CNN checkpoints, scalers, input hashes, summary, and clean/moderate-dropout adaptive logs.
- `results/runs/nested_rf_policy_fullgrid_20261022/`: completed exploratory nested eight-fold RF/policy search and final fault grid, with protocol, per-fold results, selected checkpoints, logs, aggregates, CSVs, and plots. Its one-fold pilot is saved separately as `nested_pilot_20261022_subject101` and is not an eight-fold result.

The tracked legacy `quick_1792671218_26a836` result predates dataset provenance fixes and may have used synthetic fallback or fixed export values. It is excluded from all research claims. Two interrupted earlier run directories are scratch evidence only and have no completed-run status.

## 5. Tuning procedure and final model decision

Each nested outer fold reused its historical 50-tree RF checkpoint and training-only scaler. A 100-tree, depth-20, minimum-leaf-two candidate was trained on that fold's six fitting subjects. A seventh, non-test subject selected the RF using fixed-six-class macro-F1 under all-sensor uniform fusion, weighted 0.35 clean, 0.25 moderate dropout, 0.20 moderate Gaussian noise, and 0.20 moderate combined fault. The selected RF was then paired with three prespecified policies: original; conservative switching (`u_high/u_low=0.75/0.45`, dwell 8, probe 40); and early reinforcement (`0.50/0.25`, dwell 8, probe 12). The policy objective added a 0.10 clean-cost and 0.02 switch-rate penalty. Budget and wrist anchor were unchanged. Seed 42 and deterministic condition seeds were recorded in `protocol.json`.

Validation chose the 100-tree RF in six folds, the original 50-tree RF in two, the original policy in five, early reinforcement in two, and conservative switching in one. No outer test subject was used to choose its own fold's model or policy. The nested run is exploratory because these participants had been examined in the earlier historical evaluation; it is not an external cohort. The chosen configuration did **not** improve the across-subject trade-off. The recommended documented configuration remains the original 50-tree RF and original policy, rather than a post hoc test-selected alternative.

## 6. Verified results and interpretation

The historical eight-subject run reported **observed-label** macro-F1, which averages labels appearing in the true or predicted labels of each fold. Its adaptive RF clean mean was **0.8574 ± 0.0861** at normalized cost **0.5281**; fixed wrist-plus-chest was **0.9150 ± 0.0482** at cost 0.6667, and all-uniform was **0.8891 ± 0.0726** at cost 1.0. Moderate wrist-dropout adaptive macro-F1 was **0.7919**, versus all-uniform **0.8501**. The 0.5281 cost represents a **47.2% normalized acquisition/communication proxy reduction** relative to all three sensors active. Mask-derived switching was **930 toggles / 8,273 windows = 0.1124 per window**. The mean fault-detection AUROC across 144 fold/fault/severity cells was **0.8882**, while thresholded F1 was **0.3402**. These are injected-fault metrics, not natural-failure validation.

The nested run additionally reports fixed-six-class macro-F1 with zero F1 for a selected class absent from a held-out fold. Results from the two macro-F1 definitions must not be compared as if they were the same measure. Within the nested run:

| Held-out configuration | Fixed-six clean macro-F1 mean ± subject SD | Clean cost proxy | Clean toggles |
|---|---:|---:|---:|
| Original 50-tree RF and policy, replayed | 0.8428 ± 0.1158 | 0.5281 | 930 |
| Validation-selected RF with original policy | 0.8418 ± 0.1168 | 0.5278 | 912 |
| Validation-selected RF and policy | 0.8227 ± 0.1194 | 0.5310 | 858 |
| Fixed wrist + chest, selected RF | 0.8574 ± 0.1116 | 0.6667 | 0 |
| All-sensor uniform, selected RF | 0.8533 ± 0.1108 | 1.0000 | 0 |
| All-sensor reliability weighted, selected RF | 0.8540 ± 0.1114 | 1.0000 | 0 |

Selected adaptive switching was **858 / 8,273 = 0.1037** per window. Under moderate wrist dropout, fixed-six macro-F1 was **0.7559** selected adaptive, **0.8217** all-uniform, and **0.8112** all-weighted. Under moderate noise, values were **0.6471**, **0.7630**, and **0.7922**; under moderate combined corruption, **0.6766**, **0.7661**, and **0.7978**. The full six-fault, three-severity results are in `fault_grid.csv`. The selected policy did not improve recognition at similar cost; its largest clean fold loss was subject 105 (0.7319 selected versus 0.8474 original). The validated conclusion is a negative tuning result, with reduced switching but weaker recognition.

A separate subject-105 one-factor ablation found standard clean F1 **0.754** at cost **0.645**, versus no-budget clean F1 **0.840** at cost **0.918**. That replay used an independent fault seed and is not directly cell-comparable with the main fault grid. No statistical significance is claimed from these small subject-level comparisons.

## 7. Tests and output verification

After Phase 2 source and documentation updates, `python -m pytest -q` passed **29 tests** on Python 3.13.12. `python -m compileall -q src scripts tests` passed. CLI `--help` checks for the original runner, nested runner, and summarizer passed, and the full nested runner itself completed eight folds with a `completed.json` marker. All 16 selected-policy clean/dropout logs were independently reconciled: stored predictions reproduce fixed-six F1; active masks reproduce cost and toggle counts. Each fold saved 19 test conditions (clean plus six faults at three severities). The historical 50-tree replay reproduced the original clean observed-label mean, cost, and 930 toggles. A compressed-only copy of all eight fold records regenerated an aggregate byte-for-byte identical to the original. `git diff --check` passed. Results and plots were aggregated only from saved fold records.

`results/runs/nested_rf_policy_fullgrid_20261022/PHASE1_HANDOVER.md` documents the search and validation in greater detail. The final summary is `aggregate.json`; `clean_comparison.csv`, `fault_grid.csv`, and `figures/` contain review-ready comparisons. Raw `.dat` files, 100-tree checkpoints, and local logs remain available in the local checkout even where Git excludes them.

## 8. Research report and submission assets

`overleaf_report/main.tex` is the IEEE Conference manuscript. Its three author blocks include the names, affiliations, registration numbers, and institutional email addresses supplied for the report. `overleaf_report/references.bib` contains seven bibliography entries, including PAMAP2 and six surveyed papers. `overleaf_report/figures/` contains all referenced diagrams and historical/nested plots. The report retains historical results under their original metric definition and adds the nested follow-up as a separate, negative result. Its source compiled to six pages with the Codex desktop LaTeX compiler; a separate page-count assertion verified the six-page minimum. A separately extracted `overleaf_report.zip` compiled successfully as a self-contained project. The ZIP contains `main.tex`, bibliography, README, and all referenced figures, with no build intermediates. No standalone final PDF was exported into the repository.

## 9. Reproduction

From the repository root, install dependencies with `python -m pip install -r requirements.txt` in a Python virtual environment. Obtain the official UCI PAMAP2 archive and place `subject101.dat` through `subject109.dat` in `data/raw/PAMAP2_Dataset/Protocol/`, or run `python -m src.data_download --download` where network access is available. The raw dataset is not in Git.

```bash
python -m pytest -q
python -m src.run_experiments --mode quick
python -m src.run_experiments --mode three_subject_loso
python -m src.run_experiments --mode full_fast
python -m scripts.nested_rf_policy_experiment --historical-run results/runs/full_fast_1792678712_34ed82 --output results/runs/my_nested_run --full-fault-grid
python -m scripts.summarize_nested_experiment results/runs/my_nested_run
```

Use a new output path for every nested run. To reproduce a newly generated full-fast run, pass its new run directory with `--historical-run`. The saved subject-105 ablation is generated by `python -m src.run_ablations --run-dir results/runs/quick_1792677773_f689ab --train-subject 101 --test-subject 105`; run against a copy if the current JSON must remain untouched. For the report, upload `overleaf_report.zip` to Overleaf and select `main.tex`. The desktop compiler validation is recorded above; command-line pdfLaTeX was not installed in this workspace.

## 10. Code, Git, and documentation changes

Phase 1 of the project repaired silent synthetic fallback, sensor mapping, window gaps, inactive-sensor inference, budget/anchor/probe behavior, injected-fault metrics, switching counts, and result-derived exports. It acquired and verified official PAMAP2, ran quick, three-subject, and eight-subject experiments, wrote the IEEE report, and validated its ZIP. The later optimization phase added fixed-six macro-F1 alongside historical macro-F1, a targeted metric test, the nested RF/policy runner, the aggregation script, and a complete eight-fold fault-grid run. This documentation phase updated the manuscript, README, `.gitignore`, Overleaf assets, research audit, and this summary.

Git should keep source, tests, config, documentation, the final LaTeX sources and figures, `overleaf_report.zip`, small historical summaries/logs, and nested protocol/aggregates/compressed per-fold result JSON. The plain per-fold JSON remains local for convenient inspection. Official raw data and downloaded archives, interrupted runs, pilot scratch, disposable caches, and large trained checkpoints remain local. The unverified legacy quick summary should be removed from the Git index while preserving its local copy and audit note. No history rewrite, automatic commit, or push was performed. The repository currently has no `LICENSE` file; one should only be added after the team decides its terms.

## 11. Limitations and incomplete items

- No nine-subject six-class evaluation: subject 109 has no retained rows. The six-class subset excludes other PAMAP2 activities.
- Historical observed-label and nested fixed-six macro-F1 use different denominators; compare like with like.
- Only one seed and a small RF/policy grid were run. No alternative window size, probability calibration, multi-subject inner validation, external cohort, or exhaustive search was completed.
- Cost is a normalized active-IMU proxy. Real energy, communication overhead, on-device latency and memory, natural fault detection, and clinical utility were not measured.
- The report compiled in the desktop LaTeX editor, but no standalone final PDF is present in the repository.
- The group presentation and institutional similarity report remain outstanding external submission items.

## 12. Final submission checklist and next actions

| Item | Status |
|---|---|
| Source, tests, configuration, and official-data provenance | Complete locally |
| Historical eight-subject and nested eight-subject run records | Complete and verified |
| IEEE LaTeX source, citations, figures, Overleaf ZIP | Compiled and package-validated |
| Repository README, ignore rules, handover | Updated |
| Standalone report PDF export | Not yet saved in repository |
| Presentation | Outstanding |
| Institutional similarity report | Outstanding |
| Git review, commit, push | Awaiting team review; no commit or push performed |

A new developer should read this file and the README, inspect `results/runs/nested_rf_policy_fullgrid_20261022/PHASE1_HANDOVER.md`, run the 29-test suite, and review the saved CSV/JSON before changing the report. A team member should export the final PDF if required, prepare the presentation, obtain the institutional similarity report, and review the staged Git diff before committing or pushing.
