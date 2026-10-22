# Reliability-Gated Adaptive Sensing for Fault-Resilient Wearable HAR

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/DOI-10.24432%2FC5NW2H-blue)](https://doi.org/10.24432/C5NW2H)

An end-to-end research project and interactive experimentation platform for **Reliability-Gated Adaptive Sensing in Wearable IoT Human Activity Recognition (HAR)**. 

---

## 1. Research Overview

Wearable HAR platforms deployed on resource-constrained micro-nodes face severe physical sensor vulnerabilities: complete hardware dropout, intermittent packet dropouts, electromagnetic Gaussian noise, frozen ADC flatlines, and DC bias drift. While keeping every multi-modal sensor active maximizes redundancy, it escalates battery consumption and wireless transmission loads.

### Core Research Question
> **Can a lightweight runtime policy combine per-sensor reliability, HAR prediction uncertainty, and sensing/communication cost to choose which wearable sensors should be active in the next sensing window, improving fault resilience while reducing normalized sensing cost?**

### Closed-Loop Architecture
```
 [ SENSING WINDOW t ]
         │
         ▼
  [Acquire Active Sensors] ──────> [Observable Quality Assessment]
  (Wrist, Chest, Ankle)                 │  - Missing ratio
         │                              │  - Flatline / stagnant variance
         ▼                              │  - First-diff noise variance
  [HAR Classifier Families]              │  - Saturation clipping
   - Model A: Random Forest             ▼
   - Model B: CNN-LSTM          [Reliability Score r_i(t) in [0, 1]]
         │                              │
         └───────────────┬──────────────┘
                         ▼
        [Reliability-Weighted Probability Fusion]
                         │
                         ▼
             [Predicted Activity y(t)]
             [Shannon Entropy Uncertainty u(t)]
                         │
                         ▼
        [Closed-Loop Adaptive Sensing Controller]
        (Considers: r_i(t), u(t), Normalized Cost Proxy C(t))
                         │
                         ▼
     [Active Sensor Configuration for Window t+1]
        (KEEP / ACTIVATE / DEACTIVATE / PROBE)
```

---

## 2. Experimental Realism & Normalized Cost Proxy

- **Prerecorded Data Realism**: Inactive sensors are **strictly hidden** from the controller and fusion logic. An inactive sensor's measurements and predictions cannot be read until activated or probed.
- **Next-Window Policy**: Sensor activation decisions made at window $t$ take effect strictly at window $t+1$.
- **Normalized Sensing Cost Proxy**:
  $$\mathcal{C}(t) = \frac{\sum_{i} a_i(t) \cdot w_i}{\sum_{i} w_i}$$
  *Important scientific notice*: This metric serves as a normalized sensing and communication load proxy, **not** a direct physical benchtop battery measurement.

---

## 3. Quick Start & Prerequisites

### Prerequisites
- Python 3.11, 3.12, or 3.14 on Windows, macOS, or Linux.
- 4 GB+ RAM (runs entirely on CPU; GPU is optional).

### Setup & Installation
```bash
# Clone the repository and navigate to root
git clone <repo-url>
cd reliability-gated-har

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Acquisition

The project utilizes the benchmark **PAMAP2 Physical Activity Monitoring Dataset** from the UCI Machine Learning Repository:
- **DOI**: [10.24432/C5NW2H](https://doi.org/10.24432/C5NW2H)
- **Protocol Recordings**: 9 human subjects (`subject101.dat` to `subject109.dat`), 100 Hz sampling rate, 54 columns.
- **Sensors**: 3 IMUs positioned on the **Wrist**, **Chest**, and **Ankle** (3D Acceleration $\pm 16g$ and 3D Gyroscope).

### Automated Acquisition Commands
```bash
# Option A: Calibrated Synthetic PAMAP2 Subset (Instant setup for offline verification)
python -m src.data_download --synthetic

# Option B: Download official PAMAP2 dataset from UCI (~600 MB)
python -m src.data_download --download
```

### Manual Placement Fallback
If direct internet access is restricted, download `PAMAP2_Dataset.zip` manually from the UCI repository and place the unzipped subject files (`subject101.dat` to `subject109.dat`) in `data/raw/PAMAP2_Dataset/Protocol/` or `data/raw/`.

---

## 5. Running Experiments

### Quick Development Run (~15 seconds)
Trains models on subset folds, tests all 5 fault scenarios, runs all 5 baselines, and exports artifacts:
```bash
python -m src.run_experiments --mode quick
```

### Full LOSO Cross-Validation Evaluation
Executes rigorous Leave-One-Subject-Out (LOSO) cross-validation across all 9 protocol participants:
```bash
python -m src.run_experiments --mode full
```

### Running Automated Test Suite
Executes 23 automated unit tests verifying data formatting, no-leakage subject partitioning, fault simulation, reliability indicators, probability fusion, and export integrity:
```bash
pytest
```

---

## 6. Interactive Streamlit Dashboard

Launch the research dashboard to inspect models, signals, fault corruptions, and export reports:
```bash
streamlit run app.py
```

### Dashboard Pages
1. **Overview Dashboard**: System status, top verified metrics, KPI cards, and one-click experiment execution.
2. **Dataset Explorer**: Class distributions, per-subject window counts, and IMU channel mappings.
3. **Model Laboratory**: Side-by-side Random Forest vs CNN-LSTM training curves and confusion matrices.
4. **Fault Simulation Lab**: Interactive signal corruption preview with real-time reliability indicator telemetry.
5. **Results & Comparisons**: Comprehensive Macro-F1 degradation and Cost-vs-Accuracy Pareto frontiers.
6. **Export Center**: Instant download of publication-ready 300 DPI figures, CSV tables, IEEE LaTeX tables, and the complete ZIP bundle.

---

## 7. Project Architecture

```text
reliability-gated-har/
├── app.py                      # Interactive Streamlit research dashboard
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Comprehensive documentation
├── config/
│   └── default.yaml            # Experiment hyperparameters and sensor settings
│
├── src/
│   ├── data_download.py        # Dataset acquisition and synthetic fallback generator
│   ├── data_loader.py          # 54-column PAMAP2 loader and activity filtering
│   ├── preprocessing.py        # Imputation and SubjectAwareScaler (leakage-free)
│   ├── windowing.py            # Sliding windows with label purity filtering
│   ├── feature_extraction.py   # Vectorized 72 time-domain features for Random Forest
│   ├── train_random_forest.py  # Sensor-wise Random Forest with probability alignment
│   ├── train_cnn_lstm.py       # Sensor-wise CNN-LSTM in PyTorch with early stopping
│   ├── reliability.py          # Observable signal-quality indicators & reliability
│   ├── uncertainty.py          # Shannon entropy uncertainty & disagreement
│   ├── fusion.py               # Reliability-weighted fusion with safe fallback
│   ├── fault_injection.py      # Controlled physical fault simulator
│   ├── adaptive_policy.py      # Closed-loop next-window active sensor controller
│   ├── baselines.py            # Execution harness for all 5 comparative baselines
│   ├── evaluation.py           # Classification, fault detection, and cost metrics
│   ├── plots.py                # Publication-quality 300 DPI figures
│   ├── report_exports.py       # IEEE LaTeX tables, CSVs, narratives, and ZIP bundler
│   └── run_experiments.py     # Master CLI experiment pipeline
│
├── tests/                      # 23 automated unit tests
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_no_leakage.py
│   ├── test_fault_injection.py
│   ├── test_reliability.py
│   ├── test_fusion.py
│   ├── test_adaptive_policy.py
│   └── test_exports.py
│
├── data/raw/                   # PAMAP2 raw data repository (never committed to git)
├── models/                     # Serialized model checkpoints
├── results/runs/               # Experiment run logs and JSON metric snapshots
├── report_assets/              # IEEE report assets: figures, tables, narratives, ZIP
└── scripts/                    # Convenience PowerShell execution scripts
```

---

## 8. Exported IEEE Research Artifacts

The experiment pipeline automatically populates `report_assets/` with:
- **10 Publication-Quality Tables**:
  - `table1_dataset_preprocessing.csv` / `.tex`
  - `table2_model_hyperparameters.csv` / `.tex`
  - `table3_clean_model_comparison.csv` / `.tex`
  - `table4_fault_definitions.csv` / `.tex`
  - `table5_faulted_recognition_comparison.csv` / `.tex`
  - `table6_reliability_fault_detection.csv` / `.tex`
  - `table7_sensing_cost_comparison.csv` / `.tex`
  - `table8_model_latency_computation.csv` / `.tex`
  - `table9_per_subject_loso_results.csv` / `.tex`
  - `table10_proposed_vs_baselines_summary.csv` / `.tex`
- **14 Lossless Figures (300 DPI PNG)**:
  - `fig1_activity_distribution.png`
  - `fig2_clean_confusion_matrix_rf.png`
  - `fig3_clean_confusion_matrix_cnn.png`
  - `fig4_faulted_confusion_matrix.png`
  - `fig5_macro_f1_vs_severity.png`
  - `fig7_cost_vs_macro_f1.png`
  - `fig9_reliability_and_activation_timeline.png`
  - `fig11_cnn_training_curves.png`
  - `fig12_rf_feature_importance.png`
- **Report-Ready Summaries**:
  - `results_summary.md` and `results_summary.tex`
  - `methodology_summary.md`
  - `figure_captions.md`
  - `table_captions.md`
  - `experiment_protocol.md`
- **ZIP Bundle**: `report_assets_bundle.zip`

---

## 9. References & Dataset Citations

```bibtex
@inproceedings{reiss2012introducing,
  title={Introducing a new benchmarked dataset for activity monitoring},
  author={Reiss, Attila and Stricker, Didier},
  booktitle={Proceedings of the 16th International Symposium on Wearable Computers (ISWC)},
  pages={108--109},
  year={2012},
  organization={IEEE},
  doi={10.1109/ISWC.2012.13}
}
```
Dataset DOI: [10.24432/C5NW2H](https://doi.org/10.24432/C5NW2H).
