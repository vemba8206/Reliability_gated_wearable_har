"""Report artifact exporter: publication tables (CSV, LaTeX), research narratives, and ZIP bundling."""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    return "".join(conv.get(c, c) for c in str(text))


def dataframe_to_ieee_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: Optional[str] = None
) -> str:
    """Format DataFrame into standard IEEE LaTeX table environment."""
    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)

    escaped_cols = [escape_latex(c) for c in df.columns]
    header_str = " & ".join(escaped_cols) + r" \\ \hline"

    row_strs = []
    for _, row in df.iterrows():
        escaped_vals = [escape_latex(v) for v in row.values]
        row_strs.append(" & ".join(escaped_vals) + r" \\")

    body_str = "\n".join(row_strs)

    latex_code = f"""\\begin{{table}}[htbp]
\\caption{{{escape_latex(caption)}}}
\\label{{{label}}}
\\centering
\\begin{{tabular}}{{{column_format}}}
\\hline
{header_str}
{body_str}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return latex_code


class ReportArtifactExporter:
    """Exports structured LaTeX tables, CSVs, research narrative documents, and figures."""

    def __init__(self, output_dir: str = "report_assets"):
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def export_table(
        self,
        df: pd.DataFrame,
        base_name: str,
        caption: str,
        label: str
    ) -> Tuple[Path, Path]:
        """Save DataFrame as CSV and IEEE LaTeX table."""
        csv_path = self.tables_dir / f"{base_name}.csv"
        tex_path = self.tables_dir / f"{base_name}.tex"

        df.to_csv(csv_path, index=False)
        latex_str = dataframe_to_ieee_latex(df, caption, label)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

        return csv_path, tex_path

    def export_all_tables(self, experiment_results: Dict[str, Any]) -> List[Path]:
        """Generate and export the 10 required tables from experiment results."""
        exported = []

        # Table 1: Dataset and Preprocessing Summary
        t1_data = {
            "Parameter": [
                "Dataset", "Subjects Evaluated", "Sampling Rate", "Window Length",
                "Window Stride", "Label Purity Threshold", "Sensor Locations", "Channels per IMU"
            ],
            "Specification": [
                "PAMAP2 Physical Activity Monitoring (DOI: 10.24432/C5NW2H)",
                f"{len(experiment_results['subjects'])} participants; {experiment_results['mode']} evaluation",
                "100 Hz", "2.56 s (256 samples)", "1.28 s (128 samples, 50% overlap)",
                "0.80 (transitional windows discarded)", "Wrist, Chest, Ankle",
                "6 channels (3D Accel +-16g, 3D Gyroscope)"
            ]
        }
        p1, p2 = self.export_table(
            pd.DataFrame(t1_data),
            "table1_dataset_preprocessing",
            "Dataset Acquisition and Signal Preprocessing Specifications",
            "tab:dataset_specs"
        )
        exported.extend([p1, p2])

        # Table 2: Model Architecture and Hyperparameters
        t2_data = {
            "Model Family": [
                "Random Forest (RF)", "Random Forest (RF)", "Random Forest (RF)",
                "CNN-LSTM", "CNN-LSTM", "CNN-LSTM", "CNN-LSTM"
            ],
            "Component / Hyperparameter": [
                "Number of Trees (n_estimators)", "Maximum Tree Depth", "Class Weighting",
                "Conv1D Feature Extractor", "LSTM Sequence Recurrent Layer", "Dropout Probability", "Optimizer & LR"
            ],
            "Value": [
                "50" if experiment_results["mode"] in ("quick", "three_subject_loso", "full_fast") else "100", "15", "Balanced",
                "Conv1(32, k=7) -> Conv2(64, k=5)", "64 Hidden Units (Single Layer)", "0.30", "Adam (lr=0.001, early stopping)"
            ]
        }
        p1, p2 = self.export_table(
            pd.DataFrame(t2_data),
            "table2_model_hyperparameters",
            "HAR Model Architecture and Hyperparameter Configuration",
            "tab:model_hyperparams"
        )
        exported.extend([p1, p2])

        # Table 3: Clean-Data Recognition Comparison
        clean_comp = experiment_results.get("clean_comparison", [])
        if clean_comp:
            df_clean = pd.DataFrame(clean_comp)
            p1, p2 = self.export_table(
                df_clean,
                "table3_clean_model_comparison",
                "HAR Recognition Performance Across Baselines on Clean Benchmark Data",
                "tab:clean_comparison"
            )
            exported.extend([p1, p2])

        # Table 4: Fault Scenarios & Severity Definitions
        t4_data = {
            "Fault Scenario": ["Dropout", "Intermittent Missing", "Gaussian Noise", "Stuck-at / Flatline", "Bias Drift", "Combined"],
            "Physical Phenomenon": [
                "Unavailable sensor samples (NaN marker)",
                "Transient communication packet loss",
                "Electromagnetic interference / noisy analog stage",
                "Sensor hardware lockup / frozen ADC",
                "Thermal calibration drift / DC bias shift", "Noise and intermittent loss"
            ],
            "Mild (25%)": ["25% window drop", "25% blocks dropped", "noise SD=3 scaled units", "25% samples stuck", "+5 scaled-unit drift", "noise + missing"],
            "Moderate (50%)": ["50% window drop", "50% blocks dropped", "noise SD=6 scaled units", "50% samples stuck", "+10 scaled-unit drift", "noise + missing"],
            "Severe (85%)": ["85% window drop", "85% blocks dropped", "noise SD=10.2 scaled units", "85% samples stuck", "+17 scaled-unit drift", "noise + missing"]
        }
        p1, p2 = self.export_table(
            pd.DataFrame(t4_data),
            "table4_fault_definitions",
            "Simulated Wearable Sensor Fault Scenarios and Severity Settings",
            "tab:fault_definitions"
        )
        exported.extend([p1, p2])

        # Table 5: Faulted Recognition Comparison
        faulted_comp = experiment_results.get("faulted_comparison", [])
        if faulted_comp:
            df_fault = pd.DataFrame(faulted_comp)
            p1, p2 = self.export_table(
                df_fault,
                "table5_faulted_recognition_comparison",
                "Recognition Performance Under Sensor Faults and Resilience Retention",
                "tab:faulted_comparison"
            )
            exported.extend([p1, p2])

        # Table 6: Sensor Reliability & Fault Detection Metrics
        rel_runs = [m for scenario in experiment_results["reliability_detection"].values()
                    for severity in scenario.values() for m in severity]
        rel_metrics = {key: float(np.mean([m[key] for m in rel_runs if m[key] is not None]))
                       for key in ["auroc", "precision", "recall", "f1", "false_alarm_rate"]}
        t6_data = {
            "Metric": ["AUROC", "Detection Precision", "Detection Recall", "Detection F1", "False-Alarm Rate (FAR)"],
            "Observed Value": [
                f"{rel_metrics['auroc']:.4f}",
                f"{rel_metrics['precision']:.4f}",
                f"{rel_metrics['recall']:.4f}",
                f"{rel_metrics['f1']:.4f}",
                f"{rel_metrics['false_alarm_rate']:.4f}"
            ]
        }
        p1, p2 = self.export_table(
            pd.DataFrame(t6_data),
            "table6_reliability_fault_detection",
            "SensorReliabilityEstimator Fault Detection Capability Against Ground Truth",
            "tab:rel_detection"
        )
        exported.extend([p1, p2])

        # Table 7: Sensing Cost and Active Sensor Comparison
        cost_comp = experiment_results.get("cost_comparison", [])
        if cost_comp:
            df_cost = pd.DataFrame(cost_comp)
            p1, p2 = self.export_table(
                df_cost,
                "table7_sensing_cost_comparison",
                "Sensing Cost Proxy and Dynamic Switching Statistics by Baseline",
                "tab:sensing_cost"
            )
            exported.extend([p1, p2])

        # Table 8: Model Latency and Computational Footprint
        latency_comp = experiment_results.get("latency_comparison", [])
        if latency_comp:
            df_lat = pd.DataFrame(latency_comp)
            p1, p2 = self.export_table(
                df_lat,
                "table8_model_latency_computation",
                "Inference Latency and Training Duration per Sensor Classifier",
                "tab:model_latency"
            )
            exported.extend([p1, p2])

        # Table 9: Per-Subject Evaluation Results
        per_subject = experiment_results.get("per_subject_results", [])
        if per_subject:
            df_sub = pd.DataFrame(per_subject)
            p1, p2 = self.export_table(
                df_sub,
                "table9_per_subject_loso_results",
                "Held-Out Subject Performance and Sensing-Cost Proxies (Quick Mode Uses One Fold)",
                "tab:per_subject"
            )
            exported.extend([p1, p2])

        # Table 10: Proposed vs Baselines Overall Synthesis
        overall_synthesis = experiment_results.get("overall_synthesis", [])
        if overall_synthesis:
            df_all = pd.DataFrame(overall_synthesis)
            p1, p2 = self.export_table(
                df_all,
                "table10_proposed_vs_baselines_summary",
                "Comprehensive Comparison Between Proposed Adaptive Method and All Baselines",
                "tab:overall_summary"
            )
            exported.extend([p1, p2])

        return exported

    def generate_narratives(self, experiment_results: Dict[str, Any]) -> None:
        """Generate report-ready Markdown and LaTeX methodology and result narrative summaries."""
        best_clean = experiment_results["best_clean_f1"]
        best_faulted = experiment_results["best_faulted_f1"]
        avg_cost = experiment_results["proposed_avg_cost"]
        all_uniform_clean = next(float(row["Macro-F1"].split(" +/- ")[0])
            for row in experiment_results["clean_comparison"] if row["Baseline Architecture"] == "All Sensors (Uniform Fusion)")
        rel_runs = [m for scenario in experiment_results["reliability_detection"].values()
                    for severity in scenario.values() for m in severity]
        mean_auroc = float(np.mean([m["auroc"] for m in rel_runs if m["auroc"] is not None]))
        cost_reduction = (1.0 - avg_cost) * 100.0

        # results_summary.md
        md_text = f"""# Experimental Results and Empirical Findings

## 1. Clean-Data Recognition Performance
The proposed Random Forest ensemble achieved a clean macro-F1 of **{best_clean:.4f}** on the configured evaluation split. Compare its baseline rows before claiming superiority.

## 2. Fault Resilience Under Physical Sensor Degradation
When subjecting wearable channels to simulated hardware dropout, packet loss, Gaussian noise, stuck-at flatlines, and DC bias drift:
- Mean fault-detection AUROC across configured scenarios and folds was **{mean_auroc:.4f}**, comparing clean and injected-fault windows.
- Proposed macro-F1 under moderate dropout was **{best_faulted:.4f}**. Other fault conditions may be worse; inspect the comparison table.

## 3. Sensing Cost and Energy Proxy Trade-Off
The proposed closed-loop **Adaptive Sensing Policy** activated auxiliary sensors in response to degraded reliability, high uncertainty, or periodic probing, subject to the configured budget.
- Across the evaluated continuous sequences, the proposed method maintained an average normalized sensing cost of **{avg_cost:.2f}** (relative to 1.00 for always-on multi-sensor operation), representing a **{cost_reduction:.1f}% reduction** in active sensor activation.
- Switching behavior is reported in the cost comparison table; a dwell constraint does not guarantee zero oscillation.

## 4. Key Limitations and Experimental Realism
- SENSING COST PROXY: All reported cost metrics reflect normalized sensor duty-cycling proxies, not direct physical battery hardware measurements.
- BENCHMARK DATASET: Validation was conducted on the benchmark PAMAP2 dataset under protocol conditions; clinical and unconstrained ambulatory deployments may exhibit different noise signatures.
"""
        with open(self.output_dir / "results_summary.md", "w", encoding="utf-8") as f:
            f.write(md_text)

        # results_summary.tex
        tex_text = f"""\\section{{Experimental Results and Empirical Findings}}
\\label{{sec:results}}

\\subsection{{Clean-Data Recognition Baseline}}
The proposed Random Forest method obtained clean macro-F1 {best_clean:.4f} on the configured evaluation split.

\\subsection{{Fault Resilience Under Simulated Sensor Failure}}
For moderate dropout, the proposed method obtained macro-F1 {best_faulted:.4f}. Results for other injected faults are tabulated separately.
The \\textit{{SensorReliabilityEstimator}} obtained mean AUROC {mean_auroc:.4f} when clean and injected-fault windows were compared; injected labels were used only for evaluation.

\\subsection{{Sensing Efficiency and Cost Savings}}
The runtime controller responded to reliability, entropy, budget, and periodic probes.
The normalized sensing cost proxy $C(t)$ was {avg_cost:.2f} (a {cost_reduction:.1f}\\% reduction compared to always-active sensing). The all-sensor uniform clean macro-F1 was {all_uniform_clean:.4f}, compared with {best_clean:.4f} for the adaptive method.
"""
        with open(self.output_dir / "results_summary.tex", "w", encoding="utf-8") as f:
            f.write(tex_text)

        # methodology_summary.md
        method_text = """# Research Methodology Summary

## Pipeline Architecture
1. **Acquisition & Preprocessing**: PAMAP2 protocol recordings (100 Hz, 54 columns) loaded with exact channel mappings (Wrist cols 4-6, 10-12; Chest cols 21-23, 27-29; Ankle cols 38-40, 44-46).
2. **Leakage Prevention**: Subject-wise Leave-One-Subject-Out (LOSO) partitioning across all protocol participants. Preprocessing scaling parameters are strictly fitted on training participants.
3. **Model Families**:
   - Model A: Sensor-wise Random Forest (72 statistical features per window, probability alignment).
   - Model B: Sensor-wise CNN-LSTM (1D Conv temporal-spatial filters + LSTM sequence modeling).
4. **Reliability Estimation**: Computes observable quality indicators: missing-sample ratio, flatline variance, noise first-difference variance, and sensor clipping.
5. **Uncertainty & Fusion**: Reliability-weighted probability fusion with normalized Shannon entropy quantification.
6. **Closed-Loop Controller**: Regulates active sensor configuration for the NEXT sensing window $t+1$ under physical realism rules.
"""
        with open(self.output_dir / "methodology_summary.md", "w", encoding="utf-8") as f:
            f.write(method_text)

        # figure_captions.md
        fig_cap = """# Saved Figure Captions and Provenance

- **Figure 1 (`fig1_activity_distribution.png`)**: Counts of selected activity samples in the evaluated official protocol files; these are rows, not overlapping windows.
- **Figure 2 (`fig2_clean_confusion_matrix_rf.png`)**: Adaptive random-forest confusion matrix for the clean first held-out fold only.
- **Figure 3 (`fig3_clean_confusion_matrix_cnn.png`)**: Adaptive CNN--LSTM confusion matrix for the clean first held-out fold only.
- **Figure 4 (`fig4_faulted_confusion_matrix.png`)**: Adaptive random-forest confusion matrix for the final fold under moderate wrist dropout.
- **Figure 5 (`fig5_macro_f1_vs_severity.png`)**: Mean macro-F1 for three all-sensor/adaptive strategies under clean data and three wrist-dropout severities.
- **Figure 7 (`fig7_cost_vs_macro_f1.png`)**: Clean mean macro-F1 versus normalized sensing-cost proxy for five random-forest strategies.
- **Figure 9 (`fig9_reliability_and_activation_timeline.png`)**: Reliability scores and activation masks from the final-fold moderate wrist-dropout adaptive run.
- **Figure 11 (`fig11_cnn_training_curves.png`)**: CNN--LSTM training history from the final held-out fold.
- **Figure 12 (`fig12_rf_feature_importance.png`)**: Wrist random-forest feature importance from the final held-out fold.

Figures 6, 8, 10, and 13 are not generated and are not referenced by the IEEE manuscript.
"""
        with open(self.output_dir / "figure_captions.md", "w", encoding="utf-8") as f:
            f.write(fig_cap)

        # table_captions.md
        tab_cap = """# IEEE Table Captions

- **Table 1 (`table1_dataset_preprocessing.csv` / `.tex`)**: PAMAP2 dataset and sliding-window preprocessing specifications.
- **Table 2 (`table2_model_hyperparameters.csv` / `.tex`)**: Model hyperparameters and architectural configurations.
- **Table 3 (`table3_clean_model_comparison.csv` / `.tex`)**: Clean-data recognition comparison across all 5 baselines.
- **Table 4 (`table4_fault_definitions.csv` / `.tex`)**: Fault scenarios, physical phenomena, and severity parameterization.
- **Table 5 (`table5_faulted_recognition_comparison.csv` / `.tex`)**: Faulted recognition performance and degradation in percentage points.
- **Table 6 (`table6_reliability_fault_detection.csv` / `.tex`)**: SensorReliabilityEstimator detection AUROC, precision, recall, and false-alarm rate.
- **Table 7 (`table7_sensing_cost_comparison.csv` / `.tex`)**: Normalized sensing cost proxy and switching statistics.
- **Table 8 (`table8_model_latency_computation.csv` / `.tex`)**: Host-side training duration for the final fold; per-model inference latency was not measured.
- **Table 9 (`table9_per_subject_loso_results.csv` / `.tex`)**: Leave-one-subject-out (LOSO) scores with separate clean and moderate-dropout sensing-cost proxies.
- **Table 10 (`table10_proposed_vs_baselines_summary.csv` / `.tex`)**: Comprehensive synthesis of recognition, resilience, and efficiency metrics.
"""
        with open(self.output_dir / "table_captions.md", "w", encoding="utf-8") as f:
            f.write(tab_cap)

        # experiment_protocol.md
        exp_prot = """# Experimental Protocol Specification

## Evaluation Protocol: Leave-One-Subject-Out (LOSO) Cross-Validation
- For each outer fold $k \\in \\{1, \\dots, S\\}$:
  1. Participant $k$ is held out as the test set.
  2. Remaining $S-1$ participants are partitioned into training and internal validation subjects.
  3. Preprocessing scalers (mean, std) and reliability baseline statistics are fitted strictly on the training subjects.
  4. Models (Random Forest and CNN-LSTM) are trained on the training fold.
  5. Internal validation subjects are used for CNN-LSTM early stopping.
  6. The held-out test sequence is evaluated chronologically.
  7. No future information, test subject data, or ground-truth fault masks are ever exposed to the models or controller.
"""
        with open(self.output_dir / "experiment_protocol.md", "w", encoding="utf-8") as f:
            f.write(exp_prot)

    def create_report_bundle_zip(self, zip_filename: str = "report_assets_bundle.zip") -> Path:
        """Create a downloadable ZIP archive of all report assets (figures, tables, narratives)."""
        zip_path = self.output_dir / zip_filename
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    if file == zip_filename:
                        continue
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(self.output_dir)
                    zipf.write(full_p, arcname=str(rel_p))

        logger.info(f"Report assets bundled successfully: {zip_path}")
        return zip_path
