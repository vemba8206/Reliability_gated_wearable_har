"""Reliability-Gated Adaptive Sensing for Fault-Resilient HAR - Streamlit Dashboard."""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_download import is_dataset_present, find_existing_subject_files, acquire_dataset
from src.data_loader import SENSOR_CHANNELS, DEFAULT_ACTIVITIES
from src.run_experiments import run_experiment_pipeline, load_config
from src.fault_injection import SensorFaultSimulator
from src.reliability import SensorReliabilityEstimator

st.set_page_config(
    page_title="Reliability-Gated Adaptive HAR",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Dark/Navy Styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
    .kpi-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; font-weight: 600; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-success { background-color: #238636; color: white; }
    .badge-warning { background-color: #d29922; color: black; }
    .badge-info { background-color: #1f6feb; color: white; }
    .callout-box { background-color: #161b22; border-left: 4px solid #58a6ff; padding: 12px; border-radius: 4px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)


def find_latest_experiment_summary() -> tuple[dict | None, Path | None]:
    runs_dir = Path("results/runs")
    if not runs_dir.exists():
        return None, None
    summary_files = list(runs_dir.rglob("experiment_summary.json"))
    if not summary_files:
        return None, None
    # Sort by modification time
    summary_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest = summary_files[0]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f), latest.parent


# Sidebar Navigation
st.sidebar.title("📡 Reliability-Gated HAR")
st.sidebar.caption("Fault-Resilient Wearable IoT Sensing")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Overview Dashboard",
        "Dataset Explorer",
        "Model Laboratory",
        "Fault Simulation Lab",
        "Results & Comparisons",
        "Export Center"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
> **Note on Experimental Realism:**
> Sensing cost is calculated via a normalized duty-cycling proxy $C(t)$, **not** direct physical battery dissipation.
""")

# Load latest run results if available
latest_summary, latest_run_dir = find_latest_experiment_summary()

# ==============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==============================================================================
if nav_choice == "Overview Dashboard":
    st.title("Reliability-Gated Adaptive Sensing for Wearable HAR")
    st.markdown("""
    A closed-loop AI-IoT framework dynamically adjusting active wearable sensor nodes based on 
    **observable signal reliability**, **prediction uncertainty**, and a **normalized sensing cost budget**.
    """)

    st.markdown("""
    <div class="callout-box">
        <strong>Research Question:</strong> Can a lightweight runtime policy combine per-sensor reliability, HAR prediction uncertainty, 
        and sensing/communication cost to choose which wearable sensors should be active in the next sensing window, improving fault resilience 
        while reducing normalized sensing cost?
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    dataset_ok = is_dataset_present()
    col1.metric("Dataset Availability", "PAMAP2 Ready" if dataset_ok else "Missing", delta="9 Subjects" if dataset_ok else "Not downloaded")
    
    if latest_summary:
        best_clean = latest_summary.get("best_clean_f1", 0.0)
        best_fault = latest_summary.get("best_faulted_f1", 0.0)
        avg_cost = latest_summary.get("proposed_avg_cost", 1.0)

        col2.metric("Best Clean Macro-F1", f"{best_clean:.4f}", delta="Ensemble")
        col3.metric("Faulted Macro-F1", f"{best_fault:.4f}", delta="Resilient")
        col4.metric("Avg Sensing Cost C(t)", f"{avg_cost:.2f}", delta=f"-{(1.0 - avg_cost)*100:.1f}% Active Nodes", delta_color="inverse")
    else:
        col2.metric("Best Clean Macro-F1", "Not Run", delta=None)
        col3.metric("Faulted Macro-F1", "Not Run", delta=None)
        col4.metric("Avg Sensing Cost C(t)", "Not Run", delta=None)

    st.markdown("---")
    st.subheader("Interactive Experiment Execution")
    col_act1, col_act2, col_act3 = st.columns([1, 1, 2])

    with col_act1:
        if st.button("🚀 Run Quick Dev Experiment", use_container_width=True):
            with st.spinner("Executing Quick Development Experiment (~15-20s)..."):
                res = run_experiment_pipeline(mode="quick")
                st.success(f"Quick Run complete! Run ID: {res['run_id']}")
                st.rerun()

    with col_act2:
        if st.button("⚙️ Run Full LOSO Evaluation", use_container_width=True):
            with st.spinner("Executing Full LOSO Evaluation across all subjects..."):
                res = run_experiment_pipeline(mode="full")
                st.success(f"Full LOSO Run complete! Run ID: {res['run_id']}")
                st.rerun()

    with col_act3:
        if st.button("📥 Verify / Reacquire PAMAP2 Dataset", use_container_width=True):
            status, msg = acquire_dataset()
            st.info(f"Dataset status: {msg}")

    # Display summary of last run
    if latest_summary:
        st.markdown("---")
        st.subheader(f"Latest Completed Experiment (`{latest_summary['run_id']}`)")
        
        tab_sum1, tab_sum2 = st.tabs(["Clean vs Faulted Synthesis", "Per-Subject Results"])
        with tab_sum1:
            if "overall_synthesis" in latest_summary:
                st.dataframe(pd.DataFrame(latest_summary["overall_synthesis"]), use_container_width=True)
        with tab_sum2:
            if "per_subject_results" in latest_summary:
                st.dataframe(pd.DataFrame(latest_summary["per_subject_results"]), use_container_width=True)


# ==============================================================================
# PAGE 2: DATASET EXPLORER
# ==============================================================================
elif nav_choice == "Dataset Explorer":
    st.title("PAMAP2 Dataset & Preprocessing Explorer")
    st.markdown("Official Physical Activity Monitoring Dataset (100 Hz, 3 IMUs: Wrist, Chest, Ankle).")

    files = find_existing_subject_files("data/raw")
    st.write(f"**Discovered Subject Files:** {len(files)} files")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("### Selected Sensor Nodes & Channels")
        for sensor, cfg in SENSOR_CHANNELS.items():
            st.markdown(f"**{sensor.upper()} IMU:**")
            st.markdown(f"- Accel (±16g): `{cfg['names'][:3]}`")
            st.markdown(f"- Gyroscope: `{cfg['names'][3:]}`")

    with col_info2:
        st.markdown("### Preprocessing Specifications")
        st.markdown("- **Sampling Rate:** 100 Hz")
        st.markdown("- **Window Length:** 2.56 seconds (256 samples)")
        st.markdown("- **Sliding Stride:** 1.28 seconds (128 samples, 50% overlap)")
        st.markdown("- **Label Purity Threshold:** 0.80 (ambiguous transitions discarded)")
        st.markdown("- **Evaluation Partitioning:** Leave-One-Subject-Out (LOSO) Cross-Validation")

    # Display Figure 1 if generated
    fig1_path = Path("report_assets/figures/fig1_activity_distribution.png")
    if fig1_path.exists():
        st.markdown("---")
        st.subheader("Protocol Activity Class Distribution")
        st.image(str(fig1_path), caption="Figure 1: Usable windows retained across target activities.")


# ==============================================================================
# PAGE 3: MODEL LABORATORY
# ==============================================================================
elif nav_choice == "Model Laboratory":
    st.title("HAR Model Laboratory: Sensor-Wise RF vs CNN-LSTM")
    st.markdown("Independent classifiers trained per body location (Wrist, Chest, Ankle) with probability alignment.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### Model A: Sensor-Wise Random Forest")
        st.markdown("- 72 time-domain statistical features per sensor window.")
        st.markdown("- Gini impurity split criterion, 100 estimators, balanced class weights.")
        st.markdown("- Fast CPU inference (< 0.5 ms per window).")
        
        fig12_p = Path("report_assets/figures/fig12_rf_feature_importance.png")
        if fig12_p.exists():
            st.image(str(fig12_p), caption="Top predictive features for Random Forest.")

    with col_m2:
        st.markdown("### Model B: Sensor-Wise CNN-LSTM")
        st.markdown("- 1D Conv blocks (32 filters k=7, 64 filters k=5) + LSTM (hidden=64).")
        st.markdown("- End-to-end temporal representation learning on raw 6-channel series.")
        st.markdown("- Validation-loss early stopping to prevent overfitting.")

        fig11_p = Path("report_assets/figures/fig11_cnn_training_curves.png")
        if fig11_p.exists():
            st.image(str(fig11_p), caption="CNN-LSTM Training & Validation Loss Progression.")

    st.markdown("---")
    st.subheader("Clean-Data Baseline Comparison")
    if latest_summary and "clean_comparison" in latest_summary:
        st.dataframe(pd.DataFrame(latest_summary["clean_comparison"]), use_container_width=True)

    col_cm1, col_cm2 = st.columns(2)
    fig2_p = Path("report_assets/figures/fig2_clean_confusion_matrix_rf.png")
    fig3_p = Path("report_assets/figures/fig3_clean_confusion_matrix_cnn.png")
    with col_cm1:
        if fig2_p.exists():
            st.image(str(fig2_p), caption="Figure 2: Random Forest Clean Confusion Matrix.")
    with col_cm2:
        if fig3_p.exists():
            st.image(str(fig3_p), caption="Figure 3: CNN-LSTM Clean Confusion Matrix.")


# ==============================================================================
# PAGE 4: FAULT SIMULATION LABORATORY
# ==============================================================================
elif nav_choice == "Fault Simulation Lab":
    st.title("Controlled Wearable Fault Simulation Laboratory")
    st.markdown("Interactive inspection of signal corruption, observable quality indicators, and reliability scoring.")

    sim = SensorFaultSimulator(random_seed=42)
    est = SensorReliabilityEstimator()

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        fault_type = st.selectbox("Select Fault Scenario", ["dropout", "intermittent", "noise", "flatline", "drift", "combined"])
    with col_ctrl2:
        target_sensor = st.selectbox("Target Sensor", ["wrist", "chest", "ankle"])
    with col_ctrl3:
        severity = st.slider("Corruption Severity", 0.1, 1.0, 0.5, 0.05)

    # Generate synthetic nominal IMU window
    t = np.linspace(0, 2.56, 256)
    w_nom = np.zeros((256, 6))
    w_nom[:, 0] = np.sin(2 * np.pi * 1.5 * t) * 4.0
    w_nom[:, 1] = np.cos(2 * np.pi * 1.5 * t) * 3.0 + 9.81
    w_nom[:, 2] = np.sin(2 * np.pi * 3.0 * t) * 2.0
    w_nom[:, 3:] = np.random.randn(256, 3) * 0.5

    # Inject fault
    w_corrupted, meta = sim.inject_fault(w_nom, fault_type, severity)
    rel_out = est.estimate(w_corrupted)

    # Plot Clean vs Corrupted Signal
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(y=w_nom[:, 1], mode="lines", name="Nominal Accel Y (Clean)", line=dict(color="#58a6ff", dash="dash")))
    fig_sig.add_trace(go.Scatter(y=w_corrupted[:, 1], mode="lines", name=f"Corrupted Accel Y ({fault_type.capitalize()})", line=dict(color="#f85149", width=2)))
    fig_sig.update_layout(
        title=f"Sensor Signal Before vs. After Fault Injection: {fault_type.capitalize()} (Severity={severity:.2f})",
        xaxis_title="Time Samples (100 Hz)",
        yaxis_title="Acceleration (m/s²)",
        template="plotly_dark",
        height=380
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    # Quality indicators
    st.markdown("### Runtime Observable Quality Indicators & Reliability Score")
    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    col_q1.metric("Estimated Reliability", f"{rel_out['reliability']:.3f}", delta="Degraded" if rel_out['is_faulty'] else "Nominal")
    col_q2.metric("Missing Indicator", f"{rel_out['indicators']['missing']:.3f}")
    col_q3.metric("Flatline Indicator", f"{rel_out['indicators']['flatline']:.3f}")
    col_q4.metric("Noise Indicator", f"{rel_out['indicators']['noise']:.3f}")
    col_q5.metric("Clipping Indicator", f"{rel_out['indicators']['clipping']:.3f}")


# ==============================================================================
# PAGE 5: RESULTS & COMPARISONS
# ==============================================================================
elif nav_choice == "Results & Comparisons":
    st.title("Experimental Results & Baseline Comparison")
    st.markdown("Comprehensive performance across all 5 baselines under controlled fault modes.")

    if latest_summary and "faulted_comparison" in latest_summary:
        st.subheader("Performance Under 50% Fault Severity")
        st.dataframe(pd.DataFrame(latest_summary["faulted_comparison"]), use_container_width=True)

    col_res1, col_res2 = st.columns(2)
    fig5_p = Path("report_assets/figures/fig5_macro_f1_vs_severity.png")
    fig7_p = Path("report_assets/figures/fig7_cost_vs_macro_f1.png")

    with col_res1:
        if fig5_p.exists():
            st.image(str(fig5_p), caption="Figure 5: Macro-F1 Degradation vs. Fault Severity.")
    with col_res2:
        if fig7_p.exists():
            st.image(str(fig7_p), caption="Figure 7: Sensing Efficiency vs. Macro-F1 Pareto Trade-Off.")

    st.markdown("---")
    st.subheader("Dynamic Reliability & Closed-Loop Active Sensor Timeline")
    fig9_p = Path("report_assets/figures/fig9_reliability_and_activation_timeline.png")
    if fig9_p.exists():
        st.image(str(fig9_p), caption="Figure 9 & 10: Real-time sensor health trajectory and next-window activation.")

    if latest_summary and "cost_comparison" in latest_summary:
        st.markdown("---")
        st.subheader("Sensing Cost and Duty-Cycling Comparison")
        st.dataframe(pd.DataFrame(latest_summary["cost_comparison"]), use_container_width=True)


# ==============================================================================
# PAGE 6: EXPORT CENTER
# ==============================================================================
elif nav_choice == "Export Center":
    st.title("Report Artifact Export Center")
    st.markdown("Export 300 DPI figures, CSV tables, IEEE-compliant LaTeX tables, and the complete research bundle.")

    bundle_path = Path("report_assets/report_assets_bundle.zip")
    if bundle_path.exists():
        with open(bundle_path, "rb") as f:
            st.download_button(
                label="📦 Download Complete Report Assets ZIP Bundle",
                data=f.read(),
                file_name="report_assets_bundle.zip",
                mime="application/zip",
                use_container_width=True
            )

    st.markdown("---")
    tab_exp1, tab_exp2, tab_exp3 = st.tabs(["Exported LaTeX Tables", "Exported CSV Tables", "Research Narrative Summaries"])

    with tab_exp1:
        tex_files = list(Path("report_assets/tables").glob("*.tex"))
        if tex_files:
            selected_tex = st.selectbox("Select LaTeX Table Snippet", [f.name for f in tex_files])
            tex_content = (Path("report_assets/tables") / selected_tex).read_text(encoding="utf-8")
            st.code(tex_content, language="latex")
        else:
            st.info("No LaTeX tables exported yet. Run an experiment first.")

    with tab_exp2:
        csv_files = list(Path("report_assets/tables").glob("*.csv"))
        if csv_files:
            selected_csv = st.selectbox("Select CSV Table", [f.name for f in csv_files])
            df = pd.read_csv(Path("report_assets/tables") / selected_csv)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No CSV tables exported yet. Run an experiment first.")

    with tab_exp3:
        narrative_p = Path("report_assets/results_summary.md")
        if narrative_p.exists():
            st.markdown(narrative_p.read_text(encoding="utf-8"))
        else:
            st.info("Run an experiment to generate narrative summaries.")
