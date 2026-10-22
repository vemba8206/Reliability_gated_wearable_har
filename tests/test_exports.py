"""Unit tests for report artifact exporter: CSV, LaTeX escaping, and ZIP archiving."""

import pytest
from pathlib import Path
import pandas as pd

from src.report_exports import escape_latex, dataframe_to_ieee_latex, ReportArtifactExporter


def test_escape_latex():
    raw_s = "Accuracy & F1_Score with 10% Noise #1"
    escaped = escape_latex(raw_s)
    assert r"\&" in escaped
    assert r"\_" in escaped
    assert r"\%" in escaped
    assert r"\#" in escaped


def test_table_to_latex():
    df = pd.DataFrame({
        "Model": ["RF", "CNN"],
        "F1_Score": [0.92, 0.90]
    })
    latex_code = dataframe_to_ieee_latex(df, "Model Comparison", "tab:comp")
    assert "\\begin{table}" in latex_code
    assert "\\end{table}" in latex_code
    assert "\\caption{Model Comparison}" in latex_code


def test_artifact_exporter_bundle(tmp_path):
    exporter = ReportArtifactExporter(output_dir=str(tmp_path / "test_assets"))
    df = pd.DataFrame({"Metric": ["Accuracy"], "Value": [0.95]})
    p_csv, p_tex = exporter.export_table(df, "test_table", "Test Caption", "tab:test")

    assert p_csv.exists()
    assert p_tex.exists()

    zip_file = exporter.create_report_bundle_zip()
    assert zip_file.exists()
    assert zip_file.stat().st_size > 0
