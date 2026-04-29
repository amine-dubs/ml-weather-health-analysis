"""
After running train_all_novel_architectures.py, run this script
to inject actual results into the LaTeX report and rebuild it.

Usage:
  .venv\Scripts\python.exe update_novel_report.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "pytorch_results" / "novel_architectures_all_results.csv"
SUMMARY_CSV = ROOT / "pytorch_results" / "novel_architectures_summary.csv"
TEX_FILE = ROOT / "DNN_Novel_Architectures_Report.tex"

# ML baselines for comparison
ML_BASELINES = {
    "heart_classification":    {"metric": "ROC-AUC",   "ml": 0.9784, "ml_model": "Stacking"},
    "covid_classification":    {"metric": "AUC-ROC",   "ml": 0.8975, "ml_model": "Stacking (LR)"},
    "temperature_regression":  {"metric": "$R^2$",     "ml": 0.7766, "ml_model": "Stacking"},
    "multi_output_regression": {"metric": "Avg $R^2$", "ml": 0.9282, "ml_model": "XGBoost"},
    "weather_classification":  {"metric": "AUC",       "ml": 0.8493, "ml_model": "Random Forest"},
    "wind_forecasting":        {"metric": "$R^2$",     "ml": 0.9714, "ml_model": "Ridge"},
    "energy_forecasting":      {"metric": "$R^2$",     "ml": 0.9985, "ml_model": "Linear Reg."},
    "anomaly_employee":        {"metric": "F1",        "ml": 0.4694, "ml_model": "DBSCAN"},
    "anomaly_heart":           {"metric": "Bal.\\ Acc.", "ml": 0.6640, "ml_model": "Elliptic Env."},
    "anomaly_wine":            {"metric": "F1",        "ml": 0.9188, "ml_model": "Elliptic Env."},
}

# Previous best DNN for comparison
PREV_DNN = {
    "heart_classification": 0.9517,
    "covid_classification": 0.9009,
    "temperature_regression": 0.7520,
    "multi_output_regression": 0.4665,
    "weather_classification": 0.6407,
    "wind_forecasting": 0.9710,
    "energy_forecasting": 0.9977,
    "anomaly_employee": 0.7423,
    "anomaly_heart": 0.7711,
    "anomaly_wine": 0.9937,
}

TASK_LABELS = {
    "heart_classification": "Heart Classification",
    "covid_classification": "COVID Classification",
    "temperature_regression": "Temperature Regression",
    "multi_output_regression": "Multi-Output Regression",
    "weather_classification": "Weather Classification",
    "wind_forecasting": "Wind Forecasting",
    "energy_forecasting": "Energy Forecasting",
    "anomaly_employee": "Anomaly Employee",
    "anomaly_heart": "Anomaly Heart",
    "anomaly_wine": "Anomaly Wine",
}

def build_results_latex(df):
    lines = []

    # ── Per-task detailed tables ──
    lines.append("\\subsection{Detailed Results by Task}")
    lines.append("")

    for task in df["task"].unique():
        tdf = df[df["task"] == task].copy()
        tdf = tdf.sort_values("test_metric", ascending=False)
        bl = ML_BASELINES.get(task, {})
        prev = PREV_DNN.get(task, None)

        lines.append(f"\\subsubsection*{{{TASK_LABELS.get(task, task)}}}")
        lines.append("")
        lines.append("\\begin{center}")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Architecture} & \\textbf{Test Metric} & \\textbf{Val Metric} & \\textbf{Epochs} & \\textbf{Time (s)} \\\\")
        lines.append("\\midrule")

        for _, row in tdf.iterrows():
            tm = f"{row['test_metric']:.4f}" if pd.notna(row.get('test_metric')) else "---"
            vm = f"{row['val_metric']:.4f}" if pd.notna(row.get('val_metric')) else "---"
            ep = str(int(row['epochs'])) if pd.notna(row.get('epochs')) else "---"
            ts = f"{row['time_s']:.1f}" if pd.notna(row.get('time_s')) else "---"
            arch = row['architecture'].replace("_", "\\_")
            lines.append(f"{arch} & {tm} & {vm} & {ep} & {ts} \\\\")

        lines.append("\\midrule")
        if prev:
            lines.append(f"Previous best DNN & {prev:.4f} & --- & --- & --- \\\\")
        if bl:
            lines.append(f"ML baseline ({bl.get('ml_model','')}) & {bl['ml']:.4f} & --- & --- & --- \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{center}")
        lines.append("")

    # ── Grand summary table ──
    lines.append("\\subsection{Grand Summary: Best Novel Architecture per Task}")
    lines.append("")
    lines.append("\\begin{center}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{p{3cm}p{1.2cm}p{2.8cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1cm}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Task} & \\textbf{Metric} & \\textbf{Best Novel Arch.} & \\textbf{Novel} & \\textbf{Prev DNN} & \\textbf{ML} & \\textbf{Best?} \\\\")
    lines.append("\\midrule")

    for task in df["task"].unique():
        tdf = df[df["task"] == task].dropna(subset=["test_metric"])
        if tdf.empty: continue
        best = tdf.loc[tdf["test_metric"].idxmax()]
        bl = ML_BASELINES.get(task, {})
        prev = PREV_DNN.get(task, None)
        novel_val = best["test_metric"]
        ml_val = bl.get("ml", 0)
        prev_val = prev if prev else 0

        best_of = "Novel" if novel_val >= max(ml_val, prev_val) else ("ML" if ml_val >= prev_val else "Prev DNN")

        tl = TASK_LABELS.get(task, task)
        met = bl.get("metric", "---")
        arch = best["architecture"].replace("_", "\\_")
        lines.append(f"{tl} & {met} & {arch} & {novel_val:.4f} & {prev_val:.4f} & {ml_val:.4f} & {best_of} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("")

    return "\n".join(lines)


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run train_all_novel_architectures.py first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} result rows from {RESULTS_CSV}")

    results_latex = build_results_latex(df)

    # Read existing tex, replace placeholder
    tex = TEX_FILE.read_text(encoding="utf-8")

    placeholder = "% RESULTS_PLACEHOLDER"
    if placeholder in tex:
        # Remove the placeholder text about running the script
        old_section = """\\subsection{Placeholder: Full Results Table}

After running all architectures, the complete table will show results for all 6 configurations (3 architectures $\\times$ 2 sizes) across all 10 tasks, along with comparison to existing ML and DNN baselines.

% This section will be updated by update_report_with_results.py
% RESULTS_PLACEHOLDER"""

        if old_section in tex:
            tex = tex.replace(old_section, results_latex)
        else:
            tex = tex.replace(placeholder, results_latex)
    else:
        print("WARNING: placeholder not found, appending results before conclusion")
        tex = tex.replace("\\section{Existing Results Summary",
                          results_latex + "\n\n\\section{Existing Results Summary")

    TEX_FILE.write_text(tex, encoding="utf-8")
    print(f"Updated {TEX_FILE}")
    print("You can now compile with: pdflatex DNN_Novel_Architectures_Report.tex")


if __name__ == "__main__":
    main()
