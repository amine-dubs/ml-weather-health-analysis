"""
Scratch architecture launcher and report builder.

This script has two jobs:
  1. Run the existing benchmark scripts when asked to regenerate results.
  2. Build a clean Markdown report from the already-saved CSV outputs.

The report intentionally reuses the verified preprocessing and result files
already stored in the repository so the comparison stays apples-to-apples.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

REPORT_PATH = HERE / "clean_report.md"
BEST_MODELS_PATH = HERE / "best_models_overview.csv"
CATALOG_PATH = HERE / "architecture_catalog.csv"


def read_csv(rel_path: str) -> pd.DataFrame:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows found._"

    table = df.copy()
    for column in table.columns:
        table[column] = table[column].map(fmt)

    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(row.astype(str)) + " |" for _, row in table.iterrows()]
    return "\n".join([header, divider, *rows])


def build_catalog() -> pd.DataFrame:
    catalog_rows = []
    task_sources = [
        ("heart_classification", "dnn_results/hp_search/heart_classification_hp_search.csv", "standard-search"),
        ("covid_classification", "dnn_results/hp_search/covid_classification_hp_search.csv", "standard-search"),
        ("temperature_regression", "dnn_results/hp_search/temperature_regression_hp_search.csv", "standard-search"),
        ("multi_output_regression", "dnn_results/hp_search/multi_output_regression_hp_search.csv", "standard-search"),
        ("weather_classification", "dnn_results/hp_search/weather_classification_hp_search.csv", "standard-search"),
        ("wind_forecasting", "dnn_results/hp_search/wind_forecasting_hp_search.csv", "standard-search"),
        ("energy_forecasting", "dnn_results/hp_search/energy_forecasting_hp_search.csv", "standard-search"),
        ("anomaly_employee", "dnn_results/hp_search/anomaly_employee_hp_search.csv", "standard-search"),
        ("anomaly_heart", "dnn_results/hp_search/anomaly_heart_hp_search.csv", "standard-search"),
        ("anomaly_wine", "dnn_results/hp_search/anomaly_wine_hp_search.csv", "standard-search"),
    ]

    for task, rel_path, family in task_sources:
        df = read_csv(rel_path)
        score_col = [c for c in df.columns if c != "config"][0]
        for _, row in df.iterrows():
            catalog_rows.append(
                {
                    "task": task,
                    "family": family,
                    "config": row["config"],
                    "score_metric": score_col,
                    "score": row[score_col],
                }
            )

    family_sources = [
        ("heart_classification", "pytorch_results/heart_pytorch_results.csv", "pytorch-standard"),
        ("heart_classification", "pytorch_results/heart_advanced_results.csv", "pytorch-advanced"),
        ("heart_classification", "pytorch_results/scratch_layers_heart_results.csv", "scratch-layers"),
        ("temperature_regression", "pytorch_results/temperature_pytorch_results.csv", "pytorch-standard"),
        ("temperature_regression", "pytorch_results/temperature_advanced_results.csv", "pytorch-advanced"),
        ("temperature_regression", "pytorch_results/scratch_layers_temperature_results.csv", "scratch-layers"),
        ("wind_forecasting", "pytorch_results/wind_pytorch_results.csv", "pytorch-standard"),
        ("wind_forecasting", "pytorch_results/scratch_layers_wind_results.csv", "scratch-layers"),
    ]

    for task, rel_path, family in family_sources:
        df = read_csv(rel_path)
        metric_col = next(c for c in df.columns if c.startswith("test_") or c in {"val_auc", "val_r2"})
        for _, row in df.iterrows():
            catalog_rows.append(
                {
                    "task": task,
                    "family": family,
                    "config": row["config"],
                    "score_metric": metric_col,
                    "score": row[metric_col],
                }
            )

    catalog = pd.DataFrame(catalog_rows)
    return catalog.sort_values(["task", "family", "score"], ascending=[True, True, False])


def build_best_models_overview() -> pd.DataFrame:
    dnn = read_csv("dnn_results/dnn_task_results.csv")
    ml = read_csv("dnn_results/ml_vs_dnn_comparison.csv")
    scratch = read_csv("pytorch_results/scratch_vs_pytorch_comparison.csv")

    merged = dnn.merge(
        ml[["task", "ml_model", "ml_value", "winner"]],
        on="task",
        how="left",
    )

    scratch_cols = scratch[["task", "scratch_best_config", "scratch_best_value", "overall_best_label", "overall_best_value"]].copy()
    merged = merged.merge(scratch_cols, on="task", how="left")

    columns = [
        "task",
        "metric",
        "best_dnn_model",
        "best_dnn_value",
        "ml_model",
        "ml_value",
        "winner",
        "scratch_best_config",
        "scratch_best_value",
        "overall_best_label",
        "overall_best_value",
        "preprocessing_match",
        "preprocessing_notes",
        "model_path",
    ]

    merged = merged.rename(
        columns={
            "dnn_model": "best_dnn_model",
            "dnn_value": "best_dnn_value",
        }
    )

    for column in columns:
        if column not in merged.columns:
            merged[column] = np.nan

    return merged[columns].sort_values("task")


def build_report() -> None:
    overview = build_best_models_overview()
    catalog = build_catalog()

    BEST_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(BEST_MODELS_PATH, index=False)
    catalog.to_csv(CATALOG_PATH, index=False)

    lines: list[str] = []
    lines.append("# Scratch Architectures Report")
    lines.append("")
    lines.append("This report combines the full DNN benchmark outputs already saved in the repository with the primitive-layer scratch runs added for the professor request.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("The project now covers two levels of from-scratch work:")
    lines.append("")
    lines.append("- Architecture-level PyTorch models: `FlexMLP`, `ResidualMLP`, explicit training loops, optimizers, schedulers, and early stopping.")
    lines.append("- Primitive-layer scratch models: `MyLinear`, `MyBatchNorm1d`, `ScratchFlexMLP`, and `ScratchResidualMLP` built directly from parameters and tensor math.")
    lines.append("")
    lines.append("The best-performing model families are summarized below, with the full architecture tables archived in CSV form for easy inspection.")
    lines.append("")
    lines.append("## Best Models by Task")
    lines.append("")
    overview_view = overview.copy()
    overview_view = overview_view[
        [
            "task",
            "metric",
            "best_dnn_model",
            "best_dnn_value",
            "ml_model",
            "ml_value",
            "winner",
            "scratch_best_config",
            "scratch_best_value",
            "preprocessing_notes",
        ]
    ]
    lines.append(markdown_table(overview_view))
    lines.append("")
    lines.append("## Full Architecture Catalog")
    lines.append("")
    lines.append("The table below is the machine-readable catalog used to build the report. It combines the full hyperparameter search tables for all tasks plus the scratch-layer runs already saved in `pytorch_results/`.")
    lines.append("")

    for task in catalog["task"].unique():
        task_df = catalog[catalog["task"] == task].copy()
        lines.append(f"### {task}")
        lines.append("")
        lines.append(markdown_table(task_df.rename(columns={"score_metric": "metric", "score": "value"})))
        lines.append("")

    lines.append("## Scratch-Layer Results")
    lines.append("")
    lines.append("### Heart")
    lines.append("")
    heart_scratch = read_csv("pytorch_results/scratch_layers_heart_results.csv")
    lines.append(markdown_table(heart_scratch))
    lines.append("")
    lines.append("### Temperature")
    lines.append("")
    temp_scratch = read_csv("pytorch_results/scratch_layers_temperature_results.csv")
    lines.append(markdown_table(temp_scratch))
    lines.append("")
    lines.append("### Wind")
    lines.append("")
    wind_scratch = read_csv("pytorch_results/scratch_layers_wind_results.csv")
    lines.append(markdown_table(wind_scratch))
    lines.append("")
    lines.append("## Scratch Comparison Against Existing PyTorch Models")
    lines.append("")
    scratch_cmp = read_csv("pytorch_results/scratch_vs_pytorch_comparison.csv")
    lines.append(markdown_table(scratch_cmp))
    lines.append("")
    lines.append("## Files to Show the Professor")
    lines.append("")
    lines.append("- `scratch_architectures/best_models_overview.csv`")
    lines.append("- `scratch_architectures/architecture_catalog.csv`")
    lines.append("- `pytorch_results/scratch_vs_pytorch_comparison.csv`")
    lines.append("- `pytorch_results/scratch_layers_heart_results.csv`")
    lines.append("- `pytorch_results/scratch_layers_temperature_results.csv`")
    lines.append("- `pytorch_results/scratch_layers_wind_results.csv`")
    lines.append("- `pytorch_results/heart_pytorch_results.csv`")
    lines.append("- `pytorch_results/heart_advanced_results.csv`")
    lines.append("- `pytorch_results/temperature_pytorch_results.csv`")
    lines.append("- `pytorch_results/temperature_advanced_results.csv`")
    lines.append("- `pytorch_results/wind_pytorch_results.csv`")
    lines.append("")
    lines.append("## Run Commands")
    lines.append("")
    lines.append("```powershell")
    lines.append("& .venv\\Scripts\\Activate.ps1")
    lines.append("& \"c:/Users/LENOVO/Desktop/Sem 8 TPs/deepl/ML tp0/.venv/Scripts/python.exe\" scratch_architectures/run_all.py --full")
    lines.append("```")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_scripts(scripts: list[str]) -> None:
    for script in scripts:
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def smoke_check() -> None:
    required = [
        ROOT / "dnn_results" / "dnn_task_results.csv",
        ROOT / "dnn_results" / "ml_vs_dnn_comparison.csv",
        ROOT / "pytorch_results" / "scratch_vs_pytorch_comparison.csv",
        ROOT / "pytorch_results" / "scratch_layers_heart_results.csv",
        ROOT / "pytorch_results" / "scratch_layers_temperature_results.csv",
        ROOT / "pytorch_results" / "scratch_layers_wind_results.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s):\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scratch architecture suite and build clean report.")
    parser.add_argument("--smoke", action="store_true", help="Validate that all required result files already exist and rebuild the report.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from saved CSV files.")
    parser.add_argument("--full", action="store_true", help="Run the benchmark scripts and then rebuild the report.")
    args = parser.parse_args()

    if args.full:
        # The repository already contains verified preprocessing for each task.
        # Re-running these scripts keeps the saved artifacts synchronized with the code.
        scripts = [
            "dnn_dashboard_benchmark.py",
            "pytorch_dnn_experiments.py",
            "pytorch_advanced_experiments.py",
            "scratch_layers_quick_experiment.py",
            "scratch_layers_other_tasks.py",
        ]
        run_scripts(scripts)
        build_report()
    else:
        smoke_check()
        build_report()

    print(f"Report written to: {REPORT_PATH}")
    print(f"Best-model overview written to: {BEST_MODELS_PATH}")
    print(f"Architecture catalog written to: {CATALOG_PATH}")


if __name__ == "__main__":
    main()