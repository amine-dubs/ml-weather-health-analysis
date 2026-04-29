# Scratch Architectures

This folder keeps the clean, show-ready report for the from-scratch PyTorch work.

Contents:

- `run_all.py` - builds the report from the saved CSV outputs and can also rerun the benchmark scripts.
- `clean_report.md` - generated report with task-by-task results and architecture tables.
- `best_models_overview.csv` - compact summary of the best model for each task.
- `architecture_catalog.csv` - machine-readable catalog of all tried architectures and scores.

Quick smoke check:

```powershell
& .venv\Scripts\Activate.ps1
& "c:/Users/LENOVO/Desktop/Sem 8 TPs/deepl/ML tp0/.venv/Scripts/python.exe" scratch_architectures/run_all.py --smoke
```

Full regeneration:

```powershell
& .venv\Scripts\Activate.ps1
& "c:/Users/LENOVO/Desktop/Sem 8 TPs/deepl/ML tp0/.venv/Scripts/python.exe" scratch_architectures/run_all.py --full
```