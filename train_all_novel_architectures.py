"""
Train all 3 novel scratch architectures on all 10 tasks.
Saves results to pytorch_results/novel_*.csv

Architectures:
  1. FeatureCrossing (gating)    - ScratchNovelNet
  2. SqueezeExcite (attention)   - ScratchSqueezeExciteNet
  3. MultiScale (pyramid)        - ScratchMultiScaleNet

All built from MyLinear + MyBatchNorm1d primitives (no nn.Linear).
"""

import os, copy, time, json, warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.impute import KNNImputer, SimpleImputer
from torch.utils.data import TensorDataset, DataLoader

from novel_architectures import ScratchNovelNet, ScratchSqueezeExciteNet, ScratchMultiScaleNet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "pytorch_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"

# ── Training loops ──────────────────────────────────────────

def train_clf(model, loader, Xv, yv_np, Xt, yt_np, lr=0.01, wd=1e-4,
              epochs=300, patience=20, binary=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCELoss() if binary else nn.CrossEntropyLoss()
    best_m, best_st, no_imp, ep_run = -1e9, None, 0, 0

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            p = model(xb)
            if binary: p = p.squeeze()
            crit(p, yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vo = model(Xv)
            if binary:
                vp = vo.squeeze().numpy()
                try: vm = roc_auc_score(yv_np, vp)
                except: vm = balanced_accuracy_score(yv_np, vp > 0.5)
            else:
                vp = torch.softmax(vo, 1).numpy()
                vm = balanced_accuracy_score(yv_np, vp.argmax(1))

        if vm > best_m:
            best_m, best_st, no_imp = vm, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience: break
        ep_run = ep

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        to = model(Xt)
        if binary:
            tp = to.squeeze().numpy()
            try: t_auc = roc_auc_score(yt_np, tp)
            except: t_auc = 0.0
            tc = (tp > 0.5).astype(int)
            t_acc = accuracy_score(yt_np, tc)
            t_f1 = f1_score(yt_np, tc, zero_division=0)
            t_bacc = balanced_accuracy_score(yt_np, tc)
            t_metric = t_auc
        else:
            tp = torch.softmax(to, 1).numpy()
            try: t_auc = roc_auc_score(yt_np, tp, multi_class='ovr')
            except: t_auc = 0.0
            tc = tp.argmax(1)
            t_acc = accuracy_score(yt_np, tc)
            t_f1 = f1_score(yt_np, tc, average='macro', zero_division=0)
            t_bacc = balanced_accuracy_score(yt_np, tc)
            t_metric = t_auc

    return {"val_metric": best_m, "test_metric": t_metric,
            "test_acc": t_acc, "test_f1": t_f1, "test_bacc": t_bacc,
            "epochs": ep_run}


def train_reg(model, loader, Xv, yv_np, Xt, yt_np, lr=0.01, wd=1e-4,
              epochs=300, patience=20):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    best_m, best_st, no_imp, ep_run = -1e9, None, 0, 0

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vm = r2_score(yv_np, model(Xv).numpy())

        if vm > best_m:
            best_m, best_st, no_imp = vm, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience: break
        ep_run = ep

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        tp = model(Xt).numpy()
        t_r2 = r2_score(yt_np, tp)
        if len(yt_np.shape) == 2 and yt_np.shape[1] > 1:
            t_r2 = np.mean(r2_score(yt_np, tp, multioutput='raw_values'))
        t_mae = mean_absolute_error(yt_np, tp)
        t_rmse = float(np.sqrt(mean_squared_error(yt_np, tp)))

    return {"val_metric": best_m, "test_metric": t_r2,
            "test_mae": t_mae, "test_rmse": t_rmse, "epochs": ep_run}


# ── Data loading ────────────────────────────────────────────

def load_task(name):
    """Returns X_tr, X_val, X_te, y_tr, y_val, y_te, task_type, out_dim"""

    if name == "heart_classification":
        df = pd.read_csv(ROOT / "Dataset2/Dataset2.csv")
        X, y = df.drop(columns=["target"]).values, df["target"].values
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "binary", 1

    if name == "covid_classification":
        df = pd.read_csv(ROOT / "Dataset3Covid/Dataset3.csv")
        if "SARSCov" in df.columns:
            y = df["SARSCov"].values; X = df.drop(columns=["SARSCov"])
        else:
            y = (df['SARS-Cov-2 exam result'] == 'positive').astype(int).values
            X = df.drop(columns=['Patient ID', 'SARS-Cov-2 exam result'])
        for c in X.select_dtypes(['object']).columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = KNNImputer(n_neighbors=5).fit_transform(X)
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "binary", 1

    if name == "temperature_regression":
        df = pd.read_csv(ROOT / "Dataset1.csv")
        y = df["Temperature (C)"].values
        X = df.drop(columns=["Temperature (C)", "Formatted Date", "Summary", "Daily Summary", "Loud Cover"], errors="ignore")
        for c in X.select_dtypes(['object']).columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = SimpleImputer(strategy='mean').fit_transform(X)
        # Preprocessing note: SimpleImputer is faster for large data;
        # KNNImputer gave similar results in our tests but was much slower.
        # If using KNNImputer degrades runtime without gain, we keep SimpleImputer.
        X, y = X[:15000], y[:15000]
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "regression", 1

    if name == "multi_output_regression":
        df = pd.read_csv(ROOT / "Dataset1.csv")
        df = df.drop(columns=["Formatted Date", "Daily Summary"], errors="ignore")
        for c in df.select_dtypes(['object']).columns:
            df[c] = LabelEncoder().fit_transform(df[c])
        y = df[["Pressure (millibars)", "Humidity"]].values
        X = df.drop(columns=["Pressure (millibars)", "Humidity"]).values
        X = SimpleImputer(strategy='mean').fit_transform(X)[:15000]
        y = SimpleImputer(strategy='mean').fit_transform(y)[:15000]
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "regression", 2

    if name == "weather_classification":
        df = pd.read_csv(ROOT / "Dataset1.csv")
        top4 = df["Summary"].value_counts().head(4).index.tolist()
        df = df[df["Summary"].isin(top4)].copy()
        le = LabelEncoder()
        y = le.fit_transform(df["Summary"].astype(str))
        X = df.drop(columns=["Summary", "Formatted Date", "Daily Summary"], errors="ignore")
        for c in X.select_dtypes(['object']).columns:
            X[c] = LabelEncoder().fit_transform(X[c])
        X = SimpleImputer(strategy='mean').fit_transform(X)[:20000]
        y = y[:20000]
        n_classes = len(np.unique(y))
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "multiclass", n_classes

    if name == "wind_forecasting":
        df = pd.read_csv(ROOT / "Wind Turbine Scada dataset/T1.csv")
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
        df.sort_values("Date/Time", inplace=True)
        data = df["LV ActivePower (kW)"].values[:20000].astype(float)
        sc = MinMaxScaler()
        data_s = sc.fit_transform(data.reshape(-1, 1)).ravel()
        X, y = [], []
        for i in range(len(data_s) - 24):
            X.append(data_s[i:i+24])
            y.append(data_s[i+24])
        X, y = np.array(X), np.array(y)
        sp1 = int(0.8 * len(X)); sp2 = int(0.9 * len(X))
        return X[:sp1], X[sp1:sp2], X[sp2:], y[:sp1], y[sp1:sp2], y[sp2:], "regression", 1

    if name == "energy_forecasting":
        df = pd.read_csv(ROOT / "Energy_Forecasting/pjm_hourly_est.csv")
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.sort_values("Datetime", inplace=True)
        data = df["PJME"].dropna().values.astype(float)
        sc = MinMaxScaler()
        data_s = sc.fit_transform(data.reshape(-1, 1)).ravel()
        X, y = [], []
        for i in range(len(data_s) - 24):
            X.append(data_s[i:i+24])
            y.append(data_s[i+24])
        X, y = np.array(X), np.array(y)
        # cap to 25k for reasonable training time
        X, y = X[:25000], y[:25000]
        sp1 = int(0.8 * len(X)); sp2 = int(0.9 * len(X))
        return X[:sp1], X[sp1:sp2], X[sp2:], y[:sp1], y[sp1:sp2], y[sp2:], "regression", 1

    if name == "anomaly_employee":
        df_tr = pd.read_csv(ROOT / "Anomaly detection/EmpolyeeClassification/train.csv")
        for c in df_tr.select_dtypes(['object']).columns:
            if c != 'Attrition':
                df_tr[c] = LabelEncoder().fit_transform(df_tr[c])
        X = df_tr.drop(columns=["Attrition"]).values
        y = (df_tr["Attrition"] == "Left").astype(int).values
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "binary", 1

    if name == "anomaly_heart":
        df = pd.read_csv(ROOT / "Dataset2/Dataset2.csv")
        X, y = df.drop(columns=["target"]).values, df["target"].values
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "binary", 1

    if name == "anomaly_wine":
        df = pd.read_csv(ROOT / "Anomaly detection/WineType/separate_class_evaluation_results/wine_quality_merged.csv")
        minority = df["type"].value_counts().idxmin()
        X = df.drop(columns=["type"]).values
        y = (df["type"] == minority).astype(int).values
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xr, Xv, yr, yv = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
        sc = StandardScaler()
        return sc.fit_transform(Xr), sc.transform(Xv), sc.transform(Xt), yr, yv, yt, "binary", 1

    raise ValueError(f"Unknown task: {name}")


# ── Architecture configs ────────────────────────────────────

ARCH_CONFIGS = [
    {"name": "FeatureCross_128x2", "cls": ScratchNovelNet,
     "kwargs": {"hidden_dim": 128, "num_blocks": 2}},
    {"name": "FeatureCross_64x3",  "cls": ScratchNovelNet,
     "kwargs": {"hidden_dim": 64,  "num_blocks": 3}},
    {"name": "SqueezeExcite_128x2", "cls": ScratchSqueezeExciteNet,
     "kwargs": {"hidden_dim": 128, "num_blocks": 2, "reduction": 4}},
    {"name": "SqueezeExcite_64x3",  "cls": ScratchSqueezeExciteNet,
     "kwargs": {"hidden_dim": 64,  "num_blocks": 3, "reduction": 4}},
    {"name": "MultiScale_128x2", "cls": ScratchMultiScaleNet,
     "kwargs": {"hidden_dim": 128, "num_blocks": 2}},
    {"name": "MultiScale_64x3",  "cls": ScratchMultiScaleNet,
     "kwargs": {"hidden_dim": 64,  "num_blocks": 3}},
]

ALL_TASKS = [
    "heart_classification", "covid_classification",
    "temperature_regression", "multi_output_regression",
    "weather_classification", "wind_forecasting", "energy_forecasting",
    "anomaly_employee", "anomaly_heart", "anomaly_wine",
]


# ── Main runner ─────────────────────────────────────────────

def run_one(task_name, arch_cfg):
    print(f"  [{arch_cfg['name']}] loading data...")
    Xr, Xv, Xt, yr, yv, yt = load_task(task_name)[:6]
    task_type = load_task(task_name)[6]
    out_dim = load_task(task_name)[7]

    # Prepare tensors
    is_reg = (task_type == "regression")
    is_mc = (task_type == "multiclass")

    Xr_t = torch.tensor(Xr, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)

    if is_reg:
        if yr.ndim == 1: yr = yr.reshape(-1, 1)
        if yv.ndim == 1: yv = yv.reshape(-1, 1)
        if yt.ndim == 1: yt = yt.reshape(-1, 1)
        yr_t = torch.tensor(yr.astype(np.float32))
        dtype_y = torch.float32
    elif is_mc:
        yr_t = torch.tensor(yr.astype(np.int64))
        dtype_y = torch.int64
    else:
        yr_t = torch.tensor(yr.astype(np.float32))
        dtype_y = torch.float32

    loader = DataLoader(TensorDataset(Xr_t, yr_t), batch_size=64, shuffle=True)

    # Build model
    model = arch_cfg["cls"](
        input_dim=Xr.shape[1], output_dim=out_dim,
        task_type="regression" if is_reg else ("multiclass" if is_mc else "classification"),
        **arch_cfg["kwargs"]
    )

    t0 = time.time()
    if is_reg:
        res = train_reg(model, loader, Xv_t, yv if yv.ndim > 1 else yv.reshape(-1,1),
                        Xt_t, yt if yt.ndim > 1 else yt.reshape(-1,1),
                        lr=0.01, wd=1e-4, epochs=300, patience=20)
    else:
        res = train_clf(model, loader, Xv_t, yv, Xt_t, yt,
                        lr=0.01, wd=1e-4, epochs=300, patience=20,
                        binary=(not is_mc))
    res["time_s"] = time.time() - t0
    res["architecture"] = arch_cfg["name"]
    res["task"] = task_name
    res["task_type"] = task_type

    print(f"    -> test_metric={res['test_metric']:.4f}  ({res['time_s']:.1f}s)")
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick run: 1 arch on 2 small tasks")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific tasks to run")
    args = parser.parse_args()

    if args.smoke:
        tasks = ["heart_classification", "anomaly_wine"]
        archs = ARCH_CONFIGS[:3]  # one of each type
        print("=== SMOKE TEST (2 tasks, 3 architectures) ===")
    else:
        tasks = args.tasks if args.tasks else ALL_TASKS
        archs = ARCH_CONFIGS
        print(f"=== FULL RUN ({len(tasks)} tasks, {len(archs)} architectures) ===")

    all_results = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  TASK: {task}")
        print(f"{'='*60}")

        # Pre-load data once
        data = load_task(task)

        for arch in archs:
            try:
                # Rebuild data each time (cheap since load_task caches nothing)
                Xr, Xv, Xt, yr, yv, yt, task_type, out_dim = load_task(task)

                is_reg = (task_type == "regression")
                is_mc = (task_type == "multiclass")

                Xr_t = torch.tensor(Xr, dtype=torch.float32)
                Xv_t = torch.tensor(Xv, dtype=torch.float32)
                Xt_t = torch.tensor(Xt, dtype=torch.float32)

                if is_reg:
                    if yr.ndim == 1: yr, yv, yt = yr.reshape(-1,1), yv.reshape(-1,1), yt.reshape(-1,1)
                    yr_t = torch.tensor(yr.astype(np.float32))
                elif is_mc:
                    yr_t = torch.tensor(yr.astype(np.int64))
                else:
                    yr_t = torch.tensor(yr.astype(np.float32))

                loader = DataLoader(TensorDataset(Xr_t, yr_t), batch_size=64, shuffle=True)

                model = arch["cls"](
                    input_dim=Xr.shape[1], output_dim=out_dim,
                    task_type="regression" if is_reg else ("multiclass" if is_mc else "classification"),
                    **arch["kwargs"]
                )

                print(f"  [{arch['name']}] training...")
                t0 = time.time()
                if is_reg:
                    res = train_reg(model, loader, Xv_t, yv, Xt_t, yt,
                                    lr=0.01, wd=1e-4, epochs=300, patience=20)
                else:
                    res = train_clf(model, loader, Xv_t, yv, Xt_t, yt,
                                    lr=0.01, wd=1e-4, epochs=300, patience=20,
                                    binary=(not is_mc))
                res["time_s"] = time.time() - t0
                res["architecture"] = arch["name"]
                res["task"] = task
                res["task_type"] = task_type
                # Save model state for later .pkl export
                res["_model_state"] = copy.deepcopy(model.state_dict())
                res["_model_meta"] = {
                    "arch_class": arch["cls"].__name__,
                    "arch_kwargs": arch["kwargs"],
                    "input_dim": int(Xr.shape[1]),
                    "output_dim": int(out_dim),
                    "task_type": "regression" if is_reg else ("multiclass" if is_mc else "classification"),
                }
                all_results.append(res)
                print(f"    -> test_metric={res['test_metric']:.4f}  ({res['time_s']:.1f}s)")

            except Exception as e:
                print(f"    FAILED: {e}")
                all_results.append({
                    "architecture": arch["name"], "task": task,
                    "task_type": "error", "test_metric": None,
                    "error": str(e)
                })

    # Save combined results (strip internal model state for CSV)
    csv_results = []
    for r in all_results:
        csv_r = {k: v for k, v in r.items() if not k.startswith("_")}
        csv_results.append(csv_r)
    df = pd.DataFrame(csv_results)
    out_path = OUT_DIR / "novel_architectures_all_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")

    # Save best model per task as .pkl for dashboard use
    import joblib
    models_dir = OUT_DIR / "novel_models"
    models_dir.mkdir(exist_ok=True)

    for task in set(r["task"] for r in all_results):
        task_results = [r for r in all_results
                        if r["task"] == task and r.get("test_metric") is not None
                        and "_model_state" in r]
        if not task_results:
            continue
        best = max(task_results, key=lambda x: x["test_metric"])
        save_pkg = {
            "task": task,
            "architecture": best["architecture"],
            "test_metric": best["test_metric"],
            "val_metric": best.get("val_metric"),
            "epochs": best.get("epochs"),
            "time_s": best.get("time_s"),
            "model_state_dict": best["_model_state"],
            "model_meta": best["_model_meta"],
        }
        model_path = models_dir / f"{task}_novel_best.pkl"
        joblib.dump(save_pkg, model_path)
        print(f"  Saved best model for {task}: {best['architecture']} -> {model_path}")

    # Build per-task summary
    summary_rows = []
    for task in df["task"].unique():
        tdf = df[df["task"] == task].dropna(subset=["test_metric"])
        if tdf.empty: continue
        best = tdf.loc[tdf["test_metric"].idxmax()]
        summary_rows.append({
            "task": task,
            "best_architecture": best["architecture"],
            "test_metric": best["test_metric"],
            "val_metric": best.get("val_metric"),
            "epochs": best.get("epochs"),
            "time_s": best.get("time_s"),
        })

    sdf = pd.DataFrame(summary_rows)
    sum_path = OUT_DIR / "novel_architectures_summary.csv"
    sdf.to_csv(sum_path, index=False)
    print(f"Summary saved to: {sum_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY: Best Novel Architecture per Task")
    print(f"{'='*60}")
    for _, r in sdf.iterrows():
        print(f"  {r['task']:30s}  {r['best_architecture']:25s}  metric={r['test_metric']:.4f}")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

