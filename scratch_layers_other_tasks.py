"""
Run scratch-layer experiments for temperature regression and wind forecasting,
and generate comparison tables/plots against existing PyTorch results.
"""

from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pytorch_scratch_layers import ScratchFlexMLP, ScratchResidualMLP


torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "pytorch_results"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def label_encode_preserve_nan(df: pd.DataFrame, columns: list[str]):
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in columns:
        le = LabelEncoder()
        non_nan = df[col].dropna()
        le.fit(non_nan)
        encoders[col] = le
        df[col] = df[col].map(
            lambda v, _le=le: float(_le.transform([v])[0]) if pd.notna(v) else np.nan
        )
    return df, encoders


def train_regression(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, num_workers=0
    )

    best_score = -np.inf
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).squeeze().numpy()
        val_r2 = r2_score(y_val, val_pred)

        if val_r2 > best_score:
            best_score = val_r2
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_score, epoch, round(time.time() - t0, 1)


def run_temperature_scratch() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "Dataset1.csv")
    y = df["Temperature (C)"].values.astype(np.float32)
    X = df.drop(
        columns=[
            "Temperature (C)",
            "Apparent Temperature (C)",
            "Formatted Date",
            "Daily Summary",
        ],
        errors="ignore",
    )

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X, _ = label_encode_preserve_nan(X, cat_cols)

    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_imp.values.astype(np.float32), y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Keep runtime practical while preserving consistent held-out test evaluation.
    hp_n, hp_val = 18000, 4000
    if len(X_train_s) > hp_n:
        idx = np.random.choice(len(X_train_s), hp_n, replace=False)
        X_tr_use, y_tr_use = X_train_s[idx], y_train[idx]
    else:
        X_tr_use, y_tr_use = X_train_s, y_train
    if len(X_val_s) > hp_val:
        idxv = np.random.choice(len(X_val_s), hp_val, replace=False)
        X_val_use, y_val_use = X_val_s[idxv], y_val[idxv]
    else:
        X_val_use, y_val_use = X_val_s, y_val

    input_dim = X_train_s.shape[1]

    configs = [
        {
            "name": "scratch_temp_flex_128_64",
            "desc": "ScratchFlexMLP hidden=[128,64], dropout=0.1, BN",
            "builder": lambda: ScratchFlexMLP(
                input_dim=input_dim,
                hidden_layers=[128, 64],
                dropout=0.1,
                batch_norm=True,
                task_type="regression",
            ),
            "lr": 0.001,
            "wd": 1e-3,
            "batch_size": 256,
            "epochs": 70,
            "patience": 10,
        },
        {
            "name": "scratch_temp_flex_256_128_64",
            "desc": "ScratchFlexMLP hidden=[256,128,64], dropout=0.0",
            "builder": lambda: ScratchFlexMLP(
                input_dim=input_dim,
                hidden_layers=[256, 128, 64],
                dropout=0.0,
                batch_norm=False,
                task_type="regression",
            ),
            "lr": 0.001,
            "wd": 1e-4,
            "batch_size": 256,
            "epochs": 80,
            "patience": 10,
        },
        {
            "name": "scratch_temp_residual_128x2",
            "desc": "ScratchResidualMLP hidden=128, blocks=2, dropout=0.1",
            "builder": lambda: ScratchResidualMLP(
                input_dim=input_dim,
                hidden_dim=128,
                n_blocks=2,
                dropout=0.1,
                task_type="regression",
            ),
            "lr": 0.001,
            "wd": 1e-3,
            "batch_size": 256,
            "epochs": 80,
            "patience": 10,
        },
    ]

    rows = []
    for cfg in configs:
        print(f"[Temperature] Running {cfg['name']} ...", flush=True)
        model = cfg["builder"]()
        model, val_r2, epochs_run, time_s = train_regression(
            model,
            X_tr_use,
            y_tr_use,
            X_val_use,
            y_val_use,
            lr=cfg["lr"],
            weight_decay=cfg["wd"],
            batch_size=cfg["batch_size"],
            max_epochs=cfg["epochs"],
            patience=cfg["patience"],
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test_s, dtype=torch.float32)).squeeze().numpy()

        r2_ = r2_score(y_test, pred)
        mae_ = mean_absolute_error(y_test, pred)
        rmse_ = np.sqrt(mean_squared_error(y_test, pred))

        row = {
            "config": cfg["name"],
            "description": cfg["desc"],
            "lr": cfg["lr"],
            "weight_decay": cfg["wd"],
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["epochs"],
            "patience": cfg["patience"],
            "val_r2": round(float(val_r2), 6),
            "test_r2": round(float(r2_), 6),
            "test_mae": round(float(mae_), 4),
            "test_rmse": round(float(rmse_), 4),
            "epochs_run": int(epochs_run),
            "time_s": float(time_s),
        }
        rows.append(row)
        print(
            f"  val_r2={row['val_r2']:.4f} test_r2={row['test_r2']:.4f} "
            f"mae={row['test_mae']:.4f} rmse={row['test_rmse']:.4f}"
        )

    out_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    out_df.to_csv(OUT_DIR / "scratch_layers_temperature_results.csv", index=False)
    return out_df


def run_wind_scratch() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "Wind Turbine Scada dataset" / "T1.csv")
    target_col = "LV ActivePower (kW)"
    data = df[target_col].dropna().values.reshape(-1, 1).astype(np.float64)

    scaler = MinMaxScaler()
    data_s = scaler.fit_transform(data)

    window = 24
    X_seq, y_seq = [], []
    for i in range(len(data_s) - window):
        X_seq.append(data_s[i : i + window].flatten())
        y_seq.append(data_s[i + window, 0])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    input_dim = X_tr.shape[1]

    configs = [
        {
            "name": "scratch_wind_flex_256_128",
            "desc": "ScratchFlexMLP hidden=[256,128], mirrors baseline width",
            "builder": lambda: ScratchFlexMLP(
                input_dim=input_dim,
                hidden_layers=[256, 128],
                dropout=0.0,
                batch_norm=False,
                task_type="regression",
            ),
            "lr": 0.01,
            "wd": 1e-4,
            "batch_size": 512,
            "epochs": 90,
            "patience": 12,
        },
        {
            "name": "scratch_wind_residual_128x2",
            "desc": "ScratchResidualMLP hidden=128, blocks=2, dropout=0.1",
            "builder": lambda: ScratchResidualMLP(
                input_dim=input_dim,
                hidden_dim=128,
                n_blocks=2,
                dropout=0.1,
                task_type="regression",
            ),
            "lr": 0.005,
            "wd": 1e-3,
            "batch_size": 512,
            "epochs": 90,
            "patience": 12,
        },
    ]

    rows = []
    for cfg in configs:
        print(f"[Wind] Running {cfg['name']} ...", flush=True)
        model = cfg["builder"]()
        model, val_r2, epochs_run, time_s = train_regression(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            lr=cfg["lr"],
            weight_decay=cfg["wd"],
            batch_size=cfg["batch_size"],
            max_epochs=cfg["epochs"],
            patience=cfg["patience"],
        )

        model.eval()
        with torch.no_grad():
            pred_s = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        r2_ = r2_score(y_true, y_pred)
        mae_ = mean_absolute_error(y_true, y_pred)
        rmse_ = np.sqrt(mean_squared_error(y_true, y_pred))

        row = {
            "config": cfg["name"],
            "description": cfg["desc"],
            "lr": cfg["lr"],
            "weight_decay": cfg["wd"],
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["epochs"],
            "patience": cfg["patience"],
            "val_r2": round(float(val_r2), 6),
            "test_r2": round(float(r2_), 6),
            "test_mae": round(float(mae_), 4),
            "test_rmse": round(float(rmse_), 4),
            "epochs_run": int(epochs_run),
            "time_s": float(time_s),
        }
        rows.append(row)
        print(
            f"  val_r2={row['val_r2']:.4f} test_r2={row['test_r2']:.4f} "
            f"mae={row['test_mae']:.4f} rmse={row['test_rmse']:.4f}"
        )

    out_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    out_df.to_csv(OUT_DIR / "scratch_layers_wind_results.csv", index=False)
    return out_df


def _get_row(df: pd.DataFrame, config_name: str) -> pd.Series | None:
    match = df[df["config"] == config_name]
    if match.empty:
        return None
    return match.iloc[0]


def build_comparison_and_plots(temp_df: pd.DataFrame, wind_df: pd.DataFrame):
    heart_std = pd.read_csv(OUT_DIR / "heart_pytorch_results.csv")
    temp_std = pd.read_csv(OUT_DIR / "temperature_pytorch_results.csv")
    wind_std = pd.read_csv(OUT_DIR / "wind_pytorch_results.csv")

    heart_adv = pd.read_csv(OUT_DIR / "heart_advanced_results.csv")
    temp_adv = pd.read_csv(OUT_DIR / "temperature_advanced_results.csv")

    heart_scratch = pd.read_csv(OUT_DIR / "scratch_layers_heart_results.csv")

    heart_pt_adamw = _get_row(heart_std, "pt_adamw")
    temp_pt_adamw = _get_row(temp_std, "pt_adamw")
    wind_pt_adamw = _get_row(wind_std, "pt_adamw")

    heart_adv_res = _get_row(heart_adv, "adv_residual_wide")
    temp_adv_res = _get_row(temp_adv, "adv_residual_wide")

    heart_s_best = heart_scratch.sort_values("test_auc", ascending=False).iloc[0]
    temp_s_best = temp_df.sort_values("test_r2", ascending=False).iloc[0]
    wind_s_best = wind_df.sort_values("test_r2", ascending=False).iloc[0]

    wind_best_std = wind_std.sort_values("test_r2", ascending=False).iloc[0]

    rows = [
        {
            "task": "heart_classification",
            "metric": "test_auc",
            "pt_adamw_config": "pt_adamw",
            "pt_adamw_value": float(heart_pt_adamw["test_auc"]),
            "adv_residual_config": "adv_residual_wide",
            "adv_residual_value": float(heart_adv_res["test_auc"]),
            "scratch_best_config": heart_s_best["config"],
            "scratch_best_value": float(heart_s_best["test_auc"]),
            "overall_best_label": "adv_residual_wide",
            "overall_best_value": float(heart_adv_res["test_auc"]),
        },
        {
            "task": "temperature_regression",
            "metric": "test_r2",
            "pt_adamw_config": "pt_adamw",
            "pt_adamw_value": float(temp_pt_adamw["test_r2"]),
            "adv_residual_config": "adv_residual_wide",
            "adv_residual_value": float(temp_adv_res["test_r2"]),
            "scratch_best_config": temp_s_best["config"],
            "scratch_best_value": float(temp_s_best["test_r2"]),
            "overall_best_label": "adv_residual_wide",
            "overall_best_value": float(temp_adv_res["test_r2"]),
        },
        {
            "task": "wind_forecasting",
            "metric": "test_r2",
            "pt_adamw_config": "pt_adamw",
            "pt_adamw_value": float(wind_pt_adamw["test_r2"]),
            "adv_residual_config": "n/a",
            "adv_residual_value": np.nan,
            "scratch_best_config": wind_s_best["config"],
            "scratch_best_value": float(wind_s_best["test_r2"]),
            "overall_best_label": str(wind_best_std["config"]),
            "overall_best_value": float(wind_best_std["test_r2"]),
        },
    ]

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_DIR / "scratch_vs_pytorch_comparison.csv", index=False)

    # Plot 1: grouped bars by task with fixed variants.
    tasks = ["Heart (AUC)", "Temperature (R2)", "Wind (R2)"]
    pt_vals = [rows[0]["pt_adamw_value"], rows[1]["pt_adamw_value"], rows[2]["pt_adamw_value"]]
    adv_vals = [rows[0]["adv_residual_value"], rows[1]["adv_residual_value"], np.nan]
    scratch_vals = [rows[0]["scratch_best_value"], rows[1]["scratch_best_value"], rows[2]["scratch_best_value"]]
    best_vals = [rows[0]["overall_best_value"], rows[1]["overall_best_value"], rows[2]["overall_best_value"]]

    x = np.arange(len(tasks))
    w = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * w, pt_vals, width=w, label="pt_adamw", color="#4e79a7")
    ax.bar(x - 0.5 * w, adv_vals, width=w, label="adv_residual_wide", color="#f28e2b")
    ax.bar(x + 0.5 * w, scratch_vals, width=w, label="scratch_best", color="#59a14f")
    ax.bar(x + 1.5 * w, best_vals, width=w, label="overall_best", color="#e15759", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Main metric value")
    ax.set_title("PyTorch Config Comparison: pt_adamw vs adv_residual_wide vs scratch")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for i, v in enumerate(pt_vals):
        ax.text(x[i] - 1.5 * w, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(adv_vals):
        if not np.isnan(v):
            ax.text(x[i] - 0.5 * w, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(scratch_vals):
        ax.text(x[i] + 0.5 * w, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scratch_vs_pytorch_main_metric.png", dpi=180)
    plt.close(fig)

    # Plot 2: scratch-only detail for the two newly-run tasks.
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    temp_plot = temp_df.sort_values("test_r2", ascending=False)
    ax1.bar(temp_plot["config"], temp_plot["test_r2"], color="#76b7b2")
    ax1.set_title("Temperature scratch models (test R2)")
    ax1.set_ylabel("R2")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.25)

    wind_plot = wind_df.sort_values("test_r2", ascending=False)
    ax2.bar(wind_plot["config"], wind_plot["test_r2"], color="#edc948")
    ax2.set_title("Wind scratch models (test R2)")
    ax2.set_ylabel("R2")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.25)

    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "scratch_other_tasks_detail.png", dpi=180)
    plt.close(fig2)


def main():
    temp_df = run_temperature_scratch()
    wind_df = run_wind_scratch()
    build_comparison_and_plots(temp_df, wind_df)

    print("\nSaved files:")
    print(f"- {OUT_DIR / 'scratch_layers_temperature_results.csv'}")
    print(f"- {OUT_DIR / 'scratch_layers_wind_results.csv'}")
    print(f"- {OUT_DIR / 'scratch_vs_pytorch_comparison.csv'}")
    print(f"- {PLOTS_DIR / 'scratch_vs_pytorch_main_metric.png'}")
    print(f"- {PLOTS_DIR / 'scratch_other_tasks_detail.png'}")


if __name__ == "__main__":
    main()
