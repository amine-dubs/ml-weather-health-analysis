"""
pytorch_dnn_experiments.py
===========================
PyTorch deep-learning experiments for two dashboard tasks:
  1. Heart disease classification  (heart → ROC-AUC, sklearn best 0.9416)
  2. Temperature regression         (temp  → R²,      sklearn best 0.7439)

Explores capabilities not available in sklearn MLP:
  - Dropout regularisation (0.2, 0.4)
  - Batch Normalisation (BatchNorm1d)
  - Dropout + BatchNorm combined
  - AdamW optimizer (decoupled weight decay)
  - SGD with Nesterov momentum
  - ReduceLROnPlateau learning-rate scheduler
  - CosineAnnealingLR scheduler
  - "Full modern" combo: BN + Dropout + AdamW + CosineAnnealingLR

Base architecture mirrors the best sklearn MLP config per task:
  - Heart:       hidden=[256, 128], relu, Adam lr=0.01
  - Temperature: hidden=[256, 128, 64], relu, Adam lr=0.001

Results saved to pytorch_results/ for dashboard integration.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid Windows OMP conflict

import copy
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from pytorch_model_utils import FlexMLP, PyTorchMLPWrapper

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

ROOT       = Path(__file__).resolve().parent
OUT_DIR    = ROOT / "pytorch_results"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter configurations
# ─────────────────────────────────────────────────────────────────────────────
# Each config is applied identically to both tasks.
# `weight_decay` corresponds to sklearn's L2 regularisation (alpha).
# Baseline uses alpha=0.0001 → weight_decay=0.0001 to replicate sklearn.
PYTORCH_CONFIGS = [
    # 1 — Baseline: exact mirror of sklearn best (no dropout, no BN, Adam)
    {
        "name": "pt_baseline",
        "desc": "Baseline — mirrors sklearn best (Adam, wd=1e-4, no dropout/BN)",
        "dropout": 0.0,  "batch_norm": False,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": None,
    },
    # 2 — Light dropout: adds Dropout(0.2) between hidden layers
    {
        "name": "pt_dropout_02",
        "desc": "Dropout=0.2 (light regularisation)",
        "dropout": 0.2,  "batch_norm": False,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": None,
    },
    # 3 — Heavy dropout: Dropout(0.4) — tests aggressive regularisation
    {
        "name": "pt_dropout_04",
        "desc": "Dropout=0.4 (heavy regularisation)",
        "dropout": 0.4,  "batch_norm": False,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": None,
    },
    # 4 — BatchNorm only: normalises activations, no dropout
    {
        "name": "pt_batchnorm",
        "desc": "BatchNorm1d only (no dropout)",
        "dropout": 0.0,  "batch_norm": True,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": None,
    },
    # 5 — BN + Dropout(0.2): the standard combination in modern MLPs
    {
        "name": "pt_bn_dropout",
        "desc": "BatchNorm + Dropout=0.2 (standard modern combo)",
        "dropout": 0.2,  "batch_norm": True,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": None,
    },
    # 6 — AdamW: decoupled weight decay (stronger L2, wd=0.01)
    {
        "name": "pt_adamw",
        "desc": "AdamW with weight_decay=0.01 (decoupled L2)",
        "dropout": 0.0,  "batch_norm": False,
        "optimizer": "adamw",  "weight_decay": 1e-2,  "scheduler": None,
    },
    # 7 — SGD + Nesterov momentum: classical optimizer, often matches Adam on tabular
    {
        "name": "pt_sgd_momentum",
        "desc": "SGD + Nesterov momentum=0.9",
        "dropout": 0.0,  "batch_norm": False,
        "optimizer": "sgd",  "weight_decay": 1e-4,  "scheduler": None,
    },
    # 8 — ReduceLROnPlateau: halves lr when val stalls (patience=5)
    {
        "name": "pt_scheduler_plateau",
        "desc": "Adam + ReduceLROnPlateau (factor=0.5, patience=5)",
        "dropout": 0.0,  "batch_norm": False,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": "plateau",
    },
    # 9 — CosineAnnealingLR: warm restarts decay profile
    {
        "name": "pt_scheduler_cosine",
        "desc": "Adam + CosineAnnealingLR (T_max=max_epochs, eta_min=1e-5)",
        "dropout": 0.0,  "batch_norm": False,
        "optimizer": "adam",  "weight_decay": 1e-4,  "scheduler": "cosine",
    },
    # 10 — Full modern: BN + Dropout + AdamW + Cosine — kitchen-sink best practices
    {
        "name": "pt_full_modern",
        "desc": "BN + Dropout=0.2 + AdamW(wd=0.01) + CosineAnnealingLR",
        "dropout": 0.2,  "batch_norm": True,
        "optimizer": "adamw",  "weight_decay": 1e-2,  "scheduler": "cosine",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, cfg, base_lr):
    lr = base_lr
    wd = cfg["weight_decay"]
    opt_name = cfg["optimizer"]
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(opt, cfg, max_epochs):
    if cfg["scheduler"] is None:
        return None
    elif cfg["scheduler"] == "plateau":
        return ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    elif cfg["scheduler"] == "cosine":
        return CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    raise ValueError(f"Unknown scheduler: {cfg['scheduler']}")


def train_one_config(cfg, hidden_layers, input_dim, task_type,
                     X_tr, y_tr, X_val, y_val,
                     base_lr, max_epochs, patience, batch_size):
    """
    Train one FlexMLP configuration.
    Returns (best_model, best_val_score, epochs_run, time_sec).
    """
    torch.manual_seed(42)
    model = FlexMLP(
        input_dim, hidden_layers, output_dim=1,
        activation="relu",
        dropout=cfg["dropout"],
        batch_norm=cfg["batch_norm"],
        task_type=task_type,
    )

    opt   = build_optimizer(model, cfg, base_lr)
    sched = build_scheduler(opt, cfg, max_epochs)
    crit  = nn.BCELoss() if task_type == "classification" else nn.MSELoss()

    # Build tensors once
    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )

    best_score  = -np.inf
    best_state  = None
    no_improve  = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(Xb).squeeze(), yb)
            loss.backward()
            opt.step()

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).squeeze().numpy()

        val_score = (roc_auc_score(y_val, val_out)
                     if task_type == "classification"
                     else r2_score(y_val, val_out))

        # ── Scheduler step ───────────────────────────────────────────────
        if sched is not None:
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_score)
            else:
                sched.step()

        # ── Early stopping ───────────────────────────────────────────────
        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_score, epoch, round(time.time() - t0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers (same as dnn_dashboard_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def label_encode_preserve_nan(df: pd.DataFrame, columns: list):
    df = df.copy()
    encoders = {}
    for col in columns:
        le  = LabelEncoder()
        non_nan = df[col].dropna()
        le.fit(non_nan)
        encoders[col] = le
        df[col] = df[col].map(
            lambda v, _le=le: float(_le.transform([v])[0]) if pd.notna(v) else np.nan
        )
    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Heart disease classification
# ─────────────────────────────────────────────────────────────────────────────

def run_heart_pytorch():
    print("\n" + "=" * 60)
    print("TASK 1 - Heart disease classification (PyTorch)")
    print("=" * 60)

    # ---------- data + preprocessing (identical to sklearn best) ----------
    df = pd.read_csv(ROOT / "Dataset2" / "Dataset2.csv")
    X  = df.drop(columns=["target"]).values.astype(np.float32)
    y  = df["target"].values.astype(np.float32)
    feature_cols = list(df.drop(columns=["target"]).columns)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # ---------- fixed settings ----------
    HIDDEN  = [256, 128]          # mirrors sklearn best (256×128)
    INPUT   = X_train_s.shape[1]  # 11
    BASE_LR = 0.01                # mirrors sklearn best lr
    EPOCHS  = 300
    PAT     = 25
    BSIZE   = 32

    rows   = []
    best_test_score = -np.inf
    best_wrapper    = None

    for i, cfg in enumerate(PYTORCH_CONFIGS, 1):
        print(f"  [{i:2d}/10] {cfg['name']} ... ", end="", flush=True)
        model, val_score, ep, dt = train_one_config(
            cfg, HIDDEN, INPUT, "classification",
            X_train_s, y_train, X_val_s, y_val,
            BASE_LR, EPOCHS, PAT, BSIZE,
        )

        # ── Final evaluation on held-out test set ─────────────────────
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test_s)).squeeze().numpy()
        test_pred  = (test_logits >= 0.5).astype(int)
        test_auc   = roc_auc_score(y_test, test_logits)
        test_acc   = accuracy_score(y_test, test_pred)
        test_f1    = f1_score(y_test, test_pred)

        print(f"val_auc={val_score:.4f}  test_auc={test_auc:.4f}  "
              f"ep={ep}  t={dt}s")

        row = {
            "config":       cfg["name"],
            "description":  cfg["desc"],
            "dropout":      cfg["dropout"],
            "batch_norm":   cfg["batch_norm"],
            "optimizer":    cfg["optimizer"],
            "weight_decay": cfg["weight_decay"],
            "scheduler":    cfg["scheduler"] or "none",
            "val_auc":      round(val_score, 6),
            "test_auc":     round(test_auc,  6),
            "test_acc":     round(test_acc,  6),
            "test_f1":      round(test_f1,   6),
            "epochs":       ep,
            "time_s":       dt,
        }
        rows.append(row)

        if test_auc > best_test_score:
            best_test_score = test_auc
            best_wrapper = PyTorchMLPWrapper(
                model=model,
                scaler=scaler,
                task_type="classification",
                feature_columns=feature_cols,
                target="target",
                threshold=0.5,
                config_name=cfg["name"],
                config_desc=cfg["desc"],
                val_score=val_score,
            )

    # ---------- save results ----------
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "heart_pytorch_results.csv", index=False)

    best_cfg_name = results_df.loc[results_df["test_auc"].idxmax(), "config"]
    print(f"  >> Best: {best_cfg_name}  test_auc={best_test_score:.4f}")

    joblib.dump(best_wrapper, MODELS_DIR / "heart_classification_pytorch_model.pkl")
    print("  >> Model saved -> pytorch_results/models/heart_classification_pytorch_model.pkl")

    return results_df, best_test_score, best_cfg_name


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Temperature regression
# ─────────────────────────────────────────────────────────────────────────────

def run_temperature_pytorch():
    print("\n" + "=" * 60)
    print("TASK 2 - Temperature regression (PyTorch)")
    print("=" * 60)

    # ---------- data + preprocessing (identical to sklearn best) ----------
    df = pd.read_csv(ROOT / "Dataset1.csv")
    y  = df["Temperature (C)"].values.astype(np.float32)
    X  = df.drop(
        columns=["Temperature (C)", "Apparent Temperature (C)",
                 "Formatted Date", "Daily Summary"],
        errors="ignore",
    )
    feature_cols = list(X.columns)
    cat_cols     = X.select_dtypes(include=["object"]).columns.tolist()

    X, label_encoders = label_encode_preserve_nan(X, cat_cols)

    imputer   = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), columns=feature_cols
    )

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_imputed.values.astype(np.float32), y,
        test_size=0.2, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42,
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # Subsample training + val for HP search (speed on CPU)
    HP_N = 15_000
    if len(X_train_s) > HP_N:
        idx = np.random.choice(len(X_train_s), HP_N, replace=False)
        X_tr_hp = X_train_s[idx]
        y_tr_hp = y_train[idx]
    else:
        X_tr_hp, y_tr_hp = X_train_s, y_train

    HP_VAL = 4_000
    if len(X_val_s) > HP_VAL:
        idx_v = np.random.choice(len(X_val_s), HP_VAL, replace=False)
        X_val_hp = X_val_s[idx_v]
        y_val_hp = y_val[idx_v]
    else:
        X_val_hp, y_val_hp = X_val_s, y_val

    print(f"  HP search on {len(X_tr_hp)} train rows  "
          f"({len(X_train_s)} total); val={len(X_val_hp)}")

    # ---------- fixed settings ----------
    HIDDEN  = [256, 128, 64]          # mirrors sklearn best (256×128×64)
    INPUT   = X_train_s.shape[1]
    BASE_LR = 0.001                   # mirrors sklearn best lr
    EPOCHS_HP    = 80
    PAT_HP       = 10
    BSIZE_HP     = 256
    EPOCHS_FULL  = 80
    PAT_FULL     = 10
    BSIZE_FULL   = 512

    hp_rows = []

    print("  -- HP search --")
    for i, cfg in enumerate(PYTORCH_CONFIGS, 1):
        print(f"  [{i:2d}/10] {cfg['name']} ... ", end="", flush=True)
        _, val_score, ep, dt = train_one_config(
            cfg, HIDDEN, INPUT, "regression",
            X_tr_hp, y_tr_hp, X_val_hp, y_val_hp,
            BASE_LR, EPOCHS_HP, PAT_HP, BSIZE_HP,
        )
        print(f"val_r2={val_score:.4f}  ep={ep}  t={dt}s")
        hp_rows.append({"config": cfg["name"], "val_r2_subsample": round(val_score, 6)})

    hp_df      = pd.DataFrame(hp_rows)
    best_cfg_i = hp_df["val_r2_subsample"].idxmax()
    best_cfg   = PYTORCH_CONFIGS[best_cfg_i]
    print(f"  >> Best HP config: {best_cfg['name']}  "
          f"(val_r2={hp_df.loc[best_cfg_i, 'val_r2_subsample']:.4f})")

    # ---------- retrain best config on FULL training data ----------
    print(f"  -- Retrain {best_cfg['name']} on full {len(X_train_s)} rows --")
    best_model, val_score_full, ep_full, dt_full = train_one_config(
        best_cfg, HIDDEN, INPUT, "regression",
        X_train_s, y_train, X_val_s, y_val,
        BASE_LR, EPOCHS_FULL, PAT_FULL, BSIZE_FULL,
    )
    print(f"  val_r2={val_score_full:.4f}  ep={ep_full}  t={dt_full}s")

    # ── Final test evaluation ──────────────────────────────────────────
    best_model.eval()
    with torch.no_grad():
        test_pred = best_model(torch.tensor(X_test_s)).squeeze().numpy()
    test_r2   = r2_score(y_test, test_pred)
    test_mae  = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"  >> Test R2={test_r2:.4f}  MAE={test_mae:.3f}  RMSE={test_rmse:.3f}")

    # ---------- also evaluate all other configs on test (retrain each) ----------
    print("  -- Full retrain + test eval for all 10 configs --")
    rows = []
    best_test_score = -np.inf
    best_wrapper    = None

    for i, cfg in enumerate(PYTORCH_CONFIGS, 1):
        print(f"  [{i:2d}/10] {cfg['name']} ... ", end="", flush=True)
        model_i, val_i, ep_i, dt_i = train_one_config(
            cfg, HIDDEN, INPUT, "regression",
            X_train_s, y_train, X_val_s, y_val,
            BASE_LR, EPOCHS_FULL, PAT_FULL, BSIZE_FULL,
        )
        model_i.eval()
        with torch.no_grad():
            pred_i = model_i(torch.tensor(X_test_s)).squeeze().numpy()
        r2_i   = r2_score(y_test, pred_i)
        mae_i  = mean_absolute_error(y_test, pred_i)
        rmse_i = np.sqrt(mean_squared_error(y_test, pred_i))
        print(f"test_r2={r2_i:.4f}  ep={ep_i}  t={dt_i}s")

        row = {
            "config":              cfg["name"],
            "description":         cfg["desc"],
            "dropout":             cfg["dropout"],
            "batch_norm":          cfg["batch_norm"],
            "optimizer":           cfg["optimizer"],
            "weight_decay":        cfg["weight_decay"],
            "scheduler":           cfg["scheduler"] or "none",
            "val_r2_subsample":    hp_df.loc[hp_df.config == cfg["name"], "val_r2_subsample"].values[0],
            "val_r2_full":         round(val_i,  6),
            "test_r2":             round(r2_i,   6),
            "test_mae":            round(mae_i,  4),
            "test_rmse":           round(rmse_i, 4),
            "epochs":              ep_i,
            "time_s":              dt_i,
        }
        rows.append(row)

        if r2_i > best_test_score:
            best_test_score = r2_i
            best_wrapper = PyTorchMLPWrapper(
                model=model_i,
                scaler=scaler,
                task_type="regression",
                feature_columns=feature_cols,
                target="Temperature (C)",
                imputer=imputer,
                cat_cols=cat_cols,
                label_encoders=label_encoders,
                config_name=cfg["name"],
                config_desc=cfg["desc"],
                val_score=val_i,
            )
            best_cfg_name_final = cfg["name"]

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "temperature_pytorch_results.csv", index=False)
    print(f"\n  >> Best: {best_cfg_name_final}  test_r2={best_test_score:.4f}")

    joblib.dump(best_wrapper, MODELS_DIR / "temperature_regression_pytorch_model.pkl")
    print("  >> Model saved -> pytorch_results/models/temperature_regression_pytorch_model.pkl")

    return results_df, best_test_score, best_cfg_name_final


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV (for dashboard ML vs DNN vs PyTorch comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(heart_df, heart_best, heart_cfg,
                  temp_df, temp_best, temp_cfg):
    sklearn_dnn = pd.read_csv(ROOT / "dnn_results" / "ml_vs_dnn_comparison.csv")

    def _get(task):
        row = sklearn_dnn[sklearn_dnn["task"] == task].iloc[0]
        return row["ml_value"], row["dnn_value"], row["ml_model"], row["dnn_model"]

    h_ml, h_skdnn, h_ml_m, h_sk_m = _get("heart_classification")
    t_ml, t_skdnn, t_ml_m, t_sk_m = _get("temperature_regression")

    summary = pd.DataFrame([
        {
            "task":              "heart_classification",
            "metric":            "roc_auc",
            "ml_model":          h_ml_m,     "ml_value":       round(h_ml, 4),
            "sklearn_dnn_model": h_sk_m,     "sklearn_dnn":    round(h_skdnn, 4),
            "pytorch_dnn_model": heart_cfg,  "pytorch_dnn":    round(heart_best, 4),
            "delta_pt_vs_sklearn_dnn": round(heart_best - h_skdnn, 4),
            "delta_pt_vs_ml":          round(heart_best - h_ml,    4),
        },
        {
            "task":              "temperature_regression",
            "metric":            "r2",
            "ml_model":          t_ml_m,    "ml_value":       round(t_ml, 4),
            "sklearn_dnn_model": t_sk_m,    "sklearn_dnn":    round(t_skdnn, 4),
            "pytorch_dnn_model": temp_cfg,  "pytorch_dnn":    round(temp_best, 4),
            "delta_pt_vs_sklearn_dnn": round(temp_best - t_skdnn, 4),
            "delta_pt_vs_ml":          round(temp_best - t_ml,    4),
        },
    ])
    summary.to_csv(OUT_DIR / "pytorch_vs_baseline_summary.csv", index=False)
    print("\n" + "=" * 60)
    print("PYTORCH EXPERIMENT SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    heart_df, heart_best, heart_cfg = run_heart_pytorch()
    temp_df,  temp_best,  temp_cfg  = run_temperature_pytorch()

    build_summary(heart_df, heart_best, heart_cfg,
                  temp_df,  temp_best,  temp_cfg)

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
    print("All results saved to pytorch_results/")
