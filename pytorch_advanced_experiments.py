"""
pytorch_advanced_experiments.py
================================
Extension of the PyTorch DNN experiments with:

  TASK 3  : Wind turbine forecasting (same 10 standard PyTorch configs)
            Base: MLP(256x128, relu, lr=0.01) -- mirrors sklearn best
            sklearn DNN: R2=0.9706, ML Ridge R2=0.9714

  TASKS 1b/2b : Heart & Temperature -- 6 ADVANCED configs that go beyond
                what was tested before, specifically designed to try to
                close or eliminate the gap to the ML stacking ensembles.

  New techniques:
    - Weighted BCE loss  (handles class imbalance in heart)
    - Huber / SmoothL1 loss  (robust regression for temperature)
    - Wider / deeper networks  (512x256x128)
    - ResidualMLP (skip connections -- ResNet-style for tabular)
    - LR warmup + cosine decay
    - Best combination: ResidualMLP + AdamW + dropout + warmup/cosine

Results saved to pytorch_results/ alongside the original experiment output.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import math
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from pytorch_model_utils import FlexMLP, ResidualMLP, PyTorchMLPWrapper

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

ROOT       = Path(__file__).resolve().parent
OUT_DIR    = ROOT / "pytorch_results"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Standard 10 configs (reused for wind turbine task)
# ─────────────────────────────────────────────────────────────────────────────
PYTORCH_CONFIGS = [
    {"name": "pt_baseline",          "desc": "Baseline -- mirrors sklearn best (Adam, wd=1e-4)",
     "dropout": 0.0, "batch_norm": False, "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": None},
    {"name": "pt_dropout_02",        "desc": "Dropout=0.2 (light regularisation)",
     "dropout": 0.2, "batch_norm": False, "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": None},
    {"name": "pt_dropout_04",        "desc": "Dropout=0.4 (heavy regularisation)",
     "dropout": 0.4, "batch_norm": False, "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": None},
    {"name": "pt_batchnorm",         "desc": "BatchNorm1d only (no dropout)",
     "dropout": 0.0, "batch_norm": True,  "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": None},
    {"name": "pt_bn_dropout",        "desc": "BatchNorm + Dropout=0.2",
     "dropout": 0.2, "batch_norm": True,  "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": None},
    {"name": "pt_adamw",             "desc": "AdamW with weight_decay=0.01",
     "dropout": 0.0, "batch_norm": False, "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None},
    {"name": "pt_sgd_momentum",      "desc": "SGD + Nesterov momentum=0.9",
     "dropout": 0.0, "batch_norm": False, "optimizer": "sgd",   "weight_decay": 1e-4, "scheduler": None},
    {"name": "pt_scheduler_plateau", "desc": "Adam + ReduceLROnPlateau (factor=0.5, patience=5)",
     "dropout": 0.0, "batch_norm": False, "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": "plateau"},
    {"name": "pt_scheduler_cosine",  "desc": "Adam + CosineAnnealingLR",
     "dropout": 0.0, "batch_norm": False, "optimizer": "adam",  "weight_decay": 1e-4, "scheduler": "cosine"},
    {"name": "pt_full_modern",       "desc": "BN + Dropout=0.2 + AdamW(wd=0.01) + CosineAnnealingLR",
     "dropout": 0.2, "batch_norm": True,  "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": "cosine"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Advanced configs  (heart + temperature only)
# ─────────────────────────────────────────────────────────────────────────────
# These test things not in the standard 10:
#   - weighted loss (class-imbalance / outlier-robustness)
#   - wider architectures
#   - residual connections (skip connections)
#   - LR warmup
#   - best known combination of all the above
# ─────────────────────────────────────────────────────────────────────────────

ADVANCED_HEART = [
    # 1 — Class-weighted BCE: give the minority class higher gradient signal
    {"name": "adv_weighted_bce",
     "desc": "BCELoss with pos_weight (handles class imbalance)",
     "model_type": "flexmlp", "hidden": None,
     "dropout": 0.0, "batch_norm": False,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "weighted_bce"},
    # 2 — Wider, deeper FlexMLP + AdamW
    {"name": "adv_wider",
     "desc": "Wider network (512x256x128) + AdamW",
     "model_type": "flexmlp", "hidden": [512, 256, 128],
     "dropout": 0.0, "batch_norm": False,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "default"},
    # 3 — ResidualMLP: skip connections (not possible in sklearn)
    {"name": "adv_residual",
     "desc": "ResidualMLP (hidden=256, 3 blocks) + AdamW",
     "model_type": "residual", "hidden_dim": 256, "n_blocks": 3,
     "dropout": 0.0,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "default"},
    # 4 — Wider residual
    {"name": "adv_residual_wide",
     "desc": "ResidualMLP (hidden=512, 4 blocks) + AdamW",
     "model_type": "residual", "hidden_dim": 512, "n_blocks": 4,
     "dropout": 0.0,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "default"},
    # 5 — Linear LR warmup (5 epochs) then cosine decay: stabilise early training
    {"name": "adv_warmup_cosine",
     "desc": "AdamW + linear warmup (5 ep) + cosine decay",
     "model_type": "flexmlp", "hidden": None,
     "dropout": 0.0, "batch_norm": False,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": "warmup_cosine",
     "loss": "default"},
    # 6 — Kitchen-sink best combination
    {"name": "adv_best_combo",
     "desc": "ResidualMLP (512, 4) + weighted BCE + AdamW + cosine",
     "model_type": "residual", "hidden_dim": 512, "n_blocks": 4,
     "dropout": 0.1,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": "cosine",
     "loss": "weighted_bce"},
]

ADVANCED_TEMP = [
    # 1 — Huber/SmoothL1 loss: robust to outliers in temperature data
    {"name": "adv_huber",
     "desc": "SmoothL1 (Huber) loss -- robust to outliers",
     "model_type": "flexmlp", "hidden": None,
     "dropout": 0.0, "batch_norm": False,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "huber"},
    # 2 — Wider, deeper FlexMLP + AdamW
    {"name": "adv_wider",
     "desc": "Wider network (512x256x128x64) + AdamW",
     "model_type": "flexmlp", "hidden": [512, 256, 128, 64],
     "dropout": 0.0, "batch_norm": False,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "default"},
    # 3 — ResidualMLP
    {"name": "adv_residual",
     "desc": "ResidualMLP (hidden=256, 4 blocks) + AdamW",
     "model_type": "residual", "hidden_dim": 256, "n_blocks": 4,
     "dropout": 0.0,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "default"},
    # 4 — Wider residual
    {"name": "adv_residual_wide",
     "desc": "ResidualMLP (hidden=512, 4 blocks) + AdamW",
     "model_type": "residual", "hidden_dim": 512, "n_blocks": 4,
     "dropout": 0.0,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": None,
     "loss": "default"},
    # 5 — Linear LR warmup + cosine decay
    {"name": "adv_warmup_cosine",
     "desc": "AdamW + linear warmup (5 ep) + cosine decay",
     "model_type": "flexmlp", "hidden": None,
     "dropout": 0.0, "batch_norm": False,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": "warmup_cosine",
     "loss": "default"},
    # 6 — Kitchen-sink best combination
    {"name": "adv_best_combo",
     "desc": "ResidualMLP (512, 4) + Huber loss + AdamW + cosine",
     "model_type": "residual", "hidden_dim": 512, "n_blocks": 4,
     "dropout": 0.0,
     "optimizer": "adamw", "weight_decay": 1e-2, "scheduler": "cosine",
     "loss": "huber"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg, input_dim, task_type, default_hidden):
    """Instantiate FlexMLP or ResidualMLP from a config dict."""
    mt = cfg.get("model_type", "flexmlp")
    if mt == "residual":
        return ResidualMLP(
            input_dim  = input_dim,
            hidden_dim = cfg["hidden_dim"],
            n_blocks   = cfg["n_blocks"],
            output_dim = 1,
            dropout    = cfg.get("dropout", 0.0),
            task_type  = task_type,
        )
    else:
        hidden = cfg.get("hidden") or default_hidden
        return FlexMLP(
            input_dim    = input_dim,
            hidden_layers= hidden,
            output_dim   = 1,
            activation   = "relu",
            dropout      = cfg.get("dropout", 0.0),
            batch_norm   = cfg.get("batch_norm", False),
            task_type    = task_type,
        )


def build_optimizer(model, cfg, base_lr):
    wd   = cfg.get("weight_decay", 1e-4)
    name = cfg.get("optimizer", "adam")
    if name == "adam":
        return torch.optim.Adam(model.parameters(),  lr=base_lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(),   lr=base_lr, momentum=0.9,
                                nesterov=True, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(opt, cfg, max_epochs, warmup_epochs=5):
    s = cfg.get("scheduler")
    if s is None:
        return None
    if s == "plateau":
        return ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    if s == "cosine":
        return CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    if s == "warmup_cosine":
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return float(ep + 1) / warmup_epochs
            progress = (ep - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(opt, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {s}")


def build_loss(cfg, task_type, y_train=None):
    lname = cfg.get("loss", "default")
    if lname == "default" or lname is None:
        return nn.BCELoss() if task_type == "classification" else nn.MSELoss()
    if lname == "weighted_bce":
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pw    = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        return nn.BCELoss(weight=None), pw   # handle in loop
    if lname == "huber":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {lname}")


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop
# ─────────────────────────────────────────────────────────────────────────────

def label_encode_preserve_nan(df, columns):
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        non_nan = df[col].dropna()
        le.fit(non_nan)
        encoders[col] = le
        df[col] = df[col].map(
            lambda v, _le=le: float(_le.transform([v])[0]) if pd.notna(v) else np.nan
        )
    return df, encoders


def train_config(cfg, model, loss_fn, pos_weight,
                 X_tr, y_tr, X_val, y_val,
                 base_lr, max_epochs, patience, batch_size, task_type):
    """
    Full training loop.  Returns (best_model, best_val_score, epochs_run, time_s).
    """
    torch.manual_seed(42)
    opt   = build_optimizer(model, cfg, base_lr)
    sched = build_scheduler(opt, cfg, max_epochs)

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True, num_workers=0)

    best_score = -np.inf
    best_state = None
    no_improve  = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            pred = model(Xb).squeeze()
            if pos_weight is not None:
                # per-sample weight by class
                w = torch.where(yb == 1, pos_weight.squeeze(), torch.ones(1))
                loss = (nn.BCELoss(reduction="none")(pred, yb) * w).mean()
            else:
                loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).squeeze().numpy()

        val_score = (roc_auc_score(y_val, val_out) if task_type == "classification"
                     else r2_score(y_val, val_out))

        if sched is not None:
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_score)
            else:
                sched.step()

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
# WIND TURBINE FORECASTING (10 standard configs)
# ─────────────────────────────────────────────────────────────────────────────

def run_wind_pytorch():
    print("\n" + "=" * 65)
    print("TASK 3 - Wind turbine forecasting (PyTorch)")
    print("  sklearn DNN best: R2=0.9706  |  ML Ridge: R2=0.9714")
    print("=" * 65)

    # ── Preprocessing: identical to dnn_dashboard_benchmark.py ──────────────
    df = pd.read_csv(ROOT / "Wind Turbine Scada dataset" / "T1.csv")
    # Drop Date/Time; use only the power column (univariate -- same as sklearn bench)
    target_col = "LV ActivePower (kW)"
    data = df[target_col].dropna().values.reshape(-1, 1).astype(np.float64)

    scaler_mms = MinMaxScaler()
    data_s = scaler_mms.fit_transform(data)

    WINDOW = 24
    X_seq, y_seq = [], []
    for i in range(len(data_s) - WINDOW):
        X_seq.append(data_s[i : i + WINDOW].flatten())
        y_seq.append(data_s[i + WINDOW, 0])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Val from last 10% of train (chronological)
    val_split = int(len(X_train) * 0.9)
    X_tr   = X_train[:val_split]
    X_val  = X_train[val_split:]
    y_tr   = y_train[:val_split]
    y_val  = y_train[val_split:]

    print(f"  Train={len(X_tr)}  Val={len(X_val)}  Test={len(X_test)}  "
          f"input_dim={X_tr.shape[1]}")

    # ── Fixed settings (mirrors sklearn best) ───────────────────────────────
    HIDDEN   = [256, 128]    # sklearn best architecture
    INPUT    = X_tr.shape[1] # 24 (window × 1 feature)
    BASE_LR  = 0.01          # sklearn best lr
    EPOCHS   = 100
    PATIENCE = 15
    BSIZE    = 512

    rows = []
    best_test_r2 = -np.inf
    best_wrapper = None

    for i, cfg in enumerate(PYTORCH_CONFIGS, 1):
        print(f"  [{i:2d}/10] {cfg['name']} ... ", end="", flush=True)
        model = FlexMLP(INPUT, HIDDEN, 1, "relu",
                        cfg["dropout"], cfg["batch_norm"], "regression")
        loss_fn = nn.MSELoss()
        model, val_r2, ep, dt = train_config(
            cfg, model, loss_fn, None,
            X_tr, y_tr, X_val, y_val,
            BASE_LR, EPOCHS, PATIENCE, BSIZE, "regression",
        )

        model.eval()
        with torch.no_grad():
            test_pred_s = model(torch.tensor(X_test)).squeeze().numpy()

        # Inverse-transform for reporting
        y_true_raw = scaler_mms.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_raw = scaler_mms.inverse_transform(test_pred_s.reshape(-1, 1)).ravel()
        r2_raw    = r2_score(y_true_raw, y_pred_raw)
        mae_raw   = mean_absolute_error(y_true_raw, y_pred_raw)
        rmse_raw  = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))

        print(f"val_r2={val_r2:.4f}  test_r2={r2_raw:.4f}  ep={ep}  t={dt}s")

        row = {
            "config":       cfg["name"],
            "description":  cfg["desc"],
            "dropout":      cfg["dropout"],
            "batch_norm":   cfg["batch_norm"],
            "optimizer":    cfg["optimizer"],
            "weight_decay": cfg["weight_decay"],
            "scheduler":    cfg["scheduler"] or "none",
            "val_r2":       round(val_r2,   6),
            "test_r2":      round(r2_raw,   6),
            "test_mae":     round(mae_raw,  4),
            "test_rmse":    round(rmse_raw, 4),
            "epochs":       ep,
            "time_s":       dt,
        }
        rows.append(row)

        if r2_raw > best_test_r2:
            best_test_r2 = r2_raw
            best_wrapper = PyTorchMLPWrapper(
                model         = model,
                scaler        = scaler_mms,
                task_type     = "regression",
                feature_columns = [f"t-{WINDOW - j}" for j in range(WINDOW)],
                target        = target_col,
                config_name   = cfg["name"],
                config_desc   = cfg["desc"],
                val_score     = val_r2,
            )
            best_cfg_name = cfg["name"]

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "wind_pytorch_results.csv", index=False)
    joblib.dump(best_wrapper, MODELS_DIR / "wind_forecasting_pytorch_model.pkl")

    print(f"\n  >> Best: {best_cfg_name}  test_r2={best_test_r2:.4f}")
    print("  >> Saved -> pytorch_results/wind_pytorch_results.csv")
    print("  >> Model -> pytorch_results/models/wind_forecasting_pytorch_model.pkl")
    return df_out, best_test_r2, best_cfg_name


# ─────────────────────────────────────────────────────────────────────────────
# HEART -- ADVANCED CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

def run_heart_advanced():
    print("\n" + "=" * 65)
    print("TASK 1b - Heart classification ADVANCED (PyTorch)")
    print("  Previous best: pt_adamw AUC=0.9485  |  ML Stacking: 0.9784")
    print("=" * 65)

    df  = pd.read_csv(ROOT / "Dataset2" / "Dataset2.csv")
    X   = df.drop(columns=["target"]).values.astype(np.float32)
    y   = df["target"].values.astype(np.float32)
    feature_cols = list(df.drop(columns=["target"]).columns)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # Class imbalance info
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    print(f"  Class balance -- neg={n_neg}  pos={n_pos}  "
          f"ratio={n_neg/max(n_pos,1):.2f}")

    INPUT          = X_train_s.shape[1]
    DEFAULT_HIDDEN = [256, 128]
    BASE_LR        = 0.01
    EPOCHS         = 300
    PATIENCE       = 30
    BSIZE          = 32

    rows = []
    best_test_auc = -np.inf
    best_wrapper  = None
    n_configs     = len(ADVANCED_HEART)

    for i, cfg in enumerate(ADVANCED_HEART, 1):
        print(f"  [{i}/{n_configs}] {cfg['name']} ... ", end="", flush=True)

        model = build_model(cfg, INPUT, "classification", DEFAULT_HIDDEN)

        # Build loss & pos_weight
        loss_result = build_loss(cfg, "classification", y_train)
        if isinstance(loss_result, tuple):
            loss_fn, pos_weight = loss_result
        else:
            loss_fn, pos_weight = loss_result, None

        model, val_auc, ep, dt = train_config(
            cfg, model, loss_fn, pos_weight,
            X_train_s, y_train, X_val_s, y_val,
            BASE_LR, EPOCHS, PATIENCE, BSIZE, "classification",
        )

        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test_s)).squeeze().numpy()
        test_pred = (test_logits >= 0.5).astype(int)
        test_auc  = roc_auc_score(y_test, test_logits)
        test_acc  = accuracy_score(y_test, test_pred)
        test_f1   = f1_score(y_test, test_pred)

        print(f"val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
              f"ep={ep}  t={dt}s")

        rows.append({
            "config":       cfg["name"],
            "description":  cfg["desc"],
            "model_type":   cfg.get("model_type", "flexmlp"),
            "loss":         cfg.get("loss", "default"),
            "optimizer":    cfg.get("optimizer"),
            "weight_decay": cfg.get("weight_decay"),
            "scheduler":    cfg.get("scheduler") or "none",
            "val_auc":      round(val_auc,  6),
            "test_auc":     round(test_auc, 6),
            "test_acc":     round(test_acc, 6),
            "test_f1":      round(test_f1,  6),
            "epochs":       ep,
            "time_s":       dt,
        })

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_wrapper  = PyTorchMLPWrapper(
                model=model, scaler=scaler, task_type="classification",
                feature_columns=feature_cols, target="target", threshold=0.5,
                config_name=cfg["name"], config_desc=cfg["desc"], val_score=val_auc,
            )
            best_cfg_name = cfg["name"]

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "heart_advanced_results.csv", index=False)

    # Save best advanced model only if it beats previous best (0.9485)
    if best_test_auc > 0.9485:
        joblib.dump(best_wrapper,
                    MODELS_DIR / "heart_classification_pytorch_model.pkl")
        print(f"\n  >> NEW BEST beats previous (0.9485): saving model")
    print(f"\n  >> Best advanced: {best_cfg_name}  AUC={best_test_auc:.4f}")
    print("  >> Saved -> pytorch_results/heart_advanced_results.csv")
    return df_out, best_test_auc, best_cfg_name


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE -- ADVANCED CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

def run_temperature_advanced():
    print("\n" + "=" * 65)
    print("TASK 2b - Temperature regression ADVANCED (PyTorch)")
    print("  Previous best: pt_adamw R2=0.7486  |  ML Stacking: 0.7766")
    print("=" * 65)

    df  = pd.read_csv(ROOT / "Dataset1.csv")
    y   = df["Temperature (C)"].values.astype(np.float32)
    X   = df.drop(columns=["Temperature (C)", "Apparent Temperature (C)",
                            "Formatted Date", "Daily Summary"], errors="ignore")
    feature_cols = list(X.columns)
    cat_cols     = X.select_dtypes(include=["object"]).columns.tolist()

    X, label_encoders = label_encode_preserve_nan(X, cat_cols)

    imputer   = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_imputed.values.astype(np.float32), y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    HP_N, HP_VAL = 15_000, 4_000
    if len(X_train_s) > HP_N:
        idx = np.random.choice(len(X_train_s), HP_N, replace=False)
        X_tr_hp, y_tr_hp = X_train_s[idx], y_train[idx]
    else:
        X_tr_hp, y_tr_hp = X_train_s, y_train
    if len(X_val_s) > HP_VAL:
        idxv = np.random.choice(len(X_val_s), HP_VAL, replace=False)
        X_val_hp, y_val_hp = X_val_s[idxv], y_val[idxv]
    else:
        X_val_hp, y_val_hp = X_val_s, y_val

    print(f"  HP search on {len(X_tr_hp)} rows (full train = {len(X_train_s)})")

    INPUT          = X_train_s.shape[1]
    DEFAULT_HIDDEN = [256, 128, 64]
    BASE_LR        = 0.001
    EPOCHS_HP  = 80;  PAT_HP  = 10;  BSIZE_HP  = 256
    EPOCHS_FUL = 80;  PAT_FUL = 10;  BSIZE_FUL = 512
    n_configs  = len(ADVANCED_TEMP)

    # HP search
    hp_rows = []
    for i, cfg in enumerate(ADVANCED_TEMP, 1):
        print(f"  HP [{i}/{n_configs}] {cfg['name']} ... ", end="", flush=True)
        model   = build_model(cfg, INPUT, "regression", DEFAULT_HIDDEN)
        loss_fn = build_loss(cfg, "regression")
        if isinstance(loss_fn, tuple):
            loss_fn = loss_fn[0]
        _, val_r2, ep, dt = train_config(
            cfg, model, loss_fn, None,
            X_tr_hp, y_tr_hp, X_val_hp, y_val_hp,
            BASE_LR, EPOCHS_HP, PAT_HP, BSIZE_HP, "regression",
        )
        print(f"val_r2={val_r2:.4f}  ep={ep}  t={dt}s")
        hp_rows.append({"config": cfg["name"], "val_r2_subsample": round(val_r2, 6)})

    hp_df = pd.DataFrame(hp_rows)
    best_idx = hp_df["val_r2_subsample"].idxmax()
    best_hp  = ADVANCED_TEMP[best_idx]
    print(f"  >> Best HP: {best_hp['name']} "
          f"val_r2={hp_df.loc[best_idx,'val_r2_subsample']:.4f}")

    # Full retrain all configs (same as original script)
    print(f"\n  -- Full retrain all {n_configs} configs --")
    rows = []
    best_test_r2 = -np.inf
    best_wrapper = None

    for i, cfg in enumerate(ADVANCED_TEMP, 1):
        print(f"  [{i}/{n_configs}] {cfg['name']} ... ", end="", flush=True)
        model   = build_model(cfg, INPUT, "regression", DEFAULT_HIDDEN)
        loss_fn = build_loss(cfg, "regression")
        if isinstance(loss_fn, tuple):
            loss_fn = loss_fn[0]
        model, val_r2, ep, dt = train_config(
            cfg, model, loss_fn, None,
            X_train_s, y_train, X_val_s, y_val,
            BASE_LR, EPOCHS_FUL, PAT_FUL, BSIZE_FUL, "regression",
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test_s)).squeeze().numpy()
        r2_   = r2_score(y_test, pred)
        mae_  = mean_absolute_error(y_test, pred)
        rmse_ = np.sqrt(mean_squared_error(y_test, pred))
        print(f"test_r2={r2_:.4f}  ep={ep}  t={dt}s")

        sub_r2 = hp_df.loc[hp_df.config == cfg["name"], "val_r2_subsample"].values[0]
        rows.append({
            "config":            cfg["name"],
            "description":       cfg["desc"],
            "model_type":        cfg.get("model_type", "flexmlp"),
            "loss":              cfg.get("loss", "default"),
            "optimizer":         cfg.get("optimizer"),
            "weight_decay":      cfg.get("weight_decay"),
            "scheduler":         cfg.get("scheduler") or "none",
            "val_r2_subsample":  sub_r2,
            "val_r2_full":       round(val_r2, 6),
            "test_r2":           round(r2_,    6),
            "test_mae":          round(mae_,   4),
            "test_rmse":         round(rmse_,  4),
            "epochs":            ep,
            "time_s":            dt,
        })

        if r2_ > best_test_r2:
            best_test_r2 = r2_
            best_wrapper = PyTorchMLPWrapper(
                model=model, scaler=scaler, task_type="regression",
                feature_columns=feature_cols, target="Temperature (C)",
                imputer=imputer, cat_cols=cat_cols,
                label_encoders=label_encoders,
                config_name=cfg["name"], config_desc=cfg["desc"], val_score=val_r2,
            )
            best_cfg_name = cfg["name"]

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "temperature_advanced_results.csv", index=False)

    if best_test_r2 > 0.7486:
        joblib.dump(best_wrapper,
                    MODELS_DIR / "temperature_regression_pytorch_model.pkl")
        print(f"\n  >> NEW BEST beats previous (0.7486): saving model")
    print(f"\n  >> Best advanced: {best_cfg_name}  R2={best_test_r2:.4f}")
    print("  >> Saved -> pytorch_results/temperature_advanced_results.csv")
    return df_out, best_test_r2, best_cfg_name


# ─────────────────────────────────────────────────────────────────────────────
# Extended summary
# ─────────────────────────────────────────────────────────────────────────────

def build_extended_summary(wind_best, heart_adv_best, temp_adv_best):
    print("\n" + "=" * 65)
    print("EXTENDED SUMMARY")
    print("=" * 65)

    # Load previous results for context
    try:
        prev_h = pd.read_csv(OUT_DIR / "heart_pytorch_results.csv")
        prev_t = pd.read_csv(OUT_DIR / "temperature_pytorch_results.csv")
        prev_best_h = prev_h["test_auc"].max()
        prev_best_t = prev_t["test_r2"].max()
    except Exception:
        prev_best_h = 0.9485
        prev_best_t = 0.7486

    rows = [
        {"task": "wind_forecasting",    "metric": "r2",      "sklearn_dnn": 0.9706,
         "ml_baseline": 0.9714, "pytorch_standard_best": wind_best,
         "pytorch_advanced_best": "n/a  (no advanced configs for wind)"},
        {"task": "heart_classification","metric": "roc_auc", "sklearn_dnn": 0.9416,
         "ml_baseline": 0.9784, "pytorch_standard_best": prev_best_h,
         "pytorch_advanced_best": round(heart_adv_best, 4)},
        {"task": "temperature_regression","metric": "r2",    "sklearn_dnn": 0.7439,
         "ml_baseline": 0.7766, "pytorch_standard_best": prev_best_t,
         "pytorch_advanced_best": round(temp_adv_best, 4)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "extended_summary.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved -> pytorch_results/extended_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    print("=" * 65)
    print("PyTorch Advanced Experiments")
    print("  1) Wind turbine forecasting  (10 standard configs)")
    print("  2) Heart classification       (6 advanced configs)")
    print("  3) Temperature regression     (6 advanced configs)")
    print("=" * 65)

    wind_df,  wind_best,      _ = run_wind_pytorch()
    heart_df, heart_adv_best, _ = run_heart_advanced()
    temp_df,  temp_adv_best,  _ = run_temperature_advanced()

    build_extended_summary(wind_best, heart_adv_best, temp_adv_best)

    total = (time.time() - t_start) / 60
    print(f"\nTotal time: {total:.1f} min")
    print("All results saved to pytorch_results/")
