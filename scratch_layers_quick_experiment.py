"""
Quick benchmark using primitive custom layers built in `pytorch_scratch_layers.py`.

This is intended as proof of "layers from scratch" for consultation purposes.
Runs on the heart dataset and saves metrics to pytorch_results/.
"""

from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pytorch_scratch_layers import ScratchFlexMLP, ScratchResidualMLP


torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "pytorch_results"
OUT_DIR.mkdir(exist_ok=True)


def train_one(model, X_tr, y_tr, X_val, y_val, lr, weight_decay, batch_size, max_epochs, patience):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, num_workers=0)

    best_auc = -np.inf
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
            val_prob = model(X_val_t).squeeze().numpy()
        val_auc = roc_auc_score(y_val, val_prob)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc, epoch, round(time.time() - t0, 1)


def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(X_te, dtype=torch.float32)).squeeze().numpy()
    pred = (prob >= 0.5).astype(int)
    return {
        "test_auc": float(roc_auc_score(y_te, prob)),
        "test_acc": float(accuracy_score(y_te, pred)),
        "test_f1": float(f1_score(y_te, pred)),
    }


def main():
    df = pd.read_csv(ROOT / "Dataset2" / "Dataset2.csv")
    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    input_dim = X_train_s.shape[1]

    # Small set of runs to keep runtime short while proving custom-layer models train correctly.
    configs = [
        {
            "name": "scratch_flex_small",
            "desc": "ScratchFlexMLP hidden=[64,32], dropout=0.1",
            "builder": lambda: ScratchFlexMLP(
                input_dim=input_dim,
                hidden_layers=[64, 32],
                dropout=0.1,
                batch_norm=True,
                task_type="classification",
            ),
            "lr": 0.01,
            "wd": 1e-3,
            "batch_size": 32,
            "epochs": 80,
            "patience": 12,
        },
        {
            "name": "scratch_flex_256_128",
            "desc": "ScratchFlexMLP hidden=[256,128], dropout=0.0",
            "builder": lambda: ScratchFlexMLP(
                input_dim=input_dim,
                hidden_layers=[256, 128],
                dropout=0.0,
                batch_norm=False,
                task_type="classification",
            ),
            "lr": 0.01,
            "wd": 1e-4,
            "batch_size": 32,
            "epochs": 120,
            "patience": 16,
        },
        {
            "name": "scratch_residual_128x2",
            "desc": "ScratchResidualMLP hidden=128, blocks=2, dropout=0.1",
            "builder": lambda: ScratchResidualMLP(
                input_dim=input_dim,
                hidden_dim=128,
                n_blocks=2,
                dropout=0.1,
                task_type="classification",
            ),
            "lr": 0.005,
            "wd": 1e-3,
            "batch_size": 32,
            "epochs": 120,
            "patience": 16,
        },
    ]

    rows = []
    for cfg in configs:
        print(f"Running {cfg['name']} ...", flush=True)
        model = cfg["builder"]()
        model, val_auc, epochs_run, time_s = train_one(
            model,
            X_train_s,
            y_train,
            X_val_s,
            y_val,
            lr=cfg["lr"],
            weight_decay=cfg["wd"],
            batch_size=cfg["batch_size"],
            max_epochs=cfg["epochs"],
            patience=cfg["patience"],
        )
        metrics = evaluate(model, X_test_s, y_test)
        row = {
            "config": cfg["name"],
            "description": cfg["desc"],
            "lr": cfg["lr"],
            "weight_decay": cfg["wd"],
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["epochs"],
            "patience": cfg["patience"],
            "val_auc": round(val_auc, 6),
            "test_auc": round(metrics["test_auc"], 6),
            "test_acc": round(metrics["test_acc"], 6),
            "test_f1": round(metrics["test_f1"], 6),
            "epochs_run": epochs_run,
            "time_s": time_s,
        }
        rows.append(row)
        print(
            f"  val_auc={row['val_auc']:.4f} test_auc={row['test_auc']:.4f} "
            f"acc={row['test_acc']:.4f} f1={row['test_f1']:.4f}"
        )

    out_df = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    out_path = OUT_DIR / "scratch_layers_heart_results.csv"
    out_df.to_csv(out_path, index=False)

    best = out_df.iloc[0]
    print("\nBest scratch-layer config:")
    print(best.to_string())
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
