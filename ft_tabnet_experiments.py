"""
Quick FT-Transformer and TabNet experiments (small-scale).
Saves summary to `pytorch_results/ft_tabnet_results.csv`.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, r2_score
import joblib

# TabNet import will be attempted at runtime
try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False

# -----------------------------
# Simple FT-Transformer (lightweight)
# -----------------------------
class SimpleFTTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, output_dim=1, task='regression'):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # project each scalar feature to embedding dim
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, output_dim))
        self.task = task

    def forward(self, x):
        # x: (batch, n_features)
        b, n = x.shape
        x = x.unsqueeze(-1)          # (b, n, 1)
        x = self.input_proj(x)       # (b, n, d_model)
        x = x.permute(1, 0, 2)       # (n, b, d_model) for transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)       # (b, n, d_model)
        x = x.mean(dim=1)            # (b, d_model)
        x = self.norm(x)
        out = self.head(x).squeeze(1)
        return out

# -----------------------------
# Utilities: find target column
# -----------------------------

def find_target_col(df):
    candidates = ['target', 'Target', 'label', 'Label', 'y', 'Y', 'heart_disease', 'HeartDisease', 'Heart Disease', 'Temperature (C)']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: if a Temperature column exists
    for c in df.columns:
        if 'temp' in c.lower():
            return c
    # else use last column
    return df.columns[-1]

# -----------------------------
# Small-run trainer for FT-Transformer
# -----------------------------

def train_ft_transformer(X_train, y_train, X_val, y_val, task, epochs=30, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleFTTransformer(n_features=X_train.shape[1], d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, output_dim=1, task=task)
    model.to(device)

    if task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val = -np.inf if task == 'classification' else -1e9
    best_state = None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if task == 'classification':
                loss = criterion(out, yb)
            else:
                loss = criterion(out, yb.unsqueeze(1).squeeze(1))
            loss.backward()
            opt.step()

        # val
        model.eval()
        all_preds = []
        all_y = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                all_preds.append(out)
                all_y.append(yb.numpy())
        all_preds = np.concatenate(all_preds)
        all_y = np.concatenate(all_y)
        if task == 'classification':
            probs = 1 / (1 + np.exp(-all_preds))
            val_metric = roc_auc_score(all_y, probs)
        else:
            val_metric = r2_score(all_y, all_preds)

        # simple early save
        improved = (val_metric > best_val) if task == 'classification' else (val_metric > best_val)
        if improved:
            best_val = val_metric
            best_state = model.state_dict()

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val

# -----------------------------
# Runner for a task (TabNet + FT)
# -----------------------------

def run_task(path, task_type, sample_limit=None):
    df = pd.read_csv(path)
    target = find_target_col(df)
    print(f"Loaded {path}, target -> {target}")

    X = df.drop(columns=[target]).copy()
    y = df[target].values

    # simple preprocessing: label encode object columns
    cat_cols = []
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = X[c].astype(str).fillna('NA')
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c])
            cat_cols.append(c)
    X = X.fillna(X.median(numeric_only=True))

    if sample_limit is not None and len(X) > sample_limit:
        X, _, y, _ = train_test_split(X, y, train_size=sample_limit, stratify=y if task_type=='classification' else None, random_state=42)

    # scale numeric
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y if task_type=='classification' else None)

    results = []

    # FT-Transformer
    print('Training FT-Transformer...')
    ft_model, ft_val = train_ft_transformer(X_train, y_train, X_val, y_val, task='classification' if task_type=='classification' else 'regression', epochs=30 if task_type=='classification' else 10)
    print('FT val metric:', ft_val)
    results.append({'task': os.path.basename(path), 'model':'FTTransformer', 'metric': 'roc_auc' if task_type=='classification' else 'r2', 'value': float(ft_val)})

    # TabNet (if available)
    if HAS_TABNET:
        print('Training TabNet...')
        if task_type == 'classification':
            tab = TabNetClassifier(seed=0)
        else:
            tab = TabNetRegressor(seed=0)
        try:
            tab.fit(X_train, y_train, eval_set=[(X_val, y_val)], max_epochs=30 if task_type=='classification' else 10, patience=5, batch_size=256, virtual_batch_size=64)
            preds = tab.predict(X_val)
            if task_type=='classification':
                # tab.predict returns class labels; predict_proba may be available
                try:
                    probs = tab.predict_proba(X_val)[:,1]
                    val_metric = roc_auc_score(y_val, probs)
                except Exception:
                    val_metric = roc_auc_score(y_val, preds)
            else:
                val_metric = r2_score(y_val, preds)
            print('TabNet val metric:', val_metric)
            results.append({'task': os.path.basename(path), 'model':'TabNet', 'metric': 'roc_auc' if task_type=='classification' else 'r2', 'value': float(val_metric)})
        except Exception as e:
            print('TabNet training failed:', e)
    else:
        print('pytorch-tabnet not installed, skipping TabNet')

    return results

# -----------------------------
# Execute small experiments
# -----------------------------
if __name__ == '__main__':
    os.makedirs('pytorch_results', exist_ok=True)
    all_results = []

    # Heart classification
    heart_path = 'Dataset2/Dataset2.csv'
    if os.path.exists(heart_path):
        all_results += run_task(heart_path, 'classification', sample_limit=None)
    else:
        print('Heart dataset not found:', heart_path)

    # Temperature regression (use sample for speed)
    temp_path = 'Dataset1.csv'
    if os.path.exists(temp_path):
        all_results += run_task(temp_path, 'regression', sample_limit=8000)
    else:
        print('Temperature dataset not found:', temp_path)

    # write results
    if len(all_results) > 0:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv('pytorch_results/ft_tabnet_results.csv', index=False)
        print('\nSaved results to pytorch_results/ft_tabnet_results.csv')
    else:
        print('No results to save')
