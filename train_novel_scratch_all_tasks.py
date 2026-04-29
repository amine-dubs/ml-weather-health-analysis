import os
import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.impute import KNNImputer, SimpleImputer
from torch.utils.data import TensorDataset, DataLoader

from novel_architectures import ScratchNovelNet

OUT_DIR = Path("pytorch_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- TRAINING LOOPS ---
def train_classification(model, loader, X_val_t, y_val_t, y_val, X_test_t, y_test, lr=0.01, wd=1e-4, epochs=300, patience=20, binary=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCELoss() if binary else nn.CrossEntropyLoss()
    
    best_metric = -float('inf')
    best_state = None
    no_improve = 0
    start = time.time()
    epochs_run = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            if binary: pred = pred.squeeze()
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            if binary:
                val_prob = val_out.squeeze().numpy()
                try: val_metric = roc_auc_score(y_val, val_prob)
                except ValueError: val_metric = balanced_accuracy_score(y_val, val_prob > 0.5) # Fallback
            else:
                val_prob = torch.softmax(val_out, dim=1).numpy()
                val_pred_c = val_prob.argmax(axis=1)
                val_metric = balanced_accuracy_score(y_val, val_pred_c)
                
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        epochs_run = epoch
        
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        if binary:
            test_prob = test_out.squeeze().numpy()
            try: t_auc = roc_auc_score(y_test, test_prob)
            except: t_auc = 0.0
            test_pred_c = test_prob > 0.5
            t_acc = (test_pred_c == y_test).mean()
            t_f1 = f1_score(y_test, test_pred_c, zero_division=0)
            t_metric = t_auc
        else:
            test_prob = torch.softmax(test_out, axis=1).numpy()
            # multi-class AUC is complex without exact y format, we'll use macro multiclass
            try: t_auc = roc_auc_score(y_test, test_prob, multi_class='ovr')
            except: t_auc = 0.0
            test_pred_c = test_prob.argmax(axis=1)
            t_acc = (test_pred_c == y_test).mean()
            t_f1 = f1_score(y_test, test_pred_c, average='macro')
            t_metric = t_acc
            
    return {
        "val_metric": best_metric,
        "test_metric": t_metric,
        "test_acc": t_acc,
        "test_f1": t_f1,
        "epochs_run": epochs_run,
        "time_s": time.time() - start
    }

def train_regression(model, loader, X_val_t, y_val_t, y_val, X_test_t, y_test, lr=0.01, wd=1e-4, epochs=300, patience=20):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    
    best_metric = -float('inf')
    best_state = None
    no_improve = 0
    start = time.time()
    epochs_run = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).numpy()
            val_metric = r2_score(y_val, val_out)
                
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        epochs_run = epoch
        
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t).numpy()
        test_r2 = r2_score(y_test, test_out)
        if len(y_test.shape) == 2 and y_test.shape[1] > 1:
            test_r2_avg = np.mean(r2_score(y_test, test_out, multioutput='raw_values'))
            test_r2 = test_r2_avg
        test_mae = mean_absolute_error(y_test, test_out)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_out))
            
    return {
        "val_metric": best_metric,
        "test_metric": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "epochs_run": epochs_run,
        "time_s": time.time() - start
    }

# --- TASK EXECUTORS ---

def run_task(task_name, type_='binary', hidden=128, blocks=2):
    print(f"\\n--- Running Novel Architecture on {task_name} ---")
    if task_name == "heart_classification":
        df = pd.read_csv("Dataset2/Dataset2.csv")
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
        scaler = StandardScaler()
    elif task_name == "covid_classification":
        df = pd.read_csv("Dataset3Covid/Dataset3.csv")
        df.replace('-', np.nan, inplace=True)
        # simplistic
        cat_cols = df.select_dtypes(include=['object']).columns
        if 'SARS-Cov-2 exam result' in cat_cols:
            y = (df['SARS-Cov-2 exam result'] == 'positive').astype(int).values
            df = df.drop(columns=['Patient ID', 'SARS-Cov-2 exam result'])
            
            for c in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[c] = pd.Series(le.fit_transform(df[c].astype(str))).replace(len(le.classes_)-1, np.nan) # map 'nan' back to nan if present
            
            X = KNNImputer(n_neighbors=5).fit_transform(df)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
            scaler = StandardScaler()
        else: return
    elif task_name == "temperature_regression":
        df = pd.read_csv("Dataset1.csv")
        y = df["Temperature (C)"].values
        X = df.drop(columns=["Temperature (C)", "Formatted Date", "Summary", "Daily Summary", "Loud Cover"], errors="ignore")
        for c in X.select_dtypes(['object']).columns: X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = SimpleImputer(strategy='mean').fit_transform(X) # faster for this mass loop than KNN
        # cap to 15k
        X, y = X[:15000], y[:15000]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
        scaler = StandardScaler()
    elif task_name == "wind_forecasting":
        df = pd.read_csv("Wind Turbine Scada dataset/T1.csv")
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
        df.sort_values("Date/Time", inplace=True)
        data = df.drop(columns=["Date/Time"]).values[:20000] # cap
        X_seq, y_seq = [], []
        for i in range(len(data)-24):
            X_seq.append(data[i:i+24].flatten())
            y_seq.append(data[i+24][0]) # active power
        X, y = np.array(X_seq), np.array(y_seq)
        tr_split = int(0.8 * len(X))
        val_split = int(0.9 * len(X))
        X_tr, X_val, X_te = X[:tr_split], X[tr_split:val_split], X[val_split:]
        y_tr, y_val, y_te = y[:tr_split], y[tr_split:val_split], y[val_split:]
        scaler = MinMaxScaler()
    elif task_name == "anomaly_wine":
        df = pd.read_csv("Anomaly detection/WineType/separate_class_evaluation_results/wine_quality_merged.csv")
        minority = df["type"].value_counts().idxmin()
        X = df.drop(columns=["type"]).values
        y = (df["type"] == minority).astype(int).values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
        scaler = StandardScaler()
    elif task_name == "anomaly_heart":
        df = pd.read_csv("Dataset2/Dataset2.csv")
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
        scaler = StandardScaler()
    elif task_name == "anomaly_employee":
        df_tr = pd.read_csv("Anomaly detection/EmpolyeeClassification/train.csv")
        for c in df_tr.select_dtypes(['object']).columns: 
            if c != 'Attrition': df_tr[c] = LabelEncoder().fit_transform(df_tr[c])
        X = df_tr.drop(columns=["Attrition"]).values
        y = (df_tr["Attrition"] == "Left").astype(int).values
        # Just use train for train/val/test split to benchmark this architecture
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
        scaler = StandardScaler()
    elif task_name == "multi_output_regression":
        df = pd.read_csv("Dataset1.csv")
        df = df.drop(columns=["Formatted Date", "Daily Summary"], errors="ignore")
        for c in df.select_dtypes(['object']).columns: df[c] = LabelEncoder().fit_transform(df[c])
        y = df[["Pressure (millibars)", "Humidity"]].values
        X = df.drop(columns=["Pressure (millibars)", "Humidity"]).values
        X = SimpleImputer(strategy='mean').fit_transform(X)[:15000]
        y = SimpleImputer(strategy='mean').fit_transform(y)[:15000]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
        scaler = StandardScaler()
    elif task_name == "weather_classification":
        df = pd.read_csv("Dataset1.csv")
        top4 = df["Summary"].value_counts().head(4).index.tolist()
        df = df[df["Summary"].isin(top4)].copy()
        y = LabelEncoder().fit_transform(df["Summary"].astype(str))
        X = df.drop(columns=["Summary", "Formatted Date", "Daily Summary"], errors="ignore")
        for c in X.select_dtypes(['object']).columns: X[c] = LabelEncoder().fit_transform(X[c])
        X = SimpleImputer(strategy='mean').fit_transform(X)[:20000]
        y = y[:20000]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
        scaler = StandardScaler()
    else:
        print("Task skipped.")
        return
        
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)
    
    dtype_x = torch.float32
    if type_ == "regression":
        if len(y_tr.shape) == 1:
            y_tr = y_tr.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_te = y_te.reshape(-1, 1)
        y_tr = y_tr.astype(np.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.float32)
        out_dim = y_tr.shape[1]
    elif type_ == "binary" or type_ == "multiclass":
        y_tr = y_tr.astype(np.float32)
        if type_ == "multiclass":
            y_tr = y_tr.astype(np.int64)
            y_val_t = torch.tensor(y_val, dtype=torch.int64)
            y_te_t = torch.tensor(y_te, dtype=torch.int64)
        else:
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
            y_te_t = torch.tensor(y_te, dtype=torch.float32)
        out_dim = 1 if type_ == "binary" else len(np.unique(y_tr))
        
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32 if type_!="multiclass" else torch.int64)), batch_size=64, shuffle=True)
    
    model = ScratchNovelNet(input_dim=X_tr.shape[1], hidden_dim=hidden, output_dim=out_dim, num_blocks=blocks, task_type='regression' if type_=='regression' else type_)
    
    if type_ == 'regression':
        res = train_regression(model, loader, torch.tensor(X_val, dtype=torch.float32), y_val_t, y_val, torch.tensor(X_te, dtype=torch.float32), y_te, lr=0.01)
    else:
        res = train_classification(model, loader, torch.tensor(X_val, dtype=torch.float32), y_val_t, y_val, torch.tensor(X_te, dtype=torch.float32), y_te, lr=0.01, binary=(type_=="binary"))
        
    print(f"Done. Test metric: {res['test_metric']:.4f}")
    
    # Save to CSV
    res["config"] = f"scratch_novel_feature_cross_{hidden}x{blocks}"
    res["description"] = "Custom Gate/Cross novel scratch architecture"
    
    df_out = pd.DataFrame([res])
    out_file = OUT_DIR / f"scratch_layers_{task_name}_results.csv"
    if out_file.exists():
        existing = pd.read_csv(out_file)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(out_file, index=False)

if __name__ == "__main__":
    run_task("multi_output_regression", type_="regression")
    run_task("weather_classification", type_="multiclass")

