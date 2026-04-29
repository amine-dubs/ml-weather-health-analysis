"""
pytorch_model_utils.py
======================
Importable class definitions for PyTorch MLP experiments.
Must live in the project root so that joblib-loaded wrappers can
find the class definitions at unpickle time.

Classes
-------
FlexMLP         : Standard fully-connected MLP (dropout + BN support)
ResidualBlock   : Single BN-ReLU-Dropout-Linear residual block
ResidualMLP     : ResNet-style MLP with skip connections
PyTorchMLPWrapper : sklearn-compatible wrapper for all model types
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# SCRATCH NOVEL ARCHITECTURE (FEATURE CROSSING)
# ============================================================================

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        return x.matmul(self.weight.t()) + self.bias

class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class ScratchFeatureCrossingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lin1 = MyLinear(channels, channels)
        self.bn1 = MyBatchNorm1d(channels)
        self.lin_gate = MyLinear(channels, channels)
        self.bn_gate = MyBatchNorm1d(channels)
        self.lin_out = MyLinear(channels, channels)
        self.bn_out = MyBatchNorm1d(channels)
        
    def forward(self, x):
        h = torch.relu(self.bn1(self.lin1(x)))
        g = torch.sigmoid(self.bn_gate(self.lin_gate(x)))
        crossed = h * g
        out = torch.relu(self.bn_out(self.lin_out(crossed)))
        return out + x

class ScratchNovelNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2, task_type='binary'):
        super().__init__()
        self.in_proj = MyLinear(input_dim, hidden_dim)
        self.in_bn = MyBatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([
            ScratchFeatureCrossingBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.out_proj = MyLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.in_bn(self.in_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class FlexMLP(nn.Module):
    """
    Flexible MLP that supports:
      - Variable depth / width
      - ReLU or Tanh activation
      - Dropout between hidden layers
      - BatchNorm1d after each Linear (before activation)
      - Sigmoid output for binary classification (no activation for regression)
    """
    def __init__(self, input_dim: int, hidden_layers: list,
                 output_dim: int = 1, activation: str = "relu",
                 dropout: float = 0.0, batch_norm: bool = False,
                 task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        if task_type == "classification":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        # Kaiming (He) init for all Linear layers — appropriate for ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Residual architecture
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN
    with a skip connection.  Input and output dim are both `dim`.
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.relu_out = nn.ReLU()

        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return self.relu_out(out + x)          # skip connection


class ResidualMLP(nn.Module):
    """
    ResNet-style MLP for tabular data.

    Architecture:
      Linear(input_dim -> hidden_dim) + BN + ReLU
      N x ResidualBlock(hidden_dim)
      Linear(hidden_dim -> output_dim)  [+ Sigmoid for classification]

    Uses BatchNorm in every block for stable training with larger batches.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_blocks: int,
                 output_dim: int = 1, dropout: float = 0.0,
                 task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type

        # Input projection
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.bn_input    = nn.BatchNorm1d(hidden_dim)
        self.act_input   = nn.ReLU()

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Output head
        self.output = nn.Linear(hidden_dim, output_dim)
        if task_type == "classification":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

        # Kaiming init for projection + output
        for m in [self.input_proj, self.output]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_input(self.bn_input(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.out_act(self.output(x))


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper  (for dashboard / joblib integration)
# ---------------------------------------------------------------------------

class PyTorchMLPWrapper:
    """
    sklearn-compatible wrapper around a trained FlexMLP.

    Stores the preprocessing pipeline alongside the model so a single
    joblib.load() gives a fully functional predict / predict_proba object.

    Parameters
    ----------
    model : FlexMLP
        Trained PyTorch model (CPU).
    scaler : sklearn scaler
        Fitted StandardScaler (or MinMaxScaler).
    task_type : str
        "classification" or "regression".
    feature_columns : list[str]
        Ordered list of feature names expected by the model.
    target : str
        Name of the target column (informational).
    threshold : float
        Decision threshold for binary classification (default 0.5).
    imputer : optional sklearn imputer
        Fitted KNNImputer / IterativeImputer (temperature task).
    cat_cols : list[str] | None
        Categorical column names that were label-encoded.
    label_encoders : dict[str, LabelEncoder] | None
        Fitted encoders mapping category → int (NaN preserved as NaN).
    config_name : str
        Name of the HP configuration that produced this model.
    config_desc : str
        Human-readable description of the configuration.
    val_score : float
        Best validation score achieved during training.
    """

    def __init__(self, model, scaler, task_type, feature_columns, target,
                 threshold=0.5, imputer=None, cat_cols=None,
                 label_encoders=None, config_name="", config_desc="",
                 val_score=None):
        self.model = model
        self.scaler = scaler
        self.task_type = task_type
        self.feature_columns = feature_columns
        self.target = target
        self.threshold = threshold
        self.imputer = imputer
        self.cat_cols = cat_cols or []
        self.label_encoders = label_encoders or {}
        self.config_name = config_name
        self.config_desc = config_desc
        self.val_score = val_score

    # ------------------------------------------------------------------
    # Internal preprocessing helper
    # ------------------------------------------------------------------
    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].copy()
        else:
            X = pd.DataFrame(X, columns=self.feature_columns)

        # 1. Categorical label encoding (NaN preserved)
        for col in self.cat_cols:
            if col in X.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                known = set(le.classes_)
                X[col] = X[col].map(
                    lambda v: int(le.transform([v])[0]) if (pd.notna(v) and v in known) else np.nan
                )

        # 2. Imputation
        if self.imputer is not None:
            X_arr = self.imputer.transform(X.values.astype(np.float64))
            X = pd.DataFrame(X_arr, columns=self.feature_columns)

        # 3. Scaling
        X_scaled = self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = self._preprocess(X)
            out = self.model(X_t).squeeze().numpy()
        if self.task_type == "classification":
            return (out >= self.threshold).astype(int)
        return out

    def predict_proba(self, X):
        """Returns shape (n, 2) — compatible with sklearn predict_proba."""
        assert self.task_type == "classification", "predict_proba only for classification"
        self.model.eval()
        with torch.no_grad():
            X_t = self._preprocess(X)
            prob1 = self.model(X_t).squeeze().numpy()
        prob0 = 1.0 - prob1
        return np.column_stack([prob0, prob1])
