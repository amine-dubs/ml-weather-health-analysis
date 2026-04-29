"""
Primitive PyTorch layers and MLPs built from Parameters.

Goal: provide a clear "from-scratch layers" implementation for coursework
requirements, while still using PyTorch autograd.
"""

import torch
import torch.nn as nn


class MyLinear(nn.Module):
    """A fully-connected layer implemented directly with Parameters."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


class MyBatchNorm1d(nn.Module):
    """BatchNorm1d implemented using Parameters and running-stat buffers."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class ScratchResidualBlock(nn.Module):
    """Residual block that uses MyLinear and MyBatchNorm1d."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = MyLinear(dim, dim)
        self.bn1 = MyBatchNorm1d(dim)
        self.fc2 = MyLinear(dim, dim)
        self.bn2 = MyBatchNorm1d(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return torch.relu(out + x)


class ScratchFlexMLP(nn.Module):
    """MLP built with MyLinear (+ optional custom BN) from scratch-style layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        output_dim: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        task_type: str = "classification",
    ):
        super().__init__()
        self.task_type = task_type

        blocks = []
        prev = input_dim
        for h in hidden_layers:
            blocks.append(MyLinear(prev, h))
            if batch_norm:
                blocks.append(MyBatchNorm1d(h))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            prev = h

        self.hidden = nn.ModuleList(blocks)
        self.out = MyLinear(prev, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.hidden:
            out = layer(out)
        out = self.out(out)
        if self.task_type == "classification":
            out = torch.sigmoid(out)
        return out


class ScratchResidualMLP(nn.Module):
    """Residual MLP that uses only custom linear/BN blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_blocks: int,
        output_dim: int = 1,
        dropout: float = 0.0,
        task_type: str = "classification",
    ):
        super().__init__()
        self.task_type = task_type

        self.input_proj = MyLinear(input_dim, hidden_dim)
        self.input_bn = MyBatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList(
            [ScratchResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output = MyLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.input_bn(self.input_proj(x)))
        for block in self.blocks:
            out = block(out)
        out = self.output(out)
        if self.task_type == "classification":
            out = torch.sigmoid(out)
        return out
