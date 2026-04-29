"""
Novel from-scratch architectures for tabular data.

All three architectures are built entirely from our primitive layers
(MyLinear and MyBatchNorm1d defined in pytorch_scratch_layers.py).
None of them are standard MLPs or ResNets -- each one introduces
a structurally different computation pattern:

1. ScratchNovelNet (Feature Crossing)
   - Dual-path block: main representation + sigmoid gate
   - Element-wise multiplication fuses the two paths
   - Captures feature interactions through learned gating

2. ScratchSqueezeExciteNet (Channel Attention)
   - Inspired by SE blocks but adapted for tabular 1-D data
   - Compresses hidden repr. to a bottleneck, then expands back
   - Learns per-feature importance weights via sigmoid recalibration
   - Different from standard attention: no queries/keys/values

3. ScratchMultiScaleNet (Feature Pyramid)
   - Processes inputs at multiple resolutions simultaneously
   - Each branch projects to a different hidden dimension
   - Concatenates multi-scale features, then reduces
   - Learns both fine-grained and coarse patterns in parallel
"""

import torch
import torch.nn as nn
from pytorch_scratch_layers import MyLinear, MyBatchNorm1d


# ─────────────────────────────────────────────────────────────
# Architecture 1: Feature Crossing with Gating (existing)
# ─────────────────────────────────────────────────────────────

class ScratchFeatureCrossingBlock(nn.Module):
    """
    A completely novel block built from scratch primitives.
    Computes a main representation and a gating mechanism,
    fuses them with element-wise multiplication, then projects.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj_main = MyLinear(in_dim, out_dim)
        self.bn_main = MyBatchNorm1d(out_dim)

        self.proj_gate = MyLinear(in_dim, out_dim)
        self.bn_gate = MyBatchNorm1d(out_dim)

        self.proj_out = MyLinear(out_dim, out_dim)
        self.bn_out = MyBatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path 1: Main feature transformation
        main = torch.relu(self.bn_main(self.proj_main(x)))
        # Path 2: Gate generation
        gate = torch.sigmoid(self.bn_gate(self.proj_gate(x)))

        # Crossing: element-wise multiplication
        crossed = main * gate

        # Final projection and non-linearity
        out = torch.relu(self.bn_out(self.proj_out(crossed)))

        # Residual connection if dimensions match
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out


class ScratchNovelNet(nn.Module):
    """
    Custom Feature Crossing architecture.
    Uses dual-path gating blocks instead of standard linear stacks.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_blocks: int = 2,
                 task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type

        self.entry = MyLinear(input_dim, hidden_dim)
        self.entry_bn = MyBatchNorm1d(hidden_dim)

        self.blocks = nn.ModuleList([
            ScratchFeatureCrossingBlock(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        self.head = MyLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.entry(x))
        out = self.entry_bn(out)

        for block in self.blocks:
            out = block(out)

        out = self.head(out)

        if self.task_type == "classification" and out.shape[1] == 1:
            out = torch.sigmoid(out)
        return out


# ─────────────────────────────────────────────────────────────
# Architecture 2: Squeeze-and-Excite Tabular Net (NEW)
# ─────────────────────────────────────────────────────────────

class ScratchSEBlock(nn.Module):
    """
    Squeeze-and-Excite style block for tabular data.

    Instead of spatial pooling (like image SE blocks), we:
    1. Transform input to hidden dimension (main path)
    2. Squeeze: compress the hidden representation to a bottleneck
    3. Excite: expand back and apply sigmoid to get per-feature weights
    4. Recalibrate: multiply the main representation by the weights

    This learns which hidden features matter most for each sample,
    giving the network a form of self-attention on the feature level
    without using standard attention mechanisms (no Q/K/V).
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(dim // reduction, 8)

        # Main transform
        self.main_fc = MyLinear(dim, dim)
        self.main_bn = MyBatchNorm1d(dim)

        # SE path: squeeze -> excite
        self.squeeze = MyLinear(dim, bottleneck)
        self.excite = MyLinear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main transformation
        h = torch.relu(self.main_bn(self.main_fc(x)))

        # Squeeze: compress to bottleneck
        s = torch.relu(self.squeeze(h))
        # Excite: expand and sigmoid for per-feature importance
        e = torch.sigmoid(self.excite(s))

        # Recalibrate: scale features by learned importance
        out = h * e

        # Residual
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out


class ScratchSqueezeExciteNet(nn.Module):
    """
    Novel tabular architecture with channel-attention via SE blocks.

    Each block learns to emphasise or suppress hidden features
    adaptively per sample. This is fundamentally different from
    both MLP (uniform processing) and ResNet (skip connections only):
    here, the network dynamically re-weights its own intermediate
    representations based on what it sees in each input.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_blocks: int = 2,
                 reduction: int = 4,
                 task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type

        self.entry = MyLinear(input_dim, hidden_dim)
        self.entry_bn = MyBatchNorm1d(hidden_dim)

        self.blocks = nn.ModuleList([
            ScratchSEBlock(hidden_dim, reduction=reduction)
            for _ in range(num_blocks)
        ])

        self.head = MyLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.entry_bn(self.entry(x)))

        for block in self.blocks:
            out = block(out)

        out = self.head(out)

        if self.task_type == "classification" and out.shape[1] == 1:
            out = torch.sigmoid(out)
        return out


# ─────────────────────────────────────────────────────────────
# Architecture 3: Multi-Scale Feature Pyramid Net (NEW)
# ─────────────────────────────────────────────────────────────

class ScratchMultiScaleBranch(nn.Module):
    """A single branch that projects input to a specific scale."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = MyLinear(in_dim, out_dim)
        self.bn = MyBatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.fc(x)))


class ScratchMultiScaleBlock(nn.Module):
    """
    Multi-Scale processing block for tabular data.

    The idea: instead of processing features through a single
    hidden dimension (like MLP), we split the processing into
    three parallel branches at different scales (small, medium, large).
    Each branch captures patterns at a different granularity.
    The outputs are concatenated and then projected back down.

    This is structurally different from:
    - MLP: single processing path
    - ResNet: single path with skip connections
    - Feature Crossing: dual path with gating
    - SE blocks: single path with recalibration

    Here we get genuine multi-resolution feature extraction.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # Three branches at different scales
        scale_small = max(hidden_dim // 4, 8)
        scale_medium = hidden_dim // 2
        scale_large = hidden_dim

        self.branch_small = ScratchMultiScaleBranch(in_dim, scale_small)
        self.branch_medium = ScratchMultiScaleBranch(in_dim, scale_medium)
        self.branch_large = ScratchMultiScaleBranch(in_dim, scale_large)

        # Merge: concat all branches -> project back to hidden_dim
        concat_dim = scale_small + scale_medium + scale_large
        self.merge = MyLinear(concat_dim, hidden_dim)
        self.merge_bn = MyBatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.branch_small(x)
        m = self.branch_medium(x)
        l = self.branch_large(x)

        # Concatenate multi-scale features
        concat = torch.cat([s, m, l], dim=1)

        # Merge and project
        out = torch.relu(self.merge_bn(self.merge(concat)))

        # Residual if dims match
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out


class ScratchMultiScaleNet(nn.Module):
    """
    Novel multi-resolution architecture for tabular data.

    Uses parallel branches at different widths to capture both
    fine-grained feature details and coarse high-level patterns.
    The multi-scale design is common in computer vision (FPN, UNet)
    but has not been applied to tabular data in standard practice.
    We adapt it here using our from-scratch primitive layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_blocks: int = 2,
                 task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type

        self.entry = MyLinear(input_dim, hidden_dim)
        self.entry_bn = MyBatchNorm1d(hidden_dim)

        self.blocks = nn.ModuleList([
            ScratchMultiScaleBlock(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        self.head = MyLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.entry_bn(self.entry(x)))

        for block in self.blocks:
            out = block(out)

        out = self.head(out)

        if self.task_type == "classification" and out.shape[1] == 1:
            out = torch.sigmoid(out)
        return out
