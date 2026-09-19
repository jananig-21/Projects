"""
Kolmogorov-Arnold Networks (KAN) — full production implementation.

Based on:
  "KANs: Towards Interpretable and Efficient Function Approximation Beyond MLPs"
  Janani Giridharan, Jayachandiran U. — Springer AI Review (Under peer review)

KAN Theorem (Kolmogorov-Arnold, 1957):
  Any multivariate continuous function f: [0,1]^n → R can be represented as:
  f(x) = Σ_{q=0}^{2n} Φ_q( Σ_{p=1}^{n} φ_{q,p}(x_p) )

  Where φ_{q,p} and Φ_q are univariate continuous functions.

This implementation replaces fixed activation functions with learnable 1D
B-splines on each edge of the network graph.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# B-Spline Basis Functions
# ──────────────────────────────────────────────────────────────────────────────

def b_spline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute B-spline basis functions of order k on the given grid.

    Uses the Cox-de Boor recursion formula:
      B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1] else 0
      B_{i,k}(x) = (x - grid[i]) / (grid[i+k] - grid[i]) * B_{i,k-1}(x)
                 + (grid[i+k+1] - x) / (grid[i+k+1] - grid[i+1]) * B_{i+1,k-1}(x)

    Args:
        x:    (batch, in_features) input values
        grid: (in_features, grid_size + 2*k + 1) extended knot vector
        k:    spline order

    Returns:
        (batch, in_features, grid_size + k) basis values
    """
    x = x.unsqueeze(-1)  # (B, in_f, 1)
    grid = grid.unsqueeze(0)  # (1, in_f, G+2k+1)

    # Order 0: indicator function
    bases = (x >= grid[..., :-1]) & (x < grid[..., 1:])
    bases = bases.float()  # (B, in_f, G+2k)

    # Cox-de Boor recursion
    for j in range(1, k + 1):
        g0 = grid[..., j:-1]       # (1, in_f, G+2k-j)
        g1 = grid[..., j + 1:]     # (1, in_f, G+2k-j)
        g2 = grid[..., :-j - 1]    # left inner
        g3 = grid[..., 1:-j]       # right inner

        denom_left  = g0 - grid[..., :-(j + 1)]
        denom_right = g1 - g3

        # Safe division
        safe_left  = torch.where(denom_left.abs() > 1e-6, denom_left, torch.ones_like(denom_left))
        safe_right = torch.where(denom_right.abs() > 1e-6, denom_right, torch.ones_like(denom_right))

        left_term  = (x[..., 0:1].expand_as(bases[..., :-1]) - grid[..., :-(j + 1)]) / safe_left * bases[..., :-1]
        right_term = (grid[..., 1:-j] - x[..., 0:1].expand_as(bases[..., 1:])) / safe_right * bases[..., 1:]

        # Zero out where denominator is near zero
        left_term  = torch.where(denom_left.abs() > 1e-6, left_term, torch.zeros_like(left_term))
        right_term = torch.where(denom_right.abs() > 1e-6, right_term, torch.zeros_like(right_term))

        bases = left_term + right_term

    return bases  # (B, in_f, grid_size + k)


# ──────────────────────────────────────────────────────────────────────────────
# KAN Layer
# ──────────────────────────────────────────────────────────────────────────────

class KANLayer(nn.Module):
    """
    Single KAN layer: learns univariate B-spline functions on each edge.

    For each (input_i, output_j) pair, learns φ_{ij}(x_i) as a B-spline.
    Output: y_j = Σ_i φ_{ij}(x_i)

    Each φ_{ij}(x) = base_activation(x) * w_base + spline(x) * w_spline
    where spline(x) = Σ_k c_k B_k(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.grid_range   = grid_range

        # Base function weight
        self.weight_base = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Spline coefficients: one per (out, in, basis_fn)
        self.weight_spline = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        # Scale parameters
        self.scale_base   = nn.Parameter(torch.ones(out_features, in_features) * scale_base)
        self.scale_spline = nn.Parameter(torch.ones(out_features, in_features) * scale_spline)

        # Base activation
        self.base_act = {
            "silu":  F.silu,
            "relu":  F.relu,
            "gelu":  F.gelu,
            "tanh":  torch.tanh,
            "swish": F.silu,
        }.get(base_activation, F.silu)

        # Register extended B-spline grid (not a parameter — fixed knot vector)
        grid = self._build_grid(in_features, grid_size, spline_order, grid_range)
        self.register_buffer("grid", grid)

        self._init_weights(scale_noise)

    @staticmethod
    def _build_grid(
        in_features: int,
        grid_size: int,
        spline_order: int,
        grid_range: tuple[float, float],
    ) -> torch.Tensor:
        """Build uniform extended knot vector with clamped ends."""
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        return grid.unsqueeze(0).expand(in_features, -1).contiguous()

    def _init_weights(self, scale_noise: float) -> None:
        nn.init.kaiming_uniform_(self.weight_base, a=math.sqrt(5))
        with torch.no_grad():
            noise = torch.randn_like(self.weight_spline) * scale_noise
            self.weight_spline.data.copy_(noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features)
        Returns: (batch, out_features)
        """
        assert x.shape[-1] == self.in_features, (
            f"Expected {self.in_features} features, got {x.shape[-1]}"
        )
        batch = x.shape[0]

        # Base activation term: φ_base(x_i) * w_base
        base_output = self.base_act(x)  # (B, in)
        # y_j += Σ_i w_base[j,i] * φ(x_i)
        base_term = F.linear(base_output * self.scale_base.mean(dim=0), self.weight_base)

        # B-spline term
        x_clamped = x.clamp(self.grid_range[0], self.grid_range[1])
        bases = b_spline_basis(x_clamped, self.grid, self.spline_order)
        # bases: (B, in_features, num_basis)

        # Spline output per (j, i): Σ_k c_{j,i,k} * B_k(x_i)
        # (B, in, basis) × (out, in, basis) → (B, out, in) → sum over in
        spline_vals = torch.einsum(
            "bik,oik->boi", bases, self.weight_spline * self.scale_spline.unsqueeze(-1)
        )
        spline_term = spline_vals.sum(dim=2)  # (B, out)

        return base_term + spline_term

    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """
        Adaptive grid update: move grid points to data distribution.
        Called periodically during training for better coverage.
        """
        with torch.no_grad():
            batch = x.shape[0]
            x_sorted, _ = x.sort(dim=0)

            # Sample percentile positions
            adaptive_grid = torch.stack([
                x_sorted[int(batch * q / (self.grid_size + 1))]
                for q in range(1, self.grid_size + 1)
            ], dim=1)  # (in_features, grid_size)

            # Extend with boundary points
            h = (adaptive_grid[:, -1:] - adaptive_grid[:, :1]) / (self.grid_size - 1)
            extended = torch.cat([
                adaptive_grid[:, :1] - self.spline_order * h,
                *[adaptive_grid[:, :1] - (self.spline_order - k) * h for k in range(1, self.spline_order)],
                adaptive_grid,
                *[adaptive_grid[:, -1:] + k * h for k in range(1, self.spline_order + 1)],
            ], dim=1)

            self.grid.data.copy_(extended[:, :self.grid.shape[1]])


# ──────────────────────────────────────────────────────────────────────────────
# Full KAN Model
# ──────────────────────────────────────────────────────────────────────────────

class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network.

    Replaces MLP's fixed activation functions with learnable B-splines
    on each edge, enabling symbolic regression and interpretability.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale_noise: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_sizes = layer_sizes

        self.kan_layers = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.kan_layers.append(
                KANLayer(
                    in_features=in_f,
                    out_features=out_f,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                    scale_noise=scale_noise,
                )
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(size) for size in layer_sizes[1:-1]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.kan_layers[:-1]):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = self.dropout(x)
        x = self.kan_layers[-1](x)
        return x

    def update_grids(self, x: torch.Tensor) -> None:
        """Update all layer grids to match data distribution."""
        with torch.no_grad():
            for layer in self.kan_layers:
                layer.update_grid(x)
                x = layer(x)

    def get_symbolic_representation(self) -> list[dict]:
        """
        Attempt symbolic extraction of learned functions.
        Returns spline coefficients per layer for interpretation.
        """
        repr_list = []
        for i, layer in enumerate(self.kan_layers):
            repr_list.append({
                "layer_index": i,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "weight_spline_norm": layer.weight_spline.data.norm(dim=-1).tolist(),
                "weight_base_norm": layer.weight_base.data.abs().tolist(),
                "scale_base": layer.scale_base.data.tolist(),
                "scale_spline": layer.scale_spline.data.tolist(),
            })
        return repr_list

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────────────────────
# KAN with Residual Connections
# ──────────────────────────────────────────────────────────────────────────────

class KANResidualBlock(nn.Module):
    """Residual KAN block for deeper networks with improved gradient flow."""

    def __init__(
        self,
        features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kan1 = KANLayer(features, features, grid_size, spline_order)
        self.kan2 = KANLayer(features, features, grid_size, spline_order)
        self.norm1 = nn.LayerNorm(features)
        self.norm2 = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(self.kan1(x))
        x = self.dropout(x)
        x = self.norm2(self.kan2(x))
        return x + residual


class KANResidual(nn.Module):
    """Deep KAN with residual connections for complex function approximation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = KANLayer(input_dim, hidden_dim, grid_size, spline_order)
        self.blocks = nn.ModuleList([
            KANResidualBlock(hidden_dim, grid_size, spline_order, dropout)
            for _ in range(num_blocks)
        ])
        self.output_proj = KANLayer(hidden_dim, output_dim, grid_size, spline_order)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)
