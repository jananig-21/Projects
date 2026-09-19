"""FIR-based Quantum Error Estimator architecture."""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


class LearnableFIRLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_order=32):
        super().__init__()
        self.filter_order = filter_order
        weight = self._init_fir_weights(out_channels, in_channels, filter_order)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _init_fir_weights(self, out_ch, in_ch, order):
        weights = torch.zeros(out_ch, in_ch, order + 1)
        for i in range(out_ch):
            cutoff = (i + 1) / (out_ch + 1)
            coeffs = signal.firwin(order + 1, cutoff, window="hamming")
            for j in range(in_ch):
                weights[i, j] = torch.tensor(coeffs, dtype=torch.float32)
        return weights

    def forward(self, x):
        x_unsq = x.unsqueeze(1)
        pad = self.filter_order // 2
        out = F.conv1d(x_unsq.expand(-1, self.weight.shape[1], -1),
                       self.weight, self.bias, padding=pad)
        return out.mean(dim=-1)


class FIRQuantumErrorEstimator(nn.Module):
    def __init__(self, input_dim, num_states, hidden_dims=None,
                 filter_order=32, num_filter_channels=16, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.fir_layer = LearnableFIRLayer(1, num_filter_channels, filter_order)
        proj_in = num_filter_channels + input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in, hidden_dims[0]), nn.LayerNorm(hidden_dims[0]), nn.GELU())
        layers, prev = [], hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)]
            prev = dim
        self.backbone = nn.Sequential(*layers)
        self.distribution_head  = nn.Sequential(nn.Linear(prev, num_states), nn.Softmax(dim=-1))
        self.magnitude_head     = nn.Sequential(nn.Linear(prev, 1), nn.Softplus())
        self.noise_class_head   = nn.Linear(prev, 4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        fir_feat = self.fir_layer(x)
        combined = torch.cat([x, fir_feat], dim=-1)
        projected = self.input_proj(combined)
        backbone_out = self.backbone(projected)
        return {
            "error_distribution": self.distribution_head(backbone_out),
            "error_magnitude": self.magnitude_head(backbone_out).squeeze(-1),
            "noise_class_logits": self.noise_class_head(backbone_out),
        }
