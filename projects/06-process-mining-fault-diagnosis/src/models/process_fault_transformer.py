"""
Process Mining-Driven Fault Diagnosis Transformer.

Architecture:
  1. Temporal Encoder:       Transformer over CPS telemetry windows
  2. Process Feature Branch: MLP over extracted event-log features
  3. Fusion Layer:           Cross-attention between temporal + process streams
  4. Classification Head:    Multi-class fault diagnosis

Novel contribution: fusing process mining event-log statistics with raw
telemetry enables capturing both temporal anomalies AND process-level
deviations (control-flow, inter-event timing, trace complexity).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence transformers."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Encoder (Transformer over CPS window)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Encodes CPS telemetry window using a Transformer encoder.
    Input: (B, W, num_channels) time series
    Output: (B, d_model) aggregated representation
    """

    def __init__(
        self,
        num_channels: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        window_size: int = 50,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_channels, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=window_size + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, C) → (B, d_model)"""
        B = x.shape[0]
        x = self.input_proj(x)           # (B, W, d_model)
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, W+1, d_model)

        x = self.transformer(x)
        cls_out = x[:, 0]                # (B, d_model) — CLS token representation
        return self.norm(cls_out)


# ─────────────────────────────────────────────────────────────────────────────
# Process Feature Branch
# ─────────────────────────────────────────────────────────────────────────────

class ProcessFeatureBranch(nn.Module):
    """
    MLP branch for process mining features extracted from event logs.
    Captures control-flow and timing anomalies invisible in raw telemetry.
    """

    def __init__(
        self,
        process_feature_dim: int,
        d_model: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(process_feature_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, process_feature_dim) → (B, d_model)"""
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Modal Fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Fuses temporal CPS encoding with process mining features via cross-attention.
    Temporal stream queries process stream for complementary fault indicators.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))  # Learnable fusion gate

    def forward(
        self,
        temporal: torch.Tensor,
        process: torch.Tensor,
    ) -> torch.Tensor:
        """
        temporal: (B, d_model)  — temporal encoding
        process:  (B, d_model)  — process mining encoding
        Returns:  (B, d_model)  — fused representation
        """
        # Add sequence dim for attention
        t_seq = temporal.unsqueeze(1)  # (B, 1, d_model)
        p_seq = process.unsqueeze(1)   # (B, 1, d_model)

        # Temporal queries process features
        attn_out, _ = self.cross_attn(t_seq, p_seq, p_seq)
        fused = self.norm1(t_seq + attn_out).squeeze(1)

        # FFN
        fused = self.norm2(fused + self.ffn(fused))

        # Gated combination with raw temporal
        gate = torch.sigmoid(self.gate)
        return gate * fused + (1 - gate) * temporal


# ─────────────────────────────────────────────────────────────────────────────
# Full Process Fault Transformer
# ─────────────────────────────────────────────────────────────────────────────

class ProcessFaultTransformer(nn.Module):
    """
    Full fault diagnosis model combining process mining + CPS telemetry.

    For fault_type ∈ {normal, sensor_drift, actuator_stuck, communication_delay,
                       oscillation, overload, partial_failure, cascade_failure}
    """

    def __init__(
        self,
        num_channels: int,
        num_fault_classes: int = 8,
        process_feature_dim: int = 30,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        window_size: int = 50,
        use_process_features: bool = True,
    ) -> None:
        super().__init__()
        self.use_process_features = use_process_features

        self.temporal_encoder = TemporalEncoder(
            num_channels=num_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            window_size=window_size,
        )

        if use_process_features and process_feature_dim > 0:
            self.process_branch = ProcessFeatureBranch(process_feature_dim, d_model, dropout)
            self.fusion = CrossModalFusion(d_model, nhead=4, dropout=dropout)
        else:
            self.process_branch = None
            self.fusion = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_fault_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        telemetry: torch.Tensor,
        process_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        telemetry:        (B, W, num_channels)
        process_features: (B, process_feature_dim) or None

        Returns dict with 'logits' and 'embedding'
        """
        # Encode temporal stream
        temporal_emb = self.temporal_encoder(telemetry)  # (B, d_model)

        # Fuse with process features if available
        if (self.use_process_features
                and self.process_branch is not None
                and process_features is not None):
            process_emb = self.process_branch(process_features)  # (B, d_model)
            fused_emb   = self.fusion(temporal_emb, process_emb)
        else:
            fused_emb = temporal_emb

        logits = self.classifier(fused_emb)

        return {
            "logits": logits,
            "embedding": fused_emb,
        }
