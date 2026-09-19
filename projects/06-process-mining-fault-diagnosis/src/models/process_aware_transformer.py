"""
Process-Aware Transformer (PAT) for CPS Fault Diagnosis.

Architecture fuses two streams:
1. Temporal Transformer encoder on raw multi-sensor time-series
2. Process feature MLP on process-mining-derived features
3. Cross-attention fusion module for joint reasoning
4. Multi-task head: fault classification + fault localization

Process mining component:
- Directly embeds extracted process graph features (activity frequencies,
  transition matrices) as a process-context vector
- Cross-attention lets temporal stream attend to process-context cues
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE for the temporal encoder."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ──────────────────────────────────────────────────────────────────────────────
# Process Feature Encoder (MLP branch)
# ──────────────────────────────────────────────────────────────────────────────

class ProcessFeatureEncoder(nn.Module):
    """Encodes process mining features (activity freq + transition matrix) to a context vector."""

    def __init__(self, input_dim: int, embed_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, embed_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Sensor Patch Embedding
# ──────────────────────────────────────────────────────────────────────────────

class SensorPatchEmbedding(nn.Module):
    """
    Embeds (window_size, num_sensors) sensor window as a sequence of patches.
    Inspired by ViT: treat sensor channels as patch dimension.
    """

    def __init__(
        self,
        num_sensors: int,
        window_size: int,
        d_model: int,
        patch_size: int = 5,
    ) -> None:
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = window_size // patch_size
        patch_dim        = patch_size * num_sensors

        self.proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, S) → (B, num_patches, d_model)"""
        B, W, S  = x.shape
        # Reshape into non-overlapping patches
        P        = self.patch_size
        n_p      = W // P
        x_p      = x[:, :n_p * P, :].reshape(B, n_p, P * S)
        return self.proj(x_p)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────────────────────

class ProcessContextCrossAttention(nn.Module):
    """
    Cross-attention: temporal tokens attend to process-context vector.
    Allows the temporal stream to selectively incorporate process knowledge.
    """

    def __init__(self, d_model: int, process_dim: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.process_proj = nn.Linear(process_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        temporal_tokens: torch.Tensor,    # (B, L, D)
        process_context: torch.Tensor,    # (B, P)
    ) -> torch.Tensor:
        # Project process context to d_model and expand as "memory"
        ctx = self.process_proj(process_context).unsqueeze(1)   # (B, 1, D)
        attn_out, _ = self.cross_attn(
            query=temporal_tokens,
            key=ctx,
            value=ctx,
        )
        return self.norm(temporal_tokens + self.dropout(attn_out))


# ──────────────────────────────────────────────────────────────────────────────
# Full Process-Aware Transformer
# ──────────────────────────────────────────────────────────────────────────────

class ProcessAwareTransformer(nn.Module):
    """
    Full PAT architecture for CPS fault diagnosis.

    Stream 1: Sensor patches → positional encoding → Transformer encoder
    Stream 2: Process features → MLP encoder → context vector
    Fusion:   Cross-attention (temporal attends to process context)
    Heads:
      - Fault classification (num_classes)
      - Sensor anomaly localization (num_sensors scores)
    """

    def __init__(
        self,
        num_sensors: int,
        window_size: int,
        process_feature_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        process_embed_dim: int = 64,
        patch_size: int = 5,
    ) -> None:
        super().__init__()

        # Sensor stream
        self.patch_embed  = SensorPatchEmbedding(num_sensors, window_size, d_model, patch_size)
        self.pos_enc      = SinusoidalPositionalEncoding(d_model, max_len=window_size, dropout=dropout)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Process stream
        self.process_encoder = ProcessFeatureEncoder(process_feature_dim, process_embed_dim, dropout)

        # Fusion
        self.cross_attention = ProcessContextCrossAttention(d_model, process_embed_dim, nhead=4, dropout=dropout)

        # [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output heads
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.sensor_localizer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_sensors),
            nn.Sigmoid(),   # Output: anomaly score per sensor
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sensor_window: torch.Tensor,       # (B, W, S)
        process_features: torch.Tensor,    # (B, proc_feat_dim)
    ) -> dict[str, torch.Tensor]:
        B = sensor_window.shape[0]

        # Sensor stream: patch embedding → PE → Transformer
        patches  = self.patch_embed(sensor_window)           # (B, n_patches, D)
        patches  = self.pos_enc(patches)

        # Prepend [CLS] token
        cls      = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        seq      = torch.cat([cls, patches], dim=1)          # (B, 1+n_patches, D)
        enc_out  = self.transformer_encoder(seq)             # (B, 1+n_patches, D)

        # Process stream
        proc_ctx = self.process_encoder(process_features)   # (B, proc_embed_dim)

        # Cross-attention fusion
        fused    = self.cross_attention(enc_out, proc_ctx)  # (B, 1+n_patches, D)
        cls_repr = fused[:, 0]                              # (B, D) — CLS token

        # Output heads
        return {
            "fault_logits":        self.classifier(cls_repr),       # (B, num_classes)
            "sensor_anomaly_scores": self.sensor_localizer(cls_repr), # (B, num_sensors)
        }
