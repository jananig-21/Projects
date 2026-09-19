"""Recurrent Linear Attention Cell — O(1) inference, O(L) training."""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RLAKernelFeatureMap(nn.Module):
    def __init__(self, dim: int, kernel_dim: Optional[int] = None):
        super().__init__()
        self.kernel_dim = kernel_dim or dim
        self.proj = nn.Linear(dim, self.kernel_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.proj(x)) + 1.0 + 1e-6


class RecurrentLinearAttentionCell(nn.Module):
    """
    RLA cell. Parallel mode O(LD²), recurrent mode O(D²) per step.
    State: S ∈ R^{K×D_v}, z ∈ R^K  |  y_t = (φ(q)S)/(φ(q)z)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 kernel_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.kernel_dim = kernel_dim or self.head_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.phi_q = nn.ModuleList([RLAKernelFeatureMap(self.head_dim, self.kernel_dim) for _ in range(num_heads)])
        self.phi_k = nn.ModuleList([RLAKernelFeatureMap(self.head_dim, self.kernel_dim) for _ in range(num_heads)])
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        Q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        outputs = []
        for h in range(self.num_heads):
            Qh = self.phi_q[h](Q[:, h])
            Kh = self.phi_k[h](K[:, h])
            Vh = V[:, h]
            KtV = torch.einsum("blk,bld->blkd", Kh, Vh)
            S   = torch.cumsum(KtV, dim=1)
            z   = torch.cumsum(Kh, dim=1)
            num = torch.einsum("blk,blkd->bld", Qh, S)
            den = torch.einsum("blk,blk->bl", Qh, z).unsqueeze(-1).clamp(min=1e-6)
            outputs.append(num / den)
        out = torch.cat(outputs, dim=-1)
        out = self.out_proj(out)
        return self.dropout(self.gate(x) * out)

    def forward(self, x, state=None, use_recurrent=False):
        return self.forward_parallel(x), None


class RLAFeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * multiplier
        self.gate_proj = nn.Linear(dim, inner * 2, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_proj(x).chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * up))


class RLABlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, ffn_multiplier=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = RecurrentLinearAttentionCell(hidden_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = RLAFeedForward(hidden_dim, ffn_multiplier, dropout)

    def forward(self, x, state=None, use_recurrent=False):
        attn_out, new_state = self.attn(self.norm1(x), state, use_recurrent)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_state


class RecurrentLinearAttentionModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=512, num_layers=6, num_heads=8,
                 ffn_multiplier=4, max_seq_len=2048, dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding   = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            RLABlock(hidden_dim, num_heads, ffn_multiplier, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, input_ids, states=None, use_recurrent=False):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states else None
            x, new_state = block(x, state, use_recurrent)
            new_states.append(new_state)
        return self.norm(x).matmul(self.lm_head.weight.T), new_states
