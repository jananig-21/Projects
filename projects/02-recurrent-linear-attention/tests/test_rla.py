"""Tests for Recurrent Linear Attention."""
from __future__ import annotations
import pytest, torch
from src.models.rla_cell import (RLAKernelFeatureMap, RecurrentLinearAttentionCell,
                                  RecurrentLinearAttentionModel, RLAFeedForward)

class TestKernelFeatureMap:
    def test_positive_output(self):
        phi = RLAKernelFeatureMap(dim=32)
        out = phi(torch.randn(4, 32))
        assert (out > 0).all()

class TestRLACell:
    @pytest.fixture
    def cell(self): return RecurrentLinearAttentionCell(64, 4, dropout=0.0)

    def test_parallel_shape(self, cell):
        out, _ = cell(torch.randn(2, 16, 64))
        assert out.shape == (2, 16, 64)

class TestRLAFeedForward:
    def test_shape(self):
        ffn = RLAFeedForward(64, 4)
        assert ffn(torch.randn(2, 8, 64)).shape == (2, 8, 64)

class TestRLAModel:
    @pytest.fixture
    def model(self):
        return RecurrentLinearAttentionModel(
            vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=128)

    def test_forward_shape(self, model):
        ids = torch.randint(0, 1000, (2, 32))
        logits, states = model(ids)
        assert logits.shape == (2, 32, 1000)

    def test_logits_finite(self, model):
        ids = torch.randint(0, 1000, (2, 16))
        logits, _ = model(ids)
        assert torch.isfinite(logits).all()
