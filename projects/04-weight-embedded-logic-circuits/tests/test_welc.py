"""Tests for Weight-Embedded Logic Circuit Networks."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.welc_layer import (
    BinaryWeightSTE,
    TernaryWeightSTE,
    LogicCircuitOps,
    WELCLinear,
    WELCConv2d,
    WELCClassifier,
    binarize_weights,
    ternarize_weights,
)


class TestBinarization:
    def test_binary_ste_values(self):
        w = torch.tensor([-2.0, -0.1, 0.0, 0.5, 3.0])
        result = binarize_weights(w)
        # sign(0) = 0, but sign(-0.1) = -1, sign(0.5) = 1
        assert result[0] == -1.0
        assert result[3] ==  1.0
        assert result[4] ==  1.0

    def test_binary_gradient_passthrough(self):
        w = torch.tensor([-0.5, 0.5, 2.0], requires_grad=True)
        loss = binarize_weights(w).sum()
        loss.backward()
        # Gradient passes through for |w| <= 1
        assert w.grad is not None
        assert w.grad[0] == 1.0
        assert w.grad[1] == 1.0
        assert w.grad[2] == 0.0  # clipped

    def test_ternary_ste_sparsity(self):
        w = torch.randn(100)
        tw = ternarize_weights(w)
        unique_vals = set(tw.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
        zero_ratio = (tw == 0).float().mean().item()
        assert zero_ratio > 0.0  # some zeros expected


class TestLogicCircuitOps:
    def test_soft_and_range(self):
        a = torch.randn(10, 8)
        b = torch.randn(10, 8)
        out = LogicCircuitOps.soft_and(a, b)
        assert (out >= 0).all() and (out <= 1).all()

    def test_soft_or_range(self):
        a = torch.randn(10, 8)
        b = torch.randn(10, 8)
        out = LogicCircuitOps.soft_or(a, b)
        assert (out >= 0).all() and (out <= 1).all()

    def test_xnor_popcount_shape(self):
        a = torch.sign(torch.randn(4, 16))
        b = torch.sign(torch.randn(4, 16))
        out = LogicCircuitOps.xnor_popcount(a, b)
        assert out.shape == (4,)


class TestWELCLinear:
    @pytest.mark.parametrize("mode", ["binary", "ternary"])
    def test_forward_shape(self, mode):
        layer = WELCLinear(64, 32, weight_mode=mode)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_gradient_flow(self):
        layer = WELCLinear(32, 16, weight_mode="binary")
        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight_latent.grad is not None
        assert layer.scale.grad is not None

    def test_logic_circuit_export(self):
        layer = WELCLinear(16, 8, weight_mode="binary")
        export = layer.get_logic_circuit_representation()
        assert "and_plane" in export
        assert "or_plane" in export
        assert "sparsity" in export
        assert 0.0 <= export["sparsity"] <= 1.0


class TestWELCConv2d:
    def test_forward_shape(self):
        layer = WELCConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 16, 32, 32)


class TestWELCClassifier:
    @pytest.fixture
    def model(self):
        return WELCClassifier(num_classes=10, weight_mode="binary")

    def test_forward_shape(self, model):
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_no_nan_output(self, model):
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_efficiency_stats_keys(self, model):
        stats = model.compute_efficiency_stats()
        required = {"total_parameters", "binary_parameters", "binary_param_ratio", "estimated_speedup"}
        assert required.issubset(stats.keys())
        assert stats["binary_param_ratio"] > 0.0

    def test_ternary_forward(self):
        model = WELCClassifier(num_classes=10, weight_mode="ternary")
        x = torch.randn(2, 3, 32, 32)
        assert model(x).shape == (2, 10)
