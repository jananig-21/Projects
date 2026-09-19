"""Unit tests for Kolmogorov-Arnold Networks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.kan_layer import (
    KAN, KANLayer, KANResidual, KANResidualBlock, b_spline_basis,
)


class TestBSplineBasis:
    def test_output_shape(self):
        in_f, grid_size, k = 4, 5, 3
        h    = 2.0 / grid_size
        grid = torch.linspace(-1 - k * h, 1 + k * h, grid_size + 2 * k + 1)
        grid = grid.unsqueeze(0).expand(in_f, -1).contiguous()
        x    = torch.rand(8, in_f) * 2 - 1
        bases = b_spline_basis(x, grid, k)
        assert bases.shape == (8, in_f, grid_size + k)

    def test_non_negative(self):
        in_f, grid_size, k = 2, 5, 3
        h    = 2.0 / grid_size
        grid = torch.linspace(-1 - k * h, 1 + k * h, grid_size + 2 * k + 1)
        grid = grid.unsqueeze(0).expand(in_f, -1).contiguous()
        x    = torch.rand(16, in_f) * 2 - 1
        bases = b_spline_basis(x, grid, k)
        assert (bases >= -1e-6).all()


class TestKANLayer:
    @pytest.fixture
    def layer(self):
        return KANLayer(in_features=4, out_features=8, grid_size=5, spline_order=3)

    def test_output_shape(self, layer):
        assert layer(torch.randn(16, 4)).shape == (16, 8)

    def test_gradient_flows(self, layer):
        x = torch.randn(8, 4, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert layer.weight_spline.grad is not None

    def test_finite_output(self, layer):
        assert torch.isfinite(layer(torch.randn(16, 4))).all()

    @pytest.mark.parametrize("act", ["silu", "relu", "gelu", "tanh"])
    def test_activations(self, act):
        layer = KANLayer(4, 6, grid_size=3, spline_order=2, base_activation=act)
        assert layer(torch.randn(4, 4)).shape == (4, 6)

    def test_wrong_dim_raises(self, layer):
        with pytest.raises(AssertionError):
            layer(torch.randn(4, 10))


class TestKAN:
    @pytest.fixture
    def model(self):
        return KAN(layer_sizes=[2, 8, 8, 1], grid_size=3, spline_order=2)

    def test_forward_shape(self, model):
        assert model(torch.randn(16, 2)).shape == (16, 1)

    def test_finite_output(self, model):
        assert torch.isfinite(model(torch.randn(32, 2))).all()

    def test_parameter_count(self, model):
        p = model.count_parameters()
        assert p["total"] > 0 and p["trainable"] == p["total"]

    def test_symbolic_repr(self, model):
        r = model.get_symbolic_representation()
        assert len(r) == len(model.kan_layers)
        for lr in r:
            assert "layer_index" in lr and "weight_spline_norm" in lr

    def test_update_grids(self, model):
        model.update_grids(torch.randn(64, 2))
        assert torch.isfinite(model(torch.randn(4, 2))).all()

    def test_deep_kan(self):
        m = KAN(layer_sizes=[4, 16, 16, 16, 2], grid_size=5, spline_order=3)
        assert m(torch.randn(8, 4)).shape == (8, 2)


class TestKANResidual:
    def test_forward_shape(self):
        m = KANResidual(input_dim=4, hidden_dim=16, output_dim=1, num_blocks=2)
        assert m(torch.randn(8, 4)).shape == (8, 1)

    def test_block_preserves_shape(self):
        b = KANResidualBlock(features=16, grid_size=3, spline_order=2)
        assert b(torch.randn(4, 16)).shape == (4, 16)


class TestFunctionDataset:
    @pytest.mark.parametrize("fn", ["kat_2d", "composition", "hartmann_2d"])
    def test_item(self, fn):
        from src.data.function_dataset import FunctionApproximationDataset
        ds   = FunctionApproximationDataset(function_name=fn, num_samples=50)
        item = ds[0]
        assert item["x"].dtype == torch.float32 and item["y"].dtype == torch.float32

    def test_dataloader(self):
        from src.data.function_dataset import create_kan_dataloaders
        train, val, test, idim = create_kan_dataloaders(
            function_name="kat_2d", num_samples=200, batch_size=32, num_workers=0
        )
        batch = next(iter(train))
        assert batch["x"].shape == (32, idim) and batch["y"].shape == (32,)
