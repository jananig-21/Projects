"""Comprehensive tests for Process Mining Fault Diagnosis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.cps_simulator import (
    CPSConfig,
    CPSFaultSimulator,
    CPSEventLogGenerator,
    FAULT_TYPES,
    FAULT_LABEL_MAP,
)
from src.data.fault_dataset import CPSFaultDataset, create_fault_dataloaders
from src.models.process_fault_transformer import (
    TemporalEncoder,
    ProcessFeatureBranch,
    CrossModalFusion,
    ProcessFaultTransformer,
    SinusoidalPositionalEncoding,
)


# ─────────────────────────────────────────────────────────────────────────────
# CPS Simulator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCPSFaultSimulator:
    @pytest.fixture
    def config(self):
        return CPSConfig(num_sensors=4, num_actuators=2, window_size=20, seed=0)

    @pytest.fixture
    def simulator(self, config):
        return CPSFaultSimulator(config)

    def test_normal_window_shape(self, simulator, config):
        sensor, actuator, meta = simulator.simulate_window("normal")
        assert sensor.shape   == (config.window_size, config.num_sensors)
        assert actuator.shape == (config.window_size, config.num_actuators)

    @pytest.mark.parametrize("fault_type", FAULT_TYPES)
    def test_all_fault_types_run(self, simulator, config, fault_type):
        sensor, actuator, meta = simulator.simulate_window(fault_type)
        assert meta["fault_type"]  == fault_type
        assert meta["fault_label"] == FAULT_LABEL_MAP[fault_type]
        assert sensor.dtype   == np.float32
        assert actuator.dtype == np.float32

    def test_fault_label_map_complete(self):
        assert len(FAULT_LABEL_MAP) == len(FAULT_TYPES)
        assert set(FAULT_LABEL_MAP.keys()) == set(FAULT_TYPES)

    def test_output_is_finite(self, simulator):
        for ft in FAULT_TYPES:
            s, a, _ = simulator.simulate_window(ft)
            assert np.isfinite(s).all(), f"Sensor data has non-finite values for {ft}"
            assert np.isfinite(a).all(), f"Actuator data has non-finite values for {ft}"


# ───────────────────────────────────────────────────────────────────────────────
# Event Log Generator Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCPSEventLogGenerator:
    @pytest.fixture
    def config(self):
        return CPSConfig(num_sensors=4, num_actuators=2, window_size=20, seed=0)

    @pytest.fixture
    def event_gen(self, config):
        return CPSEventLogGenerator(config)

    @pytest.fixture
    def telemetry(self, config):
        sim = CPSFaultSimulator(config)
        s, a, _ = sim.simulate_window("normal")
        return s, a

    def test_event_log_has_required_columns(self, event_gen, telemetry):
        s, a = telemetry
        df = event_gen.generate_event_log(s, a, "normal", case_id="test")
        assert "concept:name"      in df.columns
        assert "case:concept:name" in df.columns
        assert "time:timestamp"    in df.columns

    def test_event_log_nonempty(self, event_gen, telemetry):
        s, a = telemetry
        df = event_gen.generate_event_log(s, a, "sensor_drift", case_id="test2")
        assert len(df) > 0

    def test_process_features_shape(self, event_gen, telemetry):
        s, a = telemetry
        df = event_gen.generate_event_log(s, a, "oscillation", case_id="feat_test")
        pf = event_gen.extract_process_features(df)
        assert pf.ndim  == 1
        assert pf.dtype == np.float32
        assert np.isfinite(pf).all()

    def test_process_features_consistent_size(self, event_gen, config):
        """All fault types should produce same-length feature vectors."""
        sim = CPSFaultSimulator(config)
        sizes = []
        for ft in FAULT_TYPES:
            s, a, _ = sim.simulate_window(ft)
            df = event_gen.generate_event_log(s, a, ft)
            pf = event_gen.extract_process_features(df)
            sizes.append(len(pf))
        assert len(set(sizes)) == 1, f"Inconsistent process feature sizes: {sizes}"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCPSFaultDataset:
    @pytest.fixture
    def small_dataset(self):
        cfg = CPSConfig(num_sensors=4, num_actuators=2, window_size=20, seed=0)
        return CPSFaultDataset(num_samples=80, config=cfg, use_process_features=True)

    def test_dataset_length(self, small_dataset):
        assert len(small_dataset) == 80

    def test_item_keys(self, small_dataset):
        item = small_dataset[0]
        assert "telemetry"        in item
        assert "label"            in item
        assert "process_features" in item

    def test_telemetry_shape(self, small_dataset):
        item = small_dataset[0]
        W = small_dataset.config.window_size
        C = small_dataset.num_channels
        assert item["telemetry"].shape == (W, C)

    def test_labels_in_range(self, small_dataset):
        for i in range(len(small_dataset)):
            label = small_dataset[i]["label"].item()
            assert 0 <= label < len(FAULT_TYPES)

    def test_all_fault_classes_present(self, small_dataset):
        labels = [small_dataset[i]["label"].item() for i in range(len(small_dataset))]
        assert len(set(labels)) == len(FAULT_TYPES), "Dataset must contain all fault classes"

    def test_class_weights_shape(self, small_dataset):
        weights = small_dataset.get_class_weights()
        assert weights.shape == (len(FAULT_TYPES),)
        assert (weights > 0).all()

    def test_no_nan_in_tensors(self, small_dataset):
        for i in range(min(10, len(small_dataset))):
            item = small_dataset[i]
            assert torch.isfinite(item["telemetry"]).all()
            assert torch.isfinite(item["process_features"]).all()

    def test_create_dataloaders(self):
        cfg = CPSConfig(num_sensors=4, num_actuators=2, window_size=20)
        train_l, val_l, test_l, ds = create_fault_dataloaders(
            num_samples=64, config=cfg, batch_size=16, num_workers=0
        )
        batch = next(iter(train_l))
        assert "telemetry" in batch
        assert "label"     in batch
        assert batch["telemetry"].shape[0] <= 16


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSinusoidalPositionalEncoding:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(4, 50, 64)
        out = pe(x)
        assert out.shape == (4, 50, 64)

    def test_finite_output(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=50)
        x = torch.randn(2, 30, 32)
        assert torch.isfinite(pe(x)).all()


class TestTemporalEncoder:
    @pytest.fixture
    def encoder(self):
        return TemporalEncoder(
            num_channels=6,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            window_size=20,
        )

    def test_output_shape(self, encoder):
        x = torch.randn(4, 20, 6)
        out = encoder(x)
        assert out.shape == (4, 32)

    def test_output_finite(self, encoder):
        x = torch.randn(4, 20, 6)
        assert torch.isfinite(encoder(x)).all()

    def test_gradient_flow(self, encoder):
        x = torch.randn(2, 20, 6)
        loss = encoder(x).sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestProcessFeatureBranch:
    def test_output_shape(self):
        branch = ProcessFeatureBranch(process_feature_dim=30, d_model=64)
        x = torch.randn(4, 30)
        assert branch(x).shape == (4, 64)

    def test_output_finite(self):
        branch = ProcessFeatureBranch(process_feature_dim=20, d_model=32)
        x = torch.randn(8, 20)
        assert torch.isfinite(branch(x)).all()


class TestCrossModalFusion:
    def test_output_shape(self):
        fusion = CrossModalFusion(d_model=64, nhead=4)
        t = torch.randn(4, 64)
        p = torch.randn(4, 64)
        out = fusion(t, p)
        assert out.shape == (4, 64)

    def test_residual_connection(self):
        """Output should differ from pure temporal input (fusion adds info)."""
        fusion = CrossModalFusion(d_model=32, nhead=4)
        t = torch.randn(2, 32)
        p = torch.randn(2, 32)
        out = fusion(t, p)
        assert not torch.allclose(out, t, atol=1e-5)


class TestProcessFaultTransformer:
    @pytest.fixture
    def model(self):
        return ProcessFaultTransformer(
            num_channels=6,
            num_fault_classes=8,
            process_feature_dim=30,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=64,
            window_size=20,
            use_process_features=True,
        )

    def test_forward_with_process_features(self, model):
        tel = torch.randn(4, 20, 6)
        pf  = torch.randn(4, 30)
        out = model(tel, pf)
        assert "logits"    in out
        assert "embedding" in out
        assert out["logits"].shape    == (4, 8)
        assert out["embedding"].shape == (4, 32)

    def test_forward_without_process_features(self, model):
        tel = torch.randn(4, 20, 6)
        out = model(tel, None)
        assert out["logits"].shape == (4, 8)

    def test_logits_finite(self, model):
        tel = torch.randn(4, 20, 6)
        pf  = torch.randn(4, 30)
        out = model(tel, pf)
        assert torch.isfinite(out["logits"]).all()
        assert torch.isfinite(out["embedding"]).all()

    def test_no_process_features_mode(self):
        model = ProcessFaultTransformer(
            num_channels=6,
            num_fault_classes=4,
            process_feature_dim=0,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            dim_feedforward=64,
            window_size=10,
            use_process_features=False,
        )
        tel = torch.randn(2, 10, 6)
        out = model(tel, None)
        assert out["logits"].shape == (2, 4)

    def test_gradient_flow(self, model):
        tel  = torch.randn(2, 20, 6)
        pf   = torch.randn(2, 30)
        loss = model(tel, pf)["logits"].sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0
        for g in grads:
            assert torch.isfinite(g).all()

    def test_parameter_count_reasonable(self, model):
        n = sum(p.numel() for p in model.parameters())
        assert n > 1000
        print(f"Total parameters: {n:,}")
