"""Tests for FIR Quantum Error Estimator."""
from __future__ import annotations
import numpy as np, pytest, torch
from src.data.quantum_dataset import QuantumCircuitConfig, QuantumErrorDataset, FIRFilterBank, QuantumNoiseSimulator, create_dataloaders
from src.models.fir_estimator import FIRQuantumErrorEstimator, LearnableFIRLayer

@pytest.fixture
def config(): return QuantumCircuitConfig(num_qubits=2, num_shots=64, fir_order=8)
@pytest.fixture
def small_dataset(config): return QuantumErrorDataset(num_samples=64, config=config)

class TestQuantumSimulator:
    def test_simulate_depolarizing(self, config):
        sim = QuantumNoiseSimulator(config)
        noisy, ideal, err = sim.simulate_circuit("depolarizing")
        assert noisy.shape == (2**config.num_qubits,)
        assert abs(noisy.sum() - 1.0) < 1e-4
        assert err >= 0

    @pytest.mark.parametrize("noise_type", ["bit_flip", "amplitude_damping", "phase_flip"])
    def test_noise_types(self, config, noise_type):
        sim = QuantumNoiseSimulator(config)
        noisy, ideal, err = sim.simulate_circuit(noise_type)
        assert noisy.shape[0] == 2**config.num_qubits
        assert np.isfinite(noisy).all()

class TestFIRFilterBank:
    def test_spectral_features_shape(self):
        fb = FIRFilterBank(order=8, num_filters=4)
        x = np.random.rand(16).astype(np.float32)
        features = fb.compute_spectral_features(x)
        assert features.shape == (4*3,)

class TestDataset:
    def test_length(self, small_dataset): assert len(small_dataset) == 64
    def test_keys(self, small_dataset):
        assert set(small_dataset[0].keys()) == {"features","targets","noise_type"}
    def test_loaders(self, small_dataset):
        t, v, te = create_dataloaders(small_dataset, batch_size=16, num_workers=0)
        assert next(iter(t))["features"].shape[0] <= 16

class TestLearnableFIRLayer:
    def test_shape(self):
        layer = LearnableFIRLayer(1, 8, filter_order=8)
        out = layer(torch.randn(4, 16))
        assert out.shape == (4, 8)

class TestFIREstimator:
    def test_output_keys(self):
        m = FIRQuantumErrorEstimator(input_dim=28, num_states=4, hidden_dims=[32,16], filter_order=8)
        out = m(torch.randn(4, 28))
        assert set(out.keys()) == {"error_distribution","error_magnitude","noise_class_logits"}

    def test_distribution_sums_to_one(self):
        m = FIRQuantumErrorEstimator(input_dim=28, num_states=4, hidden_dims=[32,16])
        out = m(torch.randn(8, 28))
        assert torch.allclose(out["error_distribution"].sum(-1), torch.ones(8), atol=1e-5)

    def test_magnitude_nonneg(self):
        m = FIRQuantumErrorEstimator(input_dim=28, num_states=4, hidden_dims=[32,16])
        out = m(torch.randn(4, 28))
        assert (out["error_magnitude"] >= 0).all()
