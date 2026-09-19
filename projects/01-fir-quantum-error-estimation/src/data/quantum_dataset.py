"""Quantum circuit simulation and dataset generation for FIR error estimation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset, DataLoader, random_split

@dataclass
class QuantumCircuitConfig:
    num_qubits: int = 4
    num_shots: int = 1024
    noise_prob: float = 0.01
    fir_order: int = 32

NOISE_TYPES = ["depolarizing", "bit_flip", "amplitude_damping", "phase_flip"]

class QuantumNoiseSimulator:
    def __init__(self, config: QuantumCircuitConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(42)

    def simulate_circuit(self, noise_type: str = "depolarizing"):
        dim = 2 ** self.config.num_qubits
        ideal_state = np.ones(dim) / np.sqrt(dim)
        ideal_probs = np.abs(ideal_state) ** 2

        if noise_type == "depolarizing":
            noise = self.rng.normal(0, self.config.noise_prob, dim)
            noisy_probs = np.abs(ideal_probs + noise)
            noisy_probs /= noisy_probs.sum() + 1e-12
        elif noise_type == "bit_flip":
            mask = self.rng.random(dim) < self.config.noise_prob
            noisy_probs = ideal_probs.copy()
            noisy_probs[mask] = 1.0 - noisy_probs[mask]
            noisy_probs = np.abs(noisy_probs)
            noisy_probs /= noisy_probs.sum() + 1e-12
        elif noise_type == "amplitude_damping":
            gamma = self.config.noise_prob
            noisy_probs = ideal_probs * (1 - gamma) + self.rng.exponential(gamma * 0.1, dim)
            noisy_probs = np.clip(noisy_probs, 0, 1)
            noisy_probs /= noisy_probs.sum() + 1e-12
        else:
            phase_flip = self.rng.random(dim) < self.config.noise_prob
            noisy_probs = ideal_probs.copy()
            noisy_probs[phase_flip] *= -1
            noisy_probs = np.abs(noisy_probs)
            noisy_probs /= noisy_probs.sum() + 1e-12

        counts = self.rng.multinomial(self.config.num_shots, noisy_probs)
        shot_probs = counts / self.config.num_shots
        error_magnitude = float(np.linalg.norm(ideal_probs - shot_probs))
        return shot_probs.astype(np.float32), ideal_probs.astype(np.float32), error_magnitude

class FIRFilterBank:
    def __init__(self, order: int = 32, num_filters: int = 8) -> None:
        self.order = order
        self.num_filters = num_filters
        self.filters = self._design_filter_bank()

    def _design_filter_bank(self):
        filters = []
        for i in range(self.num_filters):
            cutoff = (i + 1) / (self.num_filters + 1)
            coeffs = signal.firwin(self.order + 1, cutoff, window="hamming", pass_zero=True)
            filters.append(coeffs.astype(np.float32))
        return filters

    def apply(self, x):
        return np.stack([signal.lfilter(f, 1.0, x) for f in self.filters])

    def compute_spectral_features(self, x):
        filtered = self.apply(x)
        return np.concatenate([
            np.sum(filtered**2, axis=1),
            np.mean(np.abs(filtered), axis=1),
            np.var(filtered, axis=1),
        ]).astype(np.float32)

class QuantumErrorDataset(Dataset):
    def __init__(self, num_samples: int = 50000, config: Optional[QuantumCircuitConfig] = None):
        self.num_samples = num_samples
        self.config = config or QuantumCircuitConfig()
        self.simulator = QuantumNoiseSimulator(self.config)
        self.filter_bank = FIRFilterBank(order=self.config.fir_order)
        self._generate_data()

    def _generate_data(self):
        features_list, targets_list, labels_list = [], [], []
        for i in range(self.num_samples):
            noise_type = NOISE_TYPES[i % len(NOISE_TYPES)]
            noisy, ideal, err = self.simulator.simulate_circuit(noise_type)
            spectral = self.filter_bank.compute_spectral_features(noisy)
            feature = np.concatenate([noisy, spectral]).astype(np.float32)
            target = np.append(ideal, err).astype(np.float32)
            features_list.append(feature)
            targets_list.append(target)
            labels_list.append(NOISE_TYPES.index(noise_type))
        self.features_arr = np.stack(features_list)
        self.targets_arr = np.stack(targets_list)
        self.labels_arr = np.array(labels_list, dtype=np.int64)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.features_arr[idx]),
            "targets":  torch.from_numpy(self.targets_arr[idx]),
            "noise_type": torch.tensor(self.labels_arr[idx]),
        }
    @property
    def feature_dim(self): return self.features_arr.shape[1]
    @property
    def target_dim(self): return self.targets_arr.shape[1]

def create_dataloaders(dataset, val_split=0.15, test_split=0.15, batch_size=64, num_workers=4):
    total = len(dataset)
    test_size = int(total * test_split)
    val_size  = int(total * val_split)
    train_size = total - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs))
