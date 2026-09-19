"""
Cyber-Physical System dataset with:
1. Multi-sensor time-series simulation (normal + 4 fault types)
2. Process event log extraction (for process mining)
3. Feature engineering (time-domain, frequency-domain, process-aware)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ──────────────────────────────────────────────────────────────────────────────
# Fault simulation constants
# ──────────────────────────────────────────────────────────────────────────────

FAULT_TYPES = {
    0: "normal",
    1: "bearing_fault",
    2: "gear_fault",
    3: "rotor_imbalance",
    4: "sensor_drift",
}

FAULT_PARAMS = {
    "normal":           {"amp": 1.0,  "freq_mult": 1.0,  "noise": 0.05, "drift": 0.0},
    "bearing_fault":    {"amp": 2.5,  "freq_mult": 3.2,  "noise": 0.30, "drift": 0.0},
    "gear_fault":       {"amp": 1.8,  "freq_mult": 2.5,  "noise": 0.20, "drift": 0.01},
    "rotor_imbalance":  {"amp": 3.0,  "freq_mult": 1.0,  "noise": 0.15, "drift": 0.0},
    "sensor_drift":     {"amp": 1.2,  "freq_mult": 1.0,  "noise": 0.10, "drift": 0.5},
}


@dataclass
class CPSSimulatorConfig:
    num_sensors: int = 20
    window_size: int = 50
    sampling_rate: float = 100.0        # Hz
    fault_classes: int = 5
    rng_seed: int = 42


class CPSSignalSimulator:
    """
    Simulates multi-sensor vibration/temperature/current signals
    for an industrial CPS under normal and fault conditions.
    """

    def __init__(self, config: CPSSimulatorConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.rng_seed)

    def _base_signal(
        self, fault_type: str, length: int, sensor_idx: int
    ) -> np.ndarray:
        """Generate single-sensor signal for a given fault type."""
        t      = np.linspace(0, length / self.cfg.sampling_rate, length)
        params = FAULT_PARAMS[fault_type]
        phase  = self.rng.uniform(0, 2 * np.pi)

        # Fundamental frequency per sensor (simulates different measurement points)
        base_freq = 10.0 + sensor_idx * 2.5

        signal = (
            params["amp"] * np.sin(2 * np.pi * base_freq * params["freq_mult"] * t + phase)
            + 0.5 * np.sin(2 * np.pi * base_freq * 2 * t + phase * 0.5)
        )

        # Add harmonics for gear/bearing faults
        if fault_type in ("bearing_fault", "gear_fault"):
            fault_freq = base_freq * params["freq_mult"]
            for harm in [2, 3, 4]:
                signal += 0.3 / harm * np.sin(2 * np.pi * fault_freq * harm * t)

        # Rotor imbalance: amplitude modulation
        if fault_type == "rotor_imbalance":
            mod_freq  = 1.5
            signal   *= (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))

        # Sensor drift: slow linear trend
        drift = params["drift"] * t

        # White noise
        noise  = params["noise"] * self.rng.standard_normal(length)

        return (signal + drift + noise).astype(np.float32)

    def simulate_window(
        self, fault_type: str
    ) -> np.ndarray:
        """
        Simulate one window of multi-sensor data.
        Returns: (window_size, num_sensors) array
        """
        sensors = []
        for i in range(self.cfg.num_sensors):
            sig = self._base_signal(fault_type, self.cfg.window_size, i)
            sensors.append(sig)
        return np.stack(sensors, axis=1)   # (W, S)

    def simulate_dataset(
        self, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate full dataset. Returns X: (N, W, S), y: (N,) int labels."""
        X_list, y_list = [], []
        per_class = num_samples // self.cfg.fault_classes

        for cls_idx, fault_name in FAULT_TYPES.items():
            for _ in range(per_class):
                window = self.simulate_window(fault_name)
                X_list.append(window)
                y_list.append(cls_idx)

        X = np.stack(X_list).astype(np.float32)  # (N, W, S)
        y = np.array(y_list, dtype=np.int64)
        return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Process Event Log Generation
# ──────────────────────────────────────────────────────────────────────────────

class ProcessEventLogExtractor:
    """
    Converts CPS sensor windows into process event logs.

    Maps sensor threshold crossings and state transitions to
    discrete process activities, enabling process mining analysis.
    """

    ACTIVITY_THRESHOLDS = {
        "high_vibration":   2.0,
        "low_vibration":    0.2,
        "high_amplitude":   3.0,
        "oscillation":      1.5,
        "stable":           0.5,
    }

    def extract_events(
        self,
        window: np.ndarray,   # (W, S)
        case_id: str = "case_0",
        fault_label: int = 0,
    ) -> pd.DataFrame:
        """
        Convert a sensor window into a process event log trace.
        Returns a DataFrame with columns: case_id, activity, timestamp, sensor_id, value
        """
        W, S = window.shape
        events = []
        t_step = 1.0 / 100.0  # 100 Hz

        for t in range(W):
            for s in range(S):
                val = float(window[t, s])
                activity = self._classify_activity(val)
                events.append({
                    "case_id":    case_id,
                    "activity":   activity,
                    "timestamp":  pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=t * t_step),
                    "sensor_id":  s,
                    "value":      val,
                    "fault_label": fault_label,
                })

        return pd.DataFrame(events)

    def _classify_activity(self, value: float) -> str:
        abs_val = abs(value)
        if abs_val > self.ACTIVITY_THRESHOLDS["high_amplitude"]:
            return "CRITICAL_SPIKE"
        elif abs_val > self.ACTIVITY_THRESHOLDS["high_vibration"]:
            return "HIGH_VIBRATION"
        elif abs_val > self.ACTIVITY_THRESHOLDS["oscillation"]:
            return "OSCILLATION"
        elif abs_val > self.ACTIVITY_THRESHOLDS["stable"]:
            return "MODERATE_ACTIVITY"
        else:
            return "STABLE"

    def extract_process_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract process-aware features from a sensor window:
        - Activity frequency histogram (how often each activity appears)
        - State transition probabilities
        - Mean sojourn time per state
        """
        W, S = window.shape
        activities = ["STABLE", "MODERATE_ACTIVITY", "OSCILLATION", "HIGH_VIBRATION", "CRITICAL_SPIKE"]
        act_to_idx = {a: i for i, a in enumerate(activities)}
        n_acts = len(activities)

        freq_hist     = np.zeros((S, n_acts), dtype=np.float32)
        trans_matrix  = np.zeros((n_acts, n_acts), dtype=np.float32)

        for s in range(S):
            prev_act_idx = -1
            for t in range(W):
                act     = self._classify_activity(float(window[t, s]))
                act_idx = act_to_idx[act]
                freq_hist[s, act_idx] += 1
                if prev_act_idx >= 0:
                    trans_matrix[prev_act_idx, act_idx] += 1
                prev_act_idx = act_idx

        # Normalize
        freq_hist   = freq_hist / (W + 1e-8)
        row_sums    = trans_matrix.sum(axis=1, keepdims=True) + 1e-8
        trans_norm  = trans_matrix / row_sums

        # Flatten into feature vector
        return np.concatenate([
            freq_hist.flatten(),         # S × n_acts
            trans_norm.flatten(),        # n_acts × n_acts
        ]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CPSFaultDataset(Dataset):
    """
    Combined dataset: raw sensor windows + process-mining features.
    Returns both for the process-aware transformer model.
    """

    def __init__(
        self,
        num_samples: int = 20000,
        config: Optional[CPSSimulatorConfig] = None,
        augment: bool = True,
    ) -> None:
        self.config    = config or CPSSimulatorConfig()
        self.extractor = ProcessEventLogExtractor()

        simulator = CPSSignalSimulator(self.config)
        X_raw, y  = simulator.simulate_dataset(num_samples)

        # Normalize per sensor
        self.X_mean = X_raw.mean(axis=(0, 1), keepdims=True)
        self.X_std  = X_raw.std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm      = (X_raw - self.X_mean) / self.X_std

        # Extract process features for each sample
        proc_features = np.stack([
            self.extractor.extract_process_features(X_raw[i])
            for i in range(len(X_raw))
        ])

        self.X_sensor  = X_norm.astype(np.float32)          # (N, W, S)
        self.X_process = proc_features.astype(np.float32)   # (N, proc_feat_dim)
        self.y         = y

        # Optional: augment minority classes
        if augment:
            self._augment_minorities()

    def _augment_minorities(self) -> None:
        """Oversample fault classes to balance the dataset."""
        classes, counts = np.unique(self.y, return_counts=True)
        max_count = counts.max()
        aug_sensor, aug_proc, aug_y = [], [], []

        rng = np.random.default_rng(self.config.rng_seed)
        for cls, cnt in zip(classes, counts):
            if cnt < max_count:
                idxs    = np.where(self.y == cls)[0]
                extras  = max_count - cnt
                chosen  = rng.choice(idxs, size=extras, replace=True)
                # Add small Gaussian noise for augmentation
                aug_s   = self.X_sensor[chosen]  + rng.normal(0, 0.01, self.X_sensor[chosen].shape)
                aug_p   = self.X_process[chosen] + rng.normal(0, 0.001, self.X_process[chosen].shape)
                aug_sensor.append(aug_s.astype(np.float32))
                aug_proc.append(aug_p.astype(np.float32))
                aug_y.append(np.full(extras, cls, dtype=np.int64))

        if aug_sensor:
            self.X_sensor  = np.concatenate([self.X_sensor]  + aug_sensor)
            self.X_process = np.concatenate([self.X_process] + aug_proc)
            self.y         = np.concatenate([self.y]          + aug_y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "sensor_window":    torch.from_numpy(self.X_sensor[idx]),
            "process_features": torch.from_numpy(self.X_process[idx]),
            "label":            torch.tensor(self.y[idx], dtype=torch.long),
        }

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return self.X_sensor.shape[1], self.X_sensor.shape[2]   # (W, S)

    @property
    def process_feature_dim(self) -> int:
        return self.X_process.shape[1]

    @property
    def num_classes(self) -> int:
        return len(FAULT_TYPES)


def create_cps_dataloaders(
    num_samples: int = 20000,
    config: Optional[CPSSimulatorConfig] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, CPSFaultDataset]:
    """Create train/val/test loaders and return the full dataset for metadata."""
    dataset  = CPSFaultDataset(num_samples=num_samples, config=config)
    total    = len(dataset)
    test_sz  = int(total * test_split)
    val_sz   = int(total * val_split)
    train_sz = total - val_sz - test_sz

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(42),
    )

    kw = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        dataset,
    )
