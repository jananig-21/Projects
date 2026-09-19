"""PyTorch Dataset for CPS fault diagnosis with process mining features."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.cps_simulator import (
    CPSConfig,
    CPSFaultSimulator,
    CPSEventLogGenerator,
    FAULT_TYPES,
    FAULT_LABEL_MAP,
)


class CPSFaultDataset(Dataset):
    """
    Dataset combining:
    1. Raw CPS telemetry windows (sensor + actuator time series)
    2. Process mining features (event log statistics)
    3. Fault labels
    """

    def __init__(
        self,
        num_samples: int = 20000,
        config: Optional[CPSConfig] = None,
        use_process_features: bool = True,
        augment: bool = True,
    ) -> None:
        self.config = config or CPSConfig()
        self.use_process_features = use_process_features
        self.augment = augment
        self.num_samples = num_samples

        self.simulator = CPSFaultSimulator(self.config)
        self.event_gen  = CPSEventLogGenerator(self.config)

        self._generate()

    def _generate(self) -> None:
        """Generate all samples."""
        sensor_windows: list[np.ndarray]    = []
        actuator_windows: list[np.ndarray]  = []
        process_features: list[np.ndarray]  = []
        labels: list[int]                   = []

        for i in range(self.num_samples):
            ft = FAULT_TYPES[i % len(FAULT_TYPES)]
            sensor, actuator, meta = self.simulator.simulate_window(fault_type=ft)

            sensor_windows.append(sensor)       # (W, num_sensors)
            actuator_windows.append(actuator)   # (W, num_actuators)
            labels.append(meta["fault_label"])

            if self.use_process_features:
                ev_log = self.event_gen.generate_event_log(sensor, actuator, ft, case_id=str(i))
                pf = self.event_gen.extract_process_features(ev_log)
                process_features.append(pf)

        self.sensor_data    = np.stack(sensor_windows).astype(np.float32)
        self.actuator_data  = np.stack(actuator_windows).astype(np.float32)
        self.labels_arr     = np.array(labels, dtype=np.int64)

        if self.use_process_features and process_features:
            self.process_feat = np.stack(process_features).astype(np.float32)
            # Normalize process features
            self.pm_mean = self.process_feat.mean(axis=0)
            self.pm_std  = self.process_feat.std(axis=0) + 1e-8
            self.process_feat = (self.process_feat - self.pm_mean) / self.pm_std
        else:
            self.process_feat = None

        # Combine sensor + actuator into single telemetry tensor: (N, W, S+A)
        self.telemetry = np.concatenate(
            [self.sensor_data, self.actuator_data], axis=-1
        )

        # Normalize per-channel
        self.tel_mean = self.telemetry.mean(axis=(0, 1), keepdims=True)
        self.tel_std  = self.telemetry.std(axis=(0, 1), keepdims=True) + 1e-8
        self.telemetry = (self.telemetry - self.tel_mean) / self.tel_std

    @property
    def num_channels(self) -> int:
        return self.config.num_sensors + self.config.num_actuators

    @property
    def process_feature_dim(self) -> int:
        return self.process_feat.shape[1] if self.process_feat is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "telemetry": torch.from_numpy(self.telemetry[idx]),   # (W, S+A)
            "label": torch.tensor(self.labels_arr[idx], dtype=torch.long),
        }
        if self.process_feat is not None:
            item["process_features"] = torch.from_numpy(self.process_feat[idx])

        # Data augmentation: additive Gaussian noise + random time shift
        if self.augment and self.training if hasattr(self, "training") else False:
            noise = torch.randn_like(item["telemetry"]) * 0.01
            item["telemetry"] = item["telemetry"] + noise

        return item

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training."""
        counts = np.bincount(self.labels_arr, minlength=len(FAULT_TYPES))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(FAULT_TYPES)
        return torch.from_numpy(weights.astype(np.float32))


def create_fault_dataloaders(
    num_samples: int = 20000,
    config: Optional[CPSConfig] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    use_process_features: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, CPSFaultDataset]:
    dataset = CPSFaultDataset(
        num_samples=num_samples,
        config=config,
        use_process_features=use_process_features,
    )
    total = len(dataset)
    test_size  = int(total * test_split)
    val_size   = int(total * val_split)
    train_size = total - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs),
        dataset,
    )
