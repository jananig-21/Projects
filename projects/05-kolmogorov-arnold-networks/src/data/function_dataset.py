"""
Synthetic function approximation datasets for KAN evaluation.

Includes benchmark functions from symbolic regression literature,
plus real-world datasets (California Housing, Iris, etc.).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark functions for symbolic regression
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_FUNCTIONS = {
    # KAT Theorem: f(x1,x2) = exp(sin(pi*x1) + x2^2)
    "kat_2d":       lambda x: np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2),

    # Feynman physics: pendulum period
    "feynman_pendulum": lambda x: 2 * np.pi * np.sqrt(x[:, 0] / 9.8),

    # Hartmann-style multivariate
    "hartmann_2d":  lambda x: -np.exp(-(x[:, 0] - 0.5) ** 2 - (x[:, 1] - 0.5) ** 2),

    # Composition structure: ideal for KAN
    "composition":  lambda x: np.sin(x[:, 0] ** 2) + np.cos(x[:, 1] ** 2) + x[:, 0] * x[:, 1],

    # Smooth polynomial
    "poly_4d":      lambda x: x[:, 0] ** 3 - 2 * x[:, 1] ** 2 + np.sin(x[:, 2]) + np.exp(-x[:, 3] ** 2),

    # Symbolic regression benchmark (Nguyen-7)
    "nguyen7":      lambda x: np.log(x[:, 0] + 1) + np.log(x[:, 0] ** 2 + 1),

    # Classification boundary: concentric circles
    "circles":      lambda x: (x[:, 0] ** 2 + x[:, 1] ** 2 < 0.5).astype(float),
}


class FunctionApproximationDataset(Dataset):
    """
    Synthetic dataset for benchmarking KAN vs MLP function approximation.
    """

    def __init__(
        self,
        function_name: str = "kat_2d",
        num_samples: int = 10000,
        input_dim: Optional[int] = None,
        noise_std: float = 0.1,
        input_range: tuple[float, float] = (-1.0, 1.0),
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        fn = BENCHMARK_FUNCTIONS[function_name]

        # Determine input dim from function
        test_x = np.ones((1, 10))
        for dim in range(1, 11):
            try:
                fn(test_x[:, :dim])
                if input_dim is None:
                    input_dim = dim
                break
            except (IndexError, Exception):
                continue

        self.input_dim = input_dim or 2

        # Generate samples
        X = rng.uniform(input_range[0], input_range[1], (num_samples, self.input_dim)).astype(np.float32)
        y = fn(X).astype(np.float32)

        # Add Gaussian noise
        y += rng.normal(0, noise_std, y.shape).astype(np.float32)

        # Normalize
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std  = y.std() + 1e-8

        self.X = (X - self.X_mean) / self.X_std
        self.y = (y - self.y_mean) / self.y_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.X[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class RealWorldDataset(Dataset):
    """Real-world tabular datasets for KAN benchmarking."""

    def __init__(self, name: str = "california_housing") -> None:
        X, y = self._load(name)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.input_dim = X.shape[1]
        self.is_classification = name in {"iris", "wine", "breast_cancer"}

    def _load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        from sklearn import datasets
        loaders = {
            "california_housing": lambda: datasets.fetch_california_housing(return_X_y=True),
            "iris":               lambda: datasets.load_iris(return_X_y=True),
            "wine":               lambda: datasets.load_wine(return_X_y=True),
            "breast_cancer":      lambda: datasets.load_breast_cancer(return_X_y=True),
        }
        X, y = loaders[name]()
        return X, y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.X[idx], "y": self.y[idx]}


def create_kan_dataloaders(
    function_name: str = "kat_2d",
    num_samples: int = 10000,
    noise_std: float = 0.1,
    batch_size: int = 256,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train/val/test loaders and return input_dim."""
    dataset = FunctionApproximationDataset(
        function_name=function_name,
        num_samples=num_samples,
        noise_std=noise_std,
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
        dataset.input_dim,
    )
