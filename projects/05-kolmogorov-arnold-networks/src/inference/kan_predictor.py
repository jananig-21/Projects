"""Inference engine for KAN models with symbolic extraction support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.models.kan_layer import KAN
from src.utils.logger import get_logger

log = get_logger(__name__)


class KANPredictor:
    """
    Production predictor for trained KAN models.
    Supports single / batch inference and symbolic representation export.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load(Path(model_path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        hp   = ckpt["hyperparameters"]

        self.model = KAN(
            layer_sizes=ckpt["layer_sizes"],
            grid_size=hp.get("grid_size", 5),
            spline_order=hp.get("spline_order", 3),
            base_activation=hp.get("base_activation", "silu"),
            grid_range=tuple(hp.get("grid_range", [-1.0, 1.0])),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.input_dim  = ckpt["layer_sizes"][0]
        self.output_dim = ckpt["layer_sizes"][-1]
        self.symbolic_representation = ckpt.get("symbolic_representation", [])
        self.parameter_count = ckpt.get("parameter_count", {})
        log.info(
            "KAN model loaded",
            layer_sizes=ckpt["layer_sizes"],
            device=str(self.device),
        )

    @torch.inference_mode()
    def predict_single(self, x: np.ndarray) -> dict:
        """Single sample prediction."""
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[0]}")
        tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
        out = self.model(tensor).squeeze(0).cpu().numpy()
        return {
            "input": x.tolist(),
            "output": out.tolist() if out.ndim > 0 else [float(out)],
            "output_scalar": float(out.mean()),
        }

    @torch.inference_mode()
    def predict_batch(self, X: np.ndarray) -> list[dict]:
        """Batch prediction."""
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected (N, {self.input_dim}) input, got {X.shape}")
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        outs = self.model(tensor).cpu().numpy()
        return [
            {"input": X[i].tolist(), "output": outs[i].tolist(), "output_scalar": float(outs[i].mean())}
            for i in range(len(X))
        ]

    def get_symbolic_representation(self) -> dict:
        """
        Return learned spline edge weights for interpretability analysis.
        High-magnitude edges correspond to important input-output relationships.
        """
        sym = self.model.get_symbolic_representation()
        # Compute edge importance scores
        for layer_info in sym:
            norms = np.array(layer_info["weight_spline_norm"])
            layer_info["edge_importance"] = norms.tolist()
            layer_info["top_edges"] = np.argsort(norms.flatten())[-5:][::-1].tolist()
        return {
            "layers": sym,
            "parameter_count": self.parameter_count,
            "architecture": self.model.layer_sizes,
        }

    def compare_with_mlp(self, X: np.ndarray) -> dict:
        """
        Benchmark KAN vs equivalent MLP on given data.
        Returns parameter counts and prediction comparison.
        """
        import torch.nn as nn
        input_dim = self.input_dim
        hidden = 64
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.output_dim),
        )
        mlp_params = sum(p.numel() for p in mlp.parameters())
        kan_params = self.parameter_count.get("total", 0)

        return {
            "kan_parameters": kan_params,
            "mlp_parameters": mlp_params,
            "parameter_ratio": kan_params / max(mlp_params, 1),
            "kan_predictions": self.predict_batch(X[:10]),
        }
