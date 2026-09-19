"""Inference engine for KAN model with symbolic interpretation."""

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
    Production KAN inference with:
    - Single / batch prediction
    - Symbolic representation export
    - Feature importance via gradient attribution
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

        self.input_dim      = ckpt["input_dim"]
        self.output_dim     = ckpt["output_dim"]
        self.symbolic_repr  = ckpt.get("symbolic_repr", [])

        log.info(
            "KAN model loaded",
            device=str(self.device),
            input_dim=self.input_dim,
            layer_sizes=ckpt["layer_sizes"],
        )

    @torch.inference_mode()
    def predict_single(self, x: np.ndarray) -> dict:
        """Single-sample prediction."""
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[0]}")
        tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
        out    = self.model(tensor)
        return {
            "prediction": out.squeeze().cpu().tolist(),
            "input_dim":  self.input_dim,
        }

    @torch.inference_mode()
    def predict_batch(self, X: np.ndarray) -> list[dict]:
        """Batch prediction."""
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected (N, {self.input_dim}) array, got {X.shape}")
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        outs   = self.model(tensor)
        return [{"prediction": float(v)} for v in outs.squeeze(-1).cpu().tolist()]

    def feature_importance(self, X: np.ndarray) -> dict:
        """
        Gradient-based feature attribution: d(output)/d(input).
        Averaged over all samples in X.
        """
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        tensor.requires_grad_(True)
        out    = self.model(tensor)
        out.sum().backward()
        grads  = tensor.grad.abs().mean(dim=0).cpu().numpy()
        return {
            "feature_importance": grads.tolist(),
            "most_important_feature": int(grads.argmax()),
        }

    def get_symbolic_representation(self) -> list[dict]:
        """Return the spline coefficient structure for symbolic analysis."""
        return self.symbolic_repr

    def export_for_hardware(self) -> dict:
        """
        Export KAN as piecewise-linear approximation suitable for
        hardware implementation (FPGA/ASIC lookup tables).
        """
        lut_export = []
        x_eval = torch.linspace(-1, 1, 64, device=self.device)

        with torch.inference_mode():
            for layer_idx, layer in enumerate(self.model.kan_layers):
                layer_luts = []
                for out_idx in range(layer.out_features):
                    for in_idx in range(layer.in_features):
                        # Sample single-input response
                        x_single = torch.zeros(64, layer.in_features, device=self.device)
                        x_single[:, in_idx] = x_eval
                        response = layer(x_single)[:, out_idx].cpu().numpy()
                        layer_luts.append({
                            "out_idx": out_idx,
                            "in_idx":  in_idx,
                            "x_values": x_eval.cpu().tolist(),
                            "y_values": response.tolist(),
                        })
                lut_export.append({
                    "layer_index": layer_idx,
                    "lookup_tables": layer_luts,
                })
        return {"lut_export": lut_export, "lut_resolution": 64}
