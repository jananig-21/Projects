"""Inference engine for WELC image classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.models.welc_layer import WELCClassifier
from src.utils.logger import get_logger

log = get_logger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_TRANSFORM = transforms.Compose([
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])


class WELCPredictor:
    """Production predictor for WELC image classifier."""

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
        hp = ckpt["hyperparameters"]

        self.model = WELCClassifier(
            num_classes=hp.get("num_classes", 10),
            weight_mode=hp.get("weight_mode", "binary"),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.efficiency_stats = ckpt.get("efficiency_stats", {})
        log.info("WELC model loaded", device=str(self.device))

    def _preprocess(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Expects image_array: (H, W, C) uint8 [0-255] or float [0-1].
        Returns (1, C, H, W) normalized tensor.
        """
        img = torch.from_numpy(image_array).float()
        if img.max() > 1.0:
            img = img / 255.0
        img = img.permute(2, 0, 1)  # HWC → CHW
        img = _TRANSFORM(img)
        return img.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_single(self, image_array: np.ndarray) -> dict:
        tensor = self._preprocess(image_array)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        return {
            "class_index": pred_idx,
            "class_name": CIFAR10_CLASSES[pred_idx] if pred_idx < len(CIFAR10_CLASSES) else str(pred_idx),
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {
                CIFAR10_CLASSES[i]: float(p) for i, p in enumerate(probs)
            },
        }

    @torch.inference_mode()
    def predict_batch(self, image_arrays: list[np.ndarray]) -> list[dict]:
        tensors = torch.cat([self._preprocess(img) for img in image_arrays], dim=0)
        logits = self.model(tensors)
        probs_all = F.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for probs in probs_all:
            pred_idx = int(probs.argmax())
            results.append({
                "class_index": pred_idx,
                "class_name": CIFAR10_CLASSES[pred_idx] if pred_idx < len(CIFAR10_CLASSES) else str(pred_idx),
                "confidence": float(probs[pred_idx]),
                "class_probabilities": {
                    CIFAR10_CLASSES[i]: float(p) for i, p in enumerate(probs)
                },
            })
        return results

    def get_logic_circuit_export(self) -> dict:
        """Export all WELC layers as logic circuit representations."""
        circuits = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "get_logic_circuit_representation"):
                circuits[name] = module.get_logic_circuit_representation()
        return {
            "layers": circuits,
            "efficiency_stats": self.efficiency_stats,
        }
