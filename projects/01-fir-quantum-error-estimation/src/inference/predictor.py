"""Inference engine for FIR Quantum Error Estimator."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np, torch
from src.models.fir_estimator import FIRQuantumErrorEstimator
from src.utils.logger import get_logger
log = get_logger(__name__)
NOISE_TYPES = ["depolarizing", "bit_flip", "amplitude_damping", "phase_flip"]

class QuantumErrorPredictor:
    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load_model(Path(model_path))

    def _load_model(self, path: Path):
        if not path.exists(): raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        hp = ckpt["hyperparameters"]
        self.model = FIRQuantumErrorEstimator(
            input_dim=ckpt["feature_dim"], num_states=ckpt["num_states"],
            hidden_dims=hp.get("hidden_dims", [256,128,64]), filter_order=hp.get("filter_order",32))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device); self.model.eval()
        self.feature_dim = ckpt["feature_dim"]

    @torch.inference_mode()
    def predict_single(self, features: np.ndarray) -> dict:
        tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)
        return self._run_inference(tensor)[0]

    @torch.inference_mode()
    def predict_batch(self, features: np.ndarray) -> list[dict]:
        tensor = torch.from_numpy(features.astype(np.float32)).to(self.device)
        return self._run_inference(tensor)

    def _run_inference(self, tensor):
        outputs = self.model(tensor)
        results = []
        for i in range(tensor.shape[0]):
            dist = outputs["error_distribution"][i].cpu().numpy()
            mag  = float(outputs["error_magnitude"][i].cpu())
            cls_logits = outputs["noise_class_logits"][i].cpu()
            cls_idx = int(cls_logits.argmax())
            cls_prob = float(torch.softmax(cls_logits, dim=0).max())
            results.append({
                "error_distribution": dist.tolist(),
                "error_magnitude": mag,
                "noise_type": NOISE_TYPES[cls_idx],
                "noise_type_confidence": cls_prob,
                "noise_class_probs": {
                    nt: float(p) for nt, p in
                    zip(NOISE_TYPES, torch.softmax(cls_logits, dim=0).tolist())},
            })
        return results
