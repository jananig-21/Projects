"""Inference engine for Process Mining Fault Diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.data.cps_simulator import (
    CPSConfig,
    CPSFaultSimulator,
    CPSEventLogGenerator,
    FAULT_TYPES,
)
from src.models.process_fault_transformer import ProcessFaultTransformer
from src.utils.logger import get_logger

log = get_logger(__name__)


class FaultDiagnosisPredictor:
    """
    Production inference for CPS fault diagnosis.
    Accepts raw sensor/actuator windows, extracts process features,
    and returns fault type classification with confidence.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load(Path(model_path))

        # CPS processing utilities
        self.cps_config = CPSConfig()
        self.event_gen  = CPSEventLogGenerator(self.cps_config)

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        hp   = ckpt["hyperparameters"]

        self.model = ProcessFaultTransformer(
            num_channels=ckpt["num_channels"],
            num_fault_classes=hp.get("num_fault_classes", 8),
            process_feature_dim=ckpt.get("process_feature_dim", 30),
            d_model=hp.get("d_model", 128),
            nhead=hp.get("nhead", 8),
            num_encoder_layers=hp.get("num_encoder_layers", 4),
            dim_feedforward=hp.get("dim_feedforward", 512),
            window_size=hp.get("window_size", 50),
            use_process_features=hp.get("use_process_features", True),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Normalization stats
        self.tel_mean = np.array(ckpt["tel_mean"], dtype=np.float32)
        self.tel_std  = np.array(ckpt["tel_std"],  dtype=np.float32)
        self.pm_mean  = np.array(ckpt["pm_mean"],  dtype=np.float32) if ckpt.get("pm_mean") else None
        self.pm_std   = np.array(ckpt["pm_std"],   dtype=np.float32) if ckpt.get("pm_std")  else None

        self.num_channels         = ckpt["num_channels"]
        self.process_feature_dim  = ckpt.get("process_feature_dim", 30)
        log.info("Fault diagnosis model loaded", device=str(self.device))

    def _normalize_telemetry(self, tel: np.ndarray) -> np.ndarray:
        return (tel - self.tel_mean) / self.tel_std

    def _normalize_process_features(self, pf: np.ndarray) -> np.ndarray:
        if self.pm_mean is not None:
            return (pf - self.pm_mean) / self.pm_std
        return pf

    def _extract_process_features(
        self,
        sensor_data: np.ndarray,
        actuator_data: np.ndarray,
    ) -> np.ndarray:
        ev_log = self.event_gen.generate_event_log(sensor_data, actuator_data, "unknown")
        return self.event_gen.extract_process_features(ev_log)

    @torch.inference_mode()
    def predict_single(
        self,
        sensor_data: np.ndarray,
        actuator_data: np.ndarray,
    ) -> dict:
        """
        Diagnose fault type from a single CPS window.

        sensor_data:   (window_size, num_sensors)
        actuator_data: (window_size, num_actuators)
        """
        # Build telemetry window
        tel = np.concatenate([sensor_data, actuator_data], axis=-1)
        tel_norm = self._normalize_telemetry(tel)
        tel_t = torch.from_numpy(tel_norm).unsqueeze(0).to(self.device)

        # Extract & normalize process features
        pf = self._extract_process_features(sensor_data, actuator_data)
        pf_norm = self._normalize_process_features(pf)
        pf_t = torch.from_numpy(pf_norm).unsqueeze(0).to(self.device)

        out   = self.model(tel_t, pf_t)
        probs = F.softmax(out["logits"], dim=-1).squeeze(0).cpu().numpy()
        pred  = int(probs.argmax())

        return {
            "fault_type": FAULT_TYPES[pred],
            "fault_index": pred,
            "confidence": float(probs[pred]),
            "fault_probabilities": {ft: float(p) for ft, p in zip(FAULT_TYPES, probs)},
            "embedding_norm": float(out["embedding"].norm().item()),
        }

    @torch.inference_mode()
    def predict_batch(
        self,
        sensor_windows: list[np.ndarray],
        actuator_windows: list[np.ndarray],
    ) -> list[dict]:
        """Batch fault diagnosis."""
        results = []
        for s, a in zip(sensor_windows, actuator_windows):
            results.append(self.predict_single(s, a))
        return results
