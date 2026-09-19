"""FastAPI for WELC image classification with logic circuit export."""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.inference.predictor import WELCPredictor
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)

_predictor: WELCPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    model_path = Path(settings.model_artifact_path) / "best_model.pt"
    if model_path.exists():
        _predictor = WELCPredictor(model_path)
    else:
        log.warning("No model found — run training first")
    yield
    _predictor = None


app = FastAPI(
    title="WELC Neural Acceleration API",
    description="Weight-Embedded Logic Circuit classifier for hardware-efficient inference",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ImageB64Request(BaseModel):
    """Base64-encoded image (C×H×W float array serialized as list)."""
    image: list[list[list[float]]] = Field(
        ...,
        description="Image as nested list [H][W][C] with pixel values 0-255",
    )


class PredictionResponse(BaseModel):
    class_index: int
    class_name: str
    confidence: float
    class_probabilities: dict[str, float]
    latency_ms: float


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _predictor else "degraded",
        "model_loaded": _predictor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageB64Request):
    """
    Classify an image using WELC model.

    Example curl (3×32×32 random image):
    ```bash
    python -c "import json,numpy as np; arr=np.random.randint(0,255,(32,32,3)).tolist(); print(json.dumps({'image': arr}))" > /tmp/img.json
    curl -X POST http://localhost:8003/predict -H "Content-Type: application/json" -d @/tmp/img.json
    ```
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    try:
        img = np.array(request.image, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {img.shape}")
        start = time.perf_counter()
        result = _predictor.predict_single(img)
        latency = (time.perf_counter() - start) * 1000
        return PredictionResponse(**result, latency_ms=round(latency, 2))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/export/logic_circuits")
async def export_logic_circuits():
    """Export all WELC layers as boolean logic circuit masks (AND/OR planes)."""
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    return _predictor.get_logic_circuit_export()


@app.get("/model/efficiency")
async def model_efficiency():
    """Get hardware efficiency statistics for the WELC model."""
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    return _predictor.efficiency_stats
