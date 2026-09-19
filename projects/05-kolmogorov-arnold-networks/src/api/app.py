"""FastAPI application for KAN inference with symbolic export endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictRequest, BatchPredictResponse,
    FeatureImportanceResponse, HealthResponse,
    PredictRequest, PredictResponse,
)
from src.config.settings import settings
from src.inference.predictor import KANPredictor
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)

_predictor: KANPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    model_path = Path(settings.model_artifact_path) / "best_model.pt"
    if model_path.exists():
        _predictor = KANPredictor(model_path)
    else:
        log.warning("No model found — train first")
    yield
    _predictor = None


app = FastAPI(
    title="Kolmogorov-Arnold Network API",
    description="KAN inference, symbolic extraction, feature importance, hardware LUT export.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _predictor else "degraded",
        model_loaded=_predictor is not None,
        input_dim=_predictor.input_dim if _predictor else 0,
        layer_sizes=_predictor.model.layer_sizes if _predictor else [],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single prediction. curl: curl -X POST http://localhost:8004/predict -H 'Content-Type: application/json' -d '{"features":[0.5,-0.3]}'"""
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    try:
        x = np.array(request.features, dtype=np.float32)
        start = time.perf_counter()
        result = _predictor.predict_single(x)
        ms = (time.perf_counter() - start) * 1000
        pred = result["prediction"]
        return PredictResponse(
            prediction=pred if isinstance(pred, list) else float(pred),
            input_dim=result["input_dim"],
            latency_ms=round(ms, 2),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    try:
        X = np.array(request.samples, dtype=np.float32)
        start = time.perf_counter()
        results = _predictor.predict_batch(X)
        ms = (time.perf_counter() - start) * 1000
        preds = [r["prediction"] for r in results]
        return BatchPredictResponse(predictions=preds, count=len(preds), latency_ms=round(ms, 2))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/feature-importance", response_model=FeatureImportanceResponse)
async def feature_importance(request: BatchPredictRequest):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    X = np.array(request.samples, dtype=np.float32)
    return FeatureImportanceResponse(**_predictor.feature_importance(X))


@app.get("/symbolic")
async def symbolic():
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    return {"symbolic_layers": _predictor.get_symbolic_representation()}


@app.get("/export/lut")
async def export_lut():
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    return _predictor.export_for_hardware()
