"""FastAPI for FIR Quantum Error Estimation."""
from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.config.settings import settings
from src.inference.predictor import QuantumErrorPredictor
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)
_predictor = None

@asynccontextmanager
async def lifespan(app):
    global _predictor
    mp = Path(settings.model_artifact_path) / "best_model.pt"
    if mp.exists(): _predictor = QuantumErrorPredictor(mp)
    else: log.warning("No model found")
    yield
    _predictor = None

app = FastAPI(title="FIR Quantum Error Estimation API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Quantum measurement feature vector")

class BatchPredictRequest(BaseModel):
    samples: list[list[float]]

@app.get("/health")
async def health():
    return {"status": "healthy" if _predictor else "degraded",
            "model_loaded": _predictor is not None}

@app.post("/predict")
async def predict(request: PredictRequest):
    """Single prediction. curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [0.0625]*24}'"""
    if _predictor is None: raise HTTPException(503, "Model not loaded")
    features = np.array(request.features, dtype=np.float32)
    start = time.perf_counter()
    result = _predictor.predict_single(features)
    return {"prediction": result, "latency_ms": round((time.perf_counter()-start)*1000, 2)}

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    if _predictor is None: raise HTTPException(503, "Model not loaded")
    features = np.array(request.samples, dtype=np.float32)
    results = _predictor.predict_batch(features)
    return {"predictions": results, "count": len(results)}
