"""FastAPI application for Process Mining Fault Diagnosis."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchDiagnosisResponse,
    BatchTelemetryRequest,
    FaultDiagnosisResult,
    FaultProbabilities,
    HealthResponse,
    SingleDiagnosisResponse,
    TelemetryWindowRequest,
)
from src.config.settings import settings
from src.data.cps_simulator import FAULT_TYPES
from src.inference.fault_predictor import FaultDiagnosisPredictor
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)

_predictor: FaultDiagnosisPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    model_path = Path(settings.model_artifact_path) / "best_model.pt"
    if model_path.exists():
        _predictor = FaultDiagnosisPredictor(model_path)
        log.info("Predictor loaded", path=str(model_path))
    else:
        log.warning("No model found — start training first", path=str(model_path))
    yield
    _predictor = None


app = FastAPI(
    title="Process Mining Fault Diagnosis API",
    description=(
        "Production fault diagnosis for Cyber-Physical Systems. "
        "Combines CPS telemetry with process mining event-log features for "
        "enhanced fault identification: sensor drift, actuator faults, "
        "cascade failures, and more."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next: Any) -> Any:
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    log.info(
        "HTTP request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration, 2),
    )
    return response


def _build_result(raw: dict) -> FaultDiagnosisResult:
    probs = raw["fault_probabilities"]
    return FaultDiagnosisResult(
        fault_type=raw["fault_type"],
        fault_index=raw["fault_index"],
        confidence=raw["confidence"],
        fault_probabilities=FaultProbabilities(
            normal=probs.get("normal", 0.0),
            sensor_drift=probs.get("sensor_drift", 0.0),
            actuator_stuck=probs.get("actuator_stuck", 0.0),
            communication_delay=probs.get("communication_delay", 0.0),
            oscillation=probs.get("oscillation", 0.0),
            overload=probs.get("overload", 0.0),
            partial_failure=probs.get("partial_failure", 0.0),
            cascade_failure=probs.get("cascade_failure", 0.0),
        ),
        embedding_norm=raw["embedding_norm"],
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Service health check."""
    return HealthResponse(
        status="healthy" if _predictor else "degraded",
        model_loaded=_predictor is not None,
        device=str(_predictor.device) if _predictor else "N/A",
        fault_classes=FAULT_TYPES,
    )


@app.post("/diagnose", response_model=SingleDiagnosisResponse, tags=["Inference"])
async def diagnose_single(request: TelemetryWindowRequest) -> SingleDiagnosisResponse:
    """
    Diagnose fault type from a single CPS telemetry window.

    Accepts sensor + actuator time-series window and returns:
    - Predicted fault type
    - Per-class confidence scores
    - Process mining-derived embedding norm

    Example curl:
    ```bash
    python -c "
    import json, numpy as np
    payload = {
        'sensor_readings': (np.random.randn(50, 20) * 0.1).tolist(),
        'actuator_readings': (np.random.rand(50, 5)).tolist()
    }
    print(json.dumps(payload))
    " > /tmp/window.json
    curl -X POST http://localhost:8005/diagnose \\
      -H 'Content-Type: application/json' \\
      -d @/tmp/window.json
    ```
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run training first")

    try:
        sensor  = np.array(request.sensor_readings,   dtype=np.float32)
        actuator = np.array(request.actuator_readings, dtype=np.float32)

        if sensor.ndim != 2:
            raise ValueError(f"sensor_readings must be 2D (window_size, num_sensors), got {sensor.shape}")
        if actuator.ndim != 2:
            raise ValueError(f"actuator_readings must be 2D (window_size, num_actuators), got {actuator.shape}")

        start = time.perf_counter()
        raw   = _predictor.predict_single(sensor, actuator)
        latency = (time.perf_counter() - start) * 1000

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("Inference error", error=str(exc))
        raise HTTPException(status_code=500, detail="Inference failed")

    return SingleDiagnosisResponse(
        diagnosis=_build_result(raw),
        latency_ms=round(latency, 2),
    )


@app.post("/diagnose/batch", response_model=BatchDiagnosisResponse, tags=["Inference"])
async def diagnose_batch(request: BatchTelemetryRequest) -> BatchDiagnosisResponse:
    """
    Batch fault diagnosis for multiple CPS telemetry windows.

    Example curl:
    ```bash
    curl -X POST http://localhost:8005/diagnose/batch \\
      -H 'Content-Type: application/json' \\
      -d '{
        "windows": [
          {
            "sensor_readings": [[0.1]*20]*50,
            "actuator_readings": [[0.5]*5]*50
          }
        ]
      }'
    ```
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        sensor_windows   = [np.array(w.sensor_readings,   dtype=np.float32) for w in request.windows]
        actuator_windows = [np.array(w.actuator_readings, dtype=np.float32) for w in request.windows]

        start   = time.perf_counter()
        results = _predictor.predict_batch(sensor_windows, actuator_windows)
        latency = (time.perf_counter() - start) * 1000

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("Batch inference error", error=str(exc))
        raise HTTPException(status_code=500, detail="Batch inference failed")

    return BatchDiagnosisResponse(
        diagnoses=[_build_result(r) for r in results],
        count=len(results),
        latency_ms=round(latency, 2),
    )


@app.get("/fault-types", tags=["Metadata"])
async def list_fault_types() -> dict:
    """List all supported fault types and their indices."""
    return {
        "fault_types": [
            {"index": i, "name": ft}
            for i, ft in enumerate(FAULT_TYPES)
        ]
    }
