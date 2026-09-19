"""FastAPI for PEFT LLM inference."""
from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.config.settings import settings
from src.inference.peft_predictor import PEFTPredictor
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)
_predictor = None
BASE_MODEL = "microsoft/phi-2"

@asynccontextmanager
async def lifespan(app):
    global _predictor
    ap = Path(settings.model_artifact_path) / "peft_adapter"
    if ap.exists(): _predictor = PEFTPredictor(BASE_MODEL, ap)
    else: log.warning("No adapter found")
    yield
    _predictor = None

app = FastAPI(title="PEFT LLM API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class InstructRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    input: str = Field(default="")
    max_new_tokens: int = Field(default=256, ge=1, le=1024)

class InstructResponse(BaseModel):
    instruction: str; input: str; response: str
    num_tokens_generated: int; latency_ms: float

@app.get("/health")
async def health(): return {"status":"healthy" if _predictor else "degraded","model_loaded":_predictor is not None}

@app.post("/instruct", response_model=InstructResponse)
async def instruct(request: InstructRequest):
    """Run instruction. curl -X POST http://localhost:8002/instruct -H "Content-Type: application/json" -d '{"instruction":"Explain PEFT"}'"""
    if _predictor is None: raise HTTPException(503, "Model not loaded")
    start = time.perf_counter()
    result = _predictor.predict_single(request.instruction, request.input)
    return InstructResponse(**result, latency_ms=round((time.perf_counter()-start)*1000, 2))

@app.post("/instruct/batch")
async def instruct_batch(instructions: list[str]):
    if _predictor is None: raise HTTPException(503, "Model not loaded")
    return {"results": _predictor.predict_batch(instructions)}
