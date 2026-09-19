"""FastAPI for RLA text generation."""
from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.config.settings import settings
from src.inference.generator import RLATextGenerator
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger(__name__)
_generator = None

@asynccontextmanager
async def lifespan(app):
    global _generator
    mp = Path(settings.model_artifact_path) / "best_model.pt"
    if mp.exists(): _generator = RLATextGenerator(mp)
    else: log.warning("No model found")
    yield
    _generator = None

app = FastAPI(title="RLA Text Generation API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1)

class GenerateResponse(BaseModel):
    prompt: str; generated_text: str; num_new_tokens: int; latency_ms: float

@app.get("/health")
async def health(): return {"status": "healthy" if _generator else "degraded", "model_loaded": _generator is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text. curl -X POST http://localhost:8001/generate -H "Content-Type: application/json" -d '{"prompt":"Hello world"}'"""
    if _generator is None: raise HTTPException(503, "Model not loaded")
    start = time.perf_counter()
    result = _generator.predict_single(request.prompt, request.max_new_tokens)
    return GenerateResponse(**result, latency_ms=round((time.perf_counter()-start)*1000, 2))

@app.post("/generate/batch")
async def generate_batch(prompts: list[str]):
    if _generator is None: raise HTTPException(503, "Model not loaded")
    return {"results": _generator.predict_batch(prompts), "count": len(prompts)}
