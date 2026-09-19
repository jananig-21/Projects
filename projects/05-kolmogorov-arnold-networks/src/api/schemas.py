"""API schemas for KAN inference service."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Input feature vector")

    @field_validator("features")
    @classmethod
    def non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("features must not be empty")
        return v

    model_config = {"json_schema_extra": {"example": {"features": [0.5, -0.3]}}}


class BatchPredictRequest(BaseModel):
    samples: list[list[float]] = Field(..., description="List of input feature vectors")

    @field_validator("samples")
    @classmethod
    def validate_batch(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("samples must not be empty")
        if len(v) > 10000:
            raise ValueError("max batch size is 10000")
        return v


class PredictResponse(BaseModel):
    prediction: float | list[float]
    input_dim: int
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[float]
    count: int
    latency_ms: float


class FeatureImportanceResponse(BaseModel):
    feature_importance: list[float]
    most_important_feature: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    input_dim: int
    layer_sizes: list[int]
