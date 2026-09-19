"""API schemas for Process Mining Fault Diagnosis."""

from __future__ import annotations
from pydantic import BaseModel, Field


class TelemetryWindowRequest(BaseModel):
    """
    Single CPS telemetry window for fault diagnosis.
    sensor_readings:   list of lists, shape (window_size, num_sensors)
    actuator_readings: list of lists, shape (window_size, num_actuators)
    """
    sensor_readings:   list[list[float]] = Field(..., description="Shape: (window_size, num_sensors)")
    actuator_readings: list[list[float]] = Field(..., description="Shape: (window_size, num_actuators)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sensor_readings": [[0.1] * 20] * 50,
                "actuator_readings": [[0.5] * 5] * 50,
            }
        }
    }


class BatchTelemetryRequest(BaseModel):
    windows: list[TelemetryWindowRequest] = Field(..., max_length=100)


class FaultProbabilities(BaseModel):
    normal: float
    sensor_drift: float
    actuator_stuck: float
    communication_delay: float
    oscillation: float
    overload: float
    partial_failure: float
    cascade_failure: float


class FaultDiagnosisResult(BaseModel):
    fault_type: str
    fault_index: int
    confidence: float
    fault_probabilities: FaultProbabilities
    embedding_norm: float


class SingleDiagnosisResponse(BaseModel):
    success: bool = True
    diagnosis: FaultDiagnosisResult
    latency_ms: float


class BatchDiagnosisResponse(BaseModel):
    success: bool = True
    diagnoses: list[FaultDiagnosisResult]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    fault_classes: list[str]
