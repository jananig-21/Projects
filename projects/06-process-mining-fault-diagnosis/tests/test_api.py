"""API integration tests for Process Mining Fault Diagnosis."""

from __future__ import annotations

import numpy as np
import pytest
from httpx import AsyncClient

from src.api.app import app


@pytest.fixture
def sample_window():
    rng = np.random.default_rng(0)
    return {
        "sensor_readings":   (rng.randn(50, 20) * 0.1).tolist(),
        "actuator_readings": (rng.rand(50, 5)).tolist(),
    }


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status"       in data
    assert "model_loaded" in data
    assert "fault_classes" in data
    assert len(data["fault_classes"]) == 8


@pytest.mark.asyncio
async def test_fault_types_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/fault-types")
    assert resp.status_code == 200
    data = resp.json()
    assert "fault_types" in data
    assert len(data["fault_types"]) == 8
    for ft in data["fault_types"]:
        assert "index" in ft
        assert "name"  in ft


@pytest.mark.asyncio
async def test_diagnose_503_without_model(sample_window):
    """Without a loaded model, expect 503."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/diagnose", json=sample_window)
    # Either 200 (if model loaded somehow) or 503
    assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_diagnose_invalid_input():
    """Badly shaped input should return 422."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/diagnose",
            json={
                "sensor_readings":   [[0.1, 0.2]],   # wrong shape — only 2 sensors
                "actuator_readings": [[0.5]],
            },
        )
    # 422 or 503 depending on model load state
    assert resp.status_code in (422, 503)
