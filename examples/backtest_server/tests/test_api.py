"""Tests for server.py — Phase 5 (FastAPI TestClient, all mocked)"""
import asyncio
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from backtest_server.server import app, jobs, MODEL_METADATA
from backtest_server.config_schema import BacktestResult, SummaryStats


def _valid_request_body():
    return {
        "model": {
            "model_class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"learning_rate": 0.05},
        },
        "dataset": {
            "dataset_class": "DatasetH",
            "handler": "Alpha158",
            "segments": {
                "train": ("2020-01-01", "2024-10-31"),
                "valid": ("2024-11-01", "2024-12-31"),
                "test": ("2025-01-01", "2025-03-31"),
            },
        },
        "data_source": "yfinance",
        "target": ["INTC"],
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {"strategy_class": "TopkDropoutStrategy", "topk": 50, "n_drop": 5},
    }


def _fake_result():
    return BacktestResult(
        summary=SummaryStats(
            model_return=12.5,
            bh_return=5.0,
            win_rate=60.0,
            total_trades=3,
            test_start="2025-01-01",
            test_end="2025-03-31",
            held_days=40,
            total_days=60,
        ),
        trades=[],
        segments=[],
    )


@pytest.fixture
def client():
    # Clear jobs between tests
    jobs.clear()
    return TestClient(app)


# --- Basic endpoint tests ---

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    # Returns HTML (frontend page) when the file exists, JSON fallback otherwise
    assert "text/html" in resp.headers.get("content-type", "") or resp.json().get("service")


def test_list_models(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    assert len(models) == len(MODEL_METADATA)
    names = {m["name"] for m in models}
    assert "LGBModel" in names
    assert "GRU_ts" in names
    # Verify structure
    for m in models:
        assert "name" in m
        assert "fullName" in m
        assert "cat" in m
        assert "gpu" in m
        assert "brief" in m


# --- POST /api/backtest ---

def test_start_backtest_ok(client):
    with patch("backtest_server.server._run_backtest_job", new_callable=AsyncMock):
        resp = client.post("/api/backtest", json=_valid_request_body())
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36  # UUID format


def test_start_invalid_request(client):
    # Missing required fields
    resp = client.post("/api/backtest", json={"model": {}})
    assert resp.status_code == 422


def test_start_empty_target(client):
    body = _valid_request_body()
    body["target"] = []
    resp = client.post("/api/backtest", json=body)
    assert resp.status_code == 422


# --- GET /api/backtest/{job_id}/stream ---

def test_stream_nonexistent(client):
    resp = client.get("/api/backtest/fake-job-id/stream")
    assert resp.status_code == 404


def test_stream_receives_events(client):
    """SSE stream receives step/log/result events."""
    job_id = "test-stream-job"
    queue = asyncio.Queue()
    jobs[job_id] = {"status": "running", "queue": queue, "result": None}

    # Pre-load events into queue
    result = _fake_result()
    events_to_send = [
        {"event": "step", "data": {"step": 0, "state": "active", "label": "准备数据"}},
        {"event": "log", "data": {"msg": "测试日志", "type": "info"}},
        {"event": "step", "data": {"step": 0, "state": "done", "label": "准备数据"}},
        {"event": "result", "data": result.model_dump()},
    ]
    for ev in events_to_send:
        queue.put_nowait(ev)

    resp = client.get(f"/api/backtest/{job_id}/stream")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE body
    lines = resp.text.strip().split("\n\n")
    assert len(lines) == len(events_to_send)

    # Verify first event is step
    assert lines[0].startswith("event: step")
    assert "准备数据" in lines[0]

    # Verify last event is result
    assert lines[-1].startswith("event: result")
    assert "summary" in lines[-1]


def test_stream_receives_error(client):
    """SSE stream receives error event and closes."""
    job_id = "test-error-job"
    queue = asyncio.Queue()
    jobs[job_id] = {"status": "running", "queue": queue, "result": None}

    queue.put_nowait({"event": "step", "data": {"step": 0, "state": "active", "label": "准备数据"}})
    queue.put_nowait({"event": "error", "data": {"step": 0, "msg": "CUDA out of memory"}})

    resp = client.get(f"/api/backtest/{job_id}/stream")
    assert resp.status_code == 200
    assert "CUDA out of memory" in resp.text
    # Stream should end after error
    blocks = resp.text.strip().split("\n\n")
    last_block = blocks[-1]
    assert last_block.startswith("event: error")


# --- GET /api/backtest/{job_id}/status ---

def test_status_running(client):
    job_id = "test-running-job"
    jobs[job_id] = {"status": "running", "queue": asyncio.Queue(), "result": None}

    resp = client.get(f"/api/backtest/{job_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert "result" not in data


def test_status_completed(client):
    result_data = _fake_result().model_dump()
    job_id = "test-completed-job"
    jobs[job_id] = {"status": "completed", "queue": asyncio.Queue(), "result": result_data}

    resp = client.get(f"/api/backtest/{job_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "result" in data
    assert data["result"]["summary"]["model_return"] == 12.5


# --- CORS ---

def test_cors_headers(client):
    resp = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    # With allow_credentials=True, Starlette echoes back the specific origin
    assert resp.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
    assert "GET" in resp.headers.get("access-control-allow-methods", "")


# --- Concurrent backtests ---

def test_concurrent_backtests(client):
    """Two concurrent backtests get independent job IDs."""
    with patch("backtest_server.server._run_backtest_job", new_callable=AsyncMock):
        resp1 = client.post("/api/backtest", json=_valid_request_body())
        resp2 = client.post("/api/backtest", json=_valid_request_body())
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        id1 = resp1.json()["job_id"]
        id2 = resp2.json()["job_id"]
        assert id1 != id2
        # Both jobs exist in store
        assert id1 in jobs
        assert id2 in jobs
