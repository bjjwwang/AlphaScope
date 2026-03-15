"""Tests for scan API endpoints in server.py."""
import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from backtest_server.server import app, _scan_db_path
from backtest_server.scan_db import (
    init_db, create_scan_job, update_scan_job,
    save_model_result, get_scan_job,
)


@pytest.fixture
def scan_db(tmp_path):
    db_path = str(tmp_path / "api_scan_test.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def client(scan_db):
    """TestClient with scan DB path overridden."""
    with patch("backtest_server.server._scan_db_path", scan_db):
        yield TestClient(app)


# --- POST /api/scan ---

def test_create_scan_ok(client, scan_db):
    with patch("backtest_server.server._run_scan_job", new_callable=AsyncMock):
        resp = client.post("/api/scan", json={
            "ticker": "AAPL", "data_source": "db", "trading_style": "ultra_short",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert len(data["job_id"]) == 36


def test_create_scan_yfinance(client, scan_db):
    with patch("backtest_server.server._run_scan_job", new_callable=AsyncMock):
        resp = client.post("/api/scan", json={
            "ticker": "MSFT", "data_source": "yfinance", "trading_style": "swing",
        })
    assert resp.status_code == 200


def test_create_scan_invalid_ticker(client):
    resp = client.post("/api/scan", json={
        "ticker": "", "data_source": "db", "trading_style": "ultra_short",
    })
    assert resp.status_code == 422


def test_create_scan_invalid_style(client):
    resp = client.post("/api/scan", json={
        "ticker": "AAPL", "data_source": "db", "trading_style": "invalid",
    })
    assert resp.status_code == 422


def test_create_scan_invalid_data_source(client):
    resp = client.post("/api/scan", json={
        "ticker": "AAPL", "data_source": "invalid", "trading_style": "ultra_short",
    })
    assert resp.status_code == 422


# --- GET /api/scan/list ---

def test_list_scans_empty(client, scan_db):
    resp = client.get("/api/scan/list")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_scans(client, scan_db):
    with patch("backtest_server.server._scan_db_path", scan_db):
        create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        create_scan_job(scan_db, "MSFT", "yfinance", "swing")
        resp = client.get("/api/scan/list")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) == 2
    tickers = {j["ticker"] for j in jobs}
    assert tickers == {"AAPL", "MSFT"}


# --- GET /api/scan/{job_id} ---

def test_get_scan_running(client, scan_db):
    with patch("backtest_server.server._scan_db_path", scan_db):
        job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        update_scan_job(scan_db, job_id, status="running",
                        total_models=22, completed_models=5, current_model="GRU_ts")
        resp = client.get(f"/api/scan/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["completed_models"] == 5
    assert data["current_model"] == "GRU_ts"


def test_get_scan_completed(client, scan_db):
    with patch("backtest_server.server._scan_db_path", scan_db):
        job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        update_scan_job(scan_db, job_id, status="completed",
                        best_model="GRU_ts", best_model_reason="最高收益")
        save_model_result(scan_db, job_id, "GRU_ts", "completed",
                          result_json='{"summary": {"model_return": 15}}', duration_s=120)
        resp = client.get(f"/api/scan/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["best_model"] == "GRU_ts"
    assert "model_results" in data
    assert len(data["model_results"]) == 1


def test_get_scan_not_found(client):
    resp = client.get("/api/scan/nonexistent-id")
    assert resp.status_code == 404


# --- GET /api/scan/{job_id}/stream ---

def test_scan_stream_not_found(client):
    resp = client.get("/api/scan/nonexistent-id/stream")
    assert resp.status_code == 404


def test_scan_stream_completed(client, scan_db):
    """Completed scan SSE returns scan_complete event immediately."""
    with patch("backtest_server.server._scan_db_path", scan_db):
        job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        update_scan_job(scan_db, job_id, status="completed",
                        best_model="GRU_ts", best_model_reason="最高收益",
                        total_models=22, completed_models=22)
        save_model_result(scan_db, job_id, "GRU_ts", "completed",
                          result_json='{"summary": {"model_return": 15, "win_rate": 60}}',
                          duration_s=120)

        resp = client.get(f"/api/scan/{job_id}/stream")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "scan_complete" in resp.text


def test_scan_stream_running_receives_events(client, scan_db):
    """Running scan SSE: pre-load queue with events, verify they come through."""
    with patch("backtest_server.server._scan_db_path", scan_db):
        job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        update_scan_job(scan_db, job_id, status="running", total_models=22)

        # Import scan_queues and pre-populate
        from backtest_server.server import scan_queues
        queue = asyncio.Queue()
        queue.put_nowait({"event": "scan_model_complete", "data": {"model_class": "LGBModel"}})
        queue.put_nowait({"event": "scan_complete", "data": {"best_model": "LGBModel", "rankings": []}})
        scan_queues[job_id] = queue

        resp = client.get(f"/api/scan/{job_id}/stream")

    assert resp.status_code == 200
    assert "scan_model_complete" in resp.text
    assert "scan_complete" in resp.text

    # Cleanup
    scan_queues.pop(job_id, None)


# --- POST /api/scan/{job_id}/predict ---

def test_predict_no_best_model(client, scan_db):
    with patch("backtest_server.server._scan_db_path", scan_db):
        job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        update_scan_job(scan_db, job_id, status="completed")
        resp = client.post(f"/api/scan/{job_id}/predict")
    assert resp.status_code == 400
    assert "best_model" in resp.json()["detail"]


def test_predict_not_found(client):
    resp = client.post("/api/scan/nonexistent-id/predict")
    assert resp.status_code == 404
