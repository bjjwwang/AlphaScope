"""Tests for scan_db.py — SQLite persistence for scan jobs."""
import threading
import time
import pytest

from backtest_server.scan_db import (
    init_db,
    create_scan_job,
    get_scan_job,
    update_scan_job,
    list_scan_jobs,
    save_model_result,
    get_model_results,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_scan.db")
    init_db(db_path)
    return db_path


# --- Schema & Initialization ---

def test_init_creates_tables(tmp_path):
    db_path = str(tmp_path / "fresh.db")
    init_db(db_path)
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert "scan_jobs" in tables
    assert "scan_model_results" in tables


def test_init_idempotent(tmp_path):
    db_path = str(tmp_path / "idem.db")
    init_db(db_path)
    init_db(db_path)  # should not raise


# --- CRUD: scan_jobs ---

def test_create_scan_job(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    assert isinstance(job_id, str)
    assert len(job_id) == 36  # UUID format


def test_get_scan_job(db):
    job_id = create_scan_job(db, ticker="MSFT", data_source="yfinance", trading_style="swing")
    job = get_scan_job(db, job_id)
    assert job is not None
    assert job["id"] == job_id
    assert job["ticker"] == "MSFT"
    assert job["data_source"] == "yfinance"
    assert job["trading_style"] == "swing"
    assert job["status"] == "pending"
    assert job["total_models"] == 0
    assert job["completed_models"] == 0
    assert job["best_model"] is None
    assert job["prediction_json"] is None
    assert job["created_at"] is not None
    assert job["updated_at"] is not None


def test_get_scan_job_not_found(db):
    assert get_scan_job(db, "nonexistent-id") is None


def test_create_with_all_data_sources(db):
    for ds in ("db", "yfinance", "baostock"):
        job_id = create_scan_job(db, ticker="TEST", data_source=ds, trading_style="ultra_short")
        job = get_scan_job(db, job_id)
        assert job["data_source"] == ds


def test_update_scan_job_status(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    old = get_scan_job(db, job_id)

    update_scan_job(db, job_id, status="running", total_models=22)
    job = get_scan_job(db, job_id)
    assert job["status"] == "running"
    assert job["total_models"] == 22
    assert job["updated_at"] >= old["updated_at"]


def test_update_scan_job_progress(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    update_scan_job(db, job_id, status="running", total_models=22)

    update_scan_job(db, job_id, completed_models=5, current_model="GRU_ts")
    job = get_scan_job(db, job_id)
    assert job["completed_models"] == 5
    assert job["current_model"] == "GRU_ts"


def test_update_scan_job_best_model(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    update_scan_job(db, job_id, best_model="ALSTM_ts", best_model_reason="最高模型收益 15.2%")
    job = get_scan_job(db, job_id)
    assert job["best_model"] == "ALSTM_ts"
    assert job["best_model_reason"] == "最高模型收益 15.2%"


def test_update_scan_job_prediction(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    import json
    pred = json.dumps({"signals": [{"date": "2026-03-13", "signal": "buy"}]})
    update_scan_job(db, job_id, prediction_json=pred)
    job = get_scan_job(db, job_id)
    assert job["prediction_json"] is not None
    parsed = json.loads(job["prediction_json"])
    assert parsed["signals"][0]["signal"] == "buy"


def test_update_scan_job_error(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    update_scan_job(db, job_id, status="failed", error_msg="CUDA OOM")
    job = get_scan_job(db, job_id)
    assert job["status"] == "failed"
    assert job["error_msg"] == "CUDA OOM"


def test_list_scan_jobs_empty(db):
    assert list_scan_jobs(db) == []


def test_list_scan_jobs(db):
    create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    create_scan_job(db, ticker="MSFT", data_source="yfinance", trading_style="swing")
    create_scan_job(db, ticker="GOOG", data_source="db", trading_style="ultra_short")

    jobs = list_scan_jobs(db)
    assert len(jobs) == 3
    # Ordered by created_at DESC (most recent first)
    assert jobs[0]["ticker"] == "GOOG"
    assert jobs[2]["ticker"] == "AAPL"


# --- CRUD: scan_model_results ---

def test_save_model_result_completed(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    save_model_result(
        db, job_id,
        model_class="LGBModel",
        status="completed",
        result_json='{"summary": {"model_return": 12.5}}',
        recorder_id="rec_abc123",
        duration_s=45.2,
    )
    results = get_model_results(db, job_id)
    assert len(results) == 1
    r = results[0]
    assert r["model_class"] == "LGBModel"
    assert r["status"] == "completed"
    assert r["recorder_id"] == "rec_abc123"
    assert r["duration_s"] == pytest.approx(45.2)
    assert "model_return" in r["result_json"]


def test_save_model_result_failed(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    save_model_result(db, job_id, model_class="ADARNN", status="failed", error_msg="CUDA OOM")
    results = get_model_results(db, job_id)
    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["error_msg"] == "CUDA OOM"


def test_save_model_result_skipped(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    save_model_result(db, job_id, model_class="XGBModel", status="skipped", error_msg="缺少依赖 xgboost")
    results = get_model_results(db, job_id)
    assert len(results) == 1
    assert results[0]["status"] == "skipped"


def test_get_model_results_multiple(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    save_model_result(db, job_id, model_class="LGBModel", status="completed", duration_s=30)
    save_model_result(db, job_id, model_class="GRU_ts", status="completed", duration_s=120)
    save_model_result(db, job_id, model_class="XGBModel", status="skipped")
    results = get_model_results(db, job_id)
    assert len(results) == 3


def test_get_model_results_empty(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    assert get_model_results(db, job_id) == []


# --- Concurrent safety ---

def test_concurrent_writes(db):
    job_id = create_scan_job(db, ticker="AAPL", data_source="db", trading_style="ultra_short")
    errors = []

    def write_result(model_class):
        try:
            save_model_result(db, job_id, model_class=model_class, status="completed", duration_s=10)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=write_result, args=(f"Model_{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    results = get_model_results(db, job_id)
    assert len(results) == 10
