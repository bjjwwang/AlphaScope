"""Tests for predictor.py — prediction engine using best model."""
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from backtest_server.predictor import generate_signals, run_prediction
from backtest_server.config_schema import PredictionSignal, PredictionResult
from backtest_server.scan_db import (
    init_db, create_scan_job, update_scan_job, save_model_result,
)


@pytest.fixture
def scan_db(tmp_path):
    db_path = str(tmp_path / "pred_test.db")
    init_db(db_path)
    return db_path


# --- Signal generation ---

def test_generate_signals_basic():
    """Synthetic scores produce correct buy/sell/hold signals."""
    dates = pd.date_range("2026-03-01", periods=10, freq="B")
    scores = [0.1, 0.2, -0.1, 0.3, -0.2, 0.05, 0.15, -0.05, 0.25, 0.0]
    scores_series = pd.Series(scores, index=dates, name="score")

    signals = generate_signals(scores_series, "AAPL")
    assert len(signals) == 10
    for s in signals:
        assert isinstance(s, PredictionSignal)
        assert s.signal in ("buy", "sell", "hold")


def test_generate_signals_threshold():
    """Signals use median threshold: > median = buy, < median = sell."""
    dates = pd.date_range("2026-03-01", periods=5, freq="B")
    # Median of [1, 2, 3, 4, 5] = 3
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, name="score")

    signals = generate_signals(scores, "AAPL")
    # 1 < 3 → sell, 2 < 3 → sell, 3 ≈ 3 → hold, 4 > 3 → buy, 5 > 3 → buy
    assert signals[0].signal == "sell"
    assert signals[1].signal == "sell"
    assert signals[2].signal == "hold"
    assert signals[3].signal == "buy"
    assert signals[4].signal == "buy"


def test_generate_signals_all_same():
    """When all scores are identical, all signals should be hold."""
    dates = pd.date_range("2026-03-01", periods=3, freq="B")
    scores = pd.Series([0.5, 0.5, 0.5], index=dates, name="score")
    signals = generate_signals(scores, "AAPL")
    assert all(s.signal == "hold" for s in signals)


def test_prediction_result_schema():
    """PredictionResult validates correctly."""
    result = PredictionResult(
        ticker="AAPL",
        model_class="GRU_ts",
        prediction_range=("2026-02-01", "2026-03-13"),
        signals=[PredictionSignal(date="2026-03-13", score=0.05, signal="buy")],
        latest_signal="buy",
        latest_score=0.05,
        threshold=0.02,
        generated_at="2026-03-13T10:00:00",
    )
    assert result.ticker == "AAPL"
    assert result.latest_signal == "buy"


# --- Prediction pipeline (mocked) ---

def _setup_completed_scan(db_path, ticker="AAPL", best_model="GRU_ts"):
    """Create a completed scan job with a best model in the DB."""
    job_id = create_scan_job(db_path, ticker, "db", "ultra_short")
    update_scan_job(db_path, job_id,
                    status="completed", best_model=best_model,
                    completed_models=22, total_models=22)
    save_model_result(db_path, job_id, best_model, "completed",
                      result_json='{"summary": {"model_return": 15.0}}',
                      recorder_id="test_exp_abc")
    return job_id


@patch("backtest_server.predictor.qlib")
@patch("backtest_server.predictor.init_instance_by_config")
@patch("backtest_server.predictor.R")
def test_run_prediction_mocked(mock_R, mock_init, mock_qlib, scan_db):
    job_id = _setup_completed_scan(scan_db)

    # Mock model and prediction
    mock_model = MagicMock()
    dates = pd.date_range("2026-02-10", periods=20, freq="B")
    idx = pd.MultiIndex.from_arrays(
        [dates, ["AAPL"] * len(dates)], names=["datetime", "instrument"]
    )
    pred_df = pd.DataFrame({"score": np.random.randn(20) * 0.1}, index=idx)
    mock_model.predict.return_value = pred_df

    # Mock recorder listing — return a FINISHED recorder with the trained model
    mock_recorder = MagicMock()
    mock_recorder.load_object.return_value = mock_model
    mock_recorder.info = {"status": "FINISHED"}
    mock_R.list_recorders.return_value = {"rec1": mock_recorder}

    # Mock dataset creation
    mock_dataset = MagicMock()
    mock_init.return_value = mock_dataset

    # First call creates model (we don't use it), second creates dataset
    call_count = [0]
    def init_side_effect(config):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_model
        return mock_dataset
    mock_init.side_effect = init_side_effect

    result = run_prediction(job_id, scan_db, lambda e: None)
    assert isinstance(result, PredictionResult)
    assert result.ticker == "AAPL"
    assert result.model_class == "GRU_ts"
    assert len(result.signals) > 0
    assert result.latest_signal in ("buy", "sell", "hold")


def test_run_prediction_no_best_model(scan_db):
    """Should raise if no best_model in scan job."""
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    update_scan_job(scan_db, job_id, status="completed")

    with pytest.raises(ValueError, match="best_model"):
        run_prediction(job_id, scan_db, lambda e: None)


def test_run_prediction_job_not_found(scan_db):
    """Should raise if job doesn't exist."""
    with pytest.raises(ValueError, match="not found"):
        run_prediction("nonexistent", scan_db, lambda e: None)
