"""Tests for scanner.py — scan config generation, model list, ranking."""
import pytest
from datetime import datetime, timedelta

from backtest_server.config_schema import VALID_MODEL_CLASSES, BacktestRequest
from backtest_server.scanner import (
    MODEL_SCAN_LIST,
    TRADING_STYLE_PRESETS,
    build_scan_config,
    compute_date_segments,
    rank_results,
)


# --- Trading style presets ---

def test_ultra_short_preset_exists():
    p = TRADING_STYLE_PRESETS["ultra_short"]
    assert p["feature_set"] == "Alpha158_20"
    assert p["train_months"] == 6
    assert p["valid_months"] == 1
    assert p["test_months"] == 1
    assert "label" in p


def test_swing_preset_exists():
    p = TRADING_STYLE_PRESETS["swing"]
    assert p["feature_set"] == "Alpha158"
    assert p["train_months"] == 24
    assert p["valid_months"] == 6
    assert p["test_months"] == 3
    assert "label" in p


def test_preset_date_ranges_valid():
    for style in TRADING_STYLE_PRESETS:
        seg = compute_date_segments(style)
        # train end < valid start
        assert seg.train[1] < seg.valid[0], f"{style}: train overlaps valid"
        # valid end < test start
        assert seg.valid[1] < seg.test[0], f"{style}: valid overlaps test"
        # All dates are valid date strings
        for d in [seg.train[0], seg.train[1], seg.valid[0], seg.valid[1], seg.test[0], seg.test[1]]:
            datetime.strptime(d, "%Y-%m-%d")


# --- Model scan list ---

def test_model_scan_list_has_22():
    """22 models scannable (TRAModel, HIST excluded)."""
    assert len(MODEL_SCAN_LIST) == 22


def test_model_scan_list_plus_excluded_equals_valid():
    from backtest_server.scanner import SCAN_EXCLUDED
    scan_classes = {m["model_class"] for m in MODEL_SCAN_LIST}
    assert scan_classes | SCAN_EXCLUDED == VALID_MODEL_CLASSES


def test_model_scan_list_entries_have_required_fields():
    for m in MODEL_SCAN_LIST:
        assert "model_class" in m
        assert "module_path" in m
        assert "data_type" in m
        assert m["data_type"] in ("flat", "ts")


# --- Config generation ---

def test_build_scan_config_lgb():
    config = build_scan_config("AAPL", "db", "ultra_short", "LGBModel")
    assert isinstance(config, BacktestRequest)
    assert config.target == ["AAPL"]
    assert config.data_source == "db"
    assert config.dataset.feature_set == "Alpha158_20"
    assert config.model.model_class == "LGBModel"
    assert config.training_pool == "target_only"
    assert config.strategy.strategy_class == "SignalThreshold"
    assert config.strategy.buy_threshold == 0.0


def test_build_scan_config_gru_ts():
    config = build_scan_config("MSFT", "yfinance", "swing", "GRU_ts")
    assert config.target == ["MSFT"]
    assert config.data_source == "yfinance"
    assert config.dataset.feature_set == "Alpha158"
    assert config.dataset.dataset_class == "TSDatasetH"
    assert config.dataset.step_len == 20
    assert config.model.model_class == "GRU_ts"


def test_build_scan_config_baostock():
    config = build_scan_config("600519", "baostock", "ultra_short", "LGBModel")
    assert config.data_source == "baostock"


def test_build_scan_config_all_models_valid():
    """Every model in MODEL_SCAN_LIST produces a valid BacktestRequest."""
    for m in MODEL_SCAN_LIST:
        config = build_scan_config("AAPL", "db", "ultra_short", m["model_class"])
        assert isinstance(config, BacktestRequest), f"Failed for {m['model_class']}"


# --- Result ranking ---

def test_rank_results_by_model_return():
    results = [
        {"model_class": "A", "status": "completed", "summary": {"model_return": 5.0, "win_rate": 50}},
        {"model_class": "B", "status": "completed", "summary": {"model_return": 15.0, "win_rate": 60}},
        {"model_class": "C", "status": "completed", "summary": {"model_return": 10.0, "win_rate": 55}},
    ]
    ranked = rank_results(results)
    assert [r["model_class"] for r in ranked] == ["B", "C", "A"]


def test_rank_results_skips_failed():
    results = [
        {"model_class": "A", "status": "completed", "summary": {"model_return": 5.0, "win_rate": 50}},
        {"model_class": "B", "status": "failed", "summary": None},
        {"model_class": "C", "status": "skipped", "summary": None},
    ]
    ranked = rank_results(results)
    assert len(ranked) == 1
    assert ranked[0]["model_class"] == "A"


def test_rank_results_empty():
    assert rank_results([]) == []
    assert rank_results([{"model_class": "A", "status": "failed", "summary": None}]) == []


def test_rank_results_tiebreak_by_win_rate():
    results = [
        {"model_class": "A", "status": "completed", "summary": {"model_return": 10.0, "win_rate": 40}},
        {"model_class": "B", "status": "completed", "summary": {"model_return": 10.0, "win_rate": 70}},
    ]
    ranked = rank_results(results)
    assert ranked[0]["model_class"] == "B"
    assert ranked[1]["model_class"] == "A"


# ======================================================================
# ScanRunner tests (all mock BacktestRunner — no Qlib/GPU needed)
# ======================================================================
import json
from unittest.mock import patch, MagicMock
from backtest_server.scan_db import init_db, create_scan_job, get_scan_job, get_model_results
from backtest_server.scanner import ScanRunner
from backtest_server.config_schema import (
    BacktestResult, SummaryStats,
)


def _fake_result(model_return=10.0, win_rate=55.0, trades=5):
    return BacktestResult(
        summary=SummaryStats(
            model_return=model_return,
            bh_return=5.0,
            win_rate=win_rate,
            total_trades=trades,
            test_start="2025-01-01",
            test_end="2025-03-31",
            held_days=40,
            total_days=60,
        ),
        trades=[],
        segments=[],
    )


@pytest.fixture
def scan_db(tmp_path):
    db_path = str(tmp_path / "scan_test.db")
    init_db(db_path)
    return db_path


def _mock_runner_factory(results_by_model=None, fail_models=None):
    """Create a mock that replaces BacktestRunner.

    results_by_model: dict of model_class -> BacktestResult (or None for default)
    fail_models: set of model_class names that should raise
    """
    fail_models = fail_models or set()
    results_by_model = results_by_model or {}

    class FakeRunner:
        def __init__(self, config, emit):
            self.config = config
            self.emit = emit
            self.experiment_name = "fake_exp"

        def run_sync(self):
            mc = self.config.model.model_class
            if mc in fail_models:
                raise RuntimeError(f"Fake error for {mc}")
            result = results_by_model.get(mc, _fake_result())
            self.emit({"event": "result", "data": result.model_dump()})
            return result

    return FakeRunner


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_iterates_all_models(MockRunner, mock_dp, scan_db):
    MockRunner.side_effect = _mock_runner_factory()
    events = []
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: events.append(e), scan_db, job_id)
    runner.run_sync()

    # Should have results for all 22 scannable models
    results = get_model_results(scan_db, job_id)
    assert len(results) == 22
    completed = [r for r in results if r["status"] == "completed"]
    assert len(completed) == 22

    job = get_scan_job(scan_db, job_id)
    assert job["status"] == "completed"
    assert job["completed_models"] == 22
    assert job["best_model"] is not None


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_handles_model_failure(MockRunner, mock_dp, scan_db):
    MockRunner.side_effect = _mock_runner_factory(fail_models={"GRU_ts", "ADARNN"})
    events = []
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: events.append(e), scan_db, job_id)
    runner.run_sync()

    results = get_model_results(scan_db, job_id)
    assert len(results) == 22
    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 2
    failed_models = {r["model_class"] for r in failed}
    assert failed_models == {"GRU_ts", "ADARNN"}

    # Scan still completes
    job = get_scan_job(scan_db, job_id)
    assert job["status"] == "completed"


@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_handles_model_skip(MockRunner, mock_dp, scan_db):
    # Mark XGBModel and CatBoostModel as unavailable
    avail = {mc: True for mc in VALID_MODEL_CLASSES}
    avail["XGBModel"] = False
    avail["CatBoostModel"] = False
    with patch("backtest_server.scanner._MODEL_AVAILABILITY", avail):
        MockRunner.side_effect = _mock_runner_factory()
        events = []
        job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
        runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: events.append(e), scan_db, job_id)
        runner.run_sync()

    results = get_model_results(scan_db, job_id)
    skipped = [r for r in results if r["status"] == "skipped"]
    assert len(skipped) == 2
    skipped_models = {r["model_class"] for r in skipped}
    assert skipped_models == {"XGBModel", "CatBoostModel"}


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_selects_best_model(MockRunner, mock_dp, scan_db):
    results_map = {
        "LGBModel": _fake_result(model_return=5.0),
        "GRU_ts": _fake_result(model_return=20.0),
        "LSTM_ts": _fake_result(model_return=15.0),
    }
    MockRunner.side_effect = _mock_runner_factory(results_by_model=results_map)
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: None, scan_db, job_id)
    runner.run_sync()

    job = get_scan_job(scan_db, job_id)
    assert job["best_model"] == "GRU_ts"
    assert "20" in job["best_model_reason"]


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_emits_progress_events(MockRunner, mock_dp, scan_db):
    MockRunner.side_effect = _mock_runner_factory()
    events = []
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: events.append(e), scan_db, job_id)
    runner.run_sync()

    event_types = [e["event"] for e in events]
    assert "scan_start" in event_types
    assert "scan_model_start" in event_types
    assert "scan_model_complete" in event_types
    assert "scan_progress" in event_types
    assert "scan_complete" in event_types


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_updates_db_progress(MockRunner, mock_dp, scan_db):
    call_count = [0]
    original_factory = _mock_runner_factory()

    class TrackingRunner:
        def __init__(self, config, emit):
            self.inner = original_factory(config, emit)
            self.config = config
            self.emit = emit
            self.experiment_name = "fake_exp"

        def run_sync(self):
            result = self.inner.run_sync()
            call_count[0] += 1
            return result

    MockRunner.side_effect = TrackingRunner
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: None, scan_db, job_id)
    runner.run_sync()

    job = get_scan_job(scan_db, job_id)
    assert job["completed_models"] == 22
    assert call_count[0] == 22


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_data_source_passed_correctly(MockRunner, mock_dp, scan_db):
    configs_seen = []

    class ConfigCapture:
        def __init__(self, config, emit):
            configs_seen.append(config)
            self.config = config
            self.emit = emit
            self.experiment_name = "fake_exp"

        def run_sync(self):
            r = _fake_result()
            self.emit({"event": "result", "data": r.model_dump()})
            return r

    MockRunner.side_effect = ConfigCapture
    job_id = create_scan_job(scan_db, "AAPL", "yfinance", "ultra_short")
    runner = ScanRunner("AAPL", "yfinance", "ultra_short", lambda e: None, scan_db, job_id)
    runner.run_sync()

    # All configs should use yfinance
    assert all(c.data_source == "yfinance" for c in configs_seen)


@patch("backtest_server.scanner._MODEL_AVAILABILITY", {mc: True for mc in VALID_MODEL_CLASSES})
@patch("backtest_server.scanner.data_pipeline")
@patch("backtest_server.scanner.BacktestRunner")
def test_scanner_scan_complete_has_rankings(MockRunner, mock_dp, scan_db):
    results_map = {
        "LGBModel": _fake_result(model_return=5.0),
        "GRU_ts": _fake_result(model_return=20.0),
    }
    MockRunner.side_effect = _mock_runner_factory(results_by_model=results_map)
    events = []
    job_id = create_scan_job(scan_db, "AAPL", "db", "ultra_short")
    runner = ScanRunner("AAPL", "db", "ultra_short", lambda e: events.append(e), scan_db, job_id)
    runner.run_sync()

    complete_events = [e for e in events if e["event"] == "scan_complete"]
    assert len(complete_events) == 1
    data = complete_events[0]["data"]
    assert data["best_model"] == "GRU_ts"
    assert "rankings" in data
    assert len(data["rankings"]) > 0
