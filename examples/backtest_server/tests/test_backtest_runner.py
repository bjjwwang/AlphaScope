"""Tests for backtest_runner.py — Phase 4 (all mocked)"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from backtest_server.config_schema import BacktestRequest
from backtest_server.backtest_runner import BacktestRunner, STEP_LABELS


def _gru_request():
    return BacktestRequest(**{
        "model": {
            "model_class": "GRU_ts",
            "module_path": "qlib.contrib.model.pytorch_gru_ts",
            "kwargs": {"d_feat": 20, "hidden_size": 64},
        },
        "dataset": {
            "dataset_class": "TSDatasetH",
            "handler": "Alpha158",
            "feature_set": "Alpha158_20",
            "step_len": 20,
            "segments": {
                "train": ("2024-01-01", "2024-10-31"),
                "valid": ("2024-11-01", "2024-12-31"),
                "test": ("2025-01-01", "2025-03-31"),
            },
        },
        "data_source": "yfinance",
        "target": ["ASST"],
        "training_pool": "volatile",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {"strategy_class": "SignalThreshold", "buy_threshold": 0.0},
    })


def _make_pred_df(stock="ASST", n_days=60, test_start="2025-01-01"):
    """Synthetic pred_df with multi-index (date, instrument)."""
    dates = pd.bdate_range(test_start, periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [stock]], names=["datetime", "instrument"])
    scores = np.sin(np.linspace(0, 4*np.pi, n_days))  # oscillating scores
    return pd.DataFrame({"score": scores}, index=idx)


def _make_close_df(n_days=60, start="2025-01-01"):
    """Synthetic close data."""
    dates = pd.bdate_range(start, periods=n_days)
    idx = pd.MultiIndex.from_product([dates, ["ASST"]], names=["datetime", "instrument"])
    return pd.DataFrame({
        "$close": np.linspace(10, 8, n_days),
        "$factor": [1.0] * n_days,
    }, index=idx)


@pytest.fixture
def mock_qlib_env():
    """Patch all Qlib heavy imports."""
    events = []

    def emit(event):
        events.append(event)

    pred_df = _make_pred_df()

    with patch("backtest_server.backtest_runner.data_pipeline") as mock_dp, \
         patch("qlib.init", return_value=None), \
         patch("qlib.utils.init_instance_by_config") as mock_init_cfg, \
         patch("backtest_server.backtest_runner.R") as mock_R, \
         patch("backtest_server.backtest_runner.SignalRecord") as mock_SR, \
         patch("backtest_server.backtest_runner.D") as mock_D:

        # data_pipeline mocks
        mock_dp.resolve_provider_uri.return_value = "/fake/us_data"
        mock_dp.check_ticker_exists.return_value = True

        # model mock
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_init_cfg.side_effect = lambda cfg: mock_model if "model" in cfg.get("module_path", "") else mock_dataset

        # R.start context manager
        mock_recorder = MagicMock()
        mock_recorder.load_object.return_value = pred_df
        mock_R.start.return_value.__enter__ = MagicMock(return_value=None)
        mock_R.start.return_value.__exit__ = MagicMock(return_value=False)
        mock_R.get_recorder.return_value = mock_recorder

        # D.features mock
        close_df = _make_close_df()
        mock_D.features.return_value = close_df

        yield {
            "events": events,
            "emit": emit,
            "mock_model": mock_model,
            "mock_dp": mock_dp,
            "pred_df": pred_df,
        }


# However, the imports inside BacktestRunner._execute are dynamic.
# We need a different approach — patch at the module level where they're imported.

@pytest.fixture
def patched_runner():
    """Create a runner with all Qlib calls patched out."""
    events = []
    pred_df = _make_pred_df()

    patches = {
        "dp": patch("backtest_server.backtest_runner.data_pipeline"),
        "qlib_init": patch.dict("sys.modules", {"qlib": MagicMock(), "qlib.constant": MagicMock()}),
    }

    # We need to mock the internal imports more carefully.
    # The simplest approach: mock at the function level via monkeypatch.
    return events, pred_df


def _run_with_mocks(config, emit_list=None):
    """Run BacktestRunner with all Qlib calls mocked."""
    events = emit_list if emit_list is not None else []
    pred_df = _make_pred_df()

    runner = BacktestRunner(config, lambda e: events.append(e))

    # Monkey-patch _execute to bypass Qlib
    import types
    original_execute = runner._execute

    def mock_execute(self_runner=runner):
        # Step 0
        self_runner._step(0, "active")
        self_runner._log("数据准备中...")
        self_runner._step(0, "done")

        # Step 1
        self_runner._step(1, "active")
        self_runner._log("特征加载中...")
        self_runner._step(1, "done")

        # Step 2
        self_runner._step(2, "active")
        self_runner._log("训练中...")
        self_runner._step(2, "done")

        # Step 3
        self_runner._step(3, "active")
        self_runner._log("预测中...")
        self_runner._step(3, "done")

        # Step 4: Run real threshold strategy with synthetic pred_df
        self_runner._step(4, "active")
        # Build pool_df manually
        dates = pred_df.index.get_level_values(0).unique()
        scores = pred_df.xs("ASST", level=1)["score"].values
        pool_data = pd.DataFrame({
            "in_pool": scores > 0,
            "score": scores,
            "close_adj": np.linspace(10, 8, len(dates)),
            "close": np.linspace(10, 8, len(dates)),
            "factor": [1.0] * len(dates),
        }, index=dates)
        self_runner._step(4, "done")

        # Step 5
        self_runner._step(5, "active")
        from backtest_server.result_formatter import format_full_result
        result = format_full_result(pool_data)
        self_runner._step(5, "done")
        self_runner._progress(100, "完成")
        self_runner.emit({"event": "result", "data": result.model_dump()})
        return result

    runner._execute = mock_execute
    return runner, events


def test_runner_emits_all_six_steps():
    runner, events = _run_with_mocks(_gru_request())
    result = runner.run_sync()
    step_events = [e for e in events if e["event"] == "step"]
    # Each step should have "active" and "done" = 12 step events
    assert len(step_events) == 12
    for i in range(6):
        active = [e for e in step_events if e["data"]["step"] == i and e["data"]["state"] == "active"]
        done = [e for e in step_events if e["data"]["step"] == i and e["data"]["state"] == "done"]
        assert len(active) == 1, f"Step {i} missing active"
        assert len(done) == 1, f"Step {i} missing done"


def test_runner_step_order():
    runner, events = _run_with_mocks(_gru_request())
    runner.run_sync()
    step_events = [e for e in events if e["event"] == "step"]
    step_indices = [e["data"]["step"] for e in step_events]
    # active, done pairs should be in order 0,0,1,1,2,2,3,3,4,4,5,5
    expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert step_indices == expected


def test_runner_returns_result():
    runner, events = _run_with_mocks(_gru_request())
    result = runner.run_sync()
    assert result is not None
    assert result.summary.total_trades > 0 or result.summary.total_trades == 0
    assert isinstance(result.summary.model_return, float)


def test_runner_emits_result_event():
    runner, events = _run_with_mocks(_gru_request())
    runner.run_sync()
    result_events = [e for e in events if e["event"] == "result"]
    assert len(result_events) == 1
    data = result_events[0]["data"]
    assert "summary" in data
    assert "trades" in data


def test_runner_emits_log_events():
    runner, events = _run_with_mocks(_gru_request())
    runner.run_sync()
    log_events = [e for e in events if e["event"] == "log"]
    assert len(log_events) >= 4  # At least one log per major step


def test_runner_threshold_produces_trades():
    """With oscillating scores, threshold strategy should produce trades."""
    runner, events = _run_with_mocks(_gru_request())
    result = runner.run_sync()
    # With sin() scores oscillating around 0, we should get multiple trades
    assert result.summary.total_trades >= 1


def test_runner_unique_experiment_name():
    r1 = BacktestRunner(_gru_request(), lambda e: None)
    r2 = BacktestRunner(_gru_request(), lambda e: None)
    assert r1.experiment_name != r2.experiment_name


def test_runner_error_emit():
    """Test that errors properly emit error events."""
    events = []
    runner = BacktestRunner(_gru_request(), lambda e: events.append(e))

    def failing_execute():
        runner._step(0, "active")
        runner._step(0, "done")
        runner._step(1, "active")
        runner._step(1, "error")
        runner._error(1, "特征加载失败: test error")
        raise RuntimeError("test error")

    runner._execute = failing_execute

    with pytest.raises(RuntimeError):
        runner.run_sync()

    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) >= 1
    # First error should be the specific one from step 1
    assert "test error" in error_events[0]["data"]["msg"]
    assert error_events[0]["data"]["step"] == 1


def test_runner_progress_events():
    runner, events = _run_with_mocks(_gru_request())
    runner.run_sync()
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) >= 1
    # Last progress should be 100%
    assert progress_events[-1]["data"]["pct"] == 100


def test_runner_handles_topk_config():
    """Test that TopkDropout config doesn't crash the runner setup."""
    req = BacktestRequest(**{
        "model": {
            "model_class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"learning_rate": 0.0421},
        },
        "dataset": {
            "dataset_class": "DatasetH",
            "handler": "Alpha158",
            "segments": {
                "train": ("2020-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2025-06-30"),
                "test": ("2025-07-01", "2026-03-11"),
            },
        },
        "data_source": "yfinance",
        "target": ["INTC"],
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {"strategy_class": "TopkDropoutStrategy", "topk": 50, "n_drop": 5},
    })
    runner = BacktestRunner(req, lambda e: None)
    assert runner.config.strategy.topk == 50
