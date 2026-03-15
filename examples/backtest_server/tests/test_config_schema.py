"""Tests for config_schema.py — Phase 1 RED→GREEN"""
import pytest
from pydantic import ValidationError
from backtest_server.config_schema import (
    BacktestRequest, ModelConfig, DatasetConfig, SegmentsConfig,
    StrategyConfig, BacktestResult, SummaryStats, TradeRecord,
    SegmentRecord, HoldingStats, EmptyStats, TimelineSegment,
)


def _make_valid_request(**overrides):
    """Helper to build a valid BacktestRequest dict with optional overrides."""
    base = {
        "model": {
            "model_class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"learning_rate": 0.0421, "max_depth": 8},
        },
        "dataset": {
            "dataset_class": "DatasetH",
            "handler": "Alpha158",
            "feature_set": "Alpha158",
            "segments": {
                "train": ("2020-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2025-06-30"),
                "test": ("2025-07-01", "2026-03-11"),
            },
        },
        "data_source": "yfinance",
        "kline": "daily",
        "target": ["INTC"],
        "training_pool": "all",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {
            "strategy_class": "TopkDropoutStrategy",
            "topk": 50,
            "n_drop": 5,
            "capital": 100000,
        },
    }
    base.update(overrides)
    return base


# ===== Valid Requests =====

def test_valid_lgb_request():
    req = BacktestRequest(**_make_valid_request())
    assert req.model.model_class == "LGBModel"
    assert req.target == ["INTC"]
    assert req.strategy.topk == 50


def test_valid_gru_ts_request():
    req = BacktestRequest(**_make_valid_request(
        model={
            "model_class": "GRU_ts",
            "module_path": "qlib.contrib.model.pytorch_gru_ts",
            "kwargs": {"d_feat": 20, "hidden_size": 128, "GPU": 0},
        },
        dataset={
            "dataset_class": "TSDatasetH",
            "handler": "Alpha158",
            "feature_set": "Alpha158_20",
            "step_len": 20,
            "segments": {
                "train": ("2020-01-01", "2024-06-30"),
                "valid": ("2024-07-01", "2024-12-31"),
                "test": ("2025-01-01", "2026-03-06"),
            },
        },
        strategy={
            "strategy_class": "SignalThreshold",
            "buy_threshold": 0.0,
            "capital": 100000,
        },
    ))
    assert req.dataset.dataset_class == "TSDatasetH"
    assert req.dataset.step_len == 20
    assert req.dataset.feature_set == "Alpha158_20"


# ===== Invalid Requests =====

def test_missing_model_class_raises():
    d = _make_valid_request()
    del d["model"]["model_class"]
    with pytest.raises(ValidationError):
        BacktestRequest(**d)


def test_empty_target_list_raises():
    with pytest.raises(ValidationError, match="At least one target"):
        BacktestRequest(**_make_valid_request(target=[]))


def test_invalid_date_overlap_raises():
    d = _make_valid_request()
    # train end overlaps valid start
    d["dataset"]["segments"]["train"] = ("2020-01-01", "2025-06-01")
    with pytest.raises(ValidationError, match="Train end"):
        BacktestRequest(**d)


def test_topk_without_topk_value_raises():
    d = _make_valid_request()
    d["strategy"] = {"strategy_class": "TopkDropoutStrategy", "capital": 100000}
    with pytest.raises(ValidationError, match="topk is required"):
        BacktestRequest(**d)


def test_threshold_without_value_raises():
    d = _make_valid_request()
    d["strategy"] = {"strategy_class": "SignalThreshold", "capital": 100000}
    with pytest.raises(ValidationError, match="buy_threshold is required"):
        BacktestRequest(**d)


def test_invalid_model_class_raises():
    d = _make_valid_request()
    d["model"]["model_class"] = "NotARealModel"
    with pytest.raises(ValidationError, match="Unknown model"):
        BacktestRequest(**d)


def test_invalid_data_source_raises():
    with pytest.raises(ValidationError, match="Invalid data source"):
        BacktestRequest(**_make_valid_request(data_source="random"))


def test_empty_label_raises():
    with pytest.raises(ValidationError, match="Label expression cannot be empty"):
        BacktestRequest(**_make_valid_request(label="  "))


def test_negative_capital_raises():
    d = _make_valid_request()
    d["strategy"]["capital"] = -1000
    with pytest.raises(ValidationError, match="Capital must be positive"):
        BacktestRequest(**d)


def test_result_roundtrip():
    result = BacktestResult(
        summary=SummaryStats(
            model_return=59.95, bh_return=-28.63, win_rate=69.2,
            total_trades=26, test_start="2020-01-01", test_end="2026-03-06",
            start_price=60.0, end_price=25.0, held_days=180, total_days=365,
        ),
        trades=[TradeRecord(
            buy_date="2020-01-06", buy_price=60.0,
            sell_date="2020-02-10", sell_price=65.0,
            days=35, return_pct=8.33,
        )],
        segments=[SegmentRecord(
            type="hold", start="2020-01-06", end="2020-02-10",
            start_price=60.0, end_price=65.0, days=35, change_pct=8.33,
        )],
        holding_stats=HoldingStats(
            count=26, total_days=180, profit_segs=18, loss_segs=8, avg_return=15.3,
        ),
        empty_stats=EmptyStats(count=25, total_days=160, missed_up=123.4, avoided_down=-89.2),
        timeline=[TimelineSegment(type="hold", pct=30.0), TimelineSegment(type="empty", pct=70.0)],
    )
    # Serialize → deserialize
    json_str = result.model_dump_json()
    restored = BacktestResult.model_validate_json(json_str)
    assert restored.summary.model_return == 59.95
    assert len(restored.trades) == 1
    assert restored.trades[0].buy_price == 60.0
    assert restored.holding_stats.count == 26
