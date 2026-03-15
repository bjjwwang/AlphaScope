"""Tests for pretrained model system — scan_db extensions, pretrain_manager, and API."""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backtest_server.scan_db import (
    init_db,
    save_pretrained_model,
    get_pretrained_model,
    update_pretrained_model,
    find_pretrained_model,
    list_pretrained_models,
    delete_pretrained_models_by_style,
    save_cached_prediction,
    get_cached_prediction,
    list_cached_predictions,
    clear_cached_predictions,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "pretrain_test.db")
    init_db(db_path)
    return db_path


# =============================================
# Pretrained model registry DB tests
# =============================================

def test_init_creates_pretrained_tables(tmp_path):
    db_path = str(tmp_path / "fresh.db")
    init_db(db_path)
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "pretrained_models" in tables
    assert "cached_predictions" in tables


def test_save_pretrained_model(db):
    model_id = save_pretrained_model(
        db, "pt_LGB_swing_abc", "LGBModel", "Alpha158",
        "Ref($close, -6)/Ref($close, -1) - 1", "all", "swing",
        "2024-01-01", "2025-12-31", "pretrain_exp_abc",
        performance_json='{"model_return": 12.5, "win_rate": 55.0}',
        status="completed", duration_s=45.0,
    )
    assert model_id == "pt_LGB_swing_abc"


def test_get_pretrained_model(db):
    save_pretrained_model(
        db, "pt_test_1", "GRU_ts", "Alpha158_20",
        "Ref($close, -2)/Ref($close, -1) - 1", "all", "ultra_short",
        "2025-06-01", "2025-12-31", "pretrain_exp_gru",
        status="completed",
    )
    m = get_pretrained_model(db, "pt_test_1")
    assert m is not None
    assert m["model_class"] == "GRU_ts"
    assert m["feature_set"] == "Alpha158_20"
    assert m["trading_style"] == "ultra_short"
    assert m["status"] == "completed"


def test_get_pretrained_model_not_found(db):
    assert get_pretrained_model(db, "nonexistent") is None


def test_update_pretrained_model(db):
    save_pretrained_model(
        db, "pt_upd_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_1", status="training",
    )
    update_pretrained_model(db, "pt_upd_1", status="completed",
                            performance_json='{"model_return": 10.0}')
    m = get_pretrained_model(db, "pt_upd_1")
    assert m["status"] == "completed"
    assert "10.0" in m["performance_json"]


def test_find_pretrained_model(db):
    save_pretrained_model(
        db, "pt_find_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_find_1", status="completed",
    )
    save_pretrained_model(
        db, "pt_find_2", "LGBModel", "Alpha158",
        "label", "all", "ultra_short", "2025-06-01", "2025-12-31",
        "exp_find_2", status="completed",
    )
    # Should find the swing one
    m = find_pretrained_model(db, "LGBModel", "Alpha158", "swing")
    assert m is not None
    assert m["id"] == "pt_find_1"

    # Should find the ultra_short one
    m2 = find_pretrained_model(db, "LGBModel", "Alpha158", "ultra_short")
    assert m2 is not None
    assert m2["id"] == "pt_find_2"

    # Should not find non-existent combo
    assert find_pretrained_model(db, "LGBModel", "Alpha360", "swing") is None


def test_find_pretrained_model_only_completed(db):
    """Should only return completed models."""
    save_pretrained_model(
        db, "pt_fail_1", "GRU_ts", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_fail", status="failed",
    )
    assert find_pretrained_model(db, "GRU_ts", "Alpha158", "swing") is None


def test_list_pretrained_models(db):
    save_pretrained_model(
        db, "pt_list_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_1", status="completed",
    )
    save_pretrained_model(
        db, "pt_list_2", "GRU_ts", "Alpha158_20",
        "label", "all", "ultra_short", "2025-06-01", "2025-12-31",
        "exp_2", status="failed",
    )
    # All
    all_models = list_pretrained_models(db)
    assert len(all_models) == 2
    # Filtered
    completed = list_pretrained_models(db, status="completed")
    assert len(completed) == 1
    assert completed[0]["model_class"] == "LGBModel"

    failed = list_pretrained_models(db, status="failed")
    assert len(failed) == 1
    assert failed[0]["model_class"] == "GRU_ts"


def test_delete_pretrained_models_by_style(db):
    save_pretrained_model(
        db, "pt_del_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_1", status="completed",
    )
    save_pretrained_model(
        db, "pt_del_2", "GRU_ts", "Alpha158_20",
        "label", "all", "ultra_short", "2025-06-01", "2025-12-31",
        "exp_2", status="completed",
    )
    delete_pretrained_models_by_style(db, "swing")
    remaining = list_pretrained_models(db)
    assert len(remaining) == 1
    assert remaining[0]["trading_style"] == "ultra_short"


def test_save_pretrained_model_replace(db):
    """INSERT OR REPLACE should update existing model."""
    save_pretrained_model(
        db, "pt_rep_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_old", status="training",
    )
    save_pretrained_model(
        db, "pt_rep_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_new", status="completed",
    )
    m = get_pretrained_model(db, "pt_rep_1")
    assert m["experiment_name"] == "exp_new"
    assert m["status"] == "completed"
    # Should be only one entry
    all_models = list_pretrained_models(db)
    assert len(all_models) == 1


# =============================================
# Cached predictions DB tests
# =============================================

def test_save_cached_prediction(db):
    row_id = save_cached_prediction(
        db, "AAPL", "pt_model_1", "LGBModel", "swing",
        '{"latest_signal": "buy", "latest_score": 0.05}',
        '{"score": 7, "label": "bullish"}',
    )
    assert row_id > 0


def test_get_cached_prediction(db):
    save_cached_prediction(
        db, "AAPL", "pt_model_1", "LGBModel", "swing",
        '{"latest_signal": "buy"}',
    )
    cached = get_cached_prediction(db, "AAPL", "swing")
    assert cached is not None
    assert cached["ticker"] == "AAPL"
    assert cached["model_class"] == "LGBModel"
    assert cached["trading_style"] == "swing"
    pred = json.loads(cached["prediction_json"])
    assert pred["latest_signal"] == "buy"


def test_get_cached_prediction_not_found(db):
    assert get_cached_prediction(db, "AAPL", "swing") is None


def test_get_cached_prediction_case_insensitive(db):
    save_cached_prediction(
        db, "aapl", "pt_1", "LGBModel", "swing", '{"signal": "buy"}',
    )
    # Should find via uppercase
    assert get_cached_prediction(db, "AAPL", "swing") is not None


def test_cached_prediction_replaces_old(db):
    """Saving for same ticker+style should replace old entry."""
    save_cached_prediction(
        db, "AAPL", "pt_1", "LGBModel", "swing", '{"signal": "buy"}',
    )
    save_cached_prediction(
        db, "AAPL", "pt_2", "GRU_ts", "swing", '{"signal": "sell"}',
    )
    cached = get_cached_prediction(db, "AAPL", "swing")
    assert cached["model_class"] == "GRU_ts"
    pred = json.loads(cached["prediction_json"])
    assert pred["signal"] == "sell"

    # Should be only one entry
    all_cached = list_cached_predictions(db, "swing")
    aapl_entries = [c for c in all_cached if c["ticker"] == "AAPL"]
    assert len(aapl_entries) == 1


def test_list_cached_predictions(db):
    save_cached_prediction(db, "AAPL", "pt_1", "LGBModel", "swing", '{}')
    save_cached_prediction(db, "MSFT", "pt_1", "LGBModel", "swing", '{}')
    save_cached_prediction(db, "AAPL", "pt_2", "GRU_ts", "ultra_short", '{}')

    all_cached = list_cached_predictions(db)
    assert len(all_cached) == 3

    swing_only = list_cached_predictions(db, "swing")
    assert len(swing_only) == 2

    ultra_only = list_cached_predictions(db, "ultra_short")
    assert len(ultra_only) == 1


def test_clear_cached_predictions_all(db):
    save_cached_prediction(db, "AAPL", "pt_1", "LGBModel", "swing", '{}')
    save_cached_prediction(db, "MSFT", "pt_1", "LGBModel", "ultra_short", '{}')
    clear_cached_predictions(db)
    assert list_cached_predictions(db) == []


def test_clear_cached_predictions_by_style(db):
    save_cached_prediction(db, "AAPL", "pt_1", "LGBModel", "swing", '{}')
    save_cached_prediction(db, "MSFT", "pt_1", "LGBModel", "ultra_short", '{}')
    clear_cached_predictions(db, "swing")
    remaining = list_cached_predictions(db)
    assert len(remaining) == 1
    assert remaining[0]["trading_style"] == "ultra_short"


# =============================================
# Pretrain manager tests
# =============================================

def test_pretrain_matrix_has_entries():
    from backtest_server.pretrain_manager import PRETRAIN_MATRIX
    # 22 models (scan list) × 2 styles = 44
    assert len(PRETRAIN_MATRIX) >= 40


def test_quick_scan_models():
    from backtest_server.pretrain_manager import QUICK_SCAN_MODELS
    assert "LGBModel" in QUICK_SCAN_MODELS
    assert len(QUICK_SCAN_MODELS) == 5


def test_build_pretrain_config():
    from backtest_server.pretrain_manager import build_pretrain_config
    config = build_pretrain_config("LGBModel", "swing", "all")
    assert config.model.model_class == "LGBModel"
    assert config.training_pool == "all"
    assert config.dataset.feature_set == "Alpha158"
    # Swing uses Alpha158 with 24-month training period
    assert config.label == "Ref($close, -6)/Ref($close, -1) - 1"


def test_build_pretrain_config_ultra_short():
    from backtest_server.pretrain_manager import build_pretrain_config
    config = build_pretrain_config("GRU_ts", "ultra_short", "all")
    assert config.model.model_class == "GRU_ts"
    assert config.dataset.feature_set == "Alpha158_20"
    assert config.dataset.dataset_class == "TSDatasetH"
    assert config.dataset.step_len == 20


def test_build_pretrain_config_dnn():
    from backtest_server.pretrain_manager import build_pretrain_config
    config = build_pretrain_config("DNNModelPytorch", "swing", "all")
    assert "pt_model_kwargs" in config.model.kwargs
    assert config.model.kwargs["pt_model_kwargs"]["input_dim"] == 158


# =============================================
# API endpoint tests
# =============================================

@pytest.fixture
def api_db(tmp_path):
    db_path = str(tmp_path / "api_pretrain_test.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def client(api_db):
    from fastapi.testclient import TestClient
    from backtest_server.server import app
    with patch("backtest_server.server._scan_db_path", api_db):
        yield TestClient(app)


def test_api_quick_predict_no_pretrained(client):
    """Should return 404 if no pretrained models available."""
    resp = client.post("/api/quick-predict", json={
        "ticker": "AAPL", "trading_style": "swing",
    })
    assert resp.status_code == 404


def test_api_quick_predict_cache_hit(client, api_db):
    """Should return cached result if available."""
    save_cached_prediction(
        api_db, "AAPL", "pt_model_1", "LGBModel", "swing",
        json.dumps({
            "ticker": "AAPL", "model_class": "LGBModel",
            "prediction_range": ["2026-01-01", "2026-03-13"],
            "signals": [{"date": "2026-03-13", "score": 0.05, "signal": "buy"}],
            "latest_signal": "buy", "latest_score": 0.05,
            "threshold": 0.02, "generated_at": "2026-03-13T10:00:00",
        }),
    )
    resp = client.post("/api/quick-predict", json={
        "ticker": "AAPL", "trading_style": "swing",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "cache"
    assert data["prediction"]["latest_signal"] == "buy"


def test_api_quick_predict_empty_ticker(client):
    resp = client.post("/api/quick-predict", json={
        "ticker": "", "trading_style": "swing",
    })
    assert resp.status_code == 400


def test_api_list_pretrained_models_empty(client):
    resp = client.get("/api/pretrained-models")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_list_pretrained_models(client, api_db):
    save_pretrained_model(
        api_db, "pt_api_1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_1", status="completed",
        performance_json='{"model_return": 10.0}',
    )
    resp = client.get("/api/pretrained-models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["model_class"] == "LGBModel"
    assert data[0]["performance"]["model_return"] == 10.0


def test_api_list_pretrained_models_filter(client, api_db):
    save_pretrained_model(
        api_db, "pt_f1", "LGBModel", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_1", status="completed",
    )
    save_pretrained_model(
        api_db, "pt_f2", "GRU_ts", "Alpha158",
        "label", "all", "swing", "2024-01-01", "2025-12-31",
        "exp_2", status="failed",
    )
    resp = client.get("/api/pretrained-models?status=completed")
    assert len(resp.json()) == 1

    resp = client.get("/api/pretrained-models?status=failed")
    assert len(resp.json()) == 1


def test_api_cached_predictions_empty(client):
    resp = client.get("/api/cached-predictions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_cached_predictions_list(client, api_db):
    save_cached_prediction(api_db, "AAPL", "pt_1", "LGBModel", "swing", '{}')
    save_cached_prediction(api_db, "MSFT", "pt_1", "LGBModel", "ultra_short", '{}')

    resp = client.get("/api/cached-predictions")
    assert len(resp.json()) == 2

    resp = client.get("/api/cached-predictions?trading_style=swing")
    assert len(resp.json()) == 1


def test_api_score_from_cache(client, api_db):
    """Score endpoint should use cached prediction if available."""
    save_cached_prediction(
        api_db, "TSLA", "pt_1", "LGBModel", "swing",
        '{"latest_signal": "buy", "latest_score": 0.1}',
        alpha_score_json='{"score": 8, "label": "bullish", "signal": "buy"}',
    )
    resp = client.get("/api/score/TSLA?trading_style=swing")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "cached_prediction"
    assert data["alpha_score"]["score"] == 8


def test_api_score_no_data(client, api_db):
    resp = client.get("/api/score/UNKNOWN?trading_style=swing")
    assert resp.status_code == 404


def test_api_start_pretrain(client, api_db):
    """Pretrain endpoint should return job_id."""
    with patch("backtest_server.pretrain_manager.PretrainRunner") as MockRunner:
        mock_instance = MagicMock()
        mock_instance.run_sync = MagicMock()
        MockRunner.return_value = mock_instance

        resp = client.post("/api/pretrain?quick=true")
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["mode"] == "quick"


# =============================================
# BacktestRunner with pretrained experiment
# =============================================

def test_backtest_runner_accepts_pretrained_experiment():
    """BacktestRunner should accept pretrained_experiment parameter."""
    from backtest_server.backtest_runner import BacktestRunner
    from backtest_server.config_schema import (
        BacktestRequest, ModelConfig, DatasetConfig,
        SegmentsConfig, StrategyConfig,
    )
    config = BacktestRequest(
        model=ModelConfig(model_class="LGBModel", module_path="qlib.contrib.model.gbdt"),
        dataset=DatasetConfig(
            dataset_class="DatasetH", handler="Alpha158",
            segments=SegmentsConfig(
                train=("2024-01-01", "2025-06-30"),
                valid=("2025-07-01", "2025-09-30"),
                test=("2025-10-01", "2025-12-31"),
            ),
        ),
        data_source="db", target=["AAPL"],
        label="Ref($close, -2)/Ref($close, -1) - 1",
        strategy=StrategyConfig(strategy_class="SignalThreshold", buy_threshold=0.0),
    )
    runner = BacktestRunner(config, lambda e: None, pretrained_experiment="exp_abc")
    assert runner.pretrained_experiment == "exp_abc"


def test_backtest_runner_default_no_pretrained():
    """Without pretrained_experiment, it should be None."""
    from backtest_server.backtest_runner import BacktestRunner
    from backtest_server.config_schema import (
        BacktestRequest, ModelConfig, DatasetConfig,
        SegmentsConfig, StrategyConfig,
    )
    config = BacktestRequest(
        model=ModelConfig(model_class="LGBModel", module_path="qlib.contrib.model.gbdt"),
        dataset=DatasetConfig(
            dataset_class="DatasetH", handler="Alpha158",
            segments=SegmentsConfig(
                train=("2024-01-01", "2025-06-30"),
                valid=("2025-07-01", "2025-09-30"),
                test=("2025-10-01", "2025-12-31"),
            ),
        ),
        data_source="db", target=["AAPL"],
        label="Ref($close, -2)/Ref($close, -1) - 1",
        strategy=StrategyConfig(strategy_class="SignalThreshold", buy_threshold=0.0),
    )
    runner = BacktestRunner(config, lambda e: None)
    assert runner.pretrained_experiment is None


# =============================================
# Quick prediction (predictor.py extension)
# =============================================

def test_quick_prediction_no_pretrained_model(db):
    """run_quick_prediction should raise if no pretrained model found."""
    from backtest_server.predictor import run_quick_prediction
    with pytest.raises(ValueError, match="No pretrained"):
        run_quick_prediction("AAPL", "swing", db)


def test_quick_prediction_no_model_class_fallback(db):
    """Should try QUICK_SCAN_MODELS in order when model_class not specified."""
    from backtest_server.predictor import run_quick_prediction
    # No pretrained models at all
    with pytest.raises(ValueError, match="No pretrained"):
        run_quick_prediction("AAPL", "ultra_short", db)


# =============================================
# Scheduled scan module tests
# =============================================

def test_sp500_top_list():
    from backtest_server.scheduled_scan import SP500_TOP
    assert len(SP500_TOP) == 50
    assert "AAPL" in SP500_TOP
    assert "MSFT" in SP500_TOP
