"""Tests for config_converter.py — Phase 1"""
import pytest
from backtest_server.config_schema import BacktestRequest
from backtest_server.config_converter import (
    convert_model_config, convert_dataset_config, convert, TOP20_FEATURES,
)


def _lgb_request():
    return BacktestRequest(**{
        "model": {
            "model_class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"learning_rate": 0.0421, "max_depth": 8, "num_leaves": 210,
                       "lambda_l1": 205.7, "lambda_l2": 580.98, "subsample": 0.88},
        },
        "dataset": {
            "dataset_class": "DatasetH",
            "handler": "Alpha158",
            "feature_set": "Alpha158",
            "segments": {
                "train": ("2000-01-01", "2017-12-31"),
                "valid": ("2018-01-01", "2019-12-31"),
                "test": ("2020-01-01", "2026-03-06"),
            },
        },
        "data_source": "yfinance",
        "target": ["INTC"],
        "training_pool": "all",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {"strategy_class": "TopkDropoutStrategy", "topk": 50, "n_drop": 5},
    })


def _gru_request():
    return BacktestRequest(**{
        "model": {
            "model_class": "GRU_ts",
            "module_path": "qlib.contrib.model.pytorch_gru_ts",
            "kwargs": {"d_feat": 20, "hidden_size": 128, "num_layers": 2,
                       "dropout": 0.3, "n_epochs": 200, "lr": 5e-4,
                       "early_stop": 20, "batch_size": 64},
        },
        "dataset": {
            "dataset_class": "TSDatasetH",
            "handler": "Alpha158",
            "feature_set": "Alpha158_20",
            "step_len": 20,
            "segments": {
                "train": ("2015-01-01", "2024-06-30"),
                "valid": ("2024-07-01", "2024-12-31"),
                "test": ("2025-01-01", "2026-03-06"),
            },
        },
        "data_source": "yfinance",
        "target": ["ASST"],
        "training_pool": "volatile",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {"strategy_class": "SignalThreshold", "buy_threshold": 0.0},
    })


def test_lgb_model_config():
    mc = convert_model_config(_lgb_request())
    assert mc["class"] == "LGBModel"
    assert mc["module_path"] == "qlib.contrib.model.gbdt"
    assert mc["kwargs"]["learning_rate"] == 0.0421
    assert "loss" not in mc["kwargs"]  # LGB is not a PyTorch model, no loss default
    assert "GPU" not in mc["kwargs"]  # LGB doesn't need GPU


def test_gru_ts_model_config():
    mc = convert_model_config(_gru_request())
    assert mc["class"] == "GRU"  # suffix stripped
    assert mc["kwargs"]["GPU"] == 0
    assert mc["kwargs"]["d_feat"] == 20


def test_dataset_flat():
    dc = convert_dataset_config(_lgb_request())
    assert dc["class"] == "DatasetH"
    assert dc["module_path"] == "qlib.data.dataset"
    assert dc["kwargs"]["handler"]["class"] == "Alpha158"
    assert dc["kwargs"]["handler"]["kwargs"]["instruments"] == "all"
    # No step_len for flat
    assert "step_len" not in dc["kwargs"]
    # Full Alpha158: no FilterCol, but has RobustZScoreNorm + Fillna
    infer_procs = dc["kwargs"]["handler"]["kwargs"]["infer_processors"]
    assert len(infer_procs) == 2
    assert infer_procs[0]["class"] == "RobustZScoreNorm"
    assert infer_procs[1]["class"] == "Fillna"


def test_dataset_ts_with_filter():
    dc = convert_dataset_config(_gru_request())
    assert dc["class"] == "TSDatasetH"
    assert dc["kwargs"]["step_len"] == 20
    h_kwargs = dc["kwargs"]["handler"]["kwargs"]
    assert h_kwargs["instruments"] == "volatile_us"
    # Must have FilterCol + RobustZScoreNorm + Fillna
    infer_procs = h_kwargs["infer_processors"]
    assert infer_procs[0]["class"] == "FilterCol"
    assert infer_procs[0]["kwargs"]["col_list"] == TOP20_FEATURES
    assert infer_procs[1]["class"] == "RobustZScoreNorm"
    assert infer_procs[2]["class"] == "Fillna"
    # learn_processors
    learn_procs = h_kwargs["learn_processors"]
    assert learn_procs[0]["class"] == "DropnaLabel"
    assert learn_procs[1]["class"] == "CSRankNorm"


def test_custom_features():
    req = _gru_request()
    req.dataset.feature_set = "custom"
    req.dataset.custom_features = ["RESI5", "CORR5", "STD5"]
    dc = convert_dataset_config(req)
    h_kwargs = dc["kwargs"]["handler"]["kwargs"]
    assert h_kwargs["infer_processors"][0]["kwargs"]["col_list"] == ["RESI5", "CORR5", "STD5"]


def test_label_1d():
    req = _lgb_request()
    req.label = "Ref($close, -2)/Ref($close, -1) - 1"
    dc = convert_dataset_config(req)
    label = dc["kwargs"]["handler"]["kwargs"]["label"]
    assert label == (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])


def test_label_5d():
    req = _lgb_request()
    req.label = "Ref($close, -6)/Ref($close, -1) - 1"
    dc = convert_dataset_config(req)
    label = dc["kwargs"]["handler"]["kwargs"]["label"]
    assert label == (["Ref($close, -6)/Ref($close, -1) - 1"], ["LABEL0"])


def test_instruments_all():
    req = _lgb_request()
    req.training_pool = "all"
    dc = convert_dataset_config(req)
    assert dc["kwargs"]["handler"]["kwargs"]["instruments"] == "all"


def test_instruments_volatile():
    req = _gru_request()
    req.training_pool = "volatile"
    dc = convert_dataset_config(req)
    assert dc["kwargs"]["handler"]["kwargs"]["instruments"] == "volatile_us"


def test_instruments_target_only():
    req = _gru_request()
    req.training_pool = "target_only"
    dc = convert_dataset_config(req)
    assert dc["kwargs"]["handler"]["kwargs"]["instruments"] == ["ASST"]


def test_match_intc_script():
    """Config should match intc_backtest.py structure."""
    mc, dc = convert(_lgb_request())
    # Model
    assert mc["class"] == "LGBModel"
    assert mc["kwargs"]["max_depth"] == 8
    # Dataset
    assert dc["kwargs"]["segments"]["train"] == ("2000-01-01", "2017-12-31")
    assert dc["kwargs"]["segments"]["test"] == ("2020-01-01", "2026-03-06")
    assert dc["kwargs"]["handler"]["kwargs"]["start_time"] == "2000-01-01"
    assert dc["kwargs"]["handler"]["kwargs"]["fit_end_time"] == "2017-12-31"


def test_match_gru_script():
    """Config should match asst_gru_backtest.py structure."""
    mc, dc = convert(_gru_request())
    # Model
    assert mc["class"] == "GRU"
    assert mc["kwargs"]["hidden_size"] == 128
    assert mc["kwargs"]["GPU"] == 0
    # Dataset
    assert dc["class"] == "TSDatasetH"
    assert dc["kwargs"]["step_len"] == 20
    assert dc["kwargs"]["handler"]["kwargs"]["instruments"] == "volatile_us"
    assert dc["kwargs"]["segments"]["train"] == ("2015-01-01", "2024-06-30")
