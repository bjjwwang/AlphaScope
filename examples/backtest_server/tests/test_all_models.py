"""Tests for all 24 model classes — config conversion + dependency checking + kwarg filtering."""
import pytest
from backtest_server.config_schema import BacktestRequest, VALID_MODEL_CLASSES
from backtest_server.config_converter import (
    convert_model_config, convert_dataset_config, PYTORCH_MODELS,
    _filter_kwargs, MODEL_ACCEPTED_KWARGS,
)


# ===== Model definitions: (model_class, module_path, expected_qlib_class, dataType) =====
ALL_MODELS = [
    # Tree models (flat only)
    ("LGBModel", "qlib.contrib.model.gbdt", "LGBModel", "flat"),
    ("XGBModel", "qlib.contrib.model.xgboost", "XGBModel", "flat"),
    ("CatBoostModel", "qlib.contrib.model.catboost_model", "CatBoostModel", "flat"),
    ("LinearModel", "qlib.contrib.model.linear", "LinearModel", "flat"),
    ("DEnsembleModel", "qlib.contrib.model.double_ensemble", "DEnsembleModel", "flat"),
    # RNN flat
    ("LSTM_flat", "qlib.contrib.model.pytorch_lstm", "LSTM", "flat"),
    ("GRU_flat", "qlib.contrib.model.pytorch_gru", "GRU", "flat"),
    ("ALSTM_flat", "qlib.contrib.model.pytorch_alstm", "ALSTM", "flat"),
    # RNN ts
    ("LSTM_ts", "qlib.contrib.model.pytorch_lstm_ts", "LSTM", "ts"),
    ("GRU_ts", "qlib.contrib.model.pytorch_gru_ts", "GRU", "ts"),
    ("ALSTM_ts", "qlib.contrib.model.pytorch_alstm_ts", "ALSTM", "ts"),
    # ADARNN (flat)
    ("ADARNN", "qlib.contrib.model.pytorch_adarnn", "ADARNN", "flat"),
    # Attention flat
    ("Transformer_flat", "qlib.contrib.model.pytorch_transformer", "TransformerModel", "flat"),
    ("GATs_flat", "qlib.contrib.model.pytorch_gats", "GATs", "flat"),
    # Attention ts
    ("Transformer_ts", "qlib.contrib.model.pytorch_transformer_ts", "TransformerModel", "ts"),
    ("LocalFormer_ts", "qlib.contrib.model.pytorch_localformer_ts", "LocalformerModel", "ts"),
    ("GATs_ts", "qlib.contrib.model.pytorch_gats_ts", "GATs", "ts"),
    ("TRAModel", "qlib.contrib.model.pytorch_tra", "TRAModel", "ts"),
    # Other flat
    ("TCN_flat", "qlib.contrib.model.pytorch_tcn", "TCN", "flat"),
    ("TabnetModel", "qlib.contrib.model.pytorch_tabnet", "TabnetModel", "flat"),
    ("HIST", "qlib.contrib.model.pytorch_hist", "HIST", "flat"),
    ("SFM", "qlib.contrib.model.pytorch_sfm", "SFM", "flat"),
    ("DNNModelPytorch", "qlib.contrib.model.pytorch_nn", "DNNModelPytorch", "flat"),
    # Other ts
    ("TCN_ts", "qlib.contrib.model.pytorch_tcn_ts", "TCN", "ts"),
]

# Verify we cover all 24 models
assert len(ALL_MODELS) == len(VALID_MODEL_CLASSES), (
    f"Test covers {len(ALL_MODELS)} but schema has {len(VALID_MODEL_CLASSES)} models"
)
assert {m[0] for m in ALL_MODELS} == VALID_MODEL_CLASSES


def _make_request(model_class, module_path, data_type="flat"):
    """Build a minimal BacktestRequest for any model."""
    ds_class = "TSDatasetH" if data_type == "ts" else "DatasetH"
    kwargs = {}

    # PyTorch models need d_feat + basic params
    if model_class in PYTORCH_MODELS:
        kwargs = {
            "d_feat": 158, "hidden_size": 64, "num_layers": 2,
            "dropout": 0.0, "n_epochs": 5, "lr": 0.001, "batch_size": 512,
        }
    # Tree models need their own params
    elif model_class == "LGBModel":
        kwargs = {"learning_rate": 0.05, "max_depth": 8, "num_leaves": 210}
    elif model_class == "XGBModel":
        kwargs = {"eta": 0.1, "max_depth": 6}
    elif model_class == "CatBoostModel":
        kwargs = {"learning_rate": 0.03, "depth": 6, "iterations": 100}
    elif model_class == "LinearModel":
        kwargs = {"estimator": "ols"}
    elif model_class == "DEnsembleModel":
        kwargs = {"num_models": 6, "epochs": 100}

    req_dict = {
        "model": {
            "model_class": model_class,
            "module_path": module_path,
            "kwargs": kwargs,
        },
        "dataset": {
            "dataset_class": ds_class,
            "handler": "Alpha158",
            "feature_set": "Alpha158",
            "segments": {
                "train": ("2020-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2025-01-31"),
                "test": ("2025-02-01", "2025-10-31"),
            },
        },
        "data_source": "db",
        "target": ["INTC"],
        "training_pool": "target_only",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "strategy": {"strategy_class": "SignalThreshold", "buy_threshold": 0.0},
    }
    if data_type == "ts":
        req_dict["dataset"]["step_len"] = 20
    return BacktestRequest(**req_dict)


# ===== Parametrized test: config conversion for all 24 models =====

@pytest.mark.parametrize("model_class,module_path,expected_class,data_type", ALL_MODELS,
                         ids=[m[0] for m in ALL_MODELS])
def test_model_config_conversion(model_class, module_path, expected_class, data_type):
    """Each model config converts with correct class name and module_path."""
    req = _make_request(model_class, module_path, data_type)
    mc = convert_model_config(req)

    assert mc["class"] == expected_class, f"{model_class} → expected class {expected_class}, got {mc['class']}"
    assert mc["module_path"] == module_path

    # PyTorch models should get loss=mse and GPU=0
    if model_class in PYTORCH_MODELS:
        assert mc["kwargs"]["loss"] == "mse"
        assert mc["kwargs"]["GPU"] == 0
    else:
        assert "GPU" not in mc["kwargs"]


@pytest.mark.parametrize("model_class,module_path,expected_class,data_type", ALL_MODELS,
                         ids=[m[0] for m in ALL_MODELS])
def test_dataset_config_conversion(model_class, module_path, expected_class, data_type):
    """Each model gets correct dataset type and processors."""
    req = _make_request(model_class, module_path, data_type)
    dc = convert_dataset_config(req)

    expected_ds = "TSDatasetH" if data_type == "ts" else "DatasetH"
    assert dc["class"] == expected_ds

    if data_type == "ts":
        assert dc["kwargs"]["step_len"] == 20
    else:
        assert "step_len" not in dc["kwargs"]

    h_kwargs = dc["kwargs"]["handler"]["kwargs"]
    # target_only → instruments is a list
    assert h_kwargs["instruments"] == ["INTC"]
    # Small pool (1 stock) → no CSRankNorm
    lp = h_kwargs["learn_processors"]
    assert lp[0]["class"] == "DropnaLabel"
    assert len(lp) == 1  # no CSRankNorm for small pool
    # Feature processors always present
    ip = h_kwargs["infer_processors"]
    assert ip[0]["class"] == "RobustZScoreNorm"
    assert ip[1]["class"] == "Fillna"


@pytest.mark.parametrize("model_class,module_path,expected_class,data_type", ALL_MODELS,
                         ids=[m[0] for m in ALL_MODELS])
def test_d_feat_auto_correction(model_class, module_path, expected_class, data_type):
    """d_feat is auto-corrected to match feature set."""
    if model_class not in PYTORCH_MODELS:
        pytest.skip("d_feat only relevant for PyTorch models")
    # Some PyTorch models don't accept d_feat (e.g. DNNModelPytorch)
    accepted = MODEL_ACCEPTED_KWARGS.get(model_class)
    if accepted is not None and "d_feat" not in accepted:
        pytest.skip(f"{model_class} does not accept d_feat")
    # ADARNN uses d_feat differently (features per timestep, not total) — tested separately
    if model_class == "ADARNN":
        pytest.skip("ADARNN d_feat tested in test_adarnn_len_seq_auto_*")
    req = _make_request(model_class, module_path, data_type)
    mc = convert_model_config(req)
    # Alpha158 → 158 features
    assert mc["kwargs"]["d_feat"] == 158

    # Alpha158_20 → 20 features
    req.dataset.feature_set = "Alpha158_20"
    mc = convert_model_config(req)
    assert mc["kwargs"]["d_feat"] == 20

    # Alpha360 → 360 features
    req.dataset.feature_set = "Alpha360"
    mc = convert_model_config(req)
    assert mc["kwargs"]["d_feat"] == 360


# ===== Qlib module importability test =====

@pytest.mark.parametrize("model_class,module_path,expected_class,data_type", ALL_MODELS,
                         ids=[m[0] for m in ALL_MODELS])
def test_module_importable(model_class, module_path, expected_class, data_type):
    """Each model's module_path actually exists and is importable.
    Models with missing optional deps (xgboost, catboost) may fail import
    but the module file must exist."""
    import importlib
    parts = module_path.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        # Check that the expected class exists in the module
        assert hasattr(mod, expected_class), (
            f"Module {module_path} has no class {expected_class}"
        )
    except ImportError as e:
        err_msg = str(e)
        # These are OK — optional deps not installed
        if "xgboost" in err_msg or "catboost" in err_msg:
            pytest.skip(f"Optional dependency not installed: {err_msg}")
        else:
            raise


# ===== Server dependency checking =====

def test_dependency_check_available():
    from backtest_server.server import _check_model_available
    # LGBModel uses lightgbm which is installed
    assert _check_model_available("LGBModel") is True
    # All PyTorch models should be available
    assert _check_model_available("LSTM_flat") is True
    assert _check_model_available("LinearModel") is True


def test_dependency_check_missing():
    from backtest_server.server import _check_model_available, _MODEL_AVAILABILITY
    # XGBModel and CatBoostModel depend on optional packages
    # Test that the check function works (result depends on env)
    xgb_avail = _check_model_available("XGBModel")
    cat_avail = _check_model_available("CatBoostModel")
    # Whatever the result, it must be consistent with _MODEL_AVAILABILITY
    assert _MODEL_AVAILABILITY["XGBModel"] == xgb_avail
    assert _MODEL_AVAILABILITY["CatBoostModel"] == cat_avail


def test_api_models_includes_availability():
    """The /api/models endpoint includes available flag."""
    from fastapi.testclient import TestClient
    from backtest_server.server import app
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 24
    for m in models:
        assert "available" in m
        assert isinstance(m["available"], bool)


def test_api_rejects_unavailable_model():
    """POST /api/backtest rejects models with missing deps."""
    from backtest_server.server import _MODEL_AVAILABILITY
    # Find an unavailable model (if any)
    unavailable = [mc for mc, avail in _MODEL_AVAILABILITY.items() if not avail]
    if not unavailable:
        pytest.skip("All models available in this environment")

    mc = unavailable[0]
    from fastapi.testclient import TestClient
    from backtest_server.server import app
    client = TestClient(app)
    # Build minimal request
    req = _make_request(mc, f"qlib.contrib.model.{mc.lower()}", "flat")
    resp = client.post("/api/backtest", json=req.model_dump())
    assert resp.status_code == 400
    assert "不可用" in resp.json()["detail"]


# ===== Kwargs filtering tests =====

def test_tabnet_filters_hidden_size():
    """TabnetModel should NOT receive hidden_size, num_layers, dropout."""
    req = _make_request("TabnetModel", "qlib.contrib.model.pytorch_tabnet", "flat")
    mc = convert_model_config(req)
    assert "hidden_size" not in mc["kwargs"]
    assert "num_layers" not in mc["kwargs"]
    assert "dropout" not in mc["kwargs"]
    # Should keep d_feat, n_epochs, lr, batch_size
    assert mc["kwargs"]["d_feat"] == 158
    assert mc["kwargs"]["n_epochs"] == 5
    assert mc["kwargs"]["lr"] == 0.001


def test_tabnet_pretrain_disabled():
    """TabnetModel should have pretrain=False by default (no pretrain segments in dataset)."""
    req = _make_request("TabnetModel", "qlib.contrib.model.pytorch_tabnet", "flat")
    mc = convert_model_config(req)
    assert mc["kwargs"]["pretrain"] is False


def test_adarnn_len_seq_auto_alpha360():
    """ADARNN with Alpha360 (360 features) should get d_feat=6, len_seq=60."""
    req = _make_request("ADARNN", "qlib.contrib.model.pytorch_adarnn", "flat")
    req.dataset.feature_set = "Alpha360"
    req.model.kwargs["d_feat"] = 6
    mc = convert_model_config(req)
    assert mc["kwargs"]["d_feat"] == 6
    assert mc["kwargs"]["len_seq"] == 60


def test_adarnn_len_seq_auto_alpha158_20():
    """ADARNN with Alpha158_20 (20 features) should get d_feat=2, len_seq=10."""
    req = _make_request("ADARNN", "qlib.contrib.model.pytorch_adarnn", "flat")
    req.dataset.feature_set = "Alpha158_20"
    req.model.kwargs["d_feat"] = 20
    mc = convert_model_config(req)
    # 20 features: best factorization for len_seq in [10,60] is (2, 10)
    assert mc["kwargs"]["d_feat"] == 2
    assert mc["kwargs"]["len_seq"] == 10


def test_adarnn_len_seq_auto_alpha158():
    """ADARNN with Alpha158 (158 features) should get a valid factorization with len_seq >= 2."""
    req = _make_request("ADARNN", "qlib.contrib.model.pytorch_adarnn", "flat")
    req.dataset.feature_set = "Alpha158"
    req.model.kwargs["d_feat"] = 6
    mc = convert_model_config(req)
    d = mc["kwargs"]["d_feat"]
    ls = mc["kwargs"]["len_seq"]
    assert d * ls == 158
    assert ls >= 2


def test_sfm_filters_num_layers():
    """SFM should NOT receive num_layers or dropout."""
    req = _make_request("SFM", "qlib.contrib.model.pytorch_sfm", "flat")
    mc = convert_model_config(req)
    assert "num_layers" not in mc["kwargs"]
    assert "dropout" not in mc["kwargs"]
    # Should keep d_feat, hidden_size
    assert mc["kwargs"]["d_feat"] == 158
    assert mc["kwargs"]["hidden_size"] == 64


def test_dnn_filters_most_rnn_params():
    """DNNModelPytorch should NOT receive d_feat, hidden_size, num_layers, dropout."""
    req = _make_request("DNNModelPytorch", "qlib.contrib.model.pytorch_nn", "flat")
    mc = convert_model_config(req)
    assert "d_feat" not in mc["kwargs"]
    assert "hidden_size" not in mc["kwargs"]
    assert "num_layers" not in mc["kwargs"]
    assert "dropout" not in mc["kwargs"]
    # Should keep lr, batch_size
    assert mc["kwargs"]["lr"] == 0.001
    assert mc["kwargs"]["batch_size"] == 512


def test_lstm_keeps_all_params():
    """LSTM should keep all standard RNN params."""
    req = _make_request("LSTM_flat", "qlib.contrib.model.pytorch_lstm", "flat")
    mc = convert_model_config(req)
    assert mc["kwargs"]["d_feat"] == 158
    assert mc["kwargs"]["hidden_size"] == 64
    assert mc["kwargs"]["num_layers"] == 2
    assert mc["kwargs"]["dropout"] == 0.0
    assert mc["kwargs"]["n_epochs"] == 5
    assert mc["kwargs"]["lr"] == 0.001


def test_lgb_keeps_all_params():
    """LGBModel (tree) should pass through all kwargs unfiltered."""
    req = _make_request("LGBModel", "qlib.contrib.model.gbdt", "flat")
    mc = convert_model_config(req)
    assert mc["kwargs"]["learning_rate"] == 0.05
    assert mc["kwargs"]["max_depth"] == 8
    assert mc["kwargs"]["num_leaves"] == 210


@pytest.mark.parametrize("model_class,module_path,expected_class,data_type", ALL_MODELS,
                         ids=[m[0] for m in ALL_MODELS])
def test_no_unexpected_kwargs(model_class, module_path, expected_class, data_type):
    """After filtering, all remaining kwargs should be accepted by the model."""
    req = _make_request(model_class, module_path, data_type)
    mc = convert_model_config(req)
    accepted = MODEL_ACCEPTED_KWARGS.get(model_class)
    if accepted is None:
        return  # no filtering, skip check
    for k in mc["kwargs"]:
        assert k in accepted, f"{model_class}: unexpected kwarg '{k}' after filtering"
