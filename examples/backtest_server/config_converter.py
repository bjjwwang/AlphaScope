"""
Convert frontend BacktestRequest → Qlib-native config dicts.
"""
from backtest_server.config_schema import BacktestRequest

TOP20_FEATURES = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10",
    "CORR5", "CORD5", "CORR10", "ROC60", "RESI10",
    "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
]

TRAINING_POOL_MAP = {
    "all": "all",
    "sp500": "sp500",
    "volatile": "volatile_us",
    "same_sector": "all",  # handled at runtime
}

PYTORCH_MODELS = {
    "LSTM_flat", "LSTM_ts", "GRU_flat", "GRU_ts",
    "ALSTM_flat", "ALSTM_ts", "ADARNN",
    "Transformer_flat", "Transformer_ts", "LocalFormer_ts",
    "GATs_flat", "GATs_ts", "TRAModel",
    "TCN_flat", "TCN_ts", "TabnetModel", "HIST", "SFM", "DNNModelPytorch",
}


FEATURE_DIM = {
    "Alpha158": 158,
    "Alpha158_20": 20,
    "Alpha360": 360,
}

# model_class → actual Qlib class name (for cases where strip _flat/_ts is wrong)
CLASS_NAME_MAP = {
    "Transformer_flat": "TransformerModel",
    "Transformer_ts": "TransformerModel",
    "LocalFormer_ts": "LocalformerModel",
}

# Common RNN params accepted by LSTM/GRU/ALSTM/ADARNN
_RNN_PARAMS = {
    "d_feat", "hidden_size", "num_layers", "dropout",
    "n_epochs", "lr", "early_stop", "batch_size",
    "metric", "loss", "optimizer", "GPU", "seed", "n_jobs",
}

# Per-model accepted kwargs (only models that differ from _RNN_PARAMS need entries)
MODEL_ACCEPTED_KWARGS = {
    # Tree models: pass everything through (they use **kwargs)
    "LGBModel": None,  # None = no filtering
    "XGBModel": None,
    "CatBoostModel": None,
    # LinearModel: only accepts specific params (no **kwargs)
    "LinearModel": {"estimator", "alpha", "fit_intercept", "include_valid"},
    # DEnsembleModel: accepts **kwargs but we list key params
    "DEnsembleModel": None,
    # Standard RNN models: all accept _RNN_PARAMS
    "LSTM_flat": _RNN_PARAMS,
    "LSTM_ts": _RNN_PARAMS,
    "GRU_flat": _RNN_PARAMS,
    "GRU_ts": _RNN_PARAMS,
    "ALSTM_flat": _RNN_PARAMS,
    "ALSTM_ts": _RNN_PARAMS,
    # ADARNN: RNN + extras
    "ADARNN": _RNN_PARAMS | {
        "pre_epoch", "dw", "loss_type", "len_seq", "len_win", "n_splits",
    },
    # Transformer variants: d_model/nhead instead of hidden_size
    "Transformer_flat": {
        "d_feat", "hidden_size", "num_layers", "dropout",
        "n_epochs", "lr", "early_stop", "batch_size",
        "d_model", "nhead", "metric", "loss", "reg", "n_jobs", "optimizer", "GPU", "seed",
    },
    "Transformer_ts": {
        "d_feat", "hidden_size", "num_layers", "dropout",
        "n_epochs", "lr", "early_stop", "batch_size",
        "d_model", "nhead", "metric", "loss", "reg", "n_jobs", "optimizer", "GPU", "seed",
    },
    "LocalFormer_ts": {
        "d_feat", "hidden_size", "num_layers", "dropout",
        "n_epochs", "lr", "early_stop", "batch_size",
        "d_model", "nhead", "metric", "loss", "reg", "n_jobs", "optimizer", "GPU", "seed",
    },
    # GATs: RNN-like but no batch_size, has base_model
    "GATs_flat": _RNN_PARAMS | {"base_model", "model_path"},
    "GATs_ts": _RNN_PARAMS | {"base_model", "model_path"},
    # TRAModel: very different, pass through
    "TRAModel": None,
    # TCN: RNN + n_chans/kernel_size
    "TCN_flat": _RNN_PARAMS | {"n_chans", "kernel_size"},
    "TCN_ts": _RNN_PARAMS | {"n_chans", "kernel_size"},
    # TabNet: no hidden_size/num_layers/dropout
    "TabnetModel": {
        "d_feat", "out_dim", "final_out_dim", "batch_size",
        "n_d", "n_a", "n_shared", "n_ind", "n_steps",
        "n_epochs", "pretrain_n_epochs", "relax", "vbs", "seed",
        "optimizer", "loss", "metric", "early_stop", "GPU",
        "pretrain_loss", "ps", "lr", "pretrain", "pretrain_file",
    },
    # HIST: RNN-like + base_model, no batch_size
    "HIST": _RNN_PARAMS | {"base_model", "model_path", "stock2concept", "stock_index"},
    # SFM: no num_layers/dropout, has freq_dim
    "SFM": {
        "d_feat", "hidden_size", "n_epochs", "lr", "early_stop", "batch_size",
        "output_dim", "freq_dim", "dropout_W", "dropout_U",
        "metric", "eval_steps", "loss", "optimizer", "GPU", "seed",
    },
    # DNN: very different params
    "DNNModelPytorch": {
        "lr", "max_steps", "batch_size", "early_stop_rounds", "eval_steps",
        "optimizer", "loss", "weight_decay", "GPU", "seed",
        "data_parall", "scheduler", "init_model", "eval_train_metric",
        "pt_model_uri", "pt_model_kwargs", "valid_key",
    },
}


def _adarnn_factorize(total_features: int) -> tuple[int, int]:
    """Find (d_feat, len_seq) for ADARNN such that d_feat * len_seq = total_features.

    Prefers len_seq >= 10 for meaningful temporal structure.
    Falls back to any len_seq >= 2. Primes get len_seq = total_features, d_feat = 1.
    """
    # Known optimal factorizations
    if total_features == 360:
        return (6, 60)
    # Find all factor pairs, prefer len_seq in [10..60]
    best = None
    for d in range(1, total_features + 1):
        if total_features % d == 0:
            ls = total_features // d
            if ls < 2:
                continue
            if best is None or (ls <= 60 and ls >= 10 and (best[1] < 10 or d > best[0])):
                best = (d, ls)
            elif best[1] < 10 and ls >= 2:
                best = (d, ls)
    return best if best else (total_features, 1)


def _filter_kwargs(model_class: str, kwargs: dict) -> dict:
    """Filter kwargs to only include params the model accepts."""
    accepted = MODEL_ACCEPTED_KWARGS.get(model_class)
    if accepted is None:
        return kwargs  # no filtering (tree models, TRAModel, etc.)
    return {k: v for k, v in kwargs.items() if k in accepted}


def convert_model_config(req: BacktestRequest) -> dict:
    """Convert to Qlib model config dict."""
    mc = req.model
    # Map to actual Qlib class name
    class_name = CLASS_NAME_MAP.get(mc.model_class,
                                     mc.model_class.replace("_flat", "").replace("_ts", ""))

    kwargs = dict(mc.kwargs)

    if mc.model_class in PYTORCH_MODELS:
        kwargs.setdefault("loss", "mse")
        kwargs.setdefault("GPU", 0)

    # TabnetModel: disable pretrain by default — it requires "pretrain"/"pretrain_validation"
    # dataset segments which our standard train/valid/test setup doesn't provide.
    if mc.model_class == "TabnetModel":
        kwargs.setdefault("pretrain", False)

    # Auto-correct d_feat to match actual feature count
    if "d_feat" in kwargs:
        fs = req.dataset.feature_set
        if fs == "custom" and req.dataset.custom_features:
            total_features = len(req.dataset.custom_features)
        elif fs in FEATURE_DIM:
            total_features = FEATURE_DIM[fs]
        else:
            total_features = None

        if total_features is not None:
            if mc.model_class == "ADARNN":
                # ADARNN: flat features = d_feat * len_seq.
                # Find best (d_feat, len_seq) such that d_feat * len_seq = total_features
                # and len_seq >= 2 (temporal model needs sequence length > 1).
                kwargs["d_feat"], kwargs["len_seq"] = _adarnn_factorize(total_features)
            else:
                kwargs["d_feat"] = total_features

    # Filter kwargs to only include params the model actually accepts
    kwargs = _filter_kwargs(mc.model_class, kwargs)

    return {
        "class": class_name,
        "module_path": mc.module_path,
        "kwargs": kwargs,
    }


def convert_dataset_config(req: BacktestRequest) -> dict:
    """Convert to Qlib dataset config dict."""
    dc = req.dataset
    seg = dc.segments

    # Resolve instruments
    if req.training_pool == "target_only":
        instruments = req.target
    else:
        instruments = TRAINING_POOL_MAP.get(req.training_pool, "all")

    # Determine if we have a small instrument pool (< 10 stocks).
    # CSRankNorm is meaningless with few stocks — use DropnaLabel only.
    small_pool = isinstance(instruments, list) and len(instruments) < 10

    # Label expression
    label_expr = req.label.strip()

    # Build handler kwargs
    handler_kwargs = {
        "start_time": seg.train[0],
        "end_time": seg.test[1],
        "fit_start_time": seg.train[0],
        "fit_end_time": seg.train[1],
        "instruments": instruments,
        "label": ([label_expr], ["LABEL0"]),
    }

    # Label processors: skip CSRankNorm for small pools
    if small_pool:
        learn_processors = [{"class": "DropnaLabel"}]
    else:
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]

    # Processors based on feature set
    if dc.feature_set == "Alpha158_20":
        handler_kwargs["infer_processors"] = [
            {"class": "FilterCol", "kwargs": {
                "fields_group": "feature",
                "col_list": TOP20_FEATURES,
            }},
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
        handler_kwargs["learn_processors"] = learn_processors
    elif dc.feature_set == "custom" and dc.custom_features:
        handler_kwargs["infer_processors"] = [
            {"class": "FilterCol", "kwargs": {
                "fields_group": "feature",
                "col_list": dc.custom_features,
            }},
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
        handler_kwargs["learn_processors"] = learn_processors
    else:
        # Default Alpha158/Alpha360: add feature processors.
        # Alpha158's default infer_processors=[] works for tree models (handle NaN),
        # but PyTorch models need RobustZScoreNorm + Fillna to avoid NaN gradients.
        # Always add them — normalization also benefits tree models.
        handler_kwargs["infer_processors"] = [
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
        handler_kwargs["learn_processors"] = learn_processors

    dataset_kwargs = {
        "handler": {
            "class": dc.handler,
            "module_path": "qlib.contrib.data.handler",
            "kwargs": handler_kwargs,
        },
        "segments": {
            "train": tuple(seg.train),
            "valid": tuple(seg.valid),
            "test": tuple(seg.test),
        },
    }

    if dc.dataset_class == "TSDatasetH" and dc.step_len:
        dataset_kwargs["step_len"] = dc.step_len

    return {
        "class": dc.dataset_class,
        "module_path": "qlib.data.dataset",
        "kwargs": dataset_kwargs,
    }


def convert(req: BacktestRequest) -> tuple:
    """Return (model_config, dataset_config) tuple."""
    return convert_model_config(req), convert_dataset_config(req)
