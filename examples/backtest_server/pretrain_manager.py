"""
Pretrained model manager — batch training, registry, and quick prediction.

Pre-trains models with training_pool=all so they generalize across stocks.
Users can then predict on any ticker instantly without re-training.
"""
import gc
import json
import time
import threading
import traceback
import uuid
from datetime import datetime
from typing import Callable, Any

from backtest_server.config_schema import (
    BacktestRequest, ModelConfig, DatasetConfig, SegmentsConfig, StrategyConfig,
)
from backtest_server.config_converter import PYTORCH_MODELS, FEATURE_DIM
from backtest_server.scanner import (
    MODEL_SCAN_LIST, TRADING_STYLE_PRESETS, compute_date_segments,
    _MODEL_AVAILABILITY, _MODEL_DEPS,
)
from backtest_server.scan_db import (
    save_pretrained_model, find_pretrained_model, list_pretrained_models,
    update_pretrained_model,
)


# Quick scan uses only the top 5 fastest/best models
QUICK_SCAN_MODELS = [
    "LGBModel", "XGBModel", "CatBoostModel", "GRU_ts", "LSTM_ts",
]

# Standard configuration matrix for pre-training.
# Each entry produces one pretrained model per trading_style.
# training_pool=all means the model sees all ~600 stocks and generalizes.
PRETRAIN_MATRIX = []
for m in MODEL_SCAN_LIST:
    for style_name, style_preset in TRADING_STYLE_PRESETS.items():
        PRETRAIN_MATRIX.append({
            "model_class": m["model_class"],
            "module_path": m["module_path"],
            "data_type": m["data_type"],
            "trading_style": style_name,
            "feature_set": style_preset["feature_set"],
            "label": style_preset["label"],
        })

# Lookup helper
_SCAN_MODEL_MAP = {m["model_class"]: m for m in MODEL_SCAN_LIST}


def build_pretrain_config(
    model_class: str,
    trading_style: str,
    training_pool: str = "all",
) -> BacktestRequest:
    """Build a BacktestRequest for pre-training a model on the full stock pool.

    Unlike scan's build_scan_config which targets a single ticker,
    this trains on ALL stocks (or sp500) so the model generalizes.
    Target is set to a dummy ticker for strategy purposes but training
    uses the full pool.
    """
    preset = TRADING_STYLE_PRESETS[trading_style]
    model_info = _SCAN_MODEL_MAP[model_class]
    segments = compute_date_segments(trading_style)
    feat_dim = FEATURE_DIM.get(preset["feature_set"], 158)

    # Default kwargs per model type (same as scanner.py)
    kwargs = {}
    if model_class == "DNNModelPytorch":
        kwargs = {
            "lr": 0.001,
            "max_steps": 300,
            "batch_size": 2000,
            "pt_model_kwargs": {"input_dim": feat_dim, "layers": (256,)},
        }
    elif model_class in PYTORCH_MODELS:
        kwargs = {
            "d_feat": 158,  # auto-corrected by config_converter
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 5,
            "lr": 0.001,
            "batch_size": 512,
        }
    elif model_class == "LGBModel":
        kwargs = {"learning_rate": 0.05, "max_depth": 8, "num_leaves": 210}
    elif model_class == "XGBModel":
        kwargs = {"eta": 0.1, "max_depth": 6}
    elif model_class == "CatBoostModel":
        kwargs = {"learning_rate": 0.03, "depth": 6, "iterations": 100}
    elif model_class == "LinearModel":
        kwargs = {"estimator": "ols"}
    elif model_class == "DEnsembleModel":
        kwargs = {"num_models": 6, "epochs": 100, "decay": 0.5}

    ds_class = "TSDatasetH" if model_info["data_type"] == "ts" else "DatasetH"
    step_len = 20 if model_info["data_type"] == "ts" else None

    return BacktestRequest(
        model=ModelConfig(
            model_class=model_class,
            module_path=model_info["module_path"],
            kwargs=kwargs,
        ),
        dataset=DatasetConfig(
            dataset_class=ds_class,
            handler="Alpha158",
            feature_set=preset["feature_set"],
            step_len=step_len,
            segments=segments,
        ),
        data_source="db",
        target=["AAPL"],  # dummy target — training uses full pool
        training_pool=training_pool,
        label=preset["label"],
        strategy=StrategyConfig(
            strategy_class="SignalThreshold",
            buy_threshold=0.0,
        ),
    )


class PretrainRunner:
    """Batch pre-train all models in the configuration matrix."""

    def __init__(
        self,
        db_path: str,
        emit: Callable[[dict], Any] | None = None,
        training_pool: str = "all",
        models: list[str] | None = None,
        styles: list[str] | None = None,
    ):
        self.db_path = db_path
        self.emit = emit or (lambda e: None)
        self.training_pool = training_pool
        self.filter_models = set(models) if models else None
        self.filter_styles = set(styles) if styles else None

    def _start_heartbeat(self):
        stop = threading.Event()

        def _beat():
            while not stop.is_set():
                stop.wait(15)
                if not stop.is_set():
                    self.emit({"event": "ping", "data": {}})

        t = threading.Thread(target=_beat, daemon=True)
        t.start()
        return stop

    def run_sync(self):
        """Synchronous pre-training loop. Call from thread via asyncio.to_thread."""
        hb_stop = self._start_heartbeat()
        try:
            self._run_inner()
        finally:
            hb_stop.set()

    def _run_inner(self):
        # Determine which entries to train
        matrix = PRETRAIN_MATRIX
        if self.filter_models:
            matrix = [e for e in matrix if e["model_class"] in self.filter_models]
        if self.filter_styles:
            matrix = [e for e in matrix if e["trading_style"] in self.filter_styles]

        # Skip models with missing dependencies
        available = [
            e for e in matrix
            if _MODEL_AVAILABILITY.get(e["model_class"], True)
        ]
        skipped = [
            e for e in matrix
            if not _MODEL_AVAILABILITY.get(e["model_class"], True)
        ]

        total = len(matrix)
        self.emit({"event": "pretrain_start", "data": {
            "total": total,
            "available": len(available),
            "skipped": len(skipped),
        }})

        # Prepare data once
        from backtest_server import data_pipeline
        try:
            provider_uri = data_pipeline.resolve_provider_uri("db")
            data_pipeline.dump_all_db_to_qlib(provider_uri=provider_uri)
        except Exception as e:
            self.emit({"event": "pretrain_error", "data": {"msg": f"Data prep failed: {e}"}})
            return

        completed = len(skipped)
        succeeded = 0
        failed = 0

        for idx, entry in enumerate(available):
            mc = entry["model_class"]
            style = entry["trading_style"]
            feat = entry["feature_set"]
            model_id = f"pt_{mc}_{style}_{uuid.uuid4().hex[:6]}"

            # Check if already have a completed model for this config
            existing = find_pretrained_model(self.db_path, mc, feat, style)
            if existing:
                self.emit({"event": "pretrain_model_skip", "data": {
                    "model_class": mc, "trading_style": style,
                    "reason": "Already have a pretrained model",
                }})
                completed += 1
                succeeded += 1
                continue

            self.emit({"event": "pretrain_model_start", "data": {
                "model_class": mc, "trading_style": style,
                "index": idx + 1, "total": len(available),
            }})

            t0 = time.time()
            try:
                config = build_pretrain_config(mc, style, self.training_pool)
                segments = config.dataset.segments

                from backtest_server.backtest_runner import BacktestRunner
                runner = BacktestRunner(config, self.emit)
                result = runner.run_sync()
                duration = time.time() - t0

                perf = result.summary.model_dump()
                save_pretrained_model(
                    self.db_path, model_id, mc, feat,
                    entry["label"], self.training_pool, style,
                    segments.train[0], segments.train[1],
                    runner.experiment_name,
                    performance_json=json.dumps(perf),
                    status="completed",
                    duration_s=duration,
                )
                succeeded += 1
                self.emit({"event": "pretrain_model_complete", "data": {
                    "model_class": mc, "trading_style": style,
                    "model_id": model_id,
                    "duration_s": round(duration, 1),
                    "performance": perf,
                }})

            except Exception as e:
                duration = time.time() - t0
                save_pretrained_model(
                    self.db_path, model_id, mc, feat,
                    entry["label"], self.training_pool, style,
                    "", "", f"pretrain_{model_id}",
                    status="failed", error_msg=str(e),
                    duration_s=duration,
                )
                failed += 1
                self.emit({"event": "pretrain_model_failed", "data": {
                    "model_class": mc, "trading_style": style,
                    "error": str(e),
                }})

            completed += 1
            self.emit({"event": "pretrain_progress", "data": {
                "completed": completed, "total": total,
                "succeeded": succeeded, "failed": failed,
                "pct": int(completed / total * 100),
            }})

            # Free memory between models
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        self.emit({"event": "pretrain_complete", "data": {
            "total": total, "succeeded": succeeded, "failed": failed,
            "skipped": len(skipped),
        }})


def quick_predict(
    ticker: str,
    model_class: str,
    trading_style: str,
    db_path: str,
    emit: Callable[[dict], Any] | None = None,
    lookback_days: int = 60,
) -> dict:
    """Quick prediction using a pretrained model — no training needed.

    1. Find matching pretrained model in registry
    2. Load trained model from MLflow
    3. Build dataset for the target ticker
    4. Predict and generate signals
    Returns a PredictionResult dict.
    """
    emit = emit or (lambda e: None)

    # Find pretrained model
    preset = TRADING_STYLE_PRESETS[trading_style]
    feature_set = preset["feature_set"]
    pt_model = find_pretrained_model(db_path, model_class, feature_set, trading_style)
    if pt_model is None:
        raise ValueError(
            f"No pretrained model found for {model_class}/{feature_set}/{trading_style}"
        )

    experiment_name = pt_model["experiment_name"]
    emit({"event": "quick_predict_start", "data": {
        "ticker": ticker, "model_class": model_class,
        "pretrained_model_id": pt_model["id"],
    }})

    # Build config for prediction (target = user's ticker)
    from backtest_server.scanner import build_scan_config
    config = build_scan_config(ticker, "db", trading_style, model_class)
    from backtest_server.config_converter import convert
    model_config, dataset_config = convert(config)

    # Adjust test segment to recent lookback
    from datetime import timedelta
    today = datetime.now().date()
    pred_start = today - timedelta(days=lookback_days)
    dataset_config["kwargs"]["segments"]["test"] = (
        pred_start.isoformat(), today.isoformat()
    )

    # Init Qlib
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from backtest_server import data_pipeline

    provider_uri = data_pipeline.resolve_provider_uri("db")

    # Ensure ticker data exists
    if not data_pipeline.check_ticker_exists(ticker, provider_uri):
        data_pipeline.dump_db_to_qlib(ticker, provider_uri)

    qlib.init(provider_uri=provider_uri, region="us")

    # Create model and dataset
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # Load trained model from MLflow
    emit({"event": "quick_predict_progress", "data": {"msg": "Loading pretrained model..."}})
    recorders = R.list_recorders(experiment_name=experiment_name)
    finished = [r for r in recorders.values()
                if r.info.get("status", "").upper() == "FINISHED"]
    if not finished:
        raise ValueError(f"No finished recorder in experiment {experiment_name}")
    trained_model = finished[0].load_object("params.pkl")

    # Predict
    import pandas as pd
    emit({"event": "quick_predict_progress", "data": {"msg": "Generating predictions..."}})
    pred = trained_model.predict(dataset, segment=(pred_start.isoformat(), today.isoformat()))

    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    # Extract scores for target ticker
    if pred.index.nlevels > 1:
        all_instruments = pred.index.get_level_values(1).unique()
        match = [s for s in all_instruments if s.upper() == ticker.upper()]
        if not match:
            match = [s for s in all_instruments if ticker.lower() in s.lower()]
        if not match:
            raise ValueError(f"{ticker} not found in predictions")
        stock_id = match[0]
        scores = pred.xs(stock_id, level=1)["score"].dropna()
    else:
        scores = pred["score"].dropna()

    if scores.empty:
        raise ValueError("No prediction scores generated")

    # Generate signals
    from backtest_server.predictor import generate_signals
    from backtest_server.config_schema import PredictionResult
    from datetime import timezone

    signals = generate_signals(scores, ticker)
    median_threshold = float(scores.median())

    result = PredictionResult(
        ticker=ticker,
        model_class=model_class,
        prediction_range=(pred_start.isoformat(), today.isoformat()),
        signals=signals,
        latest_signal=signals[-1].signal,
        latest_score=signals[-1].score,
        threshold=round(median_threshold, 6),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    emit({"event": "quick_predict_complete", "data": result.model_dump()})
    return result.model_dump()
