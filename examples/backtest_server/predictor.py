"""Prediction engine — use the best model from a scan or pretrained model to predict future signals."""
import json
from datetime import datetime, timedelta, timezone
from typing import Callable, Any

import numpy as np
import pandas as pd

from backtest_server.config_schema import PredictionSignal, PredictionResult
from backtest_server.scan_db import (
    get_scan_job, get_model_results, update_scan_job,
    find_pretrained_model, save_cached_prediction,
)
from backtest_server.scanner import build_scan_config
from backtest_server.config_converter import convert

# Lazy imports for Qlib (may not be available in test env)
try:
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
except ImportError:
    qlib = None
    init_instance_by_config = None
    R = None


def generate_signals(scores: pd.Series, ticker: str) -> list[PredictionSignal]:
    """Convert a series of prediction scores to buy/sell/hold signals.

    Uses median of all scores as threshold.
    score > threshold → buy, score < threshold → sell, score ≈ threshold → hold
    """
    median = float(scores.median())
    # "close to threshold" = within 5% of the score range
    score_range = scores.max() - scores.min()
    tolerance = score_range * 0.05 if score_range > 0 else 0.0

    signals = []
    for date, score in scores.items():
        score_val = float(score)
        if abs(score_val - median) <= tolerance:
            signal = "hold"
        elif score_val > median:
            signal = "buy"
        else:
            signal = "sell"
        signals.append(PredictionSignal(
            date=date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            score=round(score_val, 6),
            signal=signal,
        ))
    return signals


def run_prediction(
    job_id: str,
    db_path: str,
    emit: Callable[[dict], Any],
    lookback_days: int = 60,
) -> PredictionResult:
    """Run prediction using the best model from a completed scan.

    1. Load scan job from DB
    2. Find best model's recorder_id
    3. Rebuild config, create dataset for recent date range
    4. Load trained model from MLflow
    5. Predict on recent data
    6. Generate buy/sell/hold signals
    7. Save to DB
    """
    job = get_scan_job(db_path, job_id)
    if job is None:
        raise ValueError(f"Scan job {job_id} not found")
    if not job.get("best_model"):
        raise ValueError(f"Scan job {job_id} has no best_model")

    best_model = job["best_model"]
    ticker = job["ticker"]
    data_source = job["data_source"]
    trading_style = job["trading_style"]

    # Find recorder_id for best model
    model_results = get_model_results(db_path, job_id)
    best_result = next(
        (r for r in model_results if r["model_class"] == best_model and r["status"] == "completed"),
        None,
    )
    if best_result is None:
        raise ValueError(f"No completed result for {best_model}")
    experiment_name = best_result.get("recorder_id", f"backtest_{job_id[:8]}")

    emit({"event": "predict_start", "data": {"model_class": best_model, "ticker": ticker}})

    # Build config (same as scan used)
    config = build_scan_config(ticker, data_source, trading_style, best_model)
    model_config, dataset_config = convert(config)

    # Adjust dataset date range for prediction (recent lookback_days to today)
    today = datetime.now().date()
    pred_start = today - timedelta(days=lookback_days)
    # Keep train/valid segments as-is (needed for handler fit), but test = recent range
    dataset_config["kwargs"]["segments"]["test"] = (
        pred_start.isoformat(), today.isoformat()
    )

    # Initialize Qlib
    from backtest_server import data_pipeline
    provider_uri = data_pipeline.resolve_provider_uri(data_source)
    qlib.init(provider_uri=provider_uri, region="us")

    # Create model and dataset
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # Load trained model from the existing experiment recorder
    emit({"event": "predict_progress", "data": {"msg": "加载已训练模型..."}})
    recorders = R.list_recorders(experiment_name=experiment_name)
    # Find the FINISHED recorder that has the trained model
    finished = [r for r in recorders.values()
                if r.info.get("status", "").upper() == "FINISHED"]
    if not finished:
        raise ValueError(f"实验 {experiment_name} 中没有找到已完成的训练记录")
    trained_model = finished[0].load_object("params.pkl")

    # Predict
    emit({"event": "predict_progress", "data": {"msg": "生成预测信号..."}})
    pred = trained_model.predict(dataset, segment=(pred_start.isoformat(), today.isoformat()))

    # model.predict() returns a Series; convert to DataFrame with "score" column
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    # Extract scores for target ticker
    if pred.index.nlevels > 1:
        # MultiIndex (datetime, instrument)
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
    signals = generate_signals(scores, ticker)
    median_threshold = float(scores.median())

    result = PredictionResult(
        ticker=ticker,
        model_class=best_model,
        prediction_range=(pred_start.isoformat(), today.isoformat()),
        signals=signals,
        latest_signal=signals[-1].signal,
        latest_score=signals[-1].score,
        threshold=round(median_threshold, 6),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Save to DB
    update_scan_job(db_path, job_id, prediction_json=json.dumps(result.model_dump()))
    emit({"event": "predict_complete", "data": result.model_dump()})

    return result


def run_quick_prediction(
    ticker: str,
    trading_style: str,
    db_path: str,
    emit: Callable[[dict], Any] | None = None,
    lookback_days: int = 60,
    model_class: str | None = None,
) -> PredictionResult:
    """Quick prediction using a pretrained model — no training required.

    Finds the best available pretrained model and uses it to predict
    on the target ticker. Falls back through QUICK_SCAN_MODELS if
    model_class is not specified.
    """
    emit = emit or (lambda e: None)
    from backtest_server.scanner import TRADING_STYLE_PRESETS
    from backtest_server.pretrain_manager import QUICK_SCAN_MODELS

    preset = TRADING_STYLE_PRESETS[trading_style]
    feature_set = preset["feature_set"]

    # Find a pretrained model
    if model_class:
        pt = find_pretrained_model(db_path, model_class, feature_set, trading_style)
        if pt is None:
            raise ValueError(f"No pretrained {model_class} for {trading_style}")
    else:
        # Try quick-scan models in order, pick first available
        pt = None
        for mc in QUICK_SCAN_MODELS:
            pt = find_pretrained_model(db_path, mc, feature_set, trading_style)
            if pt:
                model_class = mc
                break
        if pt is None:
            raise ValueError(
                f"No pretrained models available for {trading_style}. "
                "Run 'python -m backtest_server.scheduled_scan pretrain' first."
            )

    experiment_name = pt["experiment_name"]
    emit({"event": "quick_predict_start", "data": {
        "ticker": ticker, "model_class": model_class,
        "pretrained_model_id": pt["id"],
        "trained_at": pt.get("created_at"),
    }})

    # Build config for prediction
    config = build_scan_config(ticker, "db", trading_style, model_class)
    model_config, dataset_config = convert(config)

    today = datetime.now().date()
    pred_start = today - timedelta(days=lookback_days)
    dataset_config["kwargs"]["segments"]["test"] = (
        pred_start.isoformat(), today.isoformat()
    )

    # Init Qlib
    from backtest_server import data_pipeline
    provider_uri = data_pipeline.resolve_provider_uri("db")

    if not data_pipeline.check_ticker_exists(ticker, provider_uri):
        emit({"event": "quick_predict_progress", "data": {"msg": f"Importing {ticker} data..."}})
        data_pipeline.dump_db_to_qlib(ticker, provider_uri)

    qlib.init(provider_uri=provider_uri, region="us")

    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # Load pretrained model
    emit({"event": "quick_predict_progress", "data": {"msg": "Loading pretrained model..."}})
    recorders = R.list_recorders(experiment_name=experiment_name)
    finished = [r for r in recorders.values()
                if r.info.get("status", "").upper() == "FINISHED"]
    if not finished:
        raise ValueError(f"No finished recorder in experiment {experiment_name}")
    trained_model = finished[0].load_object("params.pkl")

    # Predict
    emit({"event": "quick_predict_progress", "data": {"msg": "Generating predictions..."}})
    pred = trained_model.predict(dataset, segment=(pred_start.isoformat(), today.isoformat()))

    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

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

    # Cache the prediction
    save_cached_prediction(
        db_path, ticker, pt["id"], model_class, trading_style,
        json.dumps(result.model_dump()),
    )

    emit({"event": "quick_predict_complete", "data": result.model_dump()})
    return result
