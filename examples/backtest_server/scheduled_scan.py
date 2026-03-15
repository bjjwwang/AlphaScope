"""
Scheduled tasks — periodic model re-training and SP500 batch prediction.

Usage:
    # Pre-train all models (one-time or weekly re-train):
    python -m backtest_server.scheduled_scan pretrain

    # Pre-train only quick-scan models:
    python -m backtest_server.scheduled_scan pretrain --quick

    # Run batch predictions for SP500 tickers using pretrained models:
    python -m backtest_server.scheduled_scan predict-batch --tickers AAPL,MSFT,GOOGL

    # Run batch predictions for top-N SP500 by market cap:
    python -m backtest_server.scheduled_scan predict-batch --top 50

    # Force re-train (delete old models first):
    python -m backtest_server.scheduled_scan pretrain --force
"""
import argparse
import json
import os
import sys
import time

from backtest_server.scan_db import (
    init_db, list_pretrained_models, delete_pretrained_models_by_style,
    clear_cached_predictions, save_cached_prediction, find_pretrained_model,
)
from backtest_server.pretrain_manager import (
    PretrainRunner, QUICK_SCAN_MODELS, quick_predict,
)
from backtest_server.scanner import TRADING_STYLE_PRESETS

# SP500 top tickers by market cap (representative subset)
SP500_TOP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "XOM", "AVGO", "MA", "JNJ", "PG", "HD",
    "COST", "MRK", "ABBV", "ADBE", "CRM", "AMD", "NFLX", "KO", "PEP",
    "CVX", "TMO", "WMT", "BAC", "ACN", "LIN", "MCD", "CSCO", "ABT",
    "ORCL", "DHR", "INTC", "CMCSA", "VZ", "DIS", "PM", "TXN", "NEE",
    "WFC", "BMY", "RTX", "QCOM", "UPS",
]

_db_path = os.path.join(os.path.dirname(__file__), "data", "scan_jobs.db")


def _print_emit(event: dict):
    """Simple emit that prints events to stdout."""
    etype = event.get("event", "")
    data = event.get("data", {})
    if etype == "ping":
        return
    ts = time.strftime("%H:%M:%S")
    if etype == "pretrain_model_start":
        print(f"[{ts}] Training {data.get('model_class')} / {data.get('trading_style')} "
              f"({data.get('index')}/{data.get('total')})")
    elif etype == "pretrain_model_complete":
        perf = data.get("performance", {})
        print(f"[{ts}]   Done in {data.get('duration_s')}s — "
              f"return={perf.get('model_return', '?')}%, "
              f"win_rate={perf.get('win_rate', '?')}%")
    elif etype == "pretrain_model_failed":
        print(f"[{ts}]   FAILED: {data.get('error', '')[:100]}")
    elif etype == "pretrain_model_skip":
        print(f"[{ts}]   Skipping {data.get('model_class')} / {data.get('trading_style')}: "
              f"{data.get('reason')}")
    elif etype == "pretrain_complete":
        print(f"\n[{ts}] Pre-training complete: "
              f"{data.get('succeeded')} succeeded, {data.get('failed')} failed, "
              f"{data.get('skipped')} skipped")
    elif etype == "pretrain_start":
        print(f"[{ts}] Starting pre-training: {data.get('total')} model configs "
              f"({data.get('available')} available, {data.get('skipped')} skipped)")
    elif etype in ("pretrain_progress",):
        pass  # too noisy
    else:
        msg = data.get("msg", "")
        if msg:
            print(f"[{ts}] [{etype}] {msg}")


def cmd_pretrain(args):
    """Pre-train models."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    init_db(_db_path)

    models = QUICK_SCAN_MODELS if args.quick else None
    styles = [args.style] if args.style else None

    if args.force:
        print("Force mode: deleting old pretrained models...")
        if styles:
            for s in styles:
                delete_pretrained_models_by_style(_db_path, s)
                clear_cached_predictions(_db_path, s)
        else:
            for s in TRADING_STYLE_PRESETS:
                delete_pretrained_models_by_style(_db_path, s)
                clear_cached_predictions(_db_path)

    runner = PretrainRunner(
        db_path=_db_path,
        emit=_print_emit,
        training_pool=args.pool,
        models=models,
        styles=styles,
    )
    runner.run_sync()


def cmd_predict_batch(args):
    """Run batch predictions using pretrained models."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    init_db(_db_path)

    # Determine ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.top:
        tickers = SP500_TOP[:args.top]
    else:
        tickers = SP500_TOP[:20]

    styles = [args.style] if args.style else list(TRADING_STYLE_PRESETS.keys())

    # Find all available pretrained models
    pt_models = list_pretrained_models(_db_path, status="completed")
    if not pt_models:
        print("No pretrained models found. Run 'pretrain' first.")
        return

    # Group by (model_class, trading_style)
    pt_by_config = {}
    for pt in pt_models:
        key = (pt["model_class"], pt["trading_style"])
        if key not in pt_by_config:
            pt_by_config[key] = pt

    total = len(tickers) * len(styles)
    done = 0
    succeeded = 0
    failed = 0

    for style in styles:
        # Find the best pretrained model for this style (prefer LGBModel as it's fastest)
        best_pt = None
        for mc in QUICK_SCAN_MODELS:
            key = (mc, style)
            if key in pt_by_config:
                best_pt = pt_by_config[key]
                break

        if best_pt is None:
            # Try any available model
            for key, pt in pt_by_config.items():
                if key[1] == style:
                    best_pt = pt
                    break

        if best_pt is None:
            print(f"No pretrained model available for style={style}, skipping")
            done += len(tickers)
            continue

        mc = best_pt["model_class"]
        print(f"\nPredicting {len(tickers)} tickers with {mc} / {style}...")

        for ticker in tickers:
            done += 1
            try:
                result = quick_predict(
                    ticker, mc, style, _db_path,
                    lookback_days=args.lookback,
                )
                save_cached_prediction(
                    _db_path, ticker, best_pt["id"], mc, style,
                    json.dumps(result),
                )
                sig = result.get("latest_signal", "?")
                score = result.get("latest_score", 0)
                print(f"  [{done}/{total}] {ticker}: {sig} (score={score:.4f})")
                succeeded += 1
            except Exception as e:
                print(f"  [{done}/{total}] {ticker}: FAILED — {str(e)[:80]}")
                failed += 1

    print(f"\nBatch prediction complete: {succeeded} succeeded, {failed} failed")


def cmd_status(args):
    """Show pretrained model status."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    init_db(_db_path)

    models = list_pretrained_models(_db_path)
    if not models:
        print("No pretrained models found.")
        return

    by_style = {}
    for m in models:
        style = m["trading_style"]
        if style not in by_style:
            by_style[style] = {"completed": 0, "failed": 0, "training": 0}
        by_style[style][m["status"]] = by_style[style].get(m["status"], 0) + 1

    print(f"Total pretrained models: {len(models)}")
    for style, counts in by_style.items():
        print(f"  {style}: {counts.get('completed', 0)} completed, "
              f"{counts.get('failed', 0)} failed, "
              f"{counts.get('training', 0)} training")

    from backtest_server.scan_db import list_cached_predictions
    cached = list_cached_predictions(_db_path)
    print(f"\nCached predictions: {len(cached)}")
    for style in TRADING_STYLE_PRESETS:
        count = sum(1 for c in cached if c["trading_style"] == style)
        if count:
            print(f"  {style}: {count} tickers")


def main():
    parser = argparse.ArgumentParser(description="AlphaScout scheduled tasks")
    subparsers = parser.add_subparsers(dest="command")

    # pretrain
    p_train = subparsers.add_parser("pretrain", help="Pre-train models")
    p_train.add_argument("--quick", action="store_true",
                         help="Only train top 5 quick-scan models")
    p_train.add_argument("--force", action="store_true",
                         help="Delete old models and re-train from scratch")
    p_train.add_argument("--style", choices=list(TRADING_STYLE_PRESETS.keys()),
                         help="Only train for one trading style")
    p_train.add_argument("--pool", default="all",
                         choices=["all", "sp500", "target_only"],
                         help="Training pool (default: all)")

    # predict-batch
    p_pred = subparsers.add_parser("predict-batch", help="Batch predict using pretrained models")
    p_pred.add_argument("--tickers", help="Comma-separated ticker list")
    p_pred.add_argument("--top", type=int, help="Use top N SP500 tickers")
    p_pred.add_argument("--style", choices=list(TRADING_STYLE_PRESETS.keys()),
                         help="Only predict for one trading style")
    p_pred.add_argument("--lookback", type=int, default=60,
                         help="Lookback days for prediction (default: 60)")

    # status
    subparsers.add_parser("status", help="Show pretrained model status")

    args = parser.parse_args()
    if args.command == "pretrain":
        cmd_pretrain(args)
    elif args.command == "predict-batch":
        cmd_predict_batch(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
