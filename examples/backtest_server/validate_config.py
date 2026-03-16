#!/usr/bin/env python3
"""
Backend config validator — run a backtest from a YAML/JSON config file
without needing the frontend.

Usage:
    cd examples
    python -m backtest_server.validate_config config.yaml
    python -m backtest_server.validate_config config.json --dry-run
"""
import argparse
import json
import sys
import yaml
import asyncio
from pathlib import Path

from backtest_server.config_schema import BacktestRequest, VALID_MODEL_CLASSES


# ── Mapping: YAML shorthand model class → (module_path, dataset_class) ──
MODEL_REGISTRY = {
    "LGBModel":          ("qlib.contrib.model.gbdt", "DatasetH"),
    "XGBModel":          ("qlib.contrib.model.xgboost", "DatasetH"),
    "CatBoostModel":     ("qlib.contrib.model.catboost_model", "DatasetH"),
    "LinearModel":       ("qlib.contrib.model.linear", "DatasetH"),
    "DEnsembleModel":    ("qlib.contrib.model.double_ensemble", "DatasetH"),
    "LSTM_flat":         ("qlib.contrib.model.pytorch_lstm", "DatasetH"),
    "LSTM_ts":           ("qlib.contrib.model.pytorch_lstm_ts", "TSDatasetH"),
    "GRU_flat":          ("qlib.contrib.model.pytorch_gru", "DatasetH"),
    "GRU_ts":            ("qlib.contrib.model.pytorch_gru_ts", "TSDatasetH"),
    "ALSTM_flat":        ("qlib.contrib.model.pytorch_alstm", "DatasetH"),
    "ALSTM_ts":          ("qlib.contrib.model.pytorch_alstm_ts", "TSDatasetH"),
    "Transformer_flat":  ("qlib.contrib.model.pytorch_transformer", "DatasetH"),
    "Transformer_ts":    ("qlib.contrib.model.pytorch_transformer_ts", "TSDatasetH"),
    "GATs_flat":         ("qlib.contrib.model.pytorch_gats", "DatasetH"),
    "GATs_ts":           ("qlib.contrib.model.pytorch_gats_ts", "TSDatasetH"),
    "LocalFormer_ts":    ("qlib.contrib.model.pytorch_localformer_ts", "TSDatasetH"),
    "TCN_flat":          ("qlib.contrib.model.pytorch_tcn", "DatasetH"),
    "TCN_ts":            ("qlib.contrib.model.pytorch_tcn_ts", "TSDatasetH"),
    "TabnetModel":       ("qlib.contrib.model.pytorch_tabnet", "DatasetH"),
    "SFM":               ("qlib.contrib.model.pytorch_sfm", "DatasetH"),
    "DNNModelPytorch":   ("qlib.contrib.model.pytorch_nn", "DatasetH"),
    "ADARNN":            ("qlib.contrib.model.adarnn", "TSDatasetH"),
    "TRAModel":          ("qlib.contrib.model.tra", "DatasetH"),
    "HIST":              ("qlib.contrib.model.pytorch_hist", "DatasetH"),
}


def parse_yaml_config(path: str) -> dict:
    """Parse a YAML or JSON config file into a BacktestRequest-compatible dict."""
    text = Path(path).read_text()
    if path.endswith(".json"):
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)

    model_class = raw["model"]["class"]
    module_path = raw["model"].get("module_path")
    if not module_path:
        entry = MODEL_REGISTRY.get(model_class)
        if entry:
            module_path = entry[0]
        else:
            print(f"ERROR: Unknown model class '{model_class}'")
            print(f"Valid models: {sorted(VALID_MODEL_CLASSES)}")
            sys.exit(1)

    kwargs = raw["model"].get("kwargs", {})

    # Dataset
    ds = raw["dataset"]
    dataset_class = ds.get("class", "DatasetH")
    if not ds.get("class"):
        entry = MODEL_REGISTRY.get(model_class)
        if entry:
            dataset_class = entry[1]

    handler = ds.get("handler", "Alpha158")
    segs = ds["segments"]

    # Strategy
    strat = raw.get("strategy", {})
    strategy_class = strat.get("class", "SignalThreshold")

    request_dict = {
        "model": {
            "model_class": model_class,
            "module_path": module_path,
            "kwargs": kwargs,
        },
        "dataset": {
            "dataset_class": dataset_class,
            "handler": handler,
            "feature_set": ds.get("feature_set", "Alpha158"),
            "segments": {
                "train": list(segs["train"]),
                "valid": list(segs["valid"]),
                "test": list(segs["test"]),
            },
        },
        "data_source": raw.get("data_source", "yfinance"),
        "kline": raw.get("kline", "daily"),
        "target": raw.get("target", []),
        "training_pool": raw.get("training_pool", "target_only"),
        "label": raw.get("label", "Ref($close, -2)/Ref($close, -1) - 1"),
        "strategy": {
            "strategy_class": strategy_class,
            "topk": strat.get("topk"),
            "n_drop": strat.get("n_drop"),
            "buy_threshold": strat.get("buy_threshold"),
            "capital": strat.get("capital", 100000),
        },
    }
    return request_dict


def validate_only(request_dict: dict) -> BacktestRequest:
    """Validate config against Pydantic schema. Raises on error."""
    return BacktestRequest(**request_dict)


def run_backtest(request: BacktestRequest):
    """Run a full backtest synchronously, printing SSE events to stdout."""
    from backtest_server.backtest_runner import BacktestRunner

    events = []

    def emit(event: dict):
        evt_type = event.get("event", "log")
        data = event.get("data", {})
        if evt_type == "log":
            msg = data.get("msg", "") if isinstance(data, dict) else str(data)
            print(f"  [{evt_type}] {msg}")
        elif evt_type == "error":
            print(f"  [ERROR] {data}")
        elif evt_type == "result":
            events.append(data)
        else:
            print(f"  [{evt_type}] {json.dumps(data, ensure_ascii=False)}")

    runner = BacktestRunner(request, emit)
    result = runner.run_sync()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate and optionally run a backtest from a YAML/JSON config file."
    )
    parser.add_argument("config", help="Path to YAML or JSON config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate the config, don't run the backtest")
    parser.add_argument("--json", action="store_true",
                        help="Output result as JSON")
    args = parser.parse_args()

    # 1. Parse
    print(f"Parsing config: {args.config}")
    try:
        request_dict = parse_yaml_config(args.config)
    except Exception as e:
        print(f"PARSE ERROR: {e}")
        sys.exit(1)

    # 2. Validate
    print("Validating against schema...")
    try:
        request = validate_only(request_dict)
    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        sys.exit(1)

    print(f"  Model:    {request.model.model_class}")
    print(f"  Target:   {request.target}")
    print(f"  Source:    {request.data_source}")
    print(f"  Train:    {request.dataset.segments.train}")
    print(f"  Valid:     {request.dataset.segments.valid}")
    print(f"  Test:     {request.dataset.segments.test}")
    print(f"  Strategy: {request.strategy.strategy_class}")
    print("Config is VALID.")

    if args.dry_run:
        print("\n--dry-run specified, skipping backtest execution.")
        sys.exit(0)

    # 3. Run
    print("\nRunning backtest...")
    try:
        result = run_backtest(request)
    except Exception as e:
        print(f"\nBACKTEST ERROR: {e}")
        sys.exit(1)

    # 4. Output
    if args.json:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    else:
        s = result.summary
        print(f"\n{'='*50}")
        print(f"  Model Return:  {s.model_return:+.2f}%")
        print(f"  B&H Return:    {s.bh_return:+.2f}%")
        print(f"  Win Rate:      {s.win_rate:.1f}%")
        print(f"  Total Trades:  {s.total_trades}")
        print(f"  Test Period:   {s.test_start} → {s.test_end}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
