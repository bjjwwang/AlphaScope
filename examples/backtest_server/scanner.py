"""Auto model scanner — config generation, model iteration, ranking."""
import gc
import json
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Callable, Any

from dateutil.relativedelta import relativedelta

from backtest_server.config_schema import (
    BacktestRequest, ModelConfig, DatasetConfig, SegmentsConfig, StrategyConfig,
    VALID_MODEL_CLASSES,
)
from backtest_server.config_converter import PYTORCH_MODELS, FEATURE_DIM
from backtest_server.backtest_runner import BacktestRunner
from backtest_server import data_pipeline
from backtest_server.scan_db import update_scan_job, save_model_result, get_model_results


# Models excluded from scanning (need fundamentally different infrastructure):
# - TRAModel: requires MTSDatasetH + nested model_config/tra_config dicts
# - HIST: requires external stock2concept.npy / stock_index.npy mapping files
SCAN_EXCLUDED = {"TRAModel", "HIST"}

# Canonical list of scannable models with metadata.
MODEL_SCAN_LIST = [
    # Tree models (flat only)
    {"model_class": "LGBModel", "module_path": "qlib.contrib.model.gbdt", "data_type": "flat"},
    {"model_class": "XGBModel", "module_path": "qlib.contrib.model.xgboost", "data_type": "flat"},
    {"model_class": "CatBoostModel", "module_path": "qlib.contrib.model.catboost_model", "data_type": "flat"},
    {"model_class": "LinearModel", "module_path": "qlib.contrib.model.linear", "data_type": "flat"},
    {"model_class": "DEnsembleModel", "module_path": "qlib.contrib.model.double_ensemble", "data_type": "flat"},
    # RNN flat
    {"model_class": "LSTM_flat", "module_path": "qlib.contrib.model.pytorch_lstm", "data_type": "flat"},
    {"model_class": "GRU_flat", "module_path": "qlib.contrib.model.pytorch_gru", "data_type": "flat"},
    {"model_class": "ALSTM_flat", "module_path": "qlib.contrib.model.pytorch_alstm", "data_type": "flat"},
    # RNN ts
    {"model_class": "LSTM_ts", "module_path": "qlib.contrib.model.pytorch_lstm_ts", "data_type": "ts"},
    {"model_class": "GRU_ts", "module_path": "qlib.contrib.model.pytorch_gru_ts", "data_type": "ts"},
    {"model_class": "ALSTM_ts", "module_path": "qlib.contrib.model.pytorch_alstm_ts", "data_type": "ts"},
    # ADARNN
    {"model_class": "ADARNN", "module_path": "qlib.contrib.model.pytorch_adarnn", "data_type": "flat"},
    # Attention flat
    {"model_class": "Transformer_flat", "module_path": "qlib.contrib.model.pytorch_transformer", "data_type": "flat"},
    {"model_class": "GATs_flat", "module_path": "qlib.contrib.model.pytorch_gats", "data_type": "flat"},
    # Attention ts
    {"model_class": "Transformer_ts", "module_path": "qlib.contrib.model.pytorch_transformer_ts", "data_type": "ts"},
    {"model_class": "LocalFormer_ts", "module_path": "qlib.contrib.model.pytorch_localformer_ts", "data_type": "ts"},
    {"model_class": "GATs_ts", "module_path": "qlib.contrib.model.pytorch_gats_ts", "data_type": "ts"},
    # Other flat
    {"model_class": "TCN_flat", "module_path": "qlib.contrib.model.pytorch_tcn", "data_type": "flat"},
    {"model_class": "TabnetModel", "module_path": "qlib.contrib.model.pytorch_tabnet", "data_type": "flat"},
    {"model_class": "SFM", "module_path": "qlib.contrib.model.pytorch_sfm", "data_type": "flat"},
    {"model_class": "DNNModelPytorch", "module_path": "qlib.contrib.model.pytorch_nn", "data_type": "flat"},
    # Other ts
    {"model_class": "TCN_ts", "module_path": "qlib.contrib.model.pytorch_tcn_ts", "data_type": "ts"},
]

# Verify: scan list + excluded = all valid model classes
assert {m["model_class"] for m in MODEL_SCAN_LIST} | SCAN_EXCLUDED == VALID_MODEL_CLASSES

# Lookup by model_class
_MODEL_MAP = {m["model_class"]: m for m in MODEL_SCAN_LIST}


TRADING_STYLE_PRESETS = {
    "ultra_short": {
        "feature_set": "Alpha158_20",
        "label": "Ref($close, -2)/Ref($close, -1) - 1",
        "train_months": 6,
        "valid_months": 1,
        "test_months": 1,
    },
    "swing": {
        "feature_set": "Alpha158",
        "label": "Ref($close, -6)/Ref($close, -1) - 1",
        "train_months": 24,
        "valid_months": 6,
        "test_months": 3,
    },
}


def compute_date_segments(trading_style: str) -> SegmentsConfig:
    """Compute train/valid/test date segments relative to today."""
    preset = TRADING_STYLE_PRESETS[trading_style]
    today = datetime.now().date()

    test_end = today
    test_start = test_end - relativedelta(months=preset["test_months"])
    valid_end = test_start - timedelta(days=1)
    valid_start = valid_end - relativedelta(months=preset["valid_months"]) + timedelta(days=1)
    train_end = valid_start - timedelta(days=1)
    train_start = train_end - relativedelta(months=preset["train_months"]) + timedelta(days=1)

    return SegmentsConfig(
        train=(train_start.isoformat(), train_end.isoformat()),
        valid=(valid_start.isoformat(), valid_end.isoformat()),
        test=(test_start.isoformat(), test_end.isoformat()),
    )


def build_scan_config(
    ticker: str,
    data_source: str,
    trading_style: str,
    model_class: str,
) -> BacktestRequest:
    """Build a BacktestRequest for a single model in a scan."""
    preset = TRADING_STYLE_PRESETS[trading_style]
    model_info = _MODEL_MAP[model_class]
    segments = compute_date_segments(trading_style)

    # Resolve actual feature dimension for this trading style
    feat_dim = FEATURE_DIM.get(preset["feature_set"], 158)

    # Default kwargs per model type
    kwargs = {}
    if model_class == "DNNModelPytorch":
        # DNN uses pt_model_kwargs instead of d_feat
        kwargs = {
            "lr": 0.001,
            "max_steps": 300,
            "batch_size": 2000,
            "pt_model_kwargs": {"input_dim": feat_dim, "layers": (256,)},
        }
    elif model_class in PYTORCH_MODELS:
        kwargs = {
            "d_feat": 158,  # will be auto-corrected by config_converter
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
        data_source=data_source,
        target=[ticker],
        training_pool="target_only",
        label=preset["label"],
        strategy=StrategyConfig(
            strategy_class="SignalThreshold",
            buy_threshold=0.0,
        ),
    )


def rank_results(results: list[dict]) -> list[dict]:
    """Rank completed model results by model_return (desc), tiebreak by win_rate (desc)."""
    completed = [
        r for r in results
        if r.get("status") == "completed" and r.get("summary")
    ]
    return sorted(
        completed,
        key=lambda r: (r["summary"]["model_return"], r["summary"]["win_rate"]),
        reverse=True,
    )


# Model availability — mirrors server.py check for optional dependencies
_MODEL_DEPS = {"XGBModel": "xgboost", "CatBoostModel": "catboost"}


def _check_avail(mc: str) -> bool:
    pkg = _MODEL_DEPS.get(mc)
    if pkg is None:
        return True
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


_MODEL_AVAILABILITY = {mc: _check_avail(mc) for mc in VALID_MODEL_CLASSES}


class ScanRunner:
    """Iterate all models for a single stock, run backtests, rank results."""

    def __init__(
        self,
        ticker: str,
        data_source: str,
        trading_style: str,
        emit: Callable[[dict], Any],
        db_path: str,
        job_id: str,
    ):
        self.ticker = ticker
        self.data_source = data_source
        self.trading_style = trading_style
        self.emit = emit
        self.db_path = db_path
        self.job_id = job_id

    def _start_heartbeat(self):
        """Start a background heartbeat thread that sends pings every 15s."""
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
        """Synchronous execution — called from thread via asyncio.to_thread."""
        hb_stop = self._start_heartbeat()
        try:
            self._run_sync_inner()
        finally:
            hb_stop.set()

    def _run_sync_inner(self):
        available = [m for m in MODEL_SCAN_LIST if _MODEL_AVAILABILITY.get(m["model_class"], True)]
        skipped = [m for m in MODEL_SCAN_LIST if not _MODEL_AVAILABILITY.get(m["model_class"], True)]
        total = len(MODEL_SCAN_LIST)

        update_scan_job(self.db_path, self.job_id, status="running", total_models=total)
        self.emit({"event": "scan_start", "data": {
            "job_id": self.job_id, "ticker": self.ticker,
            "style": self.trading_style, "total_models": total,
        }})

        # Record skipped models
        for m in skipped:
            mc = m["model_class"]
            pkg = _MODEL_DEPS.get(mc, mc)
            save_model_result(self.db_path, self.job_id, mc, "skipped",
                              error_msg=f"缺少依赖 {pkg}")
            self.emit({"event": "scan_model_skipped", "data": {
                "model_class": mc, "reason": f"缺少依赖 {pkg}",
            }})

        # Prepare data once (reuse for all models)
        try:
            provider_uri = data_pipeline.resolve_provider_uri(self.data_source)
            if self.data_source == "db":
                data_pipeline.dump_all_db_to_qlib(provider_uri=provider_uri)
            if not data_pipeline.check_ticker_exists(self.ticker, provider_uri):
                if self.data_source == "db":
                    data_pipeline.dump_db_to_qlib(self.ticker, provider_uri)
                else:
                    data_pipeline.download_and_dump_yfinance(self.ticker, provider_uri)
        except Exception as e:
            update_scan_job(self.db_path, self.job_id, status="failed",
                            error_msg=f"数据准备失败: {e}")
            self.emit({"event": "scan_error", "data": {"msg": f"数据准备失败: {e}"}})
            return

        completed_count = len(skipped)
        collected_results = []

        for idx, model_info in enumerate(available):
            mc = model_info["model_class"]
            update_scan_job(self.db_path, self.job_id,
                            current_model=mc, completed_models=completed_count)
            self.emit({"event": "scan_model_start", "data": {
                "model_class": mc, "index": idx + 1, "total": len(available),
            }})

            t0 = time.time()
            try:
                config = build_scan_config(self.ticker, self.data_source,
                                           self.trading_style, mc)
                inner_runner = BacktestRunner(config, self.emit)
                result = inner_runner.run_sync()
                duration = time.time() - t0
                summary = result.summary.model_dump()

                save_model_result(
                    self.db_path, self.job_id, mc, "completed",
                    result_json=json.dumps(result.model_dump()),
                    recorder_id=getattr(inner_runner, "experiment_name", None),
                    duration_s=duration,
                )
                collected_results.append({
                    "model_class": mc, "status": "completed", "summary": summary,
                })
                self.emit({"event": "scan_model_complete", "data": {
                    "model_class": mc, "summary": summary, "duration_s": round(duration, 1),
                }})

            except Exception as e:
                duration = time.time() - t0
                err_msg = str(e)
                save_model_result(self.db_path, self.job_id, mc, "failed",
                                  error_msg=err_msg, duration_s=duration)
                collected_results.append({
                    "model_class": mc, "status": "failed", "summary": None,
                })
                self.emit({"event": "scan_model_failed", "data": {
                    "model_class": mc, "error": err_msg,
                }})

            completed_count += 1
            pct = int(completed_count / total * 100)
            self.emit({"event": "scan_progress", "data": {
                "completed": completed_count, "total": total,
                "current_model": mc, "pct": pct,
            }})

            # Free memory between models
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Rank and select best
        rankings = rank_results(collected_results)
        if rankings:
            best = rankings[0]
            best_mc = best["model_class"]
            best_ret = best["summary"]["model_return"]
            reason = f"最高模型收益 {best_ret}%"
            update_scan_job(self.db_path, self.job_id,
                            best_model=best_mc, best_model_reason=reason,
                            status="completed", completed_models=total,
                            current_model=None)
        else:
            reason = "所有模型均失败"
            update_scan_job(self.db_path, self.job_id,
                            status="completed", completed_models=total,
                            current_model=None, best_model_reason=reason)
            best_mc = None

        total_time = sum(
            r.get("duration_s", 0) or 0
            for r in (get_model_results(self.db_path, self.job_id) or [])
        )

        # Include training period info in scan_complete
        preset = TRADING_STYLE_PRESETS[self.trading_style]
        segs = compute_date_segments(self.trading_style)

        self.emit({"event": "scan_complete", "data": {
            "best_model": best_mc,
            "best_reason": reason,
            "rankings": [
                {"model_class": r["model_class"],
                 "model_return": r["summary"]["model_return"],
                 "bh_return": r["summary"]["bh_return"],
                 "win_rate": r["summary"]["win_rate"],
                 "total_trades": r["summary"]["total_trades"],
                 "held_days": r["summary"].get("held_days"),
                 "total_days": r["summary"].get("total_days")}
                for r in rankings
            ],
            "total_time_s": round(total_time, 1),
            "train_months": preset["train_months"],
            "valid_months": preset["valid_months"],
            "test_months": preset["test_months"],
            "segments": {
                "train": list(segs.train),
                "valid": list(segs.valid),
                "test": list(segs.test),
            },
        }})
