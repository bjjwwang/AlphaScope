"""
FastAPI server with SSE for real-time backtest progress.
"""
import asyncio
import json
import uuid
import os
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backtest_server.config_schema import BacktestRequest, ScanRequest, VALID_MODEL_CLASSES
from backtest_server.backtest_runner import BacktestRunner
from backtest_server.scan_db import (
    init_db, create_scan_job, get_scan_job, update_scan_job,
    list_scan_jobs, get_model_results, get_latest_completed_scan,
    get_cached_prediction, list_pretrained_models, list_cached_predictions,
)
from backtest_server.scanner import ScanRunner

app = FastAPI(title="Qlib 回测 API", version="1.0.0")

# Model → required third-party package
MODEL_DEPENDENCIES = {
    "XGBModel": "xgboost",
    "CatBoostModel": "catboost",
}

def _check_model_available(model_class: str) -> bool:
    """Check if a model's dependencies are installed."""
    pkg = MODEL_DEPENDENCIES.get(model_class)
    if pkg is None:
        return True
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

# Check once at import time
_MODEL_AVAILABILITY = {mc: _check_model_available(mc) for mc in VALID_MODEL_CLASSES}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (for backtests)
jobs: Dict[str, Dict[str, Any]] = {}

# In-memory scan SSE queues
scan_queues: Dict[str, asyncio.Queue] = {}

# Scan DB path
_scan_db_path = os.path.join(os.path.dirname(__file__), "data", "scan_jobs.db")

# Initialize scan DB on import (creates tables if not exist)
os.makedirs(os.path.dirname(_scan_db_path), exist_ok=True)
init_db(_scan_db_path)

# Model metadata for /api/models
MODEL_METADATA = {
    "LGBModel": {"name": "LGBModel", "fullName": "LightGBM", "cat": "tree", "gpu": False,
                  "brief": "梯度提升树，速度快，最常用"},
    "XGBModel": {"name": "XGBModel", "fullName": "XGBoost", "cat": "tree", "gpu": False,
                 "brief": "经典梯度提升树，功能全面"},
    "CatBoostModel": {"name": "CatBoostModel", "fullName": "CatBoost", "cat": "tree", "gpu": False,
                      "brief": "处理类别特征强，抗过拟合"},
    "LinearModel": {"name": "LinearModel", "fullName": "线性模型", "cat": "tree", "gpu": False,
                    "brief": "最简单的基线，速度最快"},
    "DEnsembleModel": {"name": "DEnsembleModel", "fullName": "Double Ensemble", "cat": "tree", "gpu": False,
                       "brief": "双重集成，关注困难样本"},
    "GRU_ts": {"name": "GRU_ts", "fullName": "GRU (时间序列)", "cat": "rnn", "gpu": True,
               "brief": "轻量RNN，训练快，适合波动股"},
    "GRU_flat": {"name": "GRU_flat", "fullName": "GRU (截面)", "cat": "rnn", "gpu": True,
                 "brief": "GRU截面版，输入当天特征"},
    "LSTM_ts": {"name": "LSTM_ts", "fullName": "LSTM (时间序列)", "cat": "rnn", "gpu": True,
                "brief": "经典RNN，长期记忆能力强"},
    "LSTM_flat": {"name": "LSTM_flat", "fullName": "LSTM (截面)", "cat": "rnn", "gpu": True,
                  "brief": "LSTM截面版"},
    "ALSTM_ts": {"name": "ALSTM_ts", "fullName": "ALSTM (注意力LSTM)", "cat": "rnn", "gpu": True,
                 "brief": "加注意力的LSTM，关注重要时刻"},
    "ALSTM_flat": {"name": "ALSTM_flat", "fullName": "ALSTM (截面)", "cat": "rnn", "gpu": True,
                   "brief": "注意力LSTM截面版"},
    "ADARNN": {"name": "ADARNN", "fullName": "AdaRNN", "cat": "rnn", "gpu": True,
               "brief": "自适应不同市场环境"},
    "Transformer_ts": {"name": "Transformer_ts", "fullName": "Transformer", "cat": "attention", "gpu": True,
                       "brief": "自注意力机制，并行计算快"},
    "Transformer_flat": {"name": "Transformer_flat", "fullName": "Transformer (截面)", "cat": "attention", "gpu": True,
                         "brief": "Transformer截面版"},
    "LocalFormer_ts": {"name": "LocalFormer_ts", "fullName": "LocalFormer", "cat": "attention", "gpu": True,
                       "brief": "局部注意力，关注近期数据"},
    "GATs_ts": {"name": "GATs_ts", "fullName": "GATs (图注意力)", "cat": "attention", "gpu": True,
                "brief": "图网络建模股票关联关系"},
    "GATs_flat": {"name": "GATs_flat", "fullName": "GATs (截面)", "cat": "attention", "gpu": True,
                  "brief": "图注意力截面版"},
    "TRAModel": {"name": "TRAModel", "fullName": "TRA", "cat": "attention", "gpu": True,
                 "brief": "多路径选择，适应市场变化"},
    "TCN_ts": {"name": "TCN_ts", "fullName": "TCN (时间卷积)", "cat": "other", "gpu": True,
               "brief": "因果卷积，计算效率高"},
    "TCN_flat": {"name": "TCN_flat", "fullName": "TCN (截面)", "cat": "other", "gpu": True,
                 "brief": "TCN截面版"},
    "TabnetModel": {"name": "TabnetModel", "fullName": "TabNet", "cat": "other", "gpu": True,
                    "brief": "注意力选特征，可解释性好"},
    "HIST": {"name": "HIST", "fullName": "HIST", "cat": "other", "gpu": True,
             "brief": "利用共享信息和独有信息"},
    "SFM": {"name": "SFM", "fullName": "SFM", "cat": "other", "gpu": True,
            "brief": "频域分析捕捉多周期模式"},
    "DNNModelPytorch": {"name": "DNNModelPytorch", "fullName": "DNN", "cat": "other", "gpu": True,
                        "brief": "全连接网络，简单直接"},
}


_html_file = os.path.join(os.path.dirname(__file__), "..", "backtest_ui_en.html")
_html_cn_file = os.path.join(os.path.dirname(__file__), "..", "backtest_ui_v2.html")
_landing_file = os.path.join(os.path.dirname(__file__), "..", "landing.html")
_pitch_file = os.path.join(os.path.dirname(__file__), "..", "pitch.html")
_slides_file = os.path.join(os.path.dirname(__file__), "..", "pitch_slides.html")
_landing_cn_file = os.path.join(os.path.dirname(__file__), "..", "landing_cn.html")
_slides_cn_file = os.path.join(os.path.dirname(__file__), "..", "pitch_slides_cn.html")


@app.get("/")
async def root():
    if os.path.exists(_landing_file):
        return FileResponse(_landing_file, media_type="text/html")
    return {"service": "AlphaScout API", "version": "1.0.0", "models": len(MODEL_METADATA)}


@app.get("/app")
async def app_page():
    if os.path.exists(_html_file):
        return FileResponse(_html_file, media_type="text/html")
    return {"error": "App not found"}


@app.get("/app-cn")
async def app_cn_page():
    if os.path.exists(_html_cn_file):
        return FileResponse(_html_cn_file, media_type="text/html")
    return {"error": "Chinese app not found"}


@app.get("/pitch")
async def pitch_page():
    if os.path.exists(_pitch_file):
        return FileResponse(_pitch_file, media_type="text/html")
    return {"error": "Pitch deck not found"}


@app.get("/slides")
async def slides_page():
    if os.path.exists(_slides_file):
        return FileResponse(_slides_file, media_type="text/html")
    return {"error": "Slides not found"}


@app.get("/cn")
async def landing_cn_page():
    if os.path.exists(_landing_cn_file):
        return FileResponse(_landing_cn_file, media_type="text/html")
    return {"error": "Chinese landing page not found"}


@app.get("/slides-cn")
async def slides_cn_page():
    if os.path.exists(_slides_cn_file):
        return FileResponse(_slides_cn_file, media_type="text/html")
    return {"error": "Chinese slides not found"}


def compute_alpha_score(model_results: list[dict]) -> dict:
    """Compute AlphaScore (1-10) from scan model results.

    Scoring factors:
    - How many models are profitable (breadth)
    - Best model excess return over buy-and-hold (alpha)
    - Average win rate across profitable models (consistency)
    - Agreement among top models (consensus)
    """
    completed = []
    for r in model_results:
        if r.get("status") != "completed" or not r.get("result_json"):
            continue
        try:
            summary = json.loads(r["result_json"]).get("summary", {})
            completed.append(summary)
        except (json.JSONDecodeError, TypeError):
            continue

    if not completed:
        return {"score": 1, "label": "无数据", "signal": "hold",
                "reason": "暂无扫描数据", "details": {}}

    total = len(completed)
    # How many models have positive return
    profitable = [s for s in completed if (s.get("model_return") or 0) > 0]
    profitable_ratio = len(profitable) / total if total > 0 else 0

    # Best excess return
    bh_ret = completed[0].get("bh_return", 0) or 0
    excess_returns = [(s.get("model_return") or 0) - bh_ret for s in completed]
    best_excess = max(excess_returns) if excess_returns else 0
    avg_excess = sum(excess_returns) / len(excess_returns) if excess_returns else 0

    # Average win rate of profitable models
    win_rates = [s.get("win_rate", 50) or 50 for s in profitable] if profitable else [50]
    avg_wr = sum(win_rates) / len(win_rates)

    # Best model return
    best_ret = max((s.get("model_return") or 0) for s in completed)

    # Consensus: how many models agree on direction (positive vs negative)
    bullish = sum(1 for s in completed if (s.get("model_return") or 0) > 0)
    consensus = bullish / total if total > 0 else 0.5

    # Composite score (0-100 raw, then map to 1-10)
    raw = (
        profitable_ratio * 25       # 0-25: breadth of profitability
        + min(best_excess / 5, 1) * 25  # 0-25: alpha over benchmark (cap at 5%)
        + (avg_wr - 30) / 40 * 25   # 0-25: win rate (30%-70% → 0-25)
        + consensus * 25             # 0-25: model agreement
    )
    raw = max(0, min(100, raw))
    score = max(1, min(10, round(raw / 10)))

    # Signal
    if score >= 7:
        signal = "buy"
        label = "强烈看涨" if score >= 9 else "看涨"
    elif score >= 4:
        signal = "hold"
        label = "中性"
    else:
        signal = "sell"
        label = "看跌" if score >= 2 else "强烈看跌"

    # Human-readable reason
    parts = []
    if profitable_ratio >= 0.6:
        parts.append(f"{len(profitable)}/{total} 个模型盈利")
    elif profitable_ratio <= 0.3:
        parts.append(f"仅 {len(profitable)}/{total} 个模型盈利")
    if best_excess > 1:
        parts.append(f"最优模型跑赢基准 {best_excess:.1f}%")
    elif best_excess < -1:
        parts.append(f"所有模型均跑输基准")
    if avg_wr > 55:
        parts.append(f"平均胜率 {avg_wr:.0f}%")
    reason = "，".join(parts) if parts else "综合评估"

    return {
        "score": score,
        "label": label,
        "signal": signal,
        "reason": reason,
        "details": {
            "profitable_models": len(profitable),
            "total_models": total,
            "best_return": round(best_ret, 2),
            "best_excess": round(best_excess, 2),
            "avg_win_rate": round(avg_wr, 1),
            "consensus": round(consensus, 2),
            "bh_return": round(bh_ret, 2),
        },
    }


@app.get("/api/score/{ticker}")
async def quick_score(ticker: str, trading_style: str = "swing"):
    """Quick AlphaScore lookup — checks cached predictions first, then full scans."""
    ticker = ticker.upper()

    # 1. Check cached predictions (from pretrained model batch)
    cached = get_cached_prediction(_scan_db_path, ticker, trading_style)
    if cached and cached.get("alpha_score_json"):
        try:
            alpha = json.loads(cached["alpha_score_json"])
            prediction = json.loads(cached["prediction_json"]) if cached.get("prediction_json") else None
            return {
                "ticker": ticker,
                "alpha_score": alpha,
                "best_model": cached.get("model_class"),
                "trading_style": trading_style,
                "cached_at": cached.get("cached_at"),
                "source": "cached_prediction",
                "prediction": prediction,
            }
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. Check cached prediction (without alpha score — just has prediction)
    if cached and cached.get("prediction_json"):
        try:
            prediction = json.loads(cached["prediction_json"])
            return {
                "ticker": ticker,
                "alpha_score": None,
                "best_model": cached.get("model_class"),
                "trading_style": trading_style,
                "cached_at": cached.get("cached_at"),
                "source": "cached_prediction",
                "prediction": prediction,
            }
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Fall back to full scan results
    job = get_latest_completed_scan(_scan_db_path, ticker)
    if job is None:
        raise HTTPException(404, f"No scan data for {ticker}. Run a scan first.")
    results = get_model_results(_scan_db_path, job["id"])
    alpha = compute_alpha_score(results)
    return {
        "ticker": ticker,
        "alpha_score": alpha,
        "best_model": job.get("best_model"),
        "trading_style": job.get("trading_style"),
        "scan_date": job.get("created_at"),
        "scan_job_id": job["id"],
        "source": "full_scan",
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    result = []
    for m in MODEL_METADATA.values():
        entry = dict(m)
        entry["available"] = _MODEL_AVAILABILITY.get(m["name"], True)
        pkg = MODEL_DEPENDENCIES.get(m["name"])
        if pkg and not entry["available"]:
            entry["unavailable_reason"] = f"缺少依赖: pip install {pkg}"
        result.append(entry)
    return result


# =============================================
# Backtest endpoints (existing)
# =============================================

@app.post("/api/backtest")
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    # Check model dependency before starting
    mc = request.model.model_class
    if not _MODEL_AVAILABILITY.get(mc, True):
        pkg = MODEL_DEPENDENCIES.get(mc, mc)
        raise HTTPException(400, f"模型 {mc} 不可用: 缺少依赖 {pkg}。请在服务器运行 pip install {pkg}")

    job_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    jobs[job_id] = {"status": "running", "queue": queue, "result": None}

    background_tasks.add_task(_run_backtest_job, job_id, request)
    return {"job_id": job_id}


@app.get("/api/backtest/{job_id}/stream")
async def stream_backtest(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        queue = jobs[job_id]["queue"]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield f"event: ping\ndata: {{}}\n\n"
                continue

            event_type = event.get("event", "log")
            data = json.dumps(event.get("data", {}), ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("result", "error"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/backtest/{job_id}/status")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    resp = {"status": job["status"]}
    if job["result"] is not None:
        resp["result"] = job["result"]
    return resp


async def _run_backtest_job(job_id: str, request: BacktestRequest):
    queue = jobs[job_id]["queue"]
    loop = asyncio.get_event_loop()

    def emit(event: dict):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    runner = BacktestRunner(request, emit)

    try:
        result = await asyncio.to_thread(runner.run_sync)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result.model_dump()
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        await queue.put({"event": "error", "data": {"step": -1, "msg": str(e)}})


# =============================================
# Scan endpoints (new)
# =============================================

@app.post("/api/scan")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    job_id = create_scan_job(
        _scan_db_path, request.ticker, request.data_source, request.trading_style
    )
    queue = asyncio.Queue()
    scan_queues[job_id] = queue

    background_tasks.add_task(_run_scan_job, job_id, request)
    return {"job_id": job_id}


@app.get("/api/scan/list")
async def list_scans():
    return list_scan_jobs(_scan_db_path)


@app.get("/api/scan/{job_id}")
async def get_scan(job_id: str):
    job = get_scan_job(_scan_db_path, job_id)
    if job is None:
        raise HTTPException(404, "Scan job not found")

    # Include model results
    results = get_model_results(_scan_db_path, job_id)
    # Parse result_json strings into dicts for the response
    model_results = []
    for r in results:
        entry = {
            "model_class": r["model_class"],
            "status": r["status"],
            "error_msg": r.get("error_msg"),
            "duration_s": r.get("duration_s"),
        }
        if r.get("result_json"):
            try:
                entry["summary"] = json.loads(r["result_json"]).get("summary")
            except (json.JSONDecodeError, TypeError):
                entry["summary"] = None
        model_results.append(entry)

    resp = dict(job)
    resp["model_results"] = model_results

    # Parse prediction_json if present
    if resp.get("prediction_json"):
        try:
            resp["prediction"] = json.loads(resp["prediction_json"])
        except (json.JSONDecodeError, TypeError):
            resp["prediction"] = None
    else:
        resp["prediction"] = None

    return resp


@app.get("/api/scan/{job_id}/stream")
async def stream_scan(job_id: str):
    job = get_scan_job(_scan_db_path, job_id)
    if job is None:
        raise HTTPException(404, "Scan job not found")

    status = job["status"]

    # Completed/failed scan: return final event immediately
    if status in ("completed", "failed"):
        async def completed_generator():
            if status == "completed":
                # Send model results summary
                results = get_model_results(_scan_db_path, job_id)
                rankings = []
                for r in results:
                    if r["status"] == "completed" and r.get("result_json"):
                        try:
                            summary = json.loads(r["result_json"]).get("summary", {})
                            rankings.append({
                                "model_class": r["model_class"],
                                "model_return": summary.get("model_return", 0),
                                "win_rate": summary.get("win_rate", 0),
                            })
                        except (json.JSONDecodeError, TypeError):
                            pass
                rankings.sort(key=lambda x: x.get("model_return", 0), reverse=True)

                data = json.dumps({
                    "best_model": job.get("best_model"),
                    "best_reason": job.get("best_model_reason"),
                    "rankings": rankings,
                    "total_time_s": 0,
                }, ensure_ascii=False)
                yield f"event: scan_complete\ndata: {data}\n\n"
            else:
                data = json.dumps({"msg": job.get("error_msg", "Unknown error")}, ensure_ascii=False)
                yield f"event: scan_error\ndata: {data}\n\n"

        return StreamingResponse(completed_generator(), media_type="text/event-stream")

    # Running scan: stream from queue
    async def running_generator():
        # Send current progress snapshot from DB
        progress_data = json.dumps({
            "completed": job.get("completed_models", 0),
            "total": job.get("total_models", 0),
            "current_model": job.get("current_model"),
            "pct": int(job.get("completed_models", 0) / max(job.get("total_models", 1), 1) * 100),
        }, ensure_ascii=False)
        yield f"event: scan_progress\ndata: {progress_data}\n\n"

        # Send already-completed model results
        results = get_model_results(_scan_db_path, job_id)
        for r in results:
            if r["status"] == "completed" and r.get("result_json"):
                try:
                    summary = json.loads(r["result_json"]).get("summary", {})
                    data = json.dumps({
                        "model_class": r["model_class"],
                        "summary": summary,
                        "duration_s": r.get("duration_s", 0),
                    }, ensure_ascii=False)
                    yield f"event: scan_model_complete\ndata: {data}\n\n"
                except (json.JSONDecodeError, TypeError):
                    pass
            elif r["status"] == "failed":
                data = json.dumps({
                    "model_class": r["model_class"],
                    "error": r.get("error_msg", ""),
                }, ensure_ascii=False)
                yield f"event: scan_model_failed\ndata: {data}\n\n"
            elif r["status"] == "skipped":
                data = json.dumps({
                    "model_class": r["model_class"],
                    "reason": r.get("error_msg", ""),
                }, ensure_ascii=False)
                yield f"event: scan_model_skipped\ndata: {data}\n\n"

        # Stream real-time events from queue
        queue = scan_queues.get(job_id)
        if queue is None:
            queue = asyncio.Queue()
            scan_queues[job_id] = queue

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield f"event: ping\ndata: {{}}\n\n"
                continue

            event_type = event.get("event", "log")
            data = json.dumps(event.get("data", {}), ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("scan_complete", "scan_error"):
                break

    return StreamingResponse(running_generator(), media_type="text/event-stream")


@app.post("/api/scan/{job_id}/predict")
async def predict_scan(job_id: str, background_tasks: BackgroundTasks):
    job = get_scan_job(_scan_db_path, job_id)
    if job is None:
        raise HTTPException(404, "Scan job not found")
    if not job.get("best_model"):
        raise HTTPException(400, "扫描尚未完成或没有 best_model")

    # Run prediction synchronously for now (could be async later)
    from backtest_server.predictor import run_prediction
    try:
        result = run_prediction(job_id, _scan_db_path, lambda e: None)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, f"预测失败: {e}")


async def _run_scan_job(job_id: str, request: ScanRequest):
    queue = scan_queues.get(job_id)
    if queue is None:
        queue = asyncio.Queue()
        scan_queues[job_id] = queue

    loop = asyncio.get_event_loop()

    def emit(event: dict):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    runner = ScanRunner(
        request.ticker, request.data_source, request.trading_style,
        emit, _scan_db_path, job_id,
    )

    try:
        await asyncio.to_thread(runner.run_sync)
    except Exception as e:
        update_scan_job(_scan_db_path, job_id, status="failed", error_msg=str(e))
        await queue.put({"event": "scan_error", "data": {"msg": str(e)}})
    finally:
        # Clean up queue after scan completes
        scan_queues.pop(job_id, None)


# =============================================
# Quick predict endpoints (pretrained models)
# =============================================

from pydantic import BaseModel as PydanticBaseModel


class QuickPredictRequest(PydanticBaseModel):
    ticker: str
    trading_style: str = "swing"
    model_class: str | None = None


@app.post("/api/quick-predict")
async def quick_predict_endpoint(request: QuickPredictRequest):
    """Quick prediction using pretrained model — no training needed, ~5 seconds."""
    ticker = request.ticker.upper().strip()
    if not ticker:
        raise HTTPException(400, "Ticker cannot be empty")

    # Check cache first
    cached = get_cached_prediction(_scan_db_path, ticker, request.trading_style)
    if cached:
        try:
            prediction = json.loads(cached["prediction_json"])
            return {
                "prediction": prediction,
                "source": "cache",
                "cached_at": cached["cached_at"],
                "model_class": cached["model_class"],
            }
        except (json.JSONDecodeError, TypeError):
            pass

    # Run prediction
    from backtest_server.predictor import run_quick_prediction
    try:
        result = await asyncio.to_thread(
            run_quick_prediction,
            ticker, request.trading_style, _scan_db_path,
            model_class=request.model_class,
        )
        return {
            "prediction": result.model_dump(),
            "source": "live",
            "model_class": result.model_class,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.get("/api/pretrained-models")
async def get_pretrained_models(status: str | None = None):
    """List all pretrained models."""
    models = list_pretrained_models(_scan_db_path, status=status)
    # Parse performance_json for each
    result = []
    for m in models:
        entry = dict(m)
        if entry.get("performance_json"):
            try:
                entry["performance"] = json.loads(entry["performance_json"])
            except (json.JSONDecodeError, TypeError):
                entry["performance"] = None
        result.append(entry)
    return result


@app.get("/api/cached-predictions")
async def get_cached_predictions(trading_style: str | None = None):
    """List all cached predictions."""
    return list_cached_predictions(_scan_db_path, trading_style=trading_style)


@app.post("/api/pretrain")
async def start_pretrain(background_tasks: BackgroundTasks, quick: bool = False):
    """Start pre-training models in the background."""
    from backtest_server.pretrain_manager import PretrainRunner, QUICK_SCAN_MODELS

    job_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    jobs[job_id] = {"status": "running", "queue": queue, "result": None, "type": "pretrain"}

    async def _run():
        loop = asyncio.get_event_loop()

        def emit(event: dict):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        models = QUICK_SCAN_MODELS if quick else None
        runner = PretrainRunner(
            db_path=_scan_db_path,
            emit=emit,
            models=models,
        )
        try:
            await asyncio.to_thread(runner.run_sync)
            jobs[job_id]["status"] = "completed"
            await queue.put({"event": "pretrain_complete", "data": {}})
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            await queue.put({"event": "error", "data": {"msg": str(e)}})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "mode": "quick" if quick else "full"}


@app.get("/api/pretrain/{job_id}/stream")
async def stream_pretrain(job_id: str):
    """Stream pre-training progress via SSE."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        queue = jobs[job_id]["queue"]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield f"event: ping\ndata: {{}}\n\n"
                continue

            event_type = event.get("event", "log")
            data = json.dumps(event.get("data", {}), ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("pretrain_complete", "error"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print(f"""
╔══════════════════════════════════════════════╗
║       Qlib 回测实验室 API Server             ║
╠══════════════════════════════════════════════╣
║  http://localhost:8001                       ║
║  模型: {len(MODEL_METADATA)} 个                            ║
╚══════════════════════════════════════════════╝
    """)
    uvicorn.run("backtest_server.server:app", host="0.0.0.0", port=8001, reload=True)
