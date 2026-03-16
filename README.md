# AlphaScout — AI Stock Signal Platform

> 22 AI Models. 600+ US Stocks. One Clear Answer.

**AlphaScout** is an open-source AI-powered stock analysis platform built on [Microsoft Qlib](https://github.com/microsoft/qlib). It automatically trains 22 machine learning models on any US stock and tells you which model performs best — giving you a clear buy, sell, or hold signal backed by out-of-sample results.

**Live demo:** [alphascout.bjjwwangs.win](https://alphascout.bjjwwangs.win)

---

## Features

- **AlphaScore** — One number (0-10) that summarizes 22 models' consensus on any stock
- **Auto Scan** — Trains 22 ML models (LightGBM, XGBoost, GRU, LSTM, Transformer, ALSTM, etc.) and ranks them by out-of-sample performance
- **Quick Predict** — Instant signals from pre-trained models (seconds, not minutes)
- **Out-of-Sample Proof** — All results use blind test data the models never saw during training
- **Bilingual UI** — Full English and Chinese support
- **Pitch Slides** — Built-in investor pitch deck with speaker notes

## Architecture

```
examples/
  backtest_server/          # FastAPI backend
    config_schema.py        # Pydantic models (24 model classes)
    config_converter.py     # Frontend JSON → Qlib config
    data_pipeline.py        # Data: yfinance / SQLite DB / baostock
    backtest_runner.py      # 6-step pipeline + SSE streaming
    result_formatter.py     # DataFrame → BacktestResult
    scanner.py              # Auto scan: 22 models ranked
    predictor.py            # Future signal prediction
    pretrain_manager.py     # Batch pre-training + quick predict
    scheduled_scan.py       # CLI: pretrain, predict-batch, status
    scan_db.py              # SQLite persistence (WAL mode)
    server.py               # FastAPI + SSE + job store
    tests/                  # 298 tests, all passing
    data/                   # Data pipeline
      cold_start.py         # One-time: download all US stock history
      run_daily_update.py   # Daily: update prices + refresh predictions
      stock_data_manager.py # SQLite data manager (yfinance)
      us_stocks.txt         # 23,494 US stock tickers
  landing.html              # English landing page
  backtest_ui_en.html       # English app (Backtest + Quick Predict)
  backtest_ui_v2.html       # Chinese app
  pitch_slides.html         # English pitch deck
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended for PyTorch models, not required for tree models)

### 1. Install

```bash
git clone https://github.com/bjjwwang/AlphaScout.git
cd AlphaScout
pip install -e ".[dev]"
```

### 2. Prepare Qlib data

```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### 3. Cold Start — Stock Database

Download historical price data for all US stocks into a local SQLite database. This is a one-time setup (~23k tickers, takes several hours depending on network):

```bash
cd examples/backtest_server/data
python cold_start.py
```

This creates `stock_data.db` (several GB). The script supports resumption — if interrupted, just re-run it.

### 4. Cold Start — Pre-train Models

Quick Predict requires pre-trained models. Train them once on first setup:

```bash
cd examples

# Quick mode: train top 5 models (~20 min on GPU)
python -m backtest_server.scheduled_scan pretrain --quick

# Full mode: train all 22 models (~2-3 hours)
python -m backtest_server.scheduled_scan pretrain

# Cache predictions for top SP500 stocks
python -m backtest_server.scheduled_scan predict-batch --top 50

# Verify everything is ready
python -m backtest_server.scheduled_scan status
```

### 5. Start the server

```bash
cd examples
python -m backtest_server
```

The server starts at `http://localhost:8001`. Open it in your browser.

### Routes

| Route | Description |
|-------|-------------|
| `/` | English landing page |
| `/cn` | Chinese landing page |
| `/app` | English app (Backtest + Quick Predict) |
| `/app-cn` | Chinese app |
| `/slides` | English pitch deck |
| `/slides-cn` | Chinese pitch deck |

---

## Daily Auto-Update

After the initial cold start, set up a daily cron job to keep data and predictions fresh. The pipeline runs **after US market close** (recommended: **5:30 PM Eastern**, which is 90 minutes after close to ensure data has settled):

### Option A: Crontab (recommended)

```bash
crontab -e
```

Add the following line (adjust path and timezone):

```cron
# AlphaScout daily update — 5:30 PM ET, weekdays only
# If your server is UTC, use 21:30 (= 17:30 ET during EDT)
# If your server is US/Eastern, use 17:30
30 21 * * 1-5 cd /path/to/AlphaScout/examples/backtest_server/data && python run_daily_update.py >> /tmp/alphascout_daily.log 2>&1
```

### Option B: Daemon mode

```bash
cd examples/backtest_server/data
python run_daily_update.py --daemon
```

Runs in the foreground, scheduling itself for 17:30 every weekday (server local time). Use `tmux` or `screen` to keep it alive.

### What the daily pipeline does

| Step | Action | Time |
|------|--------|------|
| 1 | Fetch latest prices from yfinance into `stock_data.db` | ~10 min |
| 2 | Rebuild Qlib binary format from updated DB | ~2 min |
| 3 | Re-generate cached predictions for top 20 SP500 stocks | ~5 min |

---

## 22 Models

### Tree Models (CPU)
LightGBM, XGBoost, CatBoost, Linear, Double Ensemble

### Recurrent Neural Networks (GPU)
LSTM, GRU, ALSTM — both cross-sectional and time-series variants

### Attention Models (GPU)
Transformer, GATs, LocalFormer — both cross-sectional and time-series variants

### Other (GPU)
TCN, TabNet, SFM, DNN, ADARNN

## Running Tests

```bash
cd examples
python -m pytest backtest_server/tests/ -v
```

298 tests, all passing. Tests run without Qlib/GPU via mocks.

## Credit System (Planned)

| Tier | Price | Credits/mo | What you get |
|------|-------|------------|--------------|
| Free | $0 | 3 | Quick Predict only |
| Pro | $19/mo | 50 | Full Scan + Quick Predict, all 22 models |
| Power | $49/mo | 200 | API access, priority GPU queue |

*1 Quick Predict = 1 credit, 1 Full Scan = 5 credits*

## Tech Stack

- **Backend:** FastAPI + SSE streaming + asyncio
- **ML Engine:** [Microsoft Qlib](https://github.com/microsoft/qlib) + PyTorch
- **Models:** LightGBM, XGBoost, CatBoost, PyTorch (LSTM/GRU/Transformer/etc.)
- **Data:** Yahoo Finance, SQLite, Baostock
- **Frontend:** Single-file HTML apps, dark mode, mobile-responsive

## License

This project is built on [Microsoft Qlib](https://github.com/microsoft/qlib), licensed under the MIT License.

## Acknowledgments

- [Microsoft Qlib](https://github.com/microsoft/qlib) — Quantitative investment platform
- Built for the UNSW Founders Program 2026
