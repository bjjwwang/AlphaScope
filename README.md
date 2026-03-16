# AlphaScout — AI Stock Signal Platform

> 22 AI Models. 600+ US Stocks. One Clear Answer.

**AlphaScout** is an open-source AI-powered stock analysis platform built on [Microsoft Qlib](https://github.com/microsoft/qlib). It automatically trains 22 machine learning models on any US stock and tells you which model performs best — giving you a clear buy, sell, or hold signal backed by out-of-sample results.

**Live demo:** [alphascout.bjjwwangs.win](https://alphascout.bjjwwangs.win)

---

## Quick Start (Clone → Run)

### 1. Clone and install

```bash
git clone https://github.com/bjjwwang/AlphaScout.git
cd AlphaScout
pip install -e ".[server]"
```

The `[server]` extra installs FastAPI, uvicorn, yfinance, and other backend dependencies. If you only want the backtest server (without the full Qlib dev stack), you can instead do:

```bash
pip install -r examples/backtest_server/requirements.txt
```

### 2. Prepare Qlib market data

Download US stock binary data used by Qlib's data loaders:

```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### 3. Cold start — download stock price history

This downloads historical OHLCV data for 23,000+ US stocks into a local SQLite database (`stock_data.db`). It takes several hours on first run but supports resume — re-run if interrupted:

```bash
python examples/backtest_server/data/cold_start.py
```

### 4. Cold start — pre-train models

Quick Predict needs pre-trained models. Train them once:

```bash
cd examples

# Quick mode (~20 min on GPU, 5 models):
python -m backtest_server.scheduled_scan pretrain --quick

# Full mode (~2-3 hours, all 22 models):
python -m backtest_server.scheduled_scan pretrain

# Cache predictions for popular stocks:
python -m backtest_server.scheduled_scan predict-batch --top 50

# Verify:
python -m backtest_server.scheduled_scan status
```

### 5. Start the server

```bash
cd examples
python -m backtest_server
```

Open `http://localhost:8001` in your browser. That's it.

---

## Daily Auto-Update (Cron)

After the initial setup, schedule a daily job to refresh data and predictions. Run it **after US market close** — we recommend **5:30 PM Eastern** (90 min after close, so data has settled).

### Crontab (recommended)

```bash
crontab -e
```

```cron
# AlphaScout daily update — 5:30 PM ET (= 21:30 UTC during EDT, 22:30 UTC during EST)
30 21 * * 1-5 cd /path/to/AlphaScout/examples/backtest_server/data && /path/to/python run_daily_update.py >> /tmp/alphascout_daily.log 2>&1
```

### What the daily pipeline does

| Step | Action | Time |
|------|--------|------|
| 1 | Fetch latest prices from yfinance → `stock_data.db` | ~10 min |
| 2 | Rebuild Qlib binary format from updated DB | ~2 min |
| 3 | Re-generate cached predictions for top 20 SP500 | ~5 min |

### Alternative: daemon mode

```bash
python examples/backtest_server/data/run_daily_update.py --daemon
```

Runs in the foreground, auto-triggers at 17:30 every weekday (server local time). Use `tmux`/`screen` to keep alive.

---

## Routes

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/cn` | Landing page (Chinese) |
| `/app` | Full app — Backtest + Quick Predict |
| `/app-cn` | Full app (Chinese) |
| `/slides` | Investor pitch deck |
| `/slides-cn` | Pitch deck (Chinese) |

## Features

- **AlphaScore** — One number (0-10) summarizing 22 models' consensus
- **Auto Scan** — Trains 22 ML models and ranks by out-of-sample excess return
- **Quick Predict** — Instant signals from pre-trained models (seconds)
- **Out-of-Sample Proof** — All numbers come from data the AI never saw during training
- **Bilingual UI** — Full English and Chinese support

## Architecture

```
examples/
  backtest_server/
    server.py               # FastAPI + SSE + serves all HTML pages
    config_schema.py        # Pydantic v2 models (24 model classes)
    config_converter.py     # Frontend JSON → Qlib config dicts
    data_pipeline.py        # yfinance / SQLite DB → Qlib binary
    backtest_runner.py      # 6-step pipeline + SSE progress streaming
    result_formatter.py     # DataFrame → BacktestResult
    scanner.py              # Auto scan: 22 models ranked
    predictor.py            # Future signal prediction
    pretrain_manager.py     # Batch pre-training + quick predict
    scheduled_scan.py       # CLI: pretrain / predict-batch / status
    scan_db.py              # SQLite persistence (WAL mode)
    requirements.txt        # Standalone dependency list
    tests/                  # 298 tests, all passing
    data/
      cold_start.py         # One-time: download all US stock history
      daily_update.py       # One-time: incremental update
      run_daily_update.py   # Daily pipeline: update + rebuild + predict
      stock_data_manager.py # Core SQLite data manager
      us_stocks.txt         # 23,494 US stock tickers
  landing.html              # English landing page
  backtest_ui_en.html       # English app
  backtest_ui_v2.html       # Chinese app
  pitch_slides.html         # Pitch deck (English)
  pitch_slides_cn.html      # Pitch deck (Chinese)
```

## 22 Models

| Category | Models |
|----------|--------|
| **Tree (CPU)** | LightGBM, XGBoost, CatBoost, Linear, Double Ensemble |
| **RNN (GPU)** | LSTM, GRU, ALSTM — cross-sectional & time-series variants |
| **Attention (GPU)** | Transformer, GATs, LocalFormer — cross-sectional & time-series |
| **Other (GPU)** | TCN, TabNet, SFM, DNN, ADARNN |

## Running Tests

```bash
cd examples
python -m pytest backtest_server/tests/ -v
```

298 tests, all passing. Tests run without Qlib/GPU via mocks.

## Backend Config Validation (CLI)

You can validate and run backtests from YAML/JSON config files directly — no frontend needed. This is useful for debugging frontend issues or running backtests in CI.

### Validate only (dry run)

```bash
cd examples
python -m backtest_server.validate_config backtest_server/sample_config.yaml --dry-run
```

### Run a full backtest

```bash
python -m backtest_server.validate_config backtest_server/sample_config.yaml
# or with JSON output:
python -m backtest_server.validate_config backtest_server/sample_config.yaml --json
```

### Config format (YAML)

```yaml
data_source: "yfinance"
kline: "daily"
target: ["AAPL"]

model:
  class: "LGBModel"    # Any of the 22 model classes
  kwargs:               # Model-specific hyperparameters
    num_leaves: 128
    learning_rate: 0.05

dataset:
  handler: "Alpha158"
  segments:
    train: ["2025-02-05", "2025-10-31"]
    valid: ["2025-11-01", "2025-12-31"]
    test:  ["2026-01-01", "2026-03-11"]

label: "Ref($close, -2)/Ref($close, -1) - 1"

strategy:
  class: "SignalThreshold"
  buy_threshold: 0
  capital: 100000
```

`module_path` and `dataset.class` are auto-resolved from the model class name. See `sample_config.yaml` for a complete example.

### Debugging workflow

If the frontend produces an error, copy the config from the browser console and save it as a `.yaml` file, then:

```bash
# 1. Validate the config structure
python -m backtest_server.validate_config user_config.yaml --dry-run

# 2. Run the backtest to reproduce the error
python -m backtest_server.validate_config user_config.yaml
```

## Deployment Notes

### PyTorch (required)

All 22 models require PyTorch. For CPU-only servers (no GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This saves ~1.5 GB compared to the full CUDA bundle.

### Qlib Calendar Sync

When using `yfinance` as data source, the data pipeline automatically:
1. Downloads OHLCV data from Yahoo Finance
2. Converts to Qlib binary format via `dump_bin.py`
3. **Merges the Qlib calendar** (`calendars/day.txt`) so that new dates are available

If you see a "division by zero" error during feature loading, it usually means the Qlib calendar is stale (e.g., only goes to 2020) while your config references 2025+ dates. Re-downloading data for the target ticker via yfinance will fix this automatically.

### Division by Zero Guards

The data pipeline guards against division-by-zero when computing `factor = adj_close / close` by replacing zero close prices with NaN (then filling factor with 1.0). Similar guards exist in `technical_indicators.py`, `scanner.py`, and `pretrain_manager.py`.

## Tech Stack

- **Backend:** FastAPI + SSE streaming + asyncio
- **ML Engine:** [Microsoft Qlib](https://github.com/microsoft/qlib) + PyTorch
- **Models:** LightGBM, XGBoost, CatBoost, PyTorch (LSTM/GRU/Transformer/etc.)
- **Data:** Yahoo Finance → SQLite → Qlib binary
- **Frontend:** Single-file HTML apps, dark mode, mobile-responsive

## Credit System (Planned)

| Tier | Price | Credits/mo | Includes |
|------|-------|------------|----------|
| Free | $0 | 3 | Quick Predict only |
| Pro | $19/mo | 50 | Full Scan + Quick Predict, all 22 models |
| Power | $49/mo | 200 | API access, priority GPU queue |

*1 Quick Predict = 1 credit, 1 Full Scan = 5 credits*

## License

Built on [Microsoft Qlib](https://github.com/microsoft/qlib) (MIT License).

## Acknowledgments

- [Microsoft Qlib](https://github.com/microsoft/qlib) — Quantitative investment platform
- Built for the UNSW Founders Program 2026
