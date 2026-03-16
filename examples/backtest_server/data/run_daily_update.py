#!/usr/bin/env python3
"""
Daily data update pipeline for AlphaScout.

Runs after US market close to refresh stock data, rebuild Qlib binary
datasets, and regenerate cached predictions.

Schedule (US Eastern):
  17:30 ET — yfinance data is settled ~90 min after market close.

Usage:
  # Single run (manual):
  python run_daily_update.py

  # Daemon mode (long-running, uses schedule library):
  python run_daily_update.py --daemon

  # Recommended: crontab (17:30 ET every weekday)
  #   If server is UTC:  30 21 * * 1-5  cd /path/to/data && python run_daily_update.py >> /tmp/alphascout_daily.log 2>&1
  #   If server is US/ET: 30 17 * * 1-5  cd /path/to/data && python run_daily_update.py >> /tmp/alphascout_daily.log 2>&1
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# This script lives in backtest_server/data/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..", "..")  # examples/


def step_update_stock_data():
    """Step 1: Fetch latest prices from yfinance into stock_data.db."""
    print(f"[{datetime.now():%H:%M:%S}] Step 1/3 — Updating stock data from yfinance...")
    sys.path.insert(0, _SCRIPT_DIR)
    from stock_data_manager import StockDataManager

    manager = StockDataManager(db_path=os.path.join(_SCRIPT_DIR, "stock_data.db"))
    result = manager.daily_update()
    print(f"  Updated: {result['updated']}, Failed: {result['failed']}, Skipped: {result['skipped']}")
    return result


def step_dump_qlib_data():
    """Step 2: Convert stock_data.db → Qlib binary format."""
    print(f"[{datetime.now():%H:%M:%S}] Step 2/3 — Dumping to Qlib binary format...")
    try:
        # data_pipeline.dump_db_to_qlib() handles this
        sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
        from backtest_server.data_pipeline import dump_all_db_to_qlib
        dump_all_db_to_qlib()
        print("  Qlib data refreshed.")
    except Exception as e:
        print(f"  Warning: Qlib dump failed ({e}). Predictions will use stale data.")


def step_predict_batch():
    """Step 3: Re-generate cached predictions for top SP500 tickers."""
    print(f"[{datetime.now():%H:%M:%S}] Step 3/3 — Regenerating cached predictions...")
    cmd = [
        sys.executable, "-m", "backtest_server.scheduled_scan",
        "predict-batch", "--top", "20",
    ]
    result = subprocess.run(cmd, cwd=_PROJECT_DIR, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        # Print last few lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  {line}")
    else:
        print(f"  Warning: predict-batch failed: {result.stderr[:200]}")


def run_full_pipeline():
    """Run the complete daily update pipeline."""
    start = time.time()
    print(f"\n{'='*60}")
    print(f"AlphaScout Daily Update — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    step_update_stock_data()
    step_dump_qlib_data()
    step_predict_batch()

    elapsed = time.time() - start
    print(f"\n[{datetime.now():%H:%M:%S}] Pipeline complete in {elapsed:.0f}s")
    print(f"{'='*60}\n")


def run_daemon():
    """Daemon mode: run daily at 17:30 (server local time)."""
    import schedule as sched

    print("AlphaScout daemon started. Daily update at 17:30 (local time).")
    print("Set your server timezone to US/Eastern, or adjust the time below.")
    print("Press Ctrl+C to stop.\n")

    sched.every().monday.at("17:30").do(run_full_pipeline)
    sched.every().tuesday.at("17:30").do(run_full_pipeline)
    sched.every().wednesday.at("17:30").do(run_full_pipeline)
    sched.every().thursday.at("17:30").do(run_full_pipeline)
    sched.every().friday.at("17:30").do(run_full_pipeline)

    while True:
        sched.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="AlphaScout daily data update")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as daemon (weekdays at 17:30)")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip the batch prediction step")
    args = parser.parse_args()

    if args.daemon:
        try:
            import schedule  # noqa: F401
            run_daemon()
        except ImportError:
            print("Daemon mode requires: pip install schedule")
            return
    else:
        start = time.time()
        print(f"\n{'='*60}")
        print(f"AlphaScout Daily Update — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"{'='*60}")

        step_update_stock_data()
        step_dump_qlib_data()
        if not args.skip_predict:
            step_predict_batch()

        elapsed = time.time() - start
        print(f"\n[{datetime.now():%H:%M:%S}] Pipeline complete in {elapsed:.0f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
