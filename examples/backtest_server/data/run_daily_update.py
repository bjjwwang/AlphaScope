#!/usr/bin/env python3
"""
Daily data update pipeline for AlphaScout.

Runs after US market close to refresh stock data, rebuild Qlib binary
datasets, and regenerate cached predictions for ALL tickers (parallel).

Schedule:
  08:30 AEDT (Sydney) = 21:30 UTC = 16:30 US/ET (after market close)
  yfinance data is settled ~30 min after market close.

Usage:
  # Single run (manual):
  python run_daily_update.py

  # Daemon mode (long-running, uses schedule library):
  python run_daily_update.py --daemon

  # Recommended: crontab (08:30 AEDT every weekday = Tue-Sat Sydney)
  #   30 8 * * 2-6  cd /path/to/data && /data1/wjw/.venv/bin/python3 run_daily_update.py >> /tmp/alphascout_daily.log 2>&1
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


def _get_all_tickers():
    """Read all tickers from stock_data.db."""
    import sqlite3
    db = os.path.join(_SCRIPT_DIR, "stock_data.db")
    if not os.path.exists(db):
        return []
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT DISTINCT ticker FROM kline_daily_1y ORDER BY ticker").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _run_predict_parallel(tickers, style, workers=4):
    """Run predict-batch for one style using parallel workers."""
    import re

    chunk_size = (len(tickers) + workers - 1) // workers
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    procs = []
    for idx, chunk in enumerate(chunks):
        ticker_str = ",".join(chunk)
        cmd = [
            sys.executable, "-m", "backtest_server.scheduled_scan",
            "predict-batch", "--tickers", ticker_str, "--style", style,
        ]
        log_path = f"/tmp/alphascout_predict_{style}_{idx+1}.log"
        log_file = open(log_path, "w")
        p = subprocess.Popen(cmd, cwd=_PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((p, log_file, log_path, len(chunk)))
        print(f"    Worker {idx+1}: {len(chunk)} tickers (PID {p.pid})")

    total_ok = 0
    total_fail = 0
    for idx, (p, log_file, log_path, count) in enumerate(procs):
        p.wait()
        log_file.close()
        try:
            with open(log_path) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "succeeded" in line:
                    m = re.search(r"(\d+) succeeded.*?(\d+) failed", line)
                    if m:
                        total_ok += int(m.group(1))
                        total_fail += int(m.group(2))
                    break
        except Exception:
            pass

    return total_ok, total_fail


def step_predict_batch(workers=4):
    """Step 3: Re-generate cached predictions for ALL tickers (parallel)."""
    tickers = _get_all_tickers()
    if not tickers:
        print(f"[{datetime.now():%H:%M:%S}] Step 3/3 — No tickers found, skipping.")
        return

    styles = ["swing", "ultra_short"]
    grand_ok = 0
    grand_fail = 0

    for style in styles:
        print(f"[{datetime.now():%H:%M:%S}] Step 3 — Predicting {len(tickers)} tickers "
              f"({style}) with {workers} workers...")
        ok, fail = _run_predict_parallel(tickers, style, workers)
        grand_ok += ok
        grand_fail += fail
        print(f"  {style}: {ok} succeeded, {fail} failed")

    print(f"  Total: {grand_ok} succeeded, {grand_fail} failed")


def step_generate_board():
    """Step 4: Generate daily board picks from cached predictions."""
    print(f"[{datetime.now():%H:%M:%S}] Step 4/4 — Generating board picks...")
    try:
        sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
        from backtest_server.board_generator import generate_daily_picks
        generate_daily_picks()
    except Exception as e:
        print(f"  Warning: Board generation failed: {e}")


def run_full_pipeline():
    """Run the complete daily update pipeline."""
    start = time.time()
    print(f"\n{'='*60}")
    print(f"AlphaScout Daily Update — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    step_update_stock_data()
    step_dump_qlib_data()
    step_predict_batch()
    step_generate_board()

    elapsed = time.time() - start
    print(f"\n[{datetime.now():%H:%M:%S}] Pipeline complete in {elapsed:.0f}s")
    print(f"{'='*60}\n")


def run_daemon():
    """Daemon mode: run daily at 17:30 (server local time)."""
    import schedule as sched

    print("AlphaScout daemon started. Daily update at 08:30 AEDT (Tue-Sat).")
    print("08:30 AEDT = 16:30 US/ET (after market close)")
    print("Press Ctrl+C to stop.\n")

    # US market closes Mon-Fri → data ready by Sydney Tue-Sat morning
    sched.every().tuesday.at("08:30").do(run_full_pipeline)
    sched.every().wednesday.at("08:30").do(run_full_pipeline)
    sched.every().thursday.at("08:30").do(run_full_pipeline)
    sched.every().friday.at("08:30").do(run_full_pipeline)
    sched.every().saturday.at("08:30").do(run_full_pipeline)

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
        step_generate_board()

        elapsed = time.time() - start
        print(f"\n[{datetime.now():%H:%M:%S}] Pipeline complete in {elapsed:.0f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
