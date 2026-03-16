#!/usr/bin/env python3
"""
Cold start — download all US stock historical data into stock_data.db.

Usage (from any directory):
    python examples/backtest_server/data/cold_start.py

Or from the data/ directory:
    cd examples/backtest_server/data && python cold_start.py

Supports resume: re-run after interruption to continue where it left off.
"""
import os
import sys

# Ensure this script can be run from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from stock_data_manager import StockDataManager

# Read ticker list
_ticker_file = os.path.join(_SCRIPT_DIR, "us_stocks.txt")
with open(_ticker_file, "r") as f:
    TICKERS = [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    print("=" * 60)
    print("AlphaScout — Stock Data Cold Start")
    print("=" * 60)
    print(f"\nWill download data for {len(TICKERS)} US stocks")
    print(f"Estimated time: {len(TICKERS) * 5 / 60:.0f} minutes")
    print(f"First 10: {', '.join(TICKERS[:10])}...")

    # Non-interactive by default; pass --confirm for interactive prompt
    if "--confirm" in sys.argv:
        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            sys.exit(0)
    else:
        print("\nStarting download (re-run to resume if interrupted)...\n")

    db_path = os.path.join(_SCRIPT_DIR, "stock_data.db")
    manager = StockDataManager(db_path=db_path)
    manager.cold_start(TICKERS, resume=True)

    progress = manager.get_overall_progress()
    print(f"\nDone! Progress: {progress['progress']:.1f}%")
