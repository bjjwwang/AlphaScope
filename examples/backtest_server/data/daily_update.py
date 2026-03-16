#!/usr/bin/env python3
"""
Incremental update — fetch latest prices for stocks already in the database.

Usage (from any directory):
    python examples/backtest_server/data/daily_update.py
"""
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from stock_data_manager import StockDataManager

if __name__ == "__main__":
    print("=" * 60)
    print(f"AlphaScout — Incremental Update — {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)

    db_path = os.path.join(_SCRIPT_DIR, "stock_data.db")
    manager = StockDataManager(db_path=db_path)

    progress = manager.get_overall_progress()
    print(f"\nDatabase: {progress['total']} stocks")

    print("\nUpdating...")
    result = manager.daily_update()

    print(f"\nDone:")
    print(f"  Updated:  {result['updated']}")
    print(f"  Failed:   {result['failed']}")
    print(f"  Skipped:  {result['skipped']}")
