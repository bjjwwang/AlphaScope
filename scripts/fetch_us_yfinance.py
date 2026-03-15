"""
Fetch US stock data (S&P500 + NASDAQ100) using yfinance, dump to Qlib format.
"""
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

# ── config ──────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE = "2026-03-08"
CSV_DIR = os.path.expanduser("~/.qlib/stock_data/us_yf_csv")
QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/us_data")
INSTRUMENTS_DIR = os.path.join(QLIB_DIR, "instruments")
CALENDAR_DIR = os.path.join(QLIB_DIR, "calendars")

# Get S&P500 + NASDAQ100 tickers
def get_sp500_tickers():
    """Get S&P500 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = table[0]
        tickers = df["Symbol"].tolist()
        # Fix tickers with dots (BRK.B -> BRK-B for yfinance)
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers
    except Exception as e:
        print(f"Failed to get S&P500 tickers from Wikipedia: {e}")
        return None

def get_nasdaq100_tickers():
    """Get NASDAQ100 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100#Components")
        # Find the right table
        for t in table:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                return t[col].tolist()
        return []
    except Exception as e:
        print(f"Failed to get NASDAQ100 tickers: {e}")
        return []

def get_hardcoded_tickers():
    """Fallback: key US stocks including INTC."""
    # Read from existing instruments files
    tickers = set()
    for fname in ["sp500.txt", "nasdaq100.txt"]:
        fpath = os.path.join(INSTRUMENTS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        tickers.add(parts[0].upper())
    print(f"Read {len(tickers)} tickers from existing instruments files")
    return list(tickers)


def fetch_one(ticker, start, end, retries=3):
    """Download one ticker from yfinance."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=False)
            if df.empty:
                return None
            # Standardize columns
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjclose",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "splits",
            })
            # Calculate factor (adjclose / close)
            if "adjclose" in df.columns and "close" in df.columns:
                df["factor"] = df["adjclose"] / df["close"]
            else:
                df["factor"] = 1.0
            # Calculate change
            df["change"] = df["close"].pct_change()
            # Clean date index
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df[["open", "high", "low", "close", "volume", "change", "factor"]]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  Failed {ticker}: {e}")
                return None


def main():
    os.makedirs(CSV_DIR, exist_ok=True)

    # Get tickers
    print("Getting ticker lists...")
    sp500 = get_sp500_tickers()
    if sp500 is None:
        sp500 = []
    nasdaq100 = get_nasdaq100_tickers()

    # Fallback to existing instruments files
    existing = get_hardcoded_tickers()

    all_tickers = sorted(set(sp500 + nasdaq100 + existing))
    # Make sure INTC is included
    if "INTC" not in all_tickers:
        all_tickers.append("INTC")

    print(f"Total tickers to fetch: {len(all_tickers)}")

    # Fetch data
    success = 0
    failed = 0
    instruments = {}  # ticker -> (start_date, end_date)
    all_dates = set()

    for i, ticker in enumerate(all_tickers):
        print(f"[{i+1}/{len(all_tickers)}] {ticker}...", end=" ", flush=True)
        df = fetch_one(ticker, START_DATE, END_DATE)
        if df is not None and len(df) > 30:
            # Save CSV
            csv_path = os.path.join(CSV_DIR, f"{ticker.lower()}.csv")
            df.to_csv(csv_path)
            success += 1
            # Record instruments range
            start_dt = df.index.min().strftime("%Y-%m-%d")
            end_dt = df.index.max().strftime("%Y-%m-%d")
            instruments[ticker.lower()] = (start_dt, end_dt)
            all_dates.update(df.index.strftime("%Y-%m-%d").tolist())
            print(f"OK ({len(df)} rows, {start_dt} ~ {end_dt})")
        else:
            failed += 1
            print("SKIP (no data or too short)")

        # Throttle to avoid rate limiting
        if (i + 1) % 50 == 0:
            print(f"  --- Progress: {success} ok, {failed} failed, sleeping 5s ---")
            time.sleep(5)

    print(f"\nDone fetching: {success} success, {failed} failed")

    # ── Dump to Qlib bin format ──
    print("\nDumping to Qlib bin format...")
    dump_script = os.path.join(os.path.dirname(__file__), "dump_bin.py")
    os.system(
        f"python {dump_script} dump_all "
        f"--csv_path {CSV_DIR} "
        f"--qlib_dir {QLIB_DIR} "
        f"--freq day "
        f"--exclude_fields date "
        f"--symbol_field_name symbol "
        f"--date_field_name date "
        f"--include_fields open,high,low,close,volume,change,factor "
    )

    # ── Write instruments files ──
    print("Writing instruments files...")
    os.makedirs(INSTRUMENTS_DIR, exist_ok=True)

    # sp500
    sp500_lower = set(t.lower() for t in sp500) if sp500 else set()
    nasdaq100_lower = set(t.lower() for t in nasdaq100) if nasdaq100 else set()

    with open(os.path.join(INSTRUMENTS_DIR, "sp500.txt"), "w") as f:
        for sym in sorted(instruments.keys()):
            if sym in sp500_lower or sym.upper() in (sp500 or []):
                s, e = instruments[sym]
                f.write(f"{sym.upper()}\t{s}\t{e}\n")

    with open(os.path.join(INSTRUMENTS_DIR, "nasdaq100.txt"), "w") as f:
        for sym in sorted(instruments.keys()):
            if sym in nasdaq100_lower or sym.upper() in (nasdaq100 or []):
                s, e = instruments[sym]
                f.write(f"{sym.upper()}\t{s}\t{e}\n")

    with open(os.path.join(INSTRUMENTS_DIR, "all.txt"), "w") as f:
        for sym in sorted(instruments.keys()):
            s, e = instruments[sym]
            f.write(f"{sym.upper()}\t{s}\t{e}\n")

    # ── Write calendar ──
    print("Writing calendar...")
    os.makedirs(CALENDAR_DIR, exist_ok=True)
    sorted_dates = sorted(all_dates)
    with open(os.path.join(CALENDAR_DIR, "day.txt"), "w") as f:
        for d in sorted_dates:
            f.write(d + "\n")

    print(f"Calendar: {sorted_dates[0]} ~ {sorted_dates[-1]} ({len(sorted_dates)} days)")
    print(f"Instruments: {len(instruments)} symbols")
    print("All done!")


if __name__ == "__main__":
    main()
