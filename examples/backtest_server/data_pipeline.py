"""
Data source management: resolve provider URIs, check/download ticker data.
"""
import os
import sqlite3
import subprocess
import tempfile
import pandas as pd


class DataDownloadError(Exception):
    pass


class InsufficientDataError(Exception):
    pass


QLIB_US_DATA = os.path.expanduser("~/.qlib/qlib_data/us_data")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "stock_data.db")


def resolve_provider_uri(data_source: str) -> str:
    """Return the Qlib provider URI for a data source."""
    if data_source in ("yfinance", "db"):
        return QLIB_US_DATA
    elif data_source == "baostock":
        return os.path.expanduser("~/.qlib/qlib_data/cn_data")
    return QLIB_US_DATA


def check_ticker_exists(ticker: str, provider_uri: str = None) -> bool:
    """Check if a ticker exists in Qlib instruments."""
    uri = provider_uri or QLIB_US_DATA
    instruments_file = os.path.join(uri, "instruments", "all.txt")
    if not os.path.exists(instruments_file):
        return False
    with open(instruments_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0].upper() == ticker.upper():
                return True
    return False


def download_and_dump_yfinance(ticker: str, provider_uri: str = None):
    """Download a ticker via yfinance and dump to Qlib bin format."""
    try:
        import yfinance as yf
    except ImportError:
        raise DataDownloadError("yfinance is not installed")

    uri = provider_uri or QLIB_US_DATA
    t = yf.Ticker(ticker)
    hist = t.history(period="max", auto_adjust=False)

    if hist is None or hist.empty:
        raise DataDownloadError(f"No data found for {ticker}")

    # Prepare CSV
    csv_dir = tempfile.mkdtemp(prefix="qlib_csv_")
    csv_path = os.path.join(csv_dir, f"{ticker}.csv")

    df = hist.copy()
    df.index.name = "date"
    df = df.reset_index()
    # Handle MultiIndex columns from newer yfinance
    if hasattr(df.columns, 'get_level_values') and df.columns.nlevels > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    df["symbol"] = ticker
    if "adj_close" in df.columns:
        df["factor"] = df["adj_close"] / df["close"]
    elif "adjclose" in df.columns:
        df["factor"] = df["adjclose"] / df["close"]
    else:
        df["factor"] = 1.0
    df["change"] = df["close"].pct_change()
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "change", "factor"]]
    df.to_csv(csv_path, index=False)

    # Find dump_bin.py
    dump_bin = _find_dump_bin()
    if not dump_bin:
        raise DataDownloadError("dump_bin.py not found")

    # Run dump_bin
    cmd = [
        "python", dump_bin, "dump_all",
        "--csv_path" if _dump_bin_uses_csv_path(dump_bin) else "--data_path", csv_dir,
        "--qlib_dir", uri,
        "--freq", "day",
        "--exclude_fields", "date,symbol",
        "--symbol_field_name", "symbol",
        "--date_field_name", "date",
        "--include_fields", "open,high,low,close,volume,change,factor",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise DataDownloadError(f"dump_bin failed: {result.stderr[:500]}")

    # Ensure ticker is in instruments file
    ensure_ticker_in_instruments(ticker, uri)


def ensure_ticker_in_instruments(ticker: str, provider_uri: str = None):
    """Ensure ticker appears in instruments/all.txt."""
    uri = provider_uri or QLIB_US_DATA
    instruments_file = os.path.join(uri, "instruments", "all.txt")
    if check_ticker_exists(ticker, uri):
        return
    # Append
    os.makedirs(os.path.dirname(instruments_file), exist_ok=True)
    with open(instruments_file, "a") as f:
        f.write(f"{ticker}\t2000-01-01\t2099-12-31\n")


def load_from_db(ticker: str, interval: str = "daily_1y", db_path: str = None) -> pd.DataFrame:
    """Load kline data from stock_data.db."""
    db = db_path or DB_PATH
    if not os.path.exists(db):
        raise DataDownloadError(f"Database not found: {db}")

    conn = sqlite3.connect(db)
    query = f"""
        SELECT ticker, timestamp, open, high, low, close, volume
        FROM kline_{interval} WHERE ticker = ? ORDER BY timestamp ASC
    """
    try:
        df = pd.read_sql_query(query, conn, params=(ticker,))
    finally:
        conn.close()

    if df.empty:
        raise DataDownloadError(f"No data for {ticker} in kline_{interval}")
    return df


def dump_db_to_qlib(ticker: str, provider_uri: str = None, db_path: str = None):
    """Read daily data from stock_data.db and dump to Qlib bin format."""
    db = db_path or DB_PATH
    uri = provider_uri or QLIB_US_DATA

    if not os.path.exists(db):
        raise DataDownloadError(f"Database not found: {db}")

    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query("""
            SELECT ticker, timestamp, open, high, low, close, volume, adj_close
            FROM kline_daily_1y WHERE ticker = ? ORDER BY timestamp ASC
        """, conn, params=(ticker,))
    finally:
        conn.close()

    if df.empty:
        raise DataDownloadError(f"No daily data for {ticker} in stock_data.db")

    # Prepare CSV for dump_bin
    df = df.rename(columns={"ticker": "symbol", "timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if df["adj_close"].notna().any() and (df["adj_close"] != 0).any():
        df["factor"] = df["adj_close"] / df["close"]
    else:
        df["factor"] = 1.0
    df["change"] = df["close"].pct_change()
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "change", "factor"]]

    csv_dir = tempfile.mkdtemp(prefix="qlib_db_")
    csv_path = os.path.join(csv_dir, f"{ticker}.csv")
    df.to_csv(csv_path, index=False)

    dump_bin = _find_dump_bin()
    if not dump_bin:
        raise DataDownloadError("dump_bin.py not found")

    cmd = [
        "python", dump_bin, "dump_all",
        "--csv_path" if _dump_bin_uses_csv_path(dump_bin) else "--data_path", csv_dir,
        "--qlib_dir", uri,
        "--freq", "day",
        "--exclude_fields", "date,symbol",
        "--symbol_field_name", "symbol",
        "--date_field_name", "date",
        "--include_fields", "open,high,low,close,volume,change,factor",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise DataDownloadError(f"dump_bin failed: {result.stderr[:500]}")

    ensure_ticker_in_instruments(ticker, uri)


def dump_all_db_to_qlib(provider_uri: str = None, db_path: str = None, log_fn=None):
    """Bulk-dump ALL daily data from stock_data.db to Qlib bin format.

    This rebuilds the Qlib bin data directory from stock_data.db, replacing
    instruments/all.txt and all feature bins.  Uses a marker file to avoid
    re-running if the db hasn't been updated since the last dump.
    """
    db = db_path or DB_PATH
    uri = provider_uri or QLIB_US_DATA
    if not os.path.exists(db):
        raise DataDownloadError(f"Database not found: {db}")

    # Check marker — skip if already dumped today (avoid re-dumping while
    # cold_start is still writing to the db, which changes mtime constantly)
    import datetime as _dt
    marker = os.path.join(uri, ".db_dump_marker")
    if os.path.exists(marker):
        with open(marker) as f:
            try:
                marker_date = f.read().strip()
                today = _dt.date.today().isoformat()
                if marker_date == today:
                    if log_fn:
                        log_fn("stock_data.db 今日已导入，跳过重新导入")
                    return
            except ValueError:
                pass

    if log_fn:
        log_fn("正在从 stock_data.db 批量导入到 Qlib 格式...")

    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query("""
            SELECT ticker, timestamp, open, high, low, close, volume, adj_close
            FROM kline_daily_1y ORDER BY ticker, timestamp
        """, conn)
    finally:
        conn.close()

    if df.empty:
        raise DataDownloadError("stock_data.db kline_daily_1y is empty")

    if log_fn:
        tickers = df["ticker"].nunique()
        log_fn(f"读取到 {len(df)} 行数据, {tickers} 只股票")

    # Prepare CSV directory with one file per ticker
    csv_dir = tempfile.mkdtemp(prefix="qlib_db_bulk_")
    df = df.rename(columns={"ticker": "symbol", "timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if df["adj_close"].notna().any() and (df["adj_close"] != 0).any():
        df["factor"] = df["adj_close"] / df["close"]
    else:
        df["factor"] = 1.0
    df["change"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "change", "factor"]]
    df.dropna(subset=["close"], inplace=True)

    # Write one CSV per ticker (dump_bin expects this structure)
    for symbol, grp in df.groupby("symbol"):
        grp.to_csv(os.path.join(csv_dir, f"{symbol}.csv"), index=False)

    if log_fn:
        log_fn("CSV 文件已准备，开始 dump_bin...")

    dump_bin = _find_dump_bin()
    if not dump_bin:
        raise DataDownloadError("dump_bin.py not found")

    cmd = [
        "python", dump_bin, "dump_all",
        "--csv_path" if _dump_bin_uses_csv_path(dump_bin) else "--data_path", csv_dir,
        "--qlib_dir", uri,
        "--freq", "day",
        "--exclude_fields", "date,symbol",
        "--symbol_field_name", "symbol",
        "--date_field_name", "date",
        "--include_fields", "open,high,low,close,volume,change,factor",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise DataDownloadError(f"dump_bin failed: {result.stderr[:500]}")

    # Update instruments/all.txt from db tickers
    instruments_file = os.path.join(uri, "instruments", "all.txt")
    os.makedirs(os.path.dirname(instruments_file), exist_ok=True)
    ticker_ranges = df.groupby("symbol")["date"].agg(["min", "max"])
    with open(instruments_file, "w") as f:
        for sym, row in ticker_ranges.iterrows():
            f.write(f"{sym}\t{row['min']}\t{row['max']}\n")

    if log_fn:
        log_fn(f"Qlib 数据已从 stock_data.db 导入完成 ({ticker_ranges.shape[0]} 只股票)")

    # Write marker (today's date — re-dump once per day at most)
    with open(marker, "w") as f:
        f.write(_dt.date.today().isoformat())

    # Cleanup temp CSV
    import shutil
    shutil.rmtree(csv_dir, ignore_errors=True)


def validate_data_coverage(ticker: str, train_start: str, test_end: str,
                           provider_uri: str = None):
    """Validate that Qlib data covers the requested time range."""
    uri = provider_uri or QLIB_US_DATA
    instruments_file = os.path.join(uri, "instruments", "all.txt")
    if not os.path.exists(instruments_file):
        raise InsufficientDataError(f"No instruments file found at {uri}")

    with open(instruments_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0].upper() == ticker.upper():
                if len(parts) >= 3:
                    data_start = parts[1]
                    data_end = parts[2]
                    if data_start > train_start:
                        raise InsufficientDataError(
                            f"{ticker} data starts at {data_start}, but train starts at {train_start}")
                return
    raise InsufficientDataError(f"{ticker} not found in instruments")


def _find_dump_bin():
    """Find dump_bin.py in the project."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "dump_bin.py"),
        os.path.expanduser("~/.qlib/qlib_data/scripts/dump_bin.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def _dump_bin_uses_csv_path(dump_bin_path: str) -> bool:
    """Check if dump_bin uses --csv_path or --data_path."""
    try:
        with open(dump_bin_path, "r") as f:
            content = f.read()
        return "--csv_path" in content
    except Exception:
        return False
