"""Tests for data_pipeline.py — Phase 3 (all mocked)"""
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from backtest_server.data_pipeline import (
    resolve_provider_uri, check_ticker_exists, ensure_ticker_in_instruments,
    load_from_db, validate_data_coverage,
    download_and_dump_yfinance, DataDownloadError, InsufficientDataError,
)


def test_resolve_provider_uri_yfinance():
    uri = resolve_provider_uri("yfinance")
    assert "us_data" in uri


def test_resolve_provider_uri_db():
    uri = resolve_provider_uri("db")
    assert "us_data" in uri


def test_resolve_provider_uri_baostock():
    uri = resolve_provider_uri("baostock")
    assert "cn_data" in uri


def test_check_ticker_exists(tmp_path):
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text("INTC\t2000-01-01\t2026-03-06\nAAPL\t2000-01-01\t2026-03-06\n")
    assert check_ticker_exists("INTC", str(tmp_path)) is True
    assert check_ticker_exists("intc", str(tmp_path)) is True  # case insensitive


def test_check_ticker_not_exists(tmp_path):
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text("INTC\t2000-01-01\t2026-03-06\n")
    assert check_ticker_exists("ASST", str(tmp_path)) is False


def test_download_yfinance_failure():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("backtest_server.data_pipeline.yf", create=True) as mock_yf:
        # We need to mock the import inside the function
        import backtest_server.data_pipeline as dp
        original_func = dp.download_and_dump_yfinance

        # Simpler: just test that empty data raises
        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import sys
            mock_yf_mod = sys.modules["yfinance"]
            mock_yf_mod.Ticker.return_value = mock_ticker
            with pytest.raises(DataDownloadError, match="No data found"):
                download_and_dump_yfinance("BADTICKER")


def test_load_from_db_success(tmp_path):
    """Test loading from a real (tiny) SQLite DB."""
    db_path = str(tmp_path / "test.db")
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE kline_daily_1y (
        ticker TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER
    )""")
    conn.execute("""INSERT INTO kline_daily_1y VALUES
        ('INTC','2026-01-01',50.0,52.0,49.0,51.0,1000000)""")
    conn.commit()
    conn.close()

    df = load_from_db("INTC", "daily_1y", db_path)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 51.0


def test_load_from_db_failure(tmp_path):
    db_path = str(tmp_path / "test.db")
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE kline_daily_1y (
        ticker TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER
    )""")
    conn.commit()
    conn.close()

    with pytest.raises(DataDownloadError, match="No data for"):
        load_from_db("NOBODY", "daily_1y", db_path)


def test_validate_coverage_ok(tmp_path):
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text("INTC\t2000-01-01\t2026-03-06\n")
    # Should not raise
    validate_data_coverage("INTC", "2020-01-01", "2026-03-06", str(tmp_path))


def test_validate_coverage_fail(tmp_path):
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text("INTC\t2020-01-01\t2026-03-06\n")
    with pytest.raises(InsufficientDataError, match="data starts at"):
        validate_data_coverage("INTC", "2015-01-01", "2026-03-06", str(tmp_path))


def test_ensure_ticker_idempotent(tmp_path):
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text("INTC\t2000-01-01\t2026-03-06\n")
    ensure_ticker_in_instruments("INTC", str(tmp_path))
    content = (instruments / "all.txt").read_text()
    # Should still have exactly one INTC line
    assert content.count("INTC") == 1
