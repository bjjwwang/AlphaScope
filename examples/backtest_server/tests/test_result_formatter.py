"""Tests for result_formatter.py — Phase 2"""
import pytest
import pandas as pd
import numpy as np
from backtest_server.result_formatter import (
    format_trades, format_segments, compute_summary,
    compute_holding_stats, compute_empty_stats, compute_timeline,
    format_full_result,
)


def _make_pool_df(in_pool_pattern, prices=None, n=None):
    """Build a synthetic pool_df."""
    if n is None:
        n = len(in_pool_pattern)
    dates = pd.bdate_range("2026-01-01", periods=n)
    if prices is None:
        prices = np.linspace(10, 12, n)
    return pd.DataFrame({
        "in_pool": in_pool_pattern[:n],
        "score": np.random.randn(n),
        "close_adj": prices,
        "close": prices,
        "factor": [1.0] * n,
    }, index=dates)


# ===== format_trades =====

def test_trades_basic():
    # 5 in, 5 out, 5 in, 5 out
    pattern = [True]*5 + [False]*5 + [True]*5 + [False]*5
    df = _make_pool_df(pattern)
    trades = format_trades(df)
    assert len(trades) == 2
    assert all(not t.is_open for t in trades)
    assert trades[0].buy_date == "2026-01-01"
    assert trades[0].days > 0


def test_trades_open_position():
    # Ends while holding
    pattern = [False]*5 + [True]*5
    df = _make_pool_df(pattern)
    trades = format_trades(df)
    assert len(trades) == 1
    assert trades[0].is_open is True


def test_trades_never_in_pool():
    pattern = [False] * 10
    df = _make_pool_df(pattern)
    trades = format_trades(df)
    assert len(trades) == 0


def test_trades_always_in_pool():
    pattern = [True] * 10
    df = _make_pool_df(pattern)
    trades = format_trades(df)
    assert len(trades) == 1
    assert trades[0].is_open is True


def test_trades_nan_prices():
    pattern = [True]*5 + [False]*5
    prices = [np.nan]*5 + [np.nan]*5
    df = _make_pool_df(pattern, prices=prices)
    trades = format_trades(df)
    assert len(trades) == 1
    assert trades[0].return_pct is None


# ===== format_segments =====

def test_segments_alternating():
    pattern = [True]*5 + [False]*5 + [True]*5
    df = _make_pool_df(pattern, n=15)
    segs = format_segments(df)
    # 3 segments (hold, empty, hold), but some may be zero-day and skipped
    types = [s.type for s in segs]
    assert "hold" in types
    assert "empty" in types
    assert all(s.days > 0 for s in segs)


def test_segments_single_day():
    # Single-day segments may have 0 calendar days → should be skipped
    pattern = [True, False, True, False, True]
    df = _make_pool_df(pattern, n=5)
    segs = format_segments(df)
    # All segments are 1 bday apart = could be 0 or more calendar days
    # The key: no segment with days=0 should appear
    assert all(s.days > 0 for s in segs)


# ===== compute_summary =====

def test_summary_profitable():
    pattern = [True]*10 + [False]*10
    prices = list(np.linspace(10, 15, 10)) + list(np.linspace(15, 14, 10))
    df = _make_pool_df(pattern, prices=prices)
    trades = format_trades(df)
    segs = format_segments(df)
    summary = compute_summary(df, trades, segs)
    assert summary.total_trades == 1
    assert summary.model_return > 0
    assert summary.win_rate == 100.0


def test_summary_losing():
    pattern = [True]*10 + [False]*10
    prices = list(np.linspace(15, 10, 10)) + list(np.linspace(10, 11, 10))
    df = _make_pool_df(pattern, prices=prices)
    trades = format_trades(df)
    segs = format_segments(df)
    summary = compute_summary(df, trades, segs)
    assert summary.model_return < 0
    assert summary.win_rate == 0.0


def test_summary_no_trades():
    pattern = [False] * 20
    df = _make_pool_df(pattern, n=20)
    trades = format_trades(df)
    segs = format_segments(df)
    summary = compute_summary(df, trades, segs)
    assert summary.total_trades == 0
    assert summary.model_return == 0.0
    assert summary.win_rate == 0.0


def test_model_return_compounding():
    """Three trades: +10%, -5%, +20% → compound = (1.1*0.95*1.2 - 1)*100 = 25.4%"""
    from backtest_server.config_schema import TradeRecord, SegmentRecord
    # We test compute_summary with pre-built trades
    dates = pd.bdate_range("2026-01-01", periods=30)
    df = pd.DataFrame({
        "in_pool": [False]*30,
        "close_adj": np.linspace(100, 120, 30),
        "close": np.linspace(100, 120, 30),
    }, index=dates)
    trades = [
        TradeRecord(buy_date="2026-01-01", sell_date="2026-01-10", days=9, return_pct=10.0),
        TradeRecord(buy_date="2026-01-13", sell_date="2026-01-20", days=7, return_pct=-5.0),
        TradeRecord(buy_date="2026-01-21", sell_date="2026-01-31", days=10, return_pct=20.0),
    ]
    summary = compute_summary(df, trades, [])
    expected = (1.1 * 0.95 * 1.2 - 1) * 100
    assert abs(summary.model_return - expected) < 0.1


# ===== stats =====

def test_holding_stats():
    from backtest_server.config_schema import SegmentRecord
    segs = [
        SegmentRecord(type="hold", start="2026-01-01", end="2026-01-10", days=9, change_pct=5.0),
        SegmentRecord(type="empty", start="2026-01-13", end="2026-01-20", days=7, change_pct=-3.0),
        SegmentRecord(type="hold", start="2026-01-21", end="2026-01-31", days=10, change_pct=-2.0),
    ]
    stats = compute_holding_stats(segs)
    assert stats.count == 2
    assert stats.total_days == 19
    assert stats.profit_segs == 1
    assert stats.loss_segs == 1
    assert stats.avg_return == 1.5  # (5 + -2) / 2


def test_empty_stats():
    from backtest_server.config_schema import SegmentRecord
    segs = [
        SegmentRecord(type="hold", start="a", end="b", days=5, change_pct=10.0),
        SegmentRecord(type="empty", start="c", end="d", days=7, change_pct=15.0),
        SegmentRecord(type="empty", start="e", end="f", days=3, change_pct=-8.0),
    ]
    stats = compute_empty_stats(segs)
    assert stats.count == 2
    assert stats.total_days == 10
    assert stats.missed_up == 15.0
    assert stats.avoided_down == -8.0


def test_timeline_proportional():
    from backtest_server.config_schema import SegmentRecord
    segs = [
        SegmentRecord(type="hold", start="a", end="b", days=10, change_pct=0),
        SegmentRecord(type="empty", start="c", end="d", days=20, change_pct=0),
        SegmentRecord(type="hold", start="e", end="f", days=30, change_pct=0),
    ]
    tl = compute_timeline(segs)
    assert len(tl) == 3
    assert abs(tl[0].pct - 16.7) < 0.1  # 10/60*100
    assert abs(tl[1].pct - 33.3) < 0.1  # 20/60*100
    assert abs(tl[2].pct - 50.0) < 0.1  # 30/60*100
