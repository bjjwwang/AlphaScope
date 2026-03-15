"""
Convert pandas DataFrames from backtest into frontend JSON structures.
Logic extracted from intc_backtest.py lines 181-365.
"""
import pandas as pd
import numpy as np
from backtest_server.config_schema import (
    TradeRecord, SegmentRecord, SummaryStats, HoldingStats,
    EmptyStats, TimelineSegment, BacktestResult,
)


def format_trades(pool_df: pd.DataFrame, ticker: str = "") -> list[TradeRecord]:
    """Extract trades from pool_df with in_pool column."""
    trades = []
    entry_date = None
    entry_price = None
    entry_price_adj = None

    for i in range(len(pool_df)):
        row = pool_df.iloc[i]
        prev_in = pool_df.iloc[i - 1]["in_pool"] if i > 0 else False

        if row["in_pool"] and not prev_in:
            entry_date = pool_df.index[i]
            entry_price = row.get("close", np.nan)
            entry_price_adj = row.get("close_adj", np.nan)

        elif not row["in_pool"] and prev_in and entry_date is not None:
            exit_date = pool_df.index[i]
            exit_price = row.get("close", np.nan)
            exit_price_adj = row.get("close_adj", np.nan)
            hold_days = (exit_date - entry_date).days
            ret = _calc_return(entry_price_adj, exit_price_adj)
            trades.append(TradeRecord(
                ticker=ticker,
                buy_date=entry_date.strftime("%Y-%m-%d"),
                buy_price=_round_or_none(entry_price),
                sell_date=exit_date.strftime("%Y-%m-%d"),
                sell_price=_round_or_none(exit_price),
                days=hold_days,
                return_pct=_round_or_none(ret),
                is_open=False,
            ))
            entry_date = None

    # Open position at end
    if entry_date is not None:
        last_row = pool_df.iloc[-1]
        exit_date = pool_df.index[-1]
        hold_days = (exit_date - entry_date).days
        exit_price_adj = last_row.get("close_adj", np.nan)
        ret = _calc_return(entry_price_adj, exit_price_adj)
        trades.append(TradeRecord(
            ticker=ticker,
            buy_date=entry_date.strftime("%Y-%m-%d"),
            buy_price=_round_or_none(entry_price),
            sell_date=exit_date.strftime("%Y-%m-%d"),
            sell_price=_round_or_none(last_row.get("close", np.nan)),
            days=hold_days,
            return_pct=_round_or_none(ret),
            is_open=True,
        ))

    return trades


def format_segments(pool_df: pd.DataFrame, ticker: str = "") -> list[SegmentRecord]:
    """Extract all hold/empty segments."""
    if len(pool_df) < 2:
        return []

    segments = []
    seg_start = 0
    seg_in_pool = pool_df.iloc[0]["in_pool"]

    for i in range(1, len(pool_df)):
        if pool_df.iloc[i]["in_pool"] != seg_in_pool:
            segments.append({"start_idx": seg_start, "end_idx": i, "in_pool": seg_in_pool})
            seg_start = i
            seg_in_pool = pool_df.iloc[i]["in_pool"]
    segments.append({"start_idx": seg_start, "end_idx": len(pool_df) - 1, "in_pool": seg_in_pool})

    result = []
    for seg in segments:
        s = pool_df.iloc[seg["start_idx"]]
        e = pool_df.iloc[seg["end_idx"]]
        start_date = pool_df.index[seg["start_idx"]]
        end_date = pool_df.index[seg["end_idx"]]
        hold_days = (end_date - start_date).days
        if hold_days == 0:
            continue
        ret = _calc_return(s.get("close_adj", np.nan), e.get("close_adj", np.nan))
        result.append(SegmentRecord(
            ticker=ticker,
            type="hold" if seg["in_pool"] else "empty",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            start_price=_round_or_none(s.get("close", np.nan)),
            end_price=_round_or_none(e.get("close", np.nan)),
            days=hold_days,
            change_pct=_round_or_none(ret),
        ))

    return result


def compute_summary(pool_df: pd.DataFrame, trades: list[TradeRecord],
                    segments: list[SegmentRecord]) -> SummaryStats:
    """Compute summary statistics."""
    # Buy-and-hold return
    first_adj = pool_df.iloc[0].get("close_adj", np.nan)
    last_adj = pool_df.iloc[-1].get("close_adj", np.nan)
    bh_return = _calc_return(first_adj, last_adj) or 0.0

    # Model return (compounding)
    model_return = 1.0
    valid_returns = [t.return_pct for t in trades if t.return_pct is not None]
    for r in valid_returns:
        model_return *= (1 + r / 100)
    model_return = (model_return - 1) * 100

    # Win rate
    wins = sum(1 for r in valid_returns if r > 0)
    win_rate = (wins / len(valid_returns) * 100) if valid_returns else 0.0

    # Days
    total_days = (pool_df.index[-1] - pool_df.index[0]).days
    held_segs = [s for s in segments if s.type == "hold"]
    held_days = sum(s.days for s in held_segs)

    return SummaryStats(
        model_return=round(model_return, 2),
        bh_return=round(bh_return, 2),
        win_rate=round(win_rate, 1),
        total_trades=len(trades),
        test_start=pool_df.index[0].strftime("%Y-%m-%d"),
        test_end=pool_df.index[-1].strftime("%Y-%m-%d"),
        start_price=_round_or_none(pool_df.iloc[0].get("close", np.nan)),
        end_price=_round_or_none(pool_df.iloc[-1].get("close", np.nan)),
        held_days=held_days,
        total_days=total_days,
    )


def compute_holding_stats(segments: list[SegmentRecord]) -> HoldingStats | None:
    """Holding period statistics."""
    held = [s for s in segments if s.type == "hold"]
    if not held:
        return None
    valid_changes = [s.change_pct for s in held if s.change_pct is not None]
    return HoldingStats(
        count=len(held),
        total_days=sum(s.days for s in held),
        profit_segs=sum(1 for c in valid_changes if c > 0),
        loss_segs=sum(1 for c in valid_changes if c <= 0),
        avg_return=round(np.mean(valid_changes), 2) if valid_changes else 0.0,
    )


def compute_empty_stats(segments: list[SegmentRecord]) -> EmptyStats | None:
    """Empty period statistics."""
    empty = [s for s in segments if s.type == "empty"]
    if not empty:
        return None
    valid_changes = [s.change_pct for s in empty if s.change_pct is not None]
    ups = [c for c in valid_changes if c > 0]
    downs = [c for c in valid_changes if c <= 0]
    return EmptyStats(
        count=len(empty),
        total_days=sum(s.days for s in empty),
        missed_up=round(sum(ups), 2) if ups else 0.0,
        avoided_down=round(sum(downs), 2) if downs else 0.0,
    )


def compute_timeline(segments: list[SegmentRecord]) -> list[TimelineSegment]:
    """Compute proportional timeline."""
    total_days = sum(s.days for s in segments)
    if total_days == 0:
        return []
    return [
        TimelineSegment(
            type=s.type,
            pct=round(s.days / total_days * 100, 1),
        )
        for s in segments
    ]


def format_full_result(pool_df: pd.DataFrame, ticker: str = "") -> BacktestResult:
    """Full pipeline: pool_df → BacktestResult."""
    trades = format_trades(pool_df, ticker=ticker)
    segments = format_segments(pool_df, ticker=ticker)
    summary = compute_summary(pool_df, trades, segments)
    return BacktestResult(
        summary=summary,
        trades=trades,
        segments=segments,
        holding_stats=compute_holding_stats(segments),
        empty_stats=compute_empty_stats(segments),
        timeline=compute_timeline(segments),
    )


def _calc_return(start_adj, end_adj):
    if pd.notna(start_adj) and pd.notna(end_adj) and start_adj != 0:
        return (end_adj - start_adj) / start_adj * 100
    return None


def _round_or_none(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(val, decimals)
