"""Generate daily board picks from cached predictions + stock prices."""
import json
import os
import sqlite3
from datetime import date, datetime, timezone

from backtest_server.scan_db import (
    init_db, list_cached_predictions, get_active_board_picks,
    save_board_pick, update_board_pick,
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_SCAN_DB = os.path.join(_DATA_DIR, "scan_jobs.db")
_STOCK_DB = os.path.join(_DATA_DIR, "stock_data.db")


def _get_close_price(conn: sqlite3.Connection, ticker: str, target_date: str):
    """Get close price on or before target_date. Returns (date_str, price) or None."""
    row = conn.execute(
        """SELECT date(timestamp) as d, close FROM kline_daily_1y
           WHERE ticker = ? AND date(timestamp) <= ?
           ORDER BY timestamp DESC LIMIT 1""",
        (ticker, target_date),
    ).fetchone()
    return (row[0], row[1]) if row else None


def _get_latest_price(conn: sqlite3.Connection, ticker: str):
    """Get most recent close price. Returns (date_str, price) or None."""
    row = conn.execute(
        """SELECT date(timestamp) as d, close FROM kline_daily_1y
           WHERE ticker = ?
           ORDER BY timestamp DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    return (row[0], row[1]) if row else None


def _alpha_score_from_signals(signals: list[dict]) -> tuple[int, str]:
    """Compute AlphaScore (1-10) and signal from prediction signals.
    Returns (score, signal_label)."""
    if not signals:
        return 5, "hold"
    buy_count = sum(1 for s in signals if s.get("signal") == "buy")
    total = len(signals)
    buy_ratio = buy_count / total if total else 0.5
    raw = buy_ratio * 10
    score = max(1, min(10, round(raw)))
    if score >= 7:
        return score, "buy"
    elif score >= 4:
        return score, "hold"
    else:
        return score, "sell"


def _find_latest_buy_start(signals: list[dict]) -> str | None:
    """Find the start date of the most recent consecutive buy streak.
    This is the 'recommendation date' — when the model started saying buy."""
    if not signals:
        return None
    # Walk backwards to find where current buy streak started
    latest = signals[-1]
    if latest.get("signal") != "buy":
        # Latest signal isn't buy — find the last buy→sell transition
        # and use that buy streak's start as recommendation
        last_buy_end = None
        for i in range(len(signals) - 1, -1, -1):
            if signals[i].get("signal") == "buy":
                last_buy_end = i
                break
        if last_buy_end is None:
            return None
        # Walk back from last_buy_end to find streak start
        start = last_buy_end
        for i in range(last_buy_end - 1, -1, -1):
            if signals[i].get("signal") == "buy":
                start = i
            else:
                break
        return signals[start].get("date")
    else:
        # Latest is buy — walk back to find streak start
        start = len(signals) - 1
        for i in range(len(signals) - 2, -1, -1):
            if signals[i].get("signal") == "buy":
                start = i
            else:
                break
        return signals[start].get("date")


def _find_sell_after(signals: list[dict], after_date: str) -> str | None:
    """Find the first sell signal after a given date."""
    found_start = False
    for s in signals:
        if s.get("date") == after_date:
            found_start = True
            continue
        if found_start and s.get("signal") == "sell":
            return s.get("date")
    return None


def generate_daily_picks(
    scan_db_path: str = None,
    stock_db_path: str = None,
    pick_date_str: str = None,
    top_n: int = 10,
    trading_style: str = "swing",
):
    """Generate today's top-N board picks and update old active picks.

    Returns dict with counts: {total, new_picks, updated}.
    """
    scan_db = scan_db_path or _SCAN_DB
    stock_db = stock_db_path or _STOCK_DB
    today = pick_date_str or date.today().isoformat()

    os.makedirs(os.path.dirname(scan_db), exist_ok=True)
    init_db(scan_db)

    # Connect to stock_data.db for price lookups
    if not os.path.exists(stock_db):
        print(f"  Warning: stock_data.db not found at {stock_db}")
        return {"total": 0, "new_picks": 0, "updated": 0}

    stock_conn = sqlite3.connect(stock_db)

    try:
        # === Part 1: Select today's top-N picks ===
        # Strategy: pick tickers whose latest signal is "buy" and have the
        # highest raw prediction score (latest_score). This measures model
        # confidence — higher score = model is more bullish.
        all_cached = list_cached_predictions(scan_db, trading_style=trading_style)
        candidates = []

        for cp in all_cached:
            try:
                pred = json.loads(cp["prediction_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            signals = pred.get("signals", [])
            if not signals:
                continue

            # Only consider tickers where latest signal is "buy"
            latest = signals[-1]
            if latest.get("signal") != "buy":
                continue

            alpha, _ = _alpha_score_from_signals(signals)
            latest_score = pred.get("latest_score", 0)
            rec_date = _find_latest_buy_start(signals)
            if not rec_date:
                continue

            candidates.append({
                "ticker": cp["ticker"],
                "model_class": cp["model_class"],
                "alpha_score": alpha,
                "signal": "buy",
                "latest_score": latest_score,
                "rec_date": rec_date,
                "signals": signals,
                "prediction_json": cp["prediction_json"],
            })

        # Sort by raw model score (highest confidence first)
        candidates.sort(key=lambda c: c["latest_score"], reverse=True)
        top_picks = candidates[:top_n]

        new_count = 0
        for pick in top_picks:
            # Get recommended price (close on rec_date)
            price_info = _get_close_price(stock_conn, pick["ticker"], pick["rec_date"])
            if not price_info:
                continue
            rec_price = price_info[1]

            # Get latest price
            latest_info = _get_latest_price(stock_conn, pick["ticker"])
            if not latest_info:
                continue
            latest_date, latest_price = latest_info

            # Check if signal has flipped to sell after rec_date
            sell_date_str = _find_sell_after(pick["signals"], pick["rec_date"])
            sell_price = None
            sell_return = None
            if sell_date_str:
                sell_info = _get_close_price(stock_conn, pick["ticker"], sell_date_str)
                if sell_info:
                    sell_price = sell_info[1]
                    sell_return = round((sell_price - rec_price) / rec_price * 100, 2) if rec_price else None

            today_return = round((latest_price - rec_price) / rec_price * 100, 2) if rec_price else None

            save_board_pick(
                scan_db,
                pick_date=today,
                ticker=pick["ticker"],
                trading_style=trading_style,
                model_class=pick["model_class"],
                alpha_score=pick["alpha_score"],
                signal=pick["signal"],
                recommended_date=pick["rec_date"],
                recommended_price=round(rec_price, 2),
                sell_date=sell_date_str,
                sell_price=round(sell_price, 2) if sell_price else None,
                sell_return_pct=sell_return,
                latest_price=round(latest_price, 2),
                latest_price_date=latest_date,
                today_return_pct=today_return,
                prediction_json=pick["prediction_json"],
            )
            new_count += 1

        # === Part 2: Update old active picks ===
        updated_count = 0
        active_picks = get_active_board_picks(scan_db)
        for ap in active_picks:
            if ap["pick_date"] == today:
                continue  # Skip today's picks (just created)
            ticker = ap["ticker"]

            # Refresh latest price
            latest_info = _get_latest_price(stock_conn, ticker)
            if not latest_info:
                continue
            latest_date, latest_price = latest_info
            rec_price = ap["recommended_price"]
            today_return = round((latest_price - rec_price) / rec_price * 100, 2) if rec_price else None

            # Check if we now have a cached prediction with a sell signal
            cp = None
            for c in all_cached:
                if c["ticker"] == ticker:
                    cp = c
                    break
            sell_date_str = None
            sell_price = None
            sell_return = None
            if cp:
                try:
                    pred = json.loads(cp["prediction_json"])
                    signals = pred.get("signals", [])
                    sell_date_str = _find_sell_after(signals, ap["pick_date"])
                    if sell_date_str:
                        sell_info = _get_close_price(stock_conn, ticker, sell_date_str)
                        if sell_info:
                            sell_price = sell_info[1]
                            sell_return = round((sell_price - rec_price) / rec_price * 100, 2) if rec_price else None
                except (json.JSONDecodeError, TypeError):
                    pass

            update_board_pick(
                scan_db, ap["id"],
                latest_price=round(latest_price, 2),
                latest_price_date=latest_date,
                today_return_pct=today_return,
                sell_date=sell_date_str,
                sell_price=round(sell_price, 2) if sell_price else None,
                sell_return_pct=sell_return,
            )
            updated_count += 1

    finally:
        stock_conn.close()

    print(f"  Board: {new_count} new picks for {today}, {updated_count} old picks updated")
    return {"total": new_count + updated_count, "new_picks": new_count, "updated": updated_count}


if __name__ == "__main__":
    result = generate_daily_picks()
    print(f"Done: {result}")
