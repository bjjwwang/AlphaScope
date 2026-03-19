"""Generate daily board picks from cached predictions + stock prices.

Supports two board types:
- "cpu": Picks from CPU models (LGBModel, XGBModel, CatBoostModel) across all tickers
- "gpu": Picks from GPU/PyTorch models on top-100 stocks, with multi-model consensus
"""
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

CPU_MODEL_CLASSES = {"LGBModel", "XGBModel", "CatBoostModel"}
GPU_MODEL_CLASSES = {
    "GRU_ts", "LSTM_ts", "ALSTM_ts", "Transformer_ts", "LocalFormer_ts",
}


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
    """Compute AlphaScore (1-10) and signal from prediction signals."""
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
    """Find the start date of the most recent consecutive buy streak."""
    if not signals:
        return None
    latest = signals[-1]
    if latest.get("signal") != "buy":
        last_buy_end = None
        for i in range(len(signals) - 1, -1, -1):
            if signals[i].get("signal") == "buy":
                last_buy_end = i
                break
        if last_buy_end is None:
            return None
        start = last_buy_end
        for i in range(last_buy_end - 1, -1, -1):
            if signals[i].get("signal") == "buy":
                start = i
            else:
                break
        return signals[start].get("date")
    else:
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
    board_type: str = "cpu",
):
    """Generate today's top-N board picks and update old active picks.

    board_type:
      "cpu" — select from CPU models (LGBModel, XGBModel, CatBoostModel)
      "gpu" — select from GPU models with multi-model consensus scoring

    Returns dict with counts: {total, new_picks, updated}.
    """
    scan_db = scan_db_path or _SCAN_DB
    stock_db = stock_db_path or _STOCK_DB
    today = pick_date_str or date.today().isoformat()
    model_filter = CPU_MODEL_CLASSES if board_type == "cpu" else GPU_MODEL_CLASSES

    os.makedirs(os.path.dirname(scan_db), exist_ok=True)
    init_db(scan_db)

    if not os.path.exists(stock_db):
        print(f"  Warning: stock_data.db not found at {stock_db}")
        return {"total": 0, "new_picks": 0, "updated": 0}

    stock_conn = sqlite3.connect(stock_db)

    try:
        # === Part 1: Select today's top-N picks ===
        all_cached = list_cached_predictions(scan_db, trading_style=trading_style)

        # Group predictions by ticker, filtering for the right model set
        by_ticker = {}  # ticker → list of (model_class, pred_dict, cached_row)
        for cp in all_cached:
            if cp["model_class"] not in model_filter:
                continue
            try:
                pred = json.loads(cp["prediction_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            signals = pred.get("signals", [])
            if not signals:
                continue
            ticker = cp["ticker"]
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append((cp["model_class"], pred, cp))

        # Score each ticker: multi-model consensus + best confidence
        candidates = []
        for ticker, model_preds in by_ticker.items():
            # Count how many models say "buy" for this ticker
            buy_models = []
            best_score = -999
            best_model = None
            best_pred = None
            best_signals = None

            for mc, pred, cp in model_preds:
                signals = pred.get("signals", [])
                latest = signals[-1] if signals else {}
                latest_score = pred.get("latest_score", 0)
                if latest.get("signal") == "buy":
                    buy_models.append(mc)
                    if latest_score > best_score:
                        best_score = latest_score
                        best_model = mc
                        best_pred = pred
                        best_signals = signals

            if not buy_models:
                continue  # No model says buy

            n_models = len(model_preds)
            n_buy = len(buy_models)
            consensus = n_buy  # absolute count of models agreeing on buy

            alpha, _ = _alpha_score_from_signals(best_signals)
            rec_date = _find_latest_buy_start(best_signals)
            if not rec_date:
                continue

            # Composite ranking: consensus first, then raw score
            candidates.append({
                "ticker": ticker,
                "model_class": best_model,
                "alpha_score": alpha,
                "signal": "buy",
                "latest_score": best_score,
                "rec_date": rec_date,
                "signals": best_signals,
                "prediction_json": json.dumps(best_pred) if best_pred else "{}",
                "consensus": consensus,
                "n_models": n_models,
                "buy_models": ",".join(sorted(buy_models)),
            })

        # Sort: consensus DESC, then latest_score DESC
        candidates.sort(key=lambda c: (c["consensus"], c["latest_score"]), reverse=True)
        top_picks = candidates[:top_n]

        new_count = 0
        for pick in top_picks:
            price_info = _get_close_price(stock_conn, pick["ticker"], pick["rec_date"])
            if not price_info:
                continue
            rec_price = price_info[1]

            latest_info = _get_latest_price(stock_conn, pick["ticker"])
            if not latest_info:
                continue
            latest_date, latest_price = latest_info

            sell_date_str = _find_sell_after(pick["signals"], pick["rec_date"])
            sell_price = None
            sell_return = None
            if sell_date_str:
                sell_info = _get_close_price(stock_conn, pick["ticker"], sell_date_str)
                if sell_info:
                    sell_price = sell_info[1]
                    sell_return = round((sell_price - rec_price) / rec_price * 100, 2) if rec_price else None

            today_return = round((latest_price - rec_price) / rec_price * 100, 2) if rec_price else None

            # model_class shows best model; buy_models in prediction_json metadata
            save_board_pick(
                scan_db,
                pick_date=today,
                ticker=pick["ticker"],
                trading_style=trading_style,
                model_class=pick["buy_models"],  # show all agreeing models
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
                board_type=board_type,
                model_consensus=pick["consensus"],
            )
            new_count += 1

        # === Part 2: Update old active picks ===
        updated_count = 0
        active_picks = get_active_board_picks(scan_db, board_type=board_type)
        for ap in active_picks:
            if ap["pick_date"] == today:
                continue
            ticker = ap["ticker"]

            latest_info = _get_latest_price(stock_conn, ticker)
            if not latest_info:
                continue
            latest_date, latest_price = latest_info
            rec_price = ap["recommended_price"]
            today_return = round((latest_price - rec_price) / rec_price * 100, 2) if rec_price else None

            # Check for sell signal from any model in the right set
            sell_date_str = None
            sell_price = None
            sell_return = None
            for cp in all_cached:
                if cp["ticker"] == ticker and cp["model_class"] in model_filter:
                    try:
                        pred = json.loads(cp["prediction_json"])
                        signals = pred.get("signals", [])
                        sd = _find_sell_after(signals, ap["recommended_date"])
                        if sd:
                            sell_date_str = sd
                            sell_info = _get_close_price(stock_conn, ticker, sd)
                            if sell_info:
                                sell_price = sell_info[1]
                                sell_return = round((sell_price - rec_price) / rec_price * 100, 2) if rec_price else None
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue

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

    print(f"  Board ({board_type}): {new_count} new picks for {today}, "
          f"{updated_count} old picks updated")
    return {"total": new_count + updated_count, "new_picks": new_count, "updated": updated_count}


if __name__ == "__main__":
    result_cpu = generate_daily_picks(board_type="cpu")
    print(f"CPU board: {result_cpu}")
    result_gpu = generate_daily_picks(board_type="gpu")
    print(f"GPU board: {result_gpu}")
