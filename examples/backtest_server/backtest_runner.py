"""
Core backtest execution pipeline with progress callbacks.
"""
import asyncio
import uuid
import traceback
import threading
import time
import pandas as pd
import numpy as np
from typing import Callable, Any

from backtest_server.config_schema import BacktestRequest, BacktestResult
from backtest_server.config_converter import convert
from backtest_server.result_formatter import format_full_result
from backtest_server import data_pipeline


STEP_LABELS = [
    "准备数据",
    "加载特征",
    "训练模型",
    "模型预测",
    "回测交易",
    "生成报告",
]


class BacktestRunner:
    def __init__(
        self,
        config: BacktestRequest,
        emit: Callable[[dict], Any],
        pretrained_experiment: str | None = None,
    ):
        self.config = config
        self.emit = emit  # callable that receives SSE event dicts
        self.experiment_name = f"backtest_{uuid.uuid4().hex[:8]}"
        # If set, skip training and load model from this experiment
        self.pretrained_experiment = pretrained_experiment

    def _step(self, idx: int, state: str):
        self.emit({"event": "step", "data": {"step": idx, "state": state, "label": STEP_LABELS[idx]}})

    def _log(self, msg: str, level: str = "info"):
        self.emit({"event": "log", "data": {"msg": msg, "type": level}})

    def _progress(self, pct: int, label: str = ""):
        self.emit({"event": "progress", "data": {"pct": pct, "label": label}})

    def _error(self, step: int, msg: str):
        self.emit({"event": "error", "data": {"step": step, "msg": msg}})

    def _heartbeat(self, label: str = ""):
        """Send periodic keep-alive events during long-running operations."""
        stop = threading.Event()
        elapsed = [0]

        def _beat():
            while not stop.is_set():
                stop.wait(15)
                if not stop.is_set():
                    elapsed[0] += 15
                    mins, secs = divmod(elapsed[0], 60)
                    t = f"{mins}分{secs:02d}秒" if mins else f"{secs}秒"
                    self.emit({"event": "ping", "data": {"msg": f"{label} ({t})"}})

        t = threading.Thread(target=_beat, daemon=True)
        t.start()
        return stop

    def run_sync(self) -> BacktestResult:
        """Synchronous execution — called from thread."""
        try:
            return self._execute()
        except Exception as e:
            self._error(-1, f"未预期的错误: {str(e)}")
            raise

    def _execute(self) -> BacktestResult:
        # Safety check: if training_pool=all but training period < 1 year,
        # auto-downgrade to target_only to avoid empty dataset
        seg = self.config.dataset.segments
        from datetime import datetime
        try:
            t0 = datetime.strptime(seg.train[0], "%Y-%m-%d")
            t1 = datetime.strptime(seg.train[1], "%Y-%m-%d")
            train_days = (t1 - t0).days
            if self.config.training_pool in ("all", "sp500") and train_days < 365:
                self._log(f"训练期仅 {train_days} 天（< 1年），自动切换为仅目标股票训练，"
                          f"避免大量股票缺失数据导致空数据集", "warn")
                self.config.training_pool = "target_only"
        except (ValueError, IndexError):
            pass

        # Step 0: Prepare data
        self._step(0, "active")
        self._progress(5, STEP_LABELS[0])
        try:
            provider_uri = data_pipeline.resolve_provider_uri(self.config.data_source)
            self._log(f"数据源: {self.config.data_source}, 路径: {provider_uri}")

            # For db data source: bulk-dump stock_data.db → Qlib bin (cached)
            if self.config.data_source == "db":
                data_pipeline.dump_all_db_to_qlib(
                    provider_uri=provider_uri,
                    log_fn=lambda msg: self._log(msg),
                )

            for ticker in self.config.target:
                if not data_pipeline.check_ticker_exists(ticker, provider_uri):
                    if self.config.data_source == "db":
                        self._log(f"{ticker} 不在Qlib数据中，从stock_data.db导入...", "warn")
                        data_pipeline.dump_db_to_qlib(ticker, provider_uri)
                    else:
                        self._log(f"{ticker} 不在Qlib数据中，正在下载...", "warn")
                        data_pipeline.download_and_dump_yfinance(ticker, provider_uri)
                    self._log(f"{ticker} 数据就绪", "success")
                else:
                    self._log(f"{ticker} 数据已就绪")
        except Exception as e:
            self._step(0, "error")
            self._error(0, f"数据准备失败: {str(e)}")
            raise
        self._step(0, "done")
        self._log("数据准备完成", "success")

        # Step 1: Load features (init qlib + create model/dataset)
        self._step(1, "active")
        self._progress(15, STEP_LABELS[1])
        try:
            import qlib
            from qlib.constant import REG_US
            qlib.init(provider_uri=provider_uri, region=REG_US)
            self._log("Qlib 初始化完成")

            model_config, dataset_config = convert(self.config)
            from qlib.utils import init_instance_by_config
            model = init_instance_by_config(model_config)
            dataset = init_instance_by_config(dataset_config)
            self._log(f"模型: {model_config['class']}, 数据集: {dataset_config['class']}")
        except Exception as e:
            self._step(1, "error")
            self._error(1, f"特征加载失败: {str(e)}")
            raise
        self._step(1, "done")
        self._log("特征加载完成", "success")

        # Step 2: Train model (or load pretrained)
        self._step(2, "active")
        self._progress(30, STEP_LABELS[2])
        try:
            from qlib.workflow import R
            from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

            if self.pretrained_experiment:
                # Skip training — load pretrained model
                self._log(f"Loading pretrained model from {self.pretrained_experiment}...")
                recorders = R.list_recorders(experiment_name=self.pretrained_experiment)
                finished = [r for r in recorders.values()
                            if r.info.get("status", "").upper() == "FINISHED"]
                if not finished:
                    raise ValueError(f"No finished recorder in {self.pretrained_experiment}")
                model = finished[0].load_object("params.pkl")
                self._log("Pretrained model loaded", "success")

                # Still need a recorder for prediction
                with R.start(experiment_name=self.experiment_name):
                    R.save_objects(**{"params.pkl": model})
                    recorder = R.get_recorder()

                    self._step(2, "done")
                    self._step(3, "active")
                    self._progress(60, STEP_LABELS[3])
                    self._log("生成预测信号...")
                    sr = SignalRecord(model, dataset, recorder)
                    sr.generate()
                    pred_df = recorder.load_object("pred.pkl")
                    self._log(f"预测完成, {len(pred_df)} 条记录", "success")
                    scores = pred_df["score"].dropna()
                    if len(scores) > 0:
                        self._log(f"预测分数范围: [{scores.min():.4f}, {scores.max():.4f}], "
                                  f"均值: {scores.mean():.4f}, 中位数: {scores.median():.4f}")
            else:
                # Normal training path
                with R.start(experiment_name=self.experiment_name):
                    self._log("开始训练模型...")
                    hb = self._heartbeat("训练中")
                    try:
                        model.fit(dataset)
                    finally:
                        hb.set()
                    self._log("模型训练完成", "success")
                    R.save_objects(**{"params.pkl": model})
                    recorder = R.get_recorder()

                    # Step 3: Predict
                    self._step(2, "done")
                    self._step(3, "active")
                    self._progress(60, STEP_LABELS[3])
                    self._log("生成预测信号...")
                    sr = SignalRecord(model, dataset, recorder)
                    sr.generate()
                    pred_df = recorder.load_object("pred.pkl")
                    self._log(f"预测完成, {len(pred_df)} 条记录", "success")
                    # Log score distribution to help user set threshold
                    scores = pred_df["score"].dropna()
                    if len(scores) > 0:
                        self._log(f"预测分数范围: [{scores.min():.4f}, {scores.max():.4f}], "
                                  f"均值: {scores.mean():.4f}, 中位数: {scores.median():.4f}")
        except Exception as e:
            tb = traceback.format_exc()
            self._step(2, "error")
            self._error(2, f"训练/预测失败: {str(e)}")
            raise
        self._step(3, "done")

        # Step 4: Backtest trading
        self._step(4, "active")
        self._progress(75, STEP_LABELS[4])
        try:
            stock_id, pool_df = self._run_strategy(pred_df)
        except Exception as e:
            self._step(4, "error")
            self._error(4, f"回测执行失败: {str(e)}")
            raise
        self._step(4, "done")
        self._log("回测交易完成", "success")

        # Step 5: Generate report
        self._step(5, "active")
        self._progress(90, STEP_LABELS[5])
        try:
            result = format_full_result(pool_df, ticker=stock_id)
            self._log(f"报告生成完成: {result.summary.total_trades} 笔交易, "
                       f"模型收益 {result.summary.model_return}%", "success")
        except Exception as e:
            self._step(5, "error")
            self._error(5, f"报告生成失败: {str(e)}")
            raise
        self._step(5, "done")
        self._progress(100, "完成")

        self.emit({"event": "result", "data": result.model_dump()})
        return result

    def _run_strategy(self, pred_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """Execute strategy and return (stock_id, pool_df) for primary target."""
        from qlib.data import D

        target = self.config.target[0]
        seg = self.config.dataset.segments

        # Find target in predictions (case-insensitive matching)
        all_instruments = pred_df.index.get_level_values(1 if pred_df.index.nlevels > 1 else 0).unique()
        match = [s for s in all_instruments if s.upper() == target.upper()]
        if not match:
            match = [s for s in all_instruments if target.lower() in s.lower()]
        if not match:
            raise ValueError(f"{target} 不在预测结果中。可用: {list(all_instruments[:10])}")
        stock_id = match[0]

        test_start = seg.test[0]
        test_end = seg.test[1]
        pred_test = pred_df[pred_df.index.get_level_values(0) >= test_start]
        dates = pred_test.index.get_level_values(0).unique().sort_values()

        if self.config.strategy.strategy_class == "SignalThreshold":
            pool_df = self._threshold_strategy(pred_test, stock_id, dates)
        else:
            pool_df = self._topk_strategy(pred_df, pred_test, stock_id, dates)

        # Attach price data
        try:
            close_data = D.features(
                [stock_id], ["$close", "$factor"],
                start_time=str(dates.min().date()),
                end_time=str(dates.max().date()),
                freq="day",
            )
            close_data = close_data.droplevel("instrument")
            close_data.columns = ["close_adj", "factor"]
            if close_data["factor"].isna().all() or (close_data["factor"] == 0).all():
                close_data["factor"] = 1.0
            close_data["close"] = close_data["close_adj"] / close_data["factor"]
            pool_df = pool_df.join(close_data, how="left")
        except Exception:
            pool_df["close_adj"] = np.nan
            pool_df["close"] = np.nan
            pool_df["factor"] = 1.0

        self._log(f"策略执行完成: {stock_id}, {len(pool_df)} 个交易日")
        return stock_id, pool_df

    def _threshold_strategy(self, pred_test, stock_id, dates):
        threshold = self.config.strategy.buy_threshold
        if threshold is None:
            threshold = 0.0

        # When threshold=0, use automatic threshold:
        # - Multi-stock: buy when stock ranks in top 50% cross-sectionally
        # - Single-stock: buy when score > time-series median of the stock's own scores
        use_auto = (threshold == 0.0)

        # Pre-collect all scores for the target stock (for single-stock median)
        if use_auto:
            stock_scores = []
            for date in dates:
                day_pred = pred_test.loc[date]
                if stock_id in day_pred.index:
                    s = float(day_pred.loc[stock_id, "score"])
                    if not np.isnan(s):
                        stock_scores.append(s)
            ts_median = float(np.median(stock_scores)) if stock_scores else 0.0

        records = []
        for date in dates:
            day_pred = pred_test.loc[date]
            if stock_id in day_pred.index:
                score = float(day_pred.loc[stock_id, "score"])
                if use_auto:
                    all_day_scores = day_pred["score"].dropna()
                    if len(all_day_scores) > 1:
                        # Multi-stock: cross-sectional ranking (top 50%)
                        day_median = float(all_day_scores.median())
                        in_pool = score > day_median
                    else:
                        # Single-stock: time-series ranking (above own median)
                        in_pool = score > ts_median
                else:
                    in_pool = score > threshold
                records.append({"date": date, "in_pool": in_pool, "score": score})
            else:
                records.append({"date": date, "in_pool": False, "score": np.nan})

        pool_df = pd.DataFrame(records).set_index("date")

        buy_days = pool_df["in_pool"].sum()
        total_days = len(pool_df)
        if use_auto:
            all_day_scores = pred_test.loc[dates[0]]["score"].dropna() if len(dates) > 0 else []
            if len(all_day_scores) > 1:
                self._log(f"信号阈值策略(自动): 按每日截面排名，排名前50%时买入 "
                          f"→ {buy_days}/{total_days} 天持仓")
            else:
                self._log(f"信号阈值策略(自动): 单股票模式，预测分数高于中位数({ts_median:.4f})时买入 "
                          f"→ {buy_days}/{total_days} 天持仓")
        else:
            self._log(f"信号阈值策略: threshold={threshold:.4f} "
                      f"→ {buy_days}/{total_days} 天持仓")
        return pool_df

    def _topk_strategy(self, pred_all, pred_test, stock_id, dates):
        topk = self.config.strategy.topk or 50
        n_drop = self.config.strategy.n_drop or 5
        current_holdings = set()
        all_dates = pred_all.index.get_level_values(0).unique().sort_values()
        records = []

        for date in all_dates:
            day_pred = pred_all.loc[date].sort_values("score", ascending=False)
            all_stocks = day_pred.index.tolist()

            if len(current_holdings) == 0:
                new_holdings = set(all_stocks[:topk])
            else:
                held_in_today = [s for s in all_stocks if s in current_holdings]
                not_held = [s for s in all_stocks if s not in current_holdings]
                candidates = held_in_today + not_held[:n_drop + topk - len(held_in_today)]
                from qlib.data import D as _  # ensure import
                candidates_scores = day_pred.reindex(candidates).sort_values("score", ascending=False)
                new_holdings = set(candidates_scores.index.tolist()[:topk])

            current_holdings = new_holdings

            if date >= dates.min():
                if stock_id in day_pred.index:
                    score = day_pred.loc[stock_id, "score"]
                    rank = day_pred.index.tolist().index(stock_id) + 1
                else:
                    score, rank = np.nan, np.nan
                records.append({
                    "date": date,
                    "in_pool": stock_id in current_holdings,
                    "score": score,
                    "rank": rank,
                })

        return pd.DataFrame(records).set_index("date")
