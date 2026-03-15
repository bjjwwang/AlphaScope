"""
比亚迪（SZ002594）回测 —— 2022年

重新训练模型，测试期覆盖2022年
训练: 2008~2019  验证: 2020  测试: 2021~2022
"""

import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data import D

# ============================================================
# 1. 初始化
# ============================================================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)
print("Qlib 初始化完成")

# ============================================================
# 2. 模型 & 数据集配置（时间窗口后移）
# ============================================================
model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2026-03-06",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2020-12-31",
                "instruments": "csi300_bs",
            },
        },
        "segments": {
            "train": ("2008-01-01", "2020-12-31"),
            "valid": ("2021-01-01", "2021-12-31"),
            "test": ("2022-01-01", "2026-03-04"),
        },
    },
}

# ============================================================
# 3. 训练 & 预测
# ============================================================
print("初始化模型和数据集...")
model = init_instance_by_config(model_config)
dataset = init_instance_by_config(dataset_config)

with R.start(experiment_name="byd_backtest_2022"):
    print("[1/3] 训练 LightGBM...")
    model.fit(dataset)
    R.save_objects(**{"params.pkl": model})

    recorder = R.get_recorder()

    print("[2/3] 生成预测信号...")
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    print("[3/3] 信号分析...")
    sar = SigAnaRecord(recorder)
    sar.generate()

    # 加载预测结果
    pred_df = recorder.load_object("pred.pkl")

print(f"\n预测数据: {pred_df.index.get_level_values(0).min().date()} ~ {pred_df.index.get_level_values(0).max().date()}")

# ============================================================
# 4. 比亚迪择时分析
# ============================================================
STOCK = "SZ002594"
STOCK_NAME = "比亚迪"
TOPK = 50
N_DROP = 5

# 取2022年开始的全部数据
pred_2022 = pred_df[pred_df.index.get_level_values(0) >= "2022-01-01"]
dates = pred_2022.index.get_level_values(0).unique().sort_values()

if STOCK not in pred_2022.index.get_level_values(1):
    print(f"警告: {STOCK} 不在预测数据中！")
    import sys; sys.exit(1)

print(f"\n2022年分析范围: {dates.min().date()} ~ {dates.max().date()}")

# 模拟选股
current_holdings = set()
daily_records = []

# 先用2021的数据预热持仓状态
pred_all_test = pred_df
all_dates = pred_all_test.index.get_level_values(0).unique().sort_values()

for date in all_dates:
    day_pred = pred_all_test.loc[date].sort_values("score", ascending=False)
    all_stocks = day_pred.index.tolist()

    if len(current_holdings) == 0:
        new_holdings = set(all_stocks[:TOPK])
    else:
        held_in_today = [s for s in all_stocks if s in current_holdings]
        not_held = [s for s in all_stocks if s not in current_holdings]
        candidates = held_in_today + not_held[:N_DROP + TOPK - len(held_in_today)]
        candidates_scores = day_pred.reindex(candidates).sort_values("score", ascending=False)
        sorted_candidates = candidates_scores.index.tolist()
        new_holdings = set(sorted_candidates[:TOPK])

    current_holdings = new_holdings

    # 只记录2022年的
    if date >= pd.Timestamp("2022-01-01"):
        if STOCK in day_pred.index:
            stock_score = day_pred.loc[STOCK, "score"]
            stock_rank = day_pred.index.tolist().index(STOCK) + 1
        else:
            stock_score = np.nan
            stock_rank = np.nan

        daily_records.append({
            "date": date,
            "in_pool": STOCK in current_holdings,
            "score": stock_score,
            "rank": stock_rank,
        })

pool_df = pd.DataFrame(daily_records).set_index("date")

# 获取真实价格
close_data = D.features(
    [STOCK], ["$close", "$factor"],
    start_time=str(dates.min().date()),
    end_time=str(dates.max().date()),
    freq="day",
)
close_data = close_data.droplevel("instrument")
close_data.columns = ["close_adj", "factor"]
close_data["close"] = close_data["close_adj"] / close_data["factor"]
pool_df = pool_df.join(close_data, how="left")

# ============================================================
# 5. 交易明细
# ============================================================
trades = []
entry_date = None

for i in range(len(pool_df)):
    row = pool_df.iloc[i]
    prev_in = pool_df.iloc[i - 1]["in_pool"] if i > 0 else False

    if row["in_pool"] and not prev_in:
        entry_date = pool_df.index[i]
        entry_price_real = row["close"]
        entry_price_adj = row["close_adj"]
        entry_rank = int(row["rank"]) if pd.notna(row["rank"]) else -1

    elif not row["in_pool"] and prev_in and entry_date is not None:
        exit_date = pool_df.index[i]
        exit_price_real = row["close"]
        exit_price_adj = row["close_adj"]
        hold_days = (exit_date - entry_date).days
        if pd.notna(entry_price_adj) and pd.notna(exit_price_adj) and entry_price_adj != 0:
            ret = (exit_price_adj - entry_price_adj) / entry_price_adj * 100
        else:
            ret = np.nan
        trades.append({
            "买入日期": entry_date.strftime("%Y-%m-%d"),
            "买入价": round(entry_price_real, 2),
            "排名": entry_rank,
            "卖出日期": exit_date.strftime("%Y-%m-%d"),
            "卖出价": round(exit_price_real, 2),
            "天数": hold_days,
            "收益率(%)": round(ret, 2) if pd.notna(ret) else np.nan,
        })
        entry_date = None

if entry_date is not None:
    last_row = pool_df.iloc[-1]
    exit_date = pool_df.index[-1]
    hold_days = (exit_date - entry_date).days
    if pd.notna(entry_price_adj) and pd.notna(last_row["close_adj"]) and entry_price_adj != 0:
        ret = (last_row["close_adj"] - entry_price_adj) / entry_price_adj * 100
    else:
        ret = np.nan
    trades.append({
        "买入日期": entry_date.strftime("%Y-%m-%d"),
        "买入价": round(entry_price_real, 2),
        "排名": entry_rank,
        "卖出日期": exit_date.strftime("%Y-%m-%d") + "(未平仓)",
        "卖出价": round(last_row["close"], 2),
        "天数": hold_days,
        "收益率(%)": round(ret, 2) if pd.notna(ret) else np.nan,
    })

trades_df = pd.DataFrame(trades)

# ============================================================
# 6. 所有区间
# ============================================================
segments = []
seg_start = 0
seg_in_pool = pool_df.iloc[0]["in_pool"]

for i in range(1, len(pool_df)):
    if pool_df.iloc[i]["in_pool"] != seg_in_pool:
        segments.append({"start_idx": seg_start, "end_idx": i, "in_pool": seg_in_pool})
        seg_start = i
        seg_in_pool = pool_df.iloc[i]["in_pool"]
segments.append({"start_idx": seg_start, "end_idx": len(pool_df) - 1, "in_pool": seg_in_pool})

all_segments = []
for seg in segments:
    s = pool_df.iloc[seg["start_idx"]]
    e = pool_df.iloc[seg["end_idx"]]
    start_date = pool_df.index[seg["start_idx"]]
    end_date = pool_df.index[seg["end_idx"]]
    hold_days = (end_date - start_date).days
    if hold_days == 0:
        continue
    if pd.notna(s["close_adj"]) and pd.notna(e["close_adj"]) and s["close_adj"] != 0:
        ret_adj = (e["close_adj"] - s["close_adj"]) / s["close_adj"] * 100
    else:
        ret_adj = np.nan
    all_segments.append({
        "类型": "持仓" if seg["in_pool"] else "空仓",
        "开始": start_date.strftime("%Y-%m-%d"),
        "结束": end_date.strftime("%Y-%m-%d"),
        "起始价": round(s["close"], 2) if pd.notna(s["close"]) else np.nan,
        "结束价": round(e["close"], 2) if pd.notna(e["close"]) else np.nan,
        "天数": hold_days,
        "涨跌(%)": round(ret_adj, 2) if pd.notna(ret_adj) else np.nan,
    })

seg_df = pd.DataFrame(all_segments)
held_seg = seg_df[seg_df["类型"] == "持仓"]
missed_seg = seg_df[seg_df["类型"] == "空仓"]

# ============================================================
# 7. 打印结果
# ============================================================
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 10)

print("\n" + "=" * 90)
print(f"{STOCK_NAME}（{STOCK}）2022年 交易明细")
print("策略：LightGBM + Alpha158 → TopkDropout（Top50, 每天换5只）")
print("=" * 90)
print(f"\n共 {len(trades_df)} 笔交易：\n")
if len(trades_df) > 0:
    print(trades_df.to_string(index=False))
    valid_returns = trades_df["收益率(%)"].dropna()
    wins = valid_returns[valid_returns > 0]
    losses = valid_returns[valid_returns <= 0]
    print(f"\n胜率: {len(wins)}/{len(valid_returns)} = {len(wins)/len(valid_returns)*100:.1f}%")
    print(f"平均收益: {valid_returns.mean():.2f}%  |  最大盈利: {valid_returns.max():.2f}%  |  最大亏损: {valid_returns.min():.2f}%")
    print(f"平均持仓: {trades_df['天数'].mean():.1f}天  |  最长: {trades_df['天数'].max()}天  |  最短: {trades_df['天数'].min()}天")

print("\n" + "=" * 90)
print(f"完整时间线：持仓 vs 空仓")
print("=" * 90)
print(seg_df.to_string(index=False))

print("\n" + "=" * 90)
print("分组统计")
print("=" * 90)

print("\n【持仓期间】")
if len(held_seg) > 0:
    h_valid = held_seg["涨跌(%)"].dropna()
    print(f"  {len(held_seg)} 段，共 {held_seg['天数'].sum()} 天")
    print(f"  上涨 {(h_valid > 0).sum()} 段 / 下跌 {(h_valid <= 0).sum()} 段")
    print(f"  平均涨跌: {h_valid.mean():.2f}%")

print("\n【空仓期间】")
if len(missed_seg) > 0:
    m_valid = missed_seg["涨跌(%)"].dropna()
    print(f"  {len(missed_seg)} 段，共 {missed_seg['天数'].sum()} 天")
    print(f"  期间上涨 {(m_valid > 0).sum()} 段 / 下跌 {(m_valid <= 0).sum()} 段")
    print(f"  平均涨跌: {m_valid.mean():.2f}%")
    missed_up = m_valid[m_valid > 0]
    missed_down = m_valid[m_valid <= 0]
    if len(missed_up) > 0:
        print(f"  错过的上涨: {missed_up.sum():.2f}%（{len(missed_up)}段）")
    if len(missed_down) > 0:
        print(f"  躲过的下跌: {missed_down.sum():.2f}%（{len(missed_down)}段）")

# 买入持有 vs 模型择时
print("\n" + "=" * 90)
print("对比：买入持有 vs 模型择时")
print("=" * 90)

first_adj = pool_df.iloc[0]["close_adj"]
last_adj = pool_df.iloc[-1]["close_adj"]
first_real = pool_df.iloc[0]["close"]
last_real = pool_df.iloc[-1]["close"]
bh_return = (last_adj - first_adj) / first_adj * 100

model_return = 1.0
for _, row in trades_df.iterrows():
    r = row["收益率(%)"]
    if pd.notna(r):
        model_return *= (1 + r / 100)
model_return = (model_return - 1) * 100

total_days = (pool_df.index[-1] - pool_df.index[0]).days
held_days = held_seg["天数"].sum() if len(held_seg) > 0 else 0

print(f"\n  回测区间: {pool_df.index[0].date()} ~ {pool_df.index[-1].date()} ({total_days}天)")
print(f"  起始价: {first_real:.2f}  →  终止价: {last_real:.2f}")
print(f"")
print(f"  【买入持有{STOCK_NAME}】")
print(f"    累计收益:  {bh_return:.2f}%")
print(f"")
print(f"  【模型择时】")
print(f"    累计收益:  {model_return:.2f}%")
print(f"    持仓天数:  {held_days}天 / {total_days}天（利用率 {held_days/total_days*100:.1f}%）")
print(f"")
diff = model_return - bh_return
if diff > 0:
    print(f"  → 模型择时胜出，多赚 {diff:.2f}%")
    print(f"    而且只用了 {held_days/total_days*100:.1f}% 的时间持仓")
else:
    print(f"  → 买入持有胜出，多赚 {-diff:.2f}%")
