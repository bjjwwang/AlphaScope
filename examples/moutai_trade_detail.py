"""
贵州茅台（SH600519）交易明细分析

从上一次回测的预测信号中，还原出：
- 茅台每次被选入/移出 Top50 的时间
- 每次持仓区间、买入价、卖出价、收益率
- 汇总统计：胜率、平均收益、最大盈亏等
"""

import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R

# ============================================================
# 1. 初始化 & 加载数据
# ============================================================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 加载上次实验的预测结果
exp = R.get_exp(experiment_name="moutai_backtest")
recorder = list(exp.list_recorders().values())[-1]
pred_df = recorder.load_object("pred.pkl")

print("预测数据加载完成")
print(f"时间范围: {pred_df.index.get_level_values(0).min().date()} ~ "
      f"{pred_df.index.get_level_values(0).max().date()}")
print(f"股票数量: {pred_df.index.get_level_values(1).nunique()}")

# ============================================================
# 2. 模拟 TopkDropout 选股逻辑，还原茅台的买卖信号
# ============================================================
TOPK = 50
N_DROP = 5
STOCK = "SH600519"

dates = pred_df.index.get_level_values(0).unique().sort_values()

# 逐日模拟选股
current_holdings = set()
moutai_in_pool = []  # 每天茅台是否在持仓中

for date in dates:
    day_pred = pred_df.loc[date].sort_values("score", ascending=False)
    all_stocks = day_pred.index.tolist()

    if len(current_holdings) == 0:
        # 第一天，直接选 top K
        new_holdings = set(all_stocks[:TOPK])
    else:
        # 已有持仓的排序
        held_in_today = [s for s in all_stocks if s in current_holdings]
        not_held = [s for s in all_stocks if s not in current_holdings]

        # 合并：已持有 + 候选新股（取 n_drop 个）
        candidates = held_in_today + not_held[:N_DROP + TOPK - len(held_in_today)]
        # 按分数重新排序
        candidates_scores = day_pred.reindex(candidates).sort_values("score", ascending=False)
        sorted_candidates = candidates_scores.index.tolist()

        # 保留 top K
        keep = set(sorted_candidates[:TOPK])
        # 卖出的：原持仓中不在 keep 里的
        sell = current_holdings - keep
        # 买入的：keep 中不在原持仓里的
        buy = keep - current_holdings
        new_holdings = keep

    current_holdings = new_holdings
    moutai_in_pool.append({
        "date": date,
        "in_pool": STOCK in current_holdings,
        "score": day_pred.loc[STOCK, "score"] if STOCK in day_pred.index else np.nan,
        "rank": (day_pred.index.tolist().index(STOCK) + 1) if STOCK in day_pred.index else np.nan,
        "total": len(day_pred),
    })

pool_df = pd.DataFrame(moutai_in_pool)
pool_df.set_index("date", inplace=True)

# ============================================================
# 3. 获取茅台的真实收盘价（前复权价 / 复权因子 = 真实价格）
# ============================================================
close_data = D.features(
    ["SH600519"],
    ["$close", "$factor"],
    start_time=str(dates.min().date()),
    end_time=str(dates.max().date()),
    freq="day",
)
close_data = close_data.droplevel("instrument")
close_data.columns = ["close_adj", "factor"]
close_data["close"] = close_data["close_adj"] / close_data["factor"]

pool_df = pool_df.join(close_data, how="left")

# ============================================================
# 4. 提取每段持仓区间（买入→卖出）
# ============================================================
trades = []
entry_date = None

for i in range(len(pool_df)):
    row = pool_df.iloc[i]
    prev_in = pool_df.iloc[i - 1]["in_pool"] if i > 0 else False

    if row["in_pool"] and not prev_in:
        # 买入信号
        entry_date = pool_df.index[i]
        entry_price_real = row["close"]       # 真实价格（用于展示）
        entry_price_adj = row["close_adj"]    # 前复权价（用于算收益率，包含分红）
        entry_rank = int(row["rank"])

    elif not row["in_pool"] and prev_in and entry_date is not None:
        # 卖出信号（上一天还持有，今天不持有了）
        exit_date = pool_df.index[i]
        exit_price_real = row["close"]
        exit_price_adj = row["close_adj"]
        hold_days = (exit_date - entry_date).days
        # 用前复权价算收益率（包含分红送转）
        ret = (exit_price_adj - entry_price_adj) / entry_price_adj * 100

        trades.append({
            "买入日期": entry_date.strftime("%Y-%m-%d"),
            "买入价(元)": round(entry_price_real, 2),
            "买入时排名": entry_rank,
            "卖出日期": exit_date.strftime("%Y-%m-%d"),
            "卖出价(元)": round(exit_price_real, 2),
            "持仓天数": hold_days,
            "收益率(%)": round(ret, 2),
        })
        entry_date = None

# 如果最后一段还在持仓中
if entry_date is not None:
    last_row = pool_df.iloc[-1]
    exit_date = pool_df.index[-1]
    exit_price_real = last_row["close"]
    exit_price_adj = last_row["close_adj"]
    hold_days = (exit_date - entry_date).days
    ret = (exit_price_adj - entry_price_adj) / entry_price_adj * 100
    trades.append({
        "买入日期": entry_date.strftime("%Y-%m-%d"),
        "买入价(元)": round(entry_price_real, 2),
        "买入时排名": entry_rank,
        "卖出日期": exit_date.strftime("%Y-%m-%d") + " (未平仓)",
        "卖出价(元)": round(exit_price_real, 2),
        "持仓天数": hold_days,
        "收益率(%)": round(ret, 2),
    })

trades_df = pd.DataFrame(trades)

# ============================================================
# 5. 打印结果
# ============================================================
print("\n" + "=" * 80)
print("贵州茅台（SH600519）交易明细")
print("策略：LightGBM + Alpha158 因子 → TopkDropout（Top50, 每天换5只）")
print("回测区间：2017-01-01 ~ 2020-08-01")
print("=" * 80)

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)

print(f"\n共发生 {len(trades_df)} 笔交易：\n")
print(trades_df.to_string(index=False))

# ============================================================
# 6. 汇总统计
# ============================================================
if len(trades_df) > 0:
    returns = trades_df["收益率(%)"]
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    print("\n" + "=" * 80)
    print("交易统计汇总")
    print("=" * 80)
    print(f"总交易次数:    {len(trades_df)}")
    print(f"盈利次数:      {len(wins)}")
    print(f"亏损次数:      {len(losses)}")
    print(f"胜率:          {len(wins)/len(trades_df)*100:.1f}%")
    print(f"")
    print(f"平均收益率:    {returns.mean():.2f}%")
    print(f"最大单笔盈利:  {returns.max():.2f}%")
    print(f"最大单笔亏损:  {returns.min():.2f}%")
    print(f"平均持仓天数:  {trades_df['持仓天数'].mean():.1f} 天")
    print(f"最长持仓:      {trades_df['持仓天数'].max()} 天")
    print(f"最短持仓:      {trades_df['持仓天数'].min()} 天")
    print(f"")
    total_return = 1.0
    for r in returns:
        total_return *= (1 + r / 100)
    total_return = (total_return - 1) * 100
    print(f"累计收益率（仅持仓期间）: {total_return:.2f}%")

    # 标注几笔典型交易
    print("\n" + "=" * 80)
    print("典型交易举例")
    print("=" * 80)
    if len(wins) > 0:
        best_idx = returns.idxmax()
        best = trades_df.loc[best_idx]
        print(f"\n【最佳交易】")
        print(f"  {best['买入日期']} 以 {best['买入价(元)']}元 买入（排名第{best['买入时排名']}）")
        print(f"  {best['卖出日期']} 以 {best['卖出价(元)']}元 卖出")
        print(f"  持仓 {best['持仓天数']} 天，收益 {best['收益率(%)']}%")

    if len(losses) > 0:
        worst_idx = returns.idxmin()
        worst = trades_df.loc[worst_idx]
        print(f"\n【最差交易】")
        print(f"  {worst['买入日期']} 以 {worst['买入价(元)']}元 买入（排名第{worst['买入时排名']}）")
        print(f"  {worst['卖出日期']} 以 {worst['卖出价(元)']}元 卖出")
        print(f"  持仓 {worst['持仓天数']} 天，亏损 {worst['收益率(%)']}%")
