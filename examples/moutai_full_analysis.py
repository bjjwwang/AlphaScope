"""
茅台回测完整分析：不只看赢的，也看错过了什么

对比三个视角：
1. 模型选中茅台的区间（持仓期收益）
2. 模型没选中茅台的区间（错过的收益）
3. 买入持有茅台 vs 跟模型操作
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

exp = R.get_exp(experiment_name="moutai_backtest")
recorder = list(exp.list_recorders().values())[-1]
pred_df = recorder.load_object("pred.pkl")

TOPK = 50
N_DROP = 5
STOCK = "SH600519"

dates = pred_df.index.get_level_values(0).unique().sort_values()

# ============================================================
# 2. 模拟选股逻辑
# ============================================================
current_holdings = set()
daily_records = []

for date in dates:
    day_pred = pred_df.loc[date].sort_values("score", ascending=False)
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
    stock_score = day_pred.loc[STOCK, "score"] if STOCK in day_pred.index else np.nan
    stock_rank = (day_pred.index.tolist().index(STOCK) + 1) if STOCK in day_pred.index else np.nan

    daily_records.append({
        "date": date,
        "in_pool": STOCK in current_holdings,
        "score": stock_score,
        "rank": stock_rank,
    })

pool_df = pd.DataFrame(daily_records).set_index("date")

# ============================================================
# 3. 获取真实价格
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
# 4. 提取所有区间（持仓 + 未持仓）
# ============================================================
segments = []
seg_start = 0
seg_in_pool = pool_df.iloc[0]["in_pool"]

for i in range(1, len(pool_df)):
    if pool_df.iloc[i]["in_pool"] != seg_in_pool:
        segments.append({
            "start_idx": seg_start,
            "end_idx": i,
            "in_pool": seg_in_pool,
        })
        seg_start = i
        seg_in_pool = pool_df.iloc[i]["in_pool"]

# 最后一段
segments.append({
    "start_idx": seg_start,
    "end_idx": len(pool_df) - 1,
    "in_pool": seg_in_pool,
})

all_segments = []
for seg in segments:
    s = pool_df.iloc[seg["start_idx"]]
    e = pool_df.iloc[seg["end_idx"]]
    start_date = pool_df.index[seg["start_idx"]]
    end_date = pool_df.index[seg["end_idx"]]
    hold_days = (end_date - start_date).days
    if hold_days == 0:
        continue
    ret_adj = (e["close_adj"] - s["close_adj"]) / s["close_adj"] * 100
    all_segments.append({
        "类型": "持仓" if seg["in_pool"] else "空仓",
        "开始": start_date.strftime("%Y-%m-%d"),
        "结束": end_date.strftime("%Y-%m-%d"),
        "起始价": round(s["close"], 2),
        "结束价": round(e["close"], 2),
        "天数": hold_days,
        "区间涨跌(%)": round(ret_adj, 2),
    })

seg_df = pd.DataFrame(all_segments)

# ============================================================
# 5. 打印完整区间
# ============================================================
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 10)

print("=" * 90)
print(f"茅台（{STOCK}）完整时间线：持仓 vs 空仓 所有区间")
print("=" * 90)
print(seg_df.to_string(index=False))

# ============================================================
# 6. 分组统计
# ============================================================
held = seg_df[seg_df["类型"] == "持仓"]
missed = seg_df[seg_df["类型"] == "空仓"]

print("\n" + "=" * 90)
print("分组统计")
print("=" * 90)

print("\n【持仓区间】（模型选中茅台的时段）")
if len(held) > 0:
    print(f"  次数:          {len(held)}")
    print(f"  盈利次数:      {(held['区间涨跌(%)'] > 0).sum()}")
    print(f"  亏损次数:      {(held['区间涨跌(%)'] <= 0).sum()}")
    print(f"  胜率:          {(held['区间涨跌(%)'] > 0).sum()/len(held)*100:.1f}%")
    print(f"  平均收益:      {held['区间涨跌(%)'].mean():.2f}%")
    print(f"  总天数:        {held['天数'].sum()} 天")

print("\n【空仓区间】（模型没选中茅台的时段，错过的涨跌）")
if len(missed) > 0:
    print(f"  次数:          {len(missed)}")
    print(f"  期间上涨次数:  {(missed['区间涨跌(%)'] > 0).sum()}")
    print(f"  期间下跌次数:  {(missed['区间涨跌(%)'] <= 0).sum()}")
    print(f"  平均涨跌:      {missed['区间涨跌(%)'].mean():.2f}%")
    print(f"  总天数:        {missed['天数'].sum()} 天")
    missed_up = missed[missed["区间涨跌(%)"] > 0]
    if len(missed_up) > 0:
        print(f"  错过的上涨合计: {missed_up['区间涨跌(%)'].sum():.2f}% （共{len(missed_up)}段）")

# ============================================================
# 7. 买入持有 vs 模型操作
# ============================================================
print("\n" + "=" * 90)
print("对比：买入持有 vs 模型择时")
print("=" * 90)

first_close_adj = pool_df.iloc[0]["close_adj"]
last_close_adj = pool_df.iloc[-1]["close_adj"]
first_close = pool_df.iloc[0]["close"]
last_close = pool_df.iloc[-1]["close"]
bh_return = (last_close_adj - first_close_adj) / first_close_adj * 100

# 模型择时的累计收益（只在持仓期间有收益）
model_return = 1.0
for _, row in held.iterrows():
    model_return *= (1 + row["区间涨跌(%)"] / 100)
model_return = (model_return - 1) * 100

# 模型实际持仓天数
total_days = (pool_df.index[-1] - pool_df.index[0]).days
held_days = held["天数"].sum()
idle_days = total_days - held_days

print(f"  回测区间:      {pool_df.index[0].strftime('%Y-%m-%d')} ~ {pool_df.index[-1].strftime('%Y-%m-%d')} ({total_days}天)")
print(f"  起始价:        {first_close:.2f} 元")
print(f"  终止价:        {last_close:.2f} 元")
print(f"")
print(f"  【买入持有茅台】")
print(f"    累计收益:    {bh_return:.2f}%")
print(f"    持仓天数:    {total_days} 天（一直满仓）")
print(f"    年化收益:    {((1+bh_return/100)**(365/total_days)-1)*100:.2f}%")
print(f"")
print(f"  【模型择时操作茅台】")
print(f"    累计收益:    {model_return:.2f}%")
print(f"    持仓天数:    {held_days} 天（空仓 {idle_days} 天）")
if held_days > 0:
    print(f"    资金利用率:  {held_days/total_days*100:.1f}%")
    # 持仓期间的年化收益
    print(f"    年化收益:    {((1+model_return/100)**(365/total_days)-1)*100:.2f}%")
    # 如果空仓期间资金做无风险收益（假设年化3%）
    rf_daily = 0.03 / 365
    rf_gain = idle_days * rf_daily * 100
    total_with_rf = model_return + rf_gain
    print(f"    空仓期间若做货基(年化3%): 额外 +{rf_gain:.2f}%")
    print(f"    合计收益:    {total_with_rf:.2f}%")

print(f"")
print(f"  【结论】")
if model_return > bh_return:
    print(f"    模型择时 > 买入持有：多赚了 {model_return - bh_return:.2f}%")
else:
    print(f"    买入持有 > 模型择时：买入持有多赚了 {bh_return - model_return:.2f}%")
    print(f"    但模型只用了 {held_days/total_days*100:.1f}% 的时间持仓，剩余资金可以投其他股票")
    print(f"    提示：这个模型是在 CSI300 全体选股，不是专门做茅台择时")
