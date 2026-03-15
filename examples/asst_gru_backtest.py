"""
ASST 妖股回测 —— GRU 时序模型 + 妖股训练集 + 信号阈值策略

与之前 LightGBM 方案的核心区别:
1. 模型: LightGBM(截面排序) → GRU(时序记忆, 看20天走势)
2. 训练集: 625只SP500大盘股 → 15只同类高波动妖股
3. 策略: TopkDropout(625选50) → 信号阈值(score>0买, ≤0卖)
4. 数据: DatasetH(单日) → TSDatasetH(20天滑动窗口)
"""

import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.data import D

# ============================================================
# 1. 初始化 (美股)
# ============================================================
provider_uri = "~/.qlib/qlib_data/us_data"
qlib.init(provider_uri=provider_uri, region=REG_US)
print("Qlib 初始化完成 (US market)")

# ============================================================
# 2. GRU 模型 & TSDatasetH 配置
# ============================================================
model_config = {
    "class": "GRU",
    "module_path": "qlib.contrib.model.pytorch_gru_ts",
    "kwargs": {
        "d_feat": 20,          # 精选20个因子 (GRU benchmark推荐)
        "hidden_size": 128,    # 隐藏层大小
        "num_layers": 2,       # GRU层数
        "dropout": 0.3,        # 防过拟合
        "n_epochs": 200,       # 最大轮数
        "lr": 5e-4,            # 学习率
        "early_stop": 20,      # 早停耐心
        "batch_size": 64,      # 小batch适合小数据
        "metric": "loss",
        "loss": "mse",
        "n_jobs": 4,
        "GPU": 0,
        "seed": 42,
    },
}

# 精选20个因子 (来自GRU官方benchmark, 适合波动股)
SELECTED_FEATURES = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10",
    "CORR5", "CORD5", "CORR10", "ROC60", "RESI10",
    "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
]

dataset_config = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2015-01-01",
                "end_time": "2026-03-08",
                "fit_start_time": "2015-01-01",
                "fit_end_time": "2024-12-31",
                "instruments": "volatile_us",  # 15只妖股
                "infer_processors": [
                    {"class": "FilterCol", "kwargs": {
                        "fields_group": "feature",
                        "col_list": SELECTED_FEATURES,
                    }},
                    {"class": "RobustZScoreNorm", "kwargs": {
                        "fields_group": "feature",
                        "clip_outlier": True,
                    }},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ],
            },
        },
        "segments": {
            "train": ("2015-01-01", "2024-06-30"),
            "valid": ("2024-07-01", "2024-12-31"),
            "test":  ("2025-01-01", "2026-03-06"),
        },
        "step_len": 20,  # 20天滑动窗口
    },
}

# ============================================================
# 3. 训练 GRU & 预测
# ============================================================
print("初始化 GRU 模型和 TSDatasetH...")
model = init_instance_by_config(model_config)
dataset = init_instance_by_config(dataset_config)

with R.start(experiment_name="asst_gru_volatile"):
    print("[1/3] 训练 GRU (GPU: RTX PRO 6000)...")
    model.fit(dataset)
    R.save_objects(**{"params.pkl": model})

    recorder = R.get_recorder()

    print("[2/3] 生成预测信号...")
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    print("[3/3] 信号分析...")
    sar = SigAnaRecord(recorder)
    sar.generate()

    pred_df = recorder.load_object("pred.pkl")

print(f"\n预测数据: {pred_df.index.get_level_values(0).min().date()} ~ {pred_df.index.get_level_values(0).max().date()}")
print(f"预测股票: {pred_df.index.get_level_values(1).unique().tolist()}")

# ============================================================
# 4. ASST 信号阈值择时
# ============================================================
STOCK = "ASST"
STOCK_NAME = "ASST (GRU模型)"

# 提取 ASST 的预测分数
all_instruments = pred_df.index.get_level_values(1).unique()
asst_match = [s for s in all_instruments if s.upper() == "ASST"]
if not asst_match:
    print(f"ASST 不在预测数据中！可用: {all_instruments.tolist()}")
    import sys; sys.exit(1)
STOCK = asst_match[0]

asst_pred = pred_df.loc[pred_df.index.get_level_values(1) == STOCK].copy()
asst_pred = asst_pred.droplevel("instrument")
print(f"\nASST 预测数据: {len(asst_pred)} 天")
print(f"Score 统计: mean={asst_pred['score'].mean():.4f}, std={asst_pred['score'].std():.4f}")
print(f"Score 分布: min={asst_pred['score'].min():.4f}, median={asst_pred['score'].median():.4f}, max={asst_pred['score'].max():.4f}")

# 获取价格数据
dates = asst_pred.index
close_data = D.features(
    [STOCK], ["$close", "$factor"],
    start_time=str(dates.min().date()),
    end_time=str(dates.max().date()),
    freq="day",
)
close_data = close_data.droplevel("instrument")
close_data.columns = ["close_adj", "factor"]
if close_data["factor"].isna().all() or (close_data["factor"] == 0).all():
    close_data["factor"] = 1.0
close_data["close"] = close_data["close_adj"] / close_data["factor"]

# 合并
pool_df = asst_pred.join(close_data, how="left")

# 信号阈值策略: score > 0 → 买入, score <= 0 → 卖出
pool_df["in_pool"] = pool_df["score"] > 0

print(f"买入天数: {pool_df['in_pool'].sum()} / {len(pool_df)} ({pool_df['in_pool'].mean()*100:.1f}%)")

# ============================================================
# 5. 交易明细
# ============================================================
trades = []
entry_date = None
entry_price_real = None
entry_price_adj = None

for i in range(len(pool_df)):
    row = pool_df.iloc[i]
    prev_in = pool_df.iloc[i - 1]["in_pool"] if i > 0 else False

    if row["in_pool"] and not prev_in:
        entry_date = pool_df.index[i]
        entry_price_real = row["close"]
        entry_price_adj = row["close_adj"]
        entry_score = row["score"]

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
            "买入价($)": round(entry_price_real, 2) if pd.notna(entry_price_real) else np.nan,
            "入场score": round(entry_score, 4),
            "卖出日期": exit_date.strftime("%Y-%m-%d"),
            "卖出价($)": round(exit_price_real, 2) if pd.notna(exit_price_real) else np.nan,
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
        "买入价($)": round(entry_price_real, 2) if pd.notna(entry_price_real) else np.nan,
        "入场score": round(entry_score, 4),
        "卖出日期": exit_date.strftime("%Y-%m-%d") + "(未平仓)",
        "卖出价($)": round(last_row["close"], 2) if pd.notna(last_row["close"]) else np.nan,
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
        "起始价($)": round(s["close"], 2) if pd.notna(s["close"]) else np.nan,
        "结束价($)": round(e["close"], 2) if pd.notna(e["close"]) else np.nan,
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
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 10)

print("\n" + "=" * 100)
print(f"{STOCK_NAME} 完整交易明细")
print("模型: GRU (2层, hidden=128, 20日窗口)")
print("训练集: 15只高波动妖股 (ASST/GME/AMC/MARA/RIOT/SMCI/MSTR...)")
print("策略: 信号阈值 (score > 0 买入, ≤ 0 卖出)")
print("=" * 100)
print(f"\n共 {len(trades_df)} 笔交易：\n")
if len(trades_df) > 0:
    print(trades_df.to_string(index=False))
    valid_returns = trades_df["收益率(%)"].dropna()
    if len(valid_returns) > 0:
        wins = valid_returns[valid_returns > 0]
        losses = valid_returns[valid_returns <= 0]
        print(f"\n胜率: {len(wins)}/{len(valid_returns)} = {len(wins)/len(valid_returns)*100:.1f}%")
        print(f"平均收益: {valid_returns.mean():.2f}%  |  最大盈利: {valid_returns.max():.2f}%  |  最大亏损: {valid_returns.min():.2f}%")
        print(f"平均持仓: {trades_df['天数'].mean():.1f}天  |  最长: {trades_df['天数'].max()}天  |  最短: {trades_df['天数'].min()}天")
else:
    print("  (无交易记录)")

print("\n" + "=" * 100)
print(f"完整时间线：持仓 vs 空仓")
print("=" * 100)
if len(seg_df) > 0:
    print(seg_df.to_string(index=False))

print("\n" + "=" * 100)
print("分组统计")
print("=" * 100)

print("\n【持仓期间】")
if len(held_seg) > 0:
    h_valid = held_seg["涨跌(%)"].dropna()
    print(f"  {len(held_seg)} 段，共 {held_seg['天数'].sum()} 天")
    if len(h_valid) > 0:
        print(f"  上涨 {(h_valid > 0).sum()} 段 / 下跌 {(h_valid <= 0).sum()} 段")
        print(f"  平均涨跌: {h_valid.mean():.2f}%")
else:
    print("  无持仓期间")

print("\n【空仓期间】")
if len(missed_seg) > 0:
    m_valid = missed_seg["涨跌(%)"].dropna()
    print(f"  {len(missed_seg)} 段，共 {missed_seg['天数'].sum()} 天")
    if len(m_valid) > 0:
        print(f"  期间上涨 {(m_valid > 0).sum()} 段 / 下跌 {(m_valid <= 0).sum()} 段")
        print(f"  平均涨跌: {m_valid.mean():.2f}%")
        missed_up = m_valid[m_valid > 0]
        missed_down = m_valid[m_valid <= 0]
        if len(missed_up) > 0:
            print(f"  错过的上涨: {missed_up.sum():.2f}%（{len(missed_up)}段）")
        if len(missed_down) > 0:
            print(f"  躲过的下跌: {missed_down.sum():.2f}%（{len(missed_down)}段）")

# 买入持有 vs 模型择时
print("\n" + "=" * 100)
print("对比：买入持有 vs GRU模型择时")
print("=" * 100)

valid_pool = pool_df.dropna(subset=["close_adj"])
if len(valid_pool) > 0:
    first_adj = valid_pool.iloc[0]["close_adj"]
    last_adj = valid_pool.iloc[-1]["close_adj"]
    first_real = valid_pool.iloc[0]["close"]
    last_real = valid_pool.iloc[-1]["close"]
    bh_return = (last_adj - first_adj) / first_adj * 100

    model_return = 1.0
    for _, row in trades_df.iterrows():
        r = row["收益率(%)"]
        if pd.notna(r):
            model_return *= (1 + r / 100)
    model_return = (model_return - 1) * 100

    total_days = (valid_pool.index[-1] - valid_pool.index[0]).days
    held_days = held_seg["天数"].sum() if len(held_seg) > 0 else 0

    print(f"\n  回测区间: {valid_pool.index[0].date()} ~ {valid_pool.index[-1].date()} ({total_days}天)")
    print(f"  起始价: ${first_real:.2f}  →  终止价: ${last_real:.2f}")
    print(f"")
    print(f"  【买入持有 ASST】")
    print(f"    累计收益:  {bh_return:.2f}%")
    print(f"")
    print(f"  【GRU 模型择时】")
    print(f"    累计收益:  {model_return:.2f}%")
    if total_days > 0:
        print(f"    持仓天数:  {held_days}天 / {total_days}天（利用率 {held_days/total_days*100:.1f}%）")
    print(f"")

    # 对比旧 LightGBM 方案
    print("=" * 100)
    print("与旧方案 (LightGBM + SP500训练 + TopkDropout) 对比")
    print("=" * 100)
    print(f"  旧方案 (LightGBM):  胜率 31.9%, 累计 -96.34%")
    print(f"  新方案 (GRU):       胜率 {len(trades_df[trades_df['收益率(%)'] > 0]) if len(trades_df) > 0 else 0}/{len(valid_returns) if len(trades_df) > 0 else 0} = {len(trades_df[trades_df['收益率(%)'] > 0])/max(len(valid_returns),1)*100:.1f}%, 累计 {model_return:.2f}%")

    diff = model_return - bh_return
    if diff > 0:
        print(f"\n  → GRU 模型择时胜出，多赚 {diff:.2f}%")
        if total_days > 0 and held_days > 0:
            print(f"    而且只用了 {held_days/total_days*100:.1f}% 的时间持仓")
    else:
        print(f"\n  → 买入持有胜出，多赚 {-diff:.2f}%")
