"""
用 akshare 采集 A 股数据，dump 成 Qlib bin 格式

采集范围：原 csi300_ext 中的全部股票 + 2008~2026
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import akshare as ak
from pathlib import Path

# ============================================================
# 配置
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
QLIB_DATA_DIR = _PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
INSTRUMENTS_FILE = QLIB_DATA_DIR / "instruments" / "csi300_ext.txt"
CSV_OUTPUT_DIR = _PROJECT_ROOT / "data" / "stock_data" / "akshare_cn"
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "20080101"
END_DATE = "20260308"

# ============================================================
# 1. 读取股票列表
# ============================================================
stocks = []
with open(INSTRUMENTS_FILE) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 1:
            symbol = parts[0]  # e.g., SH600000, SZ002594
            stocks.append(symbol)

stocks = list(set(stocks))
print(f"共 {len(stocks)} 只股票需要采集")

# ============================================================
# 2. 逐只股票采集
# ============================================================
success = 0
failed = []

for i, stock in enumerate(stocks):
    csv_path = CSV_OUTPUT_DIR / f"{stock}.csv"

    # 已有就跳过
    if csv_path.exists():
        success += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(stocks)}] {stock} 已存在，跳过")
        continue

    # 转换代码格式：SH600000 -> 600000, SZ002594 -> 002594
    code = stock[2:]

    try:
        # qfq = 前复权
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="qfq",
        )

        if df is None or len(df) == 0:
            failed.append(stock)
            continue

        # 也获取不复权数据来计算 factor
        df_raw = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="",
        )

        # 标准化列名
        df_out = pd.DataFrame()
        df_out["date"] = pd.to_datetime(df["日期"])
        df_out["symbol"] = stock
        df_out["open"] = df["开盘"].values
        df_out["close"] = df["收盘"].values
        df_out["high"] = df["最高"].values
        df_out["low"] = df["最低"].values
        df_out["volume"] = df["成交量"].values
        df_out["change"] = df["涨跌幅"].values / 100.0  # 转为小数

        # factor = 前复权价 / 不复权价
        if df_raw is not None and len(df_raw) == len(df):
            raw_close = df_raw["收盘"].values
            adj_close = df["收盘"].values
            factor = np.where(raw_close != 0, adj_close / raw_close, 1.0)
            df_out["factor"] = factor
        else:
            df_out["factor"] = 1.0

        df_out.to_csv(csv_path, index=False)
        success += 1

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(stocks)}] {stock} 完成 ({len(df_out)}行)")

        time.sleep(0.3)  # 避免被限流

    except Exception as e:
        failed.append(stock)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(stocks)}] {stock} 失败: {e}")
        time.sleep(1)

print(f"\n采集完成：成功 {success}，失败 {len(failed)}")
if failed:
    print(f"失败列表: {failed[:20]}...")

# ============================================================
# 3. 更新 instruments 文件（用实际数据日期）
# ============================================================
print("\n更新 instruments 文件...")
inst_lines = []
for csv_file in sorted(CSV_OUTPUT_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv_file, usecols=["date"], parse_dates=["date"])
        if len(df) == 0:
            continue
        symbol = csv_file.stem
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        inst_lines.append(f"{symbol}\t{start}\t{end}")
    except:
        continue

inst_path = QLIB_DATA_DIR / "instruments" / "csi300_ak.txt"
with open(inst_path, "w") as f:
    f.write("\n".join(inst_lines))
print(f"写入 {inst_path}，共 {len(inst_lines)} 只股票")

# ============================================================
# 4. Dump 成 Qlib bin 格式
# ============================================================
print("\nDump 成 Qlib bin 格式...")
dump_cmd = (
    f"python {Path(__file__).resolve().parent / 'dump_bin.py'} dump_all "
    f"--data_path {CSV_OUTPUT_DIR} "
    f"--qlib_dir {QLIB_DATA_DIR} "
    f"--freq day "
    f"--exclude_fields date,symbol "
    f"--file_suffix .csv "
    f"--max_workers 16"
)
print(f"执行: {dump_cmd}")
os.system(dump_cmd)

# 更新日历
print("\n更新日历...")
all_dates = set()
for csv_file in CSV_OUTPUT_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file, usecols=["date"])
        all_dates.update(df["date"].tolist())
    except:
        continue

sorted_dates = sorted(all_dates)
cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
with open(cal_path, "w") as f:
    f.write("\n".join(sorted_dates))
print(f"日历更新完成，共 {len(sorted_dates)} 个交易日")
print(f"范围: {sorted_dates[0]} ~ {sorted_dates[-1]}")
