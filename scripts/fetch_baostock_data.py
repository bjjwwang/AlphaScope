"""
用 baostock 采集 CSI300 成分股全历史日线数据，dump 成 Qlib bin 格式
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import baostock as bs
from pathlib import Path

# ============================================================
# 配置
# ============================================================
QLIB_DATA_DIR = Path(os.path.expanduser("~/.qlib/qlib_data/cn_data"))
INSTRUMENTS_FILE = QLIB_DATA_DIR / "instruments" / "csi300_ext.txt"
CSV_OUTPUT_DIR = Path(os.path.expanduser("~/.qlib/stock_data/baostock_cn"))
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2008-01-01"
END_DATE = "2026-03-08"

# ============================================================
# 1. 读取股票列表
# ============================================================
stocks = []
with open(INSTRUMENTS_FILE) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 1:
            symbol = parts[0]  # SH600000, SZ002594
            stocks.append(symbol)

stocks = list(set(stocks))
print(f"共 {len(stocks)} 只股票需要采集")

# ============================================================
# 2. 登录 baostock
# ============================================================
lg = bs.login()
print(f"baostock login: {lg.error_code} {lg.error_msg}")

# ============================================================
# 3. 逐只采集
# ============================================================
success = 0
failed = []

for i, stock in enumerate(stocks):
    csv_path = CSV_OUTPUT_DIR / f"{stock}.csv"

    if csv_path.exists() and os.path.getsize(csv_path) > 100:
        success += 1
        continue

    # 转换代码：SH600000 -> sh.600000, SZ002594 -> sz.002594
    bs_code = stock[:2].lower() + "." + stock[2:]

    try:
        # 前复权数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="d",
            adjustflag="2",  # 前复权
        )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if len(rows) == 0:
            failed.append(stock)
            continue

        df_qfq = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])

        # 不复权数据（用于计算 factor）
        rs_raw = bs.query_history_k_data_plus(
            bs_code,
            "date,close",
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="d",
            adjustflag="3",  # 不复权
        )
        rows_raw = []
        while rs_raw.next():
            rows_raw.append(rs_raw.get_row_data())

        # 构造输出
        df_out = pd.DataFrame()
        df_out["date"] = df_qfq["date"]
        df_out["symbol"] = stock

        for col in ["open", "high", "low", "close", "volume"]:
            df_out[col] = pd.to_numeric(df_qfq[col], errors="coerce")

        # change = 涨跌幅
        close_series = df_out["close"]
        df_out["change"] = close_series.pct_change()
        df_out.loc[df_out.index[0], "change"] = 0.0

        # factor = 前复权close / 不复权close
        if len(rows_raw) == len(rows):
            df_raw = pd.DataFrame(rows_raw, columns=["date", "raw_close"])
            raw_close = pd.to_numeric(df_raw["raw_close"], errors="coerce")
            factor = np.where(raw_close != 0, close_series / raw_close, 1.0)
            df_out["factor"] = factor
        else:
            df_out["factor"] = 1.0

        df_out.to_csv(csv_path, index=False)
        success += 1

    except Exception as e:
        failed.append(stock)

    if (i + 1) % 50 == 0:
        print(f"[{i+1}/{len(stocks)}] 已完成 {success} 只，失败 {len(failed)} 只")

bs.logout()
print(f"\n采集完成：成功 {success}，失败 {len(failed)}")
if failed:
    print(f"失败: {failed[:20]}")

# ============================================================
# 4. 生成 instruments 文件
# ============================================================
print("\n生成 instruments 文件...")
inst_lines = []
for csv_file in sorted(CSV_OUTPUT_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv_file, usecols=["date"])
        if len(df) == 0:
            continue
        symbol = csv_file.stem
        start = df["date"].min()
        end = df["date"].max()
        inst_lines.append(f"{symbol}\t{start}\t{end}")
    except:
        continue

inst_path = QLIB_DATA_DIR / "instruments" / "csi300_bs.txt"
with open(inst_path, "w") as f:
    f.write("\n".join(inst_lines))
print(f"写入 {inst_path}，共 {len(inst_lines)} 只股票")

# ============================================================
# 5. Dump 成 Qlib bin 格式
# ============================================================
print("\nDump 成 Qlib bin 格式...")
os.system(
    f"python /data1/wjw/MT5/qlib/scripts/dump_bin.py dump_all "
    f"--data_path {CSV_OUTPUT_DIR} "
    f"--qlib_dir {QLIB_DATA_DIR} "
    f"--freq day "
    f"--exclude_fields date,symbol "
    f"--file_suffix .csv "
    f"--max_workers 16"
)

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
print(f"日历: {sorted_dates[0]} ~ {sorted_dates[-1]}，共 {len(sorted_dates)} 个交易日")

print("\n全部完成！")
