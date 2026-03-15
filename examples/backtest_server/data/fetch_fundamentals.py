#!/usr/bin/env python3
"""
批量获取基本面数据脚本

使用方法:
    # 获取所有股票的基本面数据 (增量，只获取未缓存或过期的)
    python fetch_fundamentals.py

    # 强制更新所有
    python fetch_fundamentals.py --force

    # 只获取前100只测试
    python fetch_fundamentals.py --limit 100
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fundamental_data import FundamentalDataManager

DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "stock_data.db")
if not os.path.exists(DEFAULT_DB_PATH):
    ALT_DB_PATH = "/data1/wjw/MT5/USStockBase/stock_data.db"
    if os.path.exists(ALT_DB_PATH):
        DEFAULT_DB_PATH = ALT_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="批量获取基本面数据")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="数据库路径")
    parser.add_argument("--limit", type=int, help="限制获取数量")
    parser.add_argument("--force", action="store_true", help="强制更新")
    parser.add_argument("--delay", type=float, default=0.3, help="请求间隔(秒)")

    args = parser.parse_args()

    # 读取股票列表
    stock_file = os.path.join(os.path.dirname(args.db), "us_stocks.txt")
    if not os.path.exists(stock_file):
        stock_file = os.path.join(SCRIPT_DIR, "us_stocks.txt")

    if not os.path.exists(stock_file):
        print(f"错误: 找不到股票列表文件")
        return

    with open(stock_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    if args.limit:
        tickers = tickers[:args.limit]

    print(f"数据库: {args.db}")
    print(f"股票数: {len(tickers)}")
    print(f"强制更新: {args.force}")
    print(f"请求间隔: {args.delay}秒")
    print()

    manager = FundamentalDataManager(args.db)

    # 先显示当前状态
    try:
        stats = manager.get_stats()
        print(f"当前已有基本面数据: {stats['total']} 只")
    except:
        print("当前已有基本面数据: 0 只")
    print()

    manager.batch_fetch(tickers, delay=args.delay, force=args.force)

    # 显示统计
    print("\n" + "=" * 60)
    stats = manager.get_stats()
    print(f"基本面数据统计 (共 {stats['total']} 只):")
    print("\n按行业:")
    for sector, count in list(stats['by_sector'].items())[:10]:
        print(f"  {sector}: {count}")
    print("\n按市值:")
    for tier, count in stats['by_market_cap_tier'].items():
        print(f"  {tier}: {count}")
    print("\n按机构持股:")
    for level, count in stats['by_institutional_holding'].items():
        print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
