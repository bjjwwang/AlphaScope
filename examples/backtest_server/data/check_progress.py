#!/usr/bin/env python3
"""
检查下载进度

使用方法:
    python check_progress.py
    python check_progress.py AAPL     # 查看单只股票详情
"""
import sys
from stock_data_manager import StockDataManager, INTERVALS


def show_overall_progress(manager):
    """显示总体进度"""
    progress = manager.get_overall_progress()

    print("=" * 60)
    print("下载进度统计")
    print("=" * 60)

    print(f"\n总股票数: {progress['total']}")
    print(f"完全完成: {progress['completed']} ({progress['progress']:.1f}%)")

    print(f"\n各周期进度:")
    print("-" * 50)
    for interval, stats in progress['by_interval'].items():
        bar_len = 30
        filled = int(bar_len * stats['progress'] / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"{interval:15s} [{bar}] {stats['completed']:>5}/{stats['total']:<5} ({stats['progress']:>5.1f}%)")

    # 显示就绪股票数量
    ready = manager.get_ready_tickers()
    print(f"\n可用于分析的股票: {len(ready)} 只")

    # 显示数据库统计
    stats = manager.get_database_stats()
    total_rows = sum(stats['kline_counts'].values())
    print(f"数据库总行数: {total_rows:,}")


def show_ticker_detail(manager, ticker):
    """显示单只股票详情"""
    print("=" * 60)
    print(f"股票详情: {ticker}")
    print("=" * 60)

    try:
        status = manager.get_status(ticker)
    except KeyError:
        print(f"\n错误: 股票 {ticker} 不存在")
        return

    print(f"\n状态:")
    print(f"  weekly_2y:  {status.weekly_2y.value}")
    print(f"  daily_1y:   {status.daily_1y.value}")
    print(f"  hourly_60m: {status.hourly_60m.value}")
    print(f"  hourly_30m: {status.hourly_30m.value}")
    print(f"  hourly_15m: {status.hourly_15m.value}")
    print(f"  完成度: {status.completion_rate():.0f}%")

    print(f"\n数据详情:")
    print("-" * 50)
    for interval in INTERVALS.keys():
        try:
            df = manager.get_kline_data(ticker, interval)
            first = df.iloc[0]['timestamp']
            last = df.iloc[-1]['timestamp']
            print(f"{interval:15s}: {len(df):>4} 条  ({first} ~ {last})")
        except KeyError:
            print(f"{interval:15s}: 无数据")

    # 显示最新价格
    try:
        df = manager.get_kline_data(ticker, 'daily_1y')
        latest = df.iloc[-1]
        print(f"\n最新日线数据:")
        print(f"  日期: {latest['timestamp']}")
        print(f"  开盘: {latest['open']:.2f}")
        print(f"  最高: {latest['high']:.2f}")
        print(f"  最低: {latest['low']:.2f}")
        print(f"  收盘: {latest['close']:.2f}")
        print(f"  成交量: {latest['volume']:,.0f}")
    except:
        pass


if __name__ == "__main__":
    manager = StockDataManager(db_path="stock_data.db")

    if len(sys.argv) > 1:
        # 查看单只股票
        ticker = sys.argv[1].upper()
        show_ticker_detail(manager, ticker)
    else:
        # 查看总体进度
        show_overall_progress(manager)
