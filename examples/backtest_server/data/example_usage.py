"""
股票数据管理系统 - 使用示例

这个脚本演示如何：
1. 冷启动下载数据
2. 查看下载进度
3. 验证下载的数据
4. 执行增量更新
"""
import os
import sys
from datetime import datetime

from stock_data_manager import StockDataManager, DataStatus, INTERVALS


def get_us_stock_tickers():
    """
    获取美股股票列表

    实际使用时，你可以：
    1. 从文件读取股票列表
    2. 从API获取（如 yfinance 的 S&P500 列表）
    3. 手动指定
    """
    # 示例：一些常见美股
    sample_tickers = [
        # 科技股
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # 金融股
        'JPM', 'BAC', 'WFC', 'GS',
        # 消费股
        'KO', 'PEP', 'MCD', 'WMT',
        # 医疗股
        'JNJ', 'PFE', 'UNH',
        # 其他
        'XOM', 'CVX', 'DIS'
    ]
    return sample_tickers


def cold_start_demo():
    """冷启动演示"""
    print("=" * 60)
    print("股票数据管理系统 - 冷启动演示")
    print("=" * 60)

    # 初始化管理器
    db_path = "stock_data.db"
    manager = StockDataManager(db_path=db_path, data_dir="./data")

    # 获取股票列表
    tickers = get_us_stock_tickers()
    print(f"\n准备下载 {len(tickers)} 只股票的数据")
    print(f"股票列表: {', '.join(tickers[:10])}...")

    # 显示将下载的数据类型
    print("\n将下载以下时间周期的数据:")
    for interval, config in INTERVALS.items():
        print(f"  - {interval}: period={config['period']}, interval={config['interval']}")

    # 估算时间
    total_tasks = len(tickers) * len(INTERVALS)
    # API限制是60次/分钟，每个任务需要1次API调用
    estimated_minutes = total_tasks / 60
    print(f"\n预计任务数: {total_tasks} 个")
    print(f"预计耗时: {estimated_minutes:.1f} 分钟 (基于60次/分钟的API限制)")

    # 确认开始
    confirm = input("\n是否开始下载? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return

    # 开始冷启动
    print("\n开始冷启动...")
    start_time = datetime.now()

    manager.cold_start(tickers, resume=True)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n冷启动完成! 耗时: {elapsed:.1f} 秒")

    # 显示结果
    show_progress(manager)


def show_progress(manager=None):
    """显示下载进度"""
    if manager is None:
        manager = StockDataManager(db_path="stock_data.db")

    print("\n" + "=" * 60)
    print("下载进度统计")
    print("=" * 60)

    progress = manager.get_overall_progress()

    print(f"\n总体进度:")
    print(f"  - 总股票数: {progress['total']}")
    print(f"  - 完全完成: {progress['completed']}")
    print(f"  - 完成率: {progress['progress']:.2f}%")

    print(f"\n各周期进度:")
    for interval, stats in progress['by_interval'].items():
        bar_length = 20
        filled = int(bar_length * stats['progress'] / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  {interval:15s} [{bar}] {stats['completed']}/{stats['total']} ({stats['progress']:.1f}%)")

    # 显示就绪的股票
    ready_tickers = manager.get_ready_tickers()
    print(f"\n数据完整的股票 ({len(ready_tickers)} 只):")
    if ready_tickers:
        print(f"  {', '.join(ready_tickers[:20])}")
        if len(ready_tickers) > 20:
            print(f"  ... 还有 {len(ready_tickers) - 20} 只")
    else:
        print("  (暂无)")


def verify_data(ticker='AAPL'):
    """验证下载的数据"""
    print("\n" + "=" * 60)
    print(f"验证数据 - {ticker}")
    print("=" * 60)

    manager = StockDataManager(db_path="stock_data.db")

    # 检查状态
    try:
        status = manager.get_status(ticker)
        print(f"\n{ticker} 数据状态:")
        print(f"  - weekly_2y:  {status.weekly_2y.value}")
        print(f"  - daily_1y:   {status.daily_1y.value}")
        print(f"  - hourly_60m: {status.hourly_60m.value}")
        print(f"  - hourly_30m: {status.hourly_30m.value}")
        print(f"  - hourly_15m: {status.hourly_15m.value}")
        print(f"  - 完成度: {status.completion_rate():.0f}%")
    except KeyError:
        print(f"股票 {ticker} 不存在")
        return

    # 查看各周期的数据
    for interval in INTERVALS.keys():
        try:
            df = manager.get_kline_data(ticker, interval)
            print(f"\n{interval} 数据 ({len(df)} 条):")
            print("-" * 50)

            # 显示列名
            print(f"列: {', '.join(df.columns)}")

            # 显示前5条
            print("\n最早5条数据:")
            for i, row in df.head().iterrows():
                print(f"  {row['timestamp']} | O:{row['open']:.2f} H:{row['high']:.2f} "
                      f"L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:,}")

            # 显示最后5条
            print("\n最新5条数据:")
            for i, row in df.tail().iterrows():
                print(f"  {row['timestamp']} | O:{row['open']:.2f} H:{row['high']:.2f} "
                      f"L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:,}")

        except KeyError:
            print(f"\n{interval}: 暂无数据")


def daily_update_demo():
    """增量更新演示"""
    print("\n" + "=" * 60)
    print("每日增量更新")
    print("=" * 60)

    manager = StockDataManager(db_path="stock_data.db")

    print("\n开始增量更新...")
    result = manager.daily_update()

    print(f"\n更新结果:")
    print(f"  - 更新成功: {result['updated']}")
    print(f"  - 更新失败: {result['failed']}")
    print(f"  - 跳过(已是最新): {result['skipped']}")


def show_database_stats():
    """显示数据库统计"""
    print("\n" + "=" * 60)
    print("数据库统计")
    print("=" * 60)

    manager = StockDataManager(db_path="stock_data.db")
    stats = manager.get_database_stats()

    print(f"\n总股票数: {stats['total_tickers']}")

    print(f"\n各周期K线数据量:")
    for interval, count in stats['kline_counts'].items():
        print(f"  {interval:15s}: {count:,} 条")

    print(f"\nAPI调用统计:")
    print(f"  - 总调用次数: {stats['api_calls']['total']}")
    print(f"  - 成功次数: {stats['api_calls']['successful']}")
    print(f"  - 总获取行数: {stats['api_calls']['total_rows_fetched']:,}")


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 60)
        print("股票数据管理系统")
        print("=" * 60)
        print("\n请选择操作:")
        print("  1. 冷启动 (首次下载)")
        print("  2. 查看下载进度")
        print("  3. 验证数据")
        print("  4. 每日增量更新")
        print("  5. 数据库统计")
        print("  0. 退出")

        choice = input("\n请输入选项: ")

        if choice == '1':
            cold_start_demo()
        elif choice == '2':
            show_progress()
        elif choice == '3':
            ticker = input("请输入股票代码 (默认 AAPL): ").strip().upper() or 'AAPL'
            verify_data(ticker)
        elif choice == '4':
            daily_update_demo()
        elif choice == '5':
            show_database_stats()
        elif choice == '0':
            print("再见!")
            break
        else:
            print("无效选项")


if __name__ == "__main__":
    main()
