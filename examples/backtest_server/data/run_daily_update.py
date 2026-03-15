#!/usr/bin/env python3
"""
每日增量更新脚本

使用方法:
  1. 手动运行: python run_daily_update.py
  2. 定时运行: crontab -e 添加 "0 9 * * * python /path/to/run_daily_update.py"
  3. 后台持续运行: python run_daily_update.py --daemon
"""
import argparse
import time
import schedule
from datetime import datetime
from stock_data_manager import StockDataManager


def run_update():
    """执行一次增量更新"""
    print(f"\n[{datetime.now()}] 开始增量更新...")

    manager = StockDataManager(db_path="stock_data.db")
    result = manager.daily_update()

    print(f"[{datetime.now()}] 更新完成:")
    print(f"  - 更新成功: {result['updated']}")
    print(f"  - 更新失败: {result['failed']}")
    print(f"  - 跳过: {result['skipped']}")

    return result


def run_daemon():
    """后台守护进程模式，每天定时运行"""
    print("启动守护进程模式，每天 09:30 自动更新")
    print("按 Ctrl+C 退出")

    # 设置每天 09:30 运行
    schedule.every().day.at("09:30").do(run_update)

    # 也可以设置多个时间点
    # schedule.every().day.at("16:00").do(run_update)

    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次


def main():
    parser = argparse.ArgumentParser(description="股票数据增量更新")
    parser.add_argument("--daemon", action="store_true", help="守护进程模式")
    args = parser.parse_args()

    if args.daemon:
        # 需要安装 schedule: pip install schedule
        try:
            import schedule
            run_daemon()
        except ImportError:
            print("守护进程模式需要安装 schedule: pip install schedule")
            return
    else:
        # 单次运行
        run_update()


if __name__ == "__main__":
    main()
