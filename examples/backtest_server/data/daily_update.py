#!/usr/bin/env python3
"""
增量更新脚本 - 更新已有股票的最新数据

使用方法:
    python daily_update.py
"""
from datetime import datetime
from stock_data_manager import StockDataManager


if __name__ == "__main__":
    print("=" * 60)
    print(f"增量更新 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    manager = StockDataManager(db_path="stock_data.db")

    # 显示当前状态
    progress = manager.get_overall_progress()
    print(f"\n当前数据库: {progress['total']} 只股票")

    # 执行更新
    print("\n开始更新...")
    result = manager.daily_update()

    # 显示结果
    print(f"\n更新完成:")
    print(f"  - 更新成功: {result['updated']}")
    print(f"  - 更新失败: {result['failed']}")
    print(f"  - 跳过(已是最新): {result['skipped']}")
