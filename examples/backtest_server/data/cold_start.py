#!/usr/bin/env python3
"""
冷启动脚本 - 下载所有美股历史数据

使用方法:
    python cold_start.py
"""
from stock_data_manager import StockDataManager


# ============================================================
# 从 us_stocks.txt 读取股票列表
# ============================================================

with open('us_stocks.txt', 'r') as f:
    TICKERS = [line.strip() for line in f if line.strip()]


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("股票数据冷启动")
    print("=" * 60)

    print(f"\n将下载 {len(TICKERS)} 只股票的数据")
    print(f"预计耗时: {len(TICKERS) * 5 / 60:.1f} 分钟")
    print(f"\n股票列表: {', '.join(TICKERS[:10])}{'...' if len(TICKERS) > 10 else ''}")

    confirm = input("\n确认开始? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        exit()

    # 开始下载
    manager = StockDataManager(db_path="stock_data.db")
    manager.cold_start(TICKERS, resume=True)

    # 显示结果
    progress = manager.get_overall_progress()
    print(f"\n完成! 总进度: {progress['progress']:.1f}%")
