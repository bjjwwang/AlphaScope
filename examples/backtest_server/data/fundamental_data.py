#!/usr/bin/env python3
"""
基本面数据管理模块

爬取并存储股票基本面数据，杜绝 LLM 幻觉：
1. 行业分类 (Sector/Industry)
2. 市值 (Market Cap)
3. 机构持股比例 (Institutional Holding %)
4. 机构持股数量变化 (Holding Changes)
5. 内部人买卖 (Insider Transactions)

数据源: yfinance (免费但有限制)
更新策略: 基本面数据每周更新一次即可
"""
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """基本面数据结构"""
    ticker: str
    # 公司信息
    name: str
    sector: str  # 大行业: Technology, Healthcare, etc.
    industry: str  # 细分行业: Semiconductors, Biotechnology, etc.
    # 市值
    market_cap: float  # 单位: 美元
    market_cap_tier: str  # 'micro', 'small', 'mid', 'large', 'mega'
    # 机构持股
    institutional_percent: float  # 机构持股比例 0-100
    institutional_count: int  # 机构数量
    # 内部人交易
    insider_buy_count: int  # 6个月内买入次数
    insider_sell_count: int  # 6个月内卖出次数
    insider_net_shares: float  # 净买入股数
    # 元数据
    last_updated: str  # ISO format timestamp


# 市值分级标准 (单位: 十亿美元)
MARKET_CAP_TIERS = {
    'micro': (0, 0.3),        # < 300M
    'small': (0.3, 2),        # 300M - 2B
    'mid': (2, 10),           # 2B - 10B
    'large': (10, 200),       # 10B - 200B
    'mega': (200, float('inf'))  # > 200B
}

# 行业映射 (yfinance sector -> 简化分类)
SECTOR_MAPPING = {
    'Technology': 'Tech',
    'Communication Services': 'Tech',
    'Healthcare': 'Healthcare',
    'Financial Services': 'Finance',
    'Consumer Cyclical': 'Consumer',
    'Consumer Defensive': 'Consumer',
    'Industrials': 'Industrial',
    'Basic Materials': 'Industrial',
    'Energy': 'Energy',
    'Utilities': 'Utilities',
    'Real Estate': 'Real Estate'
}


def get_market_cap_tier(market_cap: float) -> str:
    """根据市值返回分级"""
    cap_billions = market_cap / 1e9
    for tier, (low, high) in MARKET_CAP_TIERS.items():
        if low <= cap_billions < high:
            return tier
    return 'unknown'


class FundamentalDataManager:
    """基本面数据管理器"""

    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 基本面数据主表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                market_cap_tier TEXT,
                institutional_percent REAL,
                institutional_count INTEGER,
                insider_buy_count INTEGER,
                insider_sell_count INTEGER,
                insider_net_shares REAL,
                last_updated TEXT
            )
        """)

        # 机构持股历史 (用于跟踪变化)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS institutional_history (
                ticker TEXT,
                quarter TEXT,
                institutional_percent REAL,
                institutional_count INTEGER,
                recorded_at TEXT,
                PRIMARY KEY (ticker, quarter)
            )
        """)

        # 内部人交易记录
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insider_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                insider_name TEXT,
                relation TEXT,
                transaction_type TEXT,
                shares REAL,
                value REAL,
                transaction_date TEXT,
                recorded_at TEXT
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fund_sector ON fundamentals(sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fund_tier ON fundamentals(market_cap_tier)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fund_inst ON fundamentals(institutional_percent)")

        conn.commit()
        conn.close()

    def fetch_fundamental(self, ticker: str, force: bool = False) -> Optional[FundamentalData]:
        """
        获取单只股票的基本面数据

        Args:
            ticker: 股票代码
            force: 是否强制更新 (忽略缓存)

        Returns:
            FundamentalData or None
        """
        # 检查缓存 (7天内的数据不重新获取)
        if not force:
            cached = self._get_cached(ticker)
            if cached:
                cache_time = datetime.fromisoformat(cached['last_updated'])
                if datetime.now() - cache_time < timedelta(days=7):
                    return self._dict_to_dataclass(cached)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or 'symbol' not in info:
                logger.warning(f"{ticker}: 无法获取股票信息")
                return None

            # 基本信息
            name = info.get('longName', info.get('shortName', ticker))
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            # 市值
            market_cap = info.get('marketCap', 0) or 0
            market_cap_tier = get_market_cap_tier(market_cap)

            # 机构持股
            inst_percent = (info.get('heldPercentInstitutions', 0) or 0) * 100

            # 获取机构数量 (从 institutional_holders)
            inst_count = 0
            try:
                inst_holders = stock.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    inst_count = len(inst_holders)
            except:
                pass

            # 内部人交易
            insider_buy = 0
            insider_sell = 0
            insider_net = 0
            try:
                insider_txns = stock.insider_transactions
                if insider_txns is not None and not insider_txns.empty:
                    # 过滤最近6个月
                    six_months_ago = datetime.now() - timedelta(days=180)

                    for _, row in insider_txns.iterrows():
                        txn_date = row.get('Start Date')
                        if txn_date and txn_date >= six_months_ago:
                            shares = row.get('Shares', 0) or 0
                            txn_type = str(row.get('Transaction', '')).lower()

                            if 'buy' in txn_type or 'purchase' in txn_type:
                                insider_buy += 1
                                insider_net += shares
                            elif 'sell' in txn_type or 'sale' in txn_type:
                                insider_sell += 1
                                insider_net -= abs(shares)
            except Exception as e:
                logger.debug(f"{ticker}: 获取内部人交易失败 - {e}")

            data = FundamentalData(
                ticker=ticker,
                name=name,
                sector=sector,
                industry=industry,
                market_cap=market_cap,
                market_cap_tier=market_cap_tier,
                institutional_percent=inst_percent,
                institutional_count=inst_count,
                insider_buy_count=insider_buy,
                insider_sell_count=insider_sell,
                insider_net_shares=insider_net,
                last_updated=datetime.now().isoformat()
            )

            # 保存到数据库
            self._save_fundamental(data)

            return data

        except Exception as e:
            logger.error(f"{ticker}: 获取基本面数据失败 - {e}")
            return None

    def _get_cached(self, ticker: str) -> Optional[Dict]:
        """从数据库获取缓存的数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM fundamentals WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        columns = ['ticker', 'name', 'sector', 'industry', 'market_cap',
                   'market_cap_tier', 'institutional_percent', 'institutional_count',
                   'insider_buy_count', 'insider_sell_count', 'insider_net_shares',
                   'last_updated']
        return dict(zip(columns, row))

    def _dict_to_dataclass(self, d: Dict) -> FundamentalData:
        """字典转数据类"""
        return FundamentalData(**d)

    def _save_fundamental(self, data: FundamentalData):
        """保存基本面数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO fundamentals
            (ticker, name, sector, industry, market_cap, market_cap_tier,
             institutional_percent, institutional_count, insider_buy_count,
             insider_sell_count, insider_net_shares, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.ticker, data.name, data.sector, data.industry,
            data.market_cap, data.market_cap_tier, data.institutional_percent,
            data.institutional_count, data.insider_buy_count, data.insider_sell_count,
            data.insider_net_shares, data.last_updated
        ))

        conn.commit()
        conn.close()

    def batch_fetch(self, tickers: List[str], delay: float = 0.5,
                    force: bool = False) -> Dict[str, FundamentalData]:
        """
        批量获取基本面数据

        Args:
            tickers: 股票代码列表
            delay: 每次请求间隔 (秒)
            force: 强制更新

        Returns:
            {ticker: FundamentalData}
        """
        results = {}
        skipped = 0
        failed = 0
        total = len(tickers)

        logger.info(f"开始获取基本面数据: {total} 只股票, 间隔 {delay}秒")
        if not force:
            logger.info(f"增量模式: 7天内已缓存的将跳过")
        else:
            logger.info(f"强制模式: 全部重新获取, 预计 {total * delay / 60:.1f} 分钟")

        start_time = time.time()

        for i, ticker in enumerate(tickers):
            # 检查是否已缓存 (非强制模式)
            if not force:
                cached = self._get_cached(ticker)
                if cached:
                    cache_time = datetime.fromisoformat(cached['last_updated'])
                    if datetime.now() - cache_time < timedelta(days=7):
                        results[ticker] = self._dict_to_dataclass(cached)
                        skipped += 1
                        # 跳过的也显示进度 (每100只)
                        if (i + 1) % 100 == 0:
                            logger.info(f"进度: {i + 1}/{total} ({(i+1)/total*100:.1f}%) - 已跳过 {skipped} 只(已缓存)")
                        continue

            # 需要获取的，显示进度 (每10只或每只)
            fetched_count = len(results) - skipped + 1
            if fetched_count % 10 == 0 or fetched_count == 1:
                logger.info(f"进度: {i + 1}/{total} ({(i+1)/total*100:.1f}%) - 正在获取 {ticker}")

            data = self.fetch_fundamental(ticker, force=force)
            if data:
                results[ticker] = data
            else:
                failed += 1

            time.sleep(delay)

        elapsed = time.time() - start_time
        logger.info(f"完成! 耗时 {elapsed/60:.1f} 分钟")
        logger.info(f"  成功获取: {len(results) - skipped} 只")
        logger.info(f"  使用缓存: {skipped} 只")
        logger.info(f"  失败: {failed} 只")
        return results

    def get_fundamental(self, ticker: str) -> Optional[FundamentalData]:
        """获取已缓存的基本面数据 (不发起网络请求)"""
        cached = self._get_cached(ticker)
        if cached:
            return self._dict_to_dataclass(cached)
        return None

    def filter_by_sector(self, sectors: List[str]) -> List[str]:
        """按行业筛选股票"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join(['?' for _ in sectors])
        cursor.execute(f"""
            SELECT ticker FROM fundamentals
            WHERE sector IN ({placeholders}) OR industry IN ({placeholders})
        """, sectors + sectors)

        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers

    def filter_by_market_cap(self, min_cap: float = 0, max_cap: float = float('inf')) -> List[str]:
        """按市值筛选股票 (单位: 十亿美元)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker FROM fundamentals
            WHERE market_cap >= ? AND market_cap <= ?
        """, (min_cap * 1e9, max_cap * 1e9))

        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers

    def filter_by_institutional(self, min_percent: float = 0) -> List[str]:
        """按机构持股比例筛选"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker FROM fundamentals
            WHERE institutional_percent >= ?
        """, (min_percent,))

        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers

    def filter_by_insider_buying(self, min_net_buys: int = 1) -> List[str]:
        """筛选有内部人净买入的股票"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker FROM fundamentals
            WHERE (insider_buy_count - insider_sell_count) >= ?
        """, (min_net_buys,))

        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers

    def get_stats(self) -> Dict:
        """获取基本面数据统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 总数
        cursor.execute("SELECT COUNT(*) FROM fundamentals")
        total = cursor.fetchone()[0]

        # 按行业统计
        cursor.execute("""
            SELECT sector, COUNT(*) as cnt
            FROM fundamentals
            GROUP BY sector
            ORDER BY cnt DESC
        """)
        by_sector = {row[0]: row[1] for row in cursor.fetchall()}

        # 按市值分级统计
        cursor.execute("""
            SELECT market_cap_tier, COUNT(*) as cnt
            FROM fundamentals
            GROUP BY market_cap_tier
        """)
        by_tier = {row[0]: row[1] for row in cursor.fetchall()}

        # 机构持股分布
        cursor.execute("""
            SELECT
                SUM(CASE WHEN institutional_percent >= 80 THEN 1 ELSE 0 END) as high,
                SUM(CASE WHEN institutional_percent >= 50 AND institutional_percent < 80 THEN 1 ELSE 0 END) as medium,
                SUM(CASE WHEN institutional_percent >= 20 AND institutional_percent < 50 THEN 1 ELSE 0 END) as low,
                SUM(CASE WHEN institutional_percent < 20 THEN 1 ELSE 0 END) as very_low
            FROM fundamentals
        """)
        inst_dist = cursor.fetchone()
        by_institutional = {
            '>=80%': inst_dist[0] or 0,
            '50-80%': inst_dist[1] or 0,
            '20-50%': inst_dist[2] or 0,
            '<20%': inst_dist[3] or 0
        }

        conn.close()

        return {
            'total': total,
            'by_sector': by_sector,
            'by_market_cap_tier': by_tier,
            'by_institutional_holding': by_institutional
        }


def main():
    """测试基本面数据获取"""
    import argparse

    parser = argparse.ArgumentParser(description="基本面数据管理")
    parser.add_argument("--db", default="stock_data.db", help="数据库路径")
    parser.add_argument("--ticker", help="获取单只股票数据")
    parser.add_argument("--batch", action="store_true", help="批量获取所有股票")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--force", action="store_true", help="强制更新")

    args = parser.parse_args()

    manager = FundamentalDataManager(args.db)

    if args.stats:
        stats = manager.get_stats()
        print("\n基本面数据统计:")
        print(f"  总数: {stats['total']}")
        print("\n  按行业:")
        for sector, count in stats['by_sector'].items():
            print(f"    {sector}: {count}")
        print("\n  按市值:")
        for tier, count in stats['by_market_cap_tier'].items():
            print(f"    {tier}: {count}")
        print("\n  按机构持股:")
        for level, count in stats['by_institutional_holding'].items():
            print(f"    {level}: {count}")

    elif args.ticker:
        data = manager.fetch_fundamental(args.ticker, force=args.force)
        if data:
            print(f"\n{data.ticker} - {data.name}")
            print(f"  行业: {data.sector} / {data.industry}")
            print(f"  市值: ${data.market_cap/1e9:.2f}B ({data.market_cap_tier})")
            print(f"  机构持股: {data.institutional_percent:.1f}% ({data.institutional_count} 家)")
            print(f"  内部人交易(6个月): 买{data.insider_buy_count}次 / 卖{data.insider_sell_count}次")
            print(f"  更新时间: {data.last_updated}")

    elif args.batch:
        # 从 us_stocks.txt 读取股票列表
        import os
        stock_file = os.path.join(os.path.dirname(args.db), "us_stocks.txt")
        if os.path.exists(stock_file):
            with open(stock_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            print(f"开始获取 {len(tickers)} 只股票的基本面数据...")
            manager.batch_fetch(tickers, force=args.force)
        else:
            print(f"找不到股票列表文件: {stock_file}")


if __name__ == "__main__":
    main()
