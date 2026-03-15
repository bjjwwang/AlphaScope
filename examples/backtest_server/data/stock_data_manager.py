"""
Stock Data Manager - A system for managing US stock historical data.

This module provides functionality for:
- Cold start: Download historical data for thousands of stocks
- Resume capability: Continue from where it left off after crashes
- API rate limiting: Respect API call limits (60 calls/minute)
- Status tracking: Know which stocks have complete data
- Incremental updates: Daily smart updates for new data
- Data consistency: Ensure different time intervals don't pollute each other
"""
import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from threading import Lock
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """Status of data download for a specific interval."""
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# Supported time intervals with their configurations
INTERVALS = {
    'weekly_2y': {'period': 'max', 'interval': '1wk', 'priority': 3},
    'daily_1y': {'period': 'max', 'interval': '1d', 'priority': 5},  # Highest priority
    'hourly_60m': {'period': '730d', 'interval': '60m', 'priority': 2},
    'hourly_30m': {'period': '60d', 'interval': '30m', 'priority': 1},
    'hourly_15m': {'period': '60d', 'interval': '15m', 'priority': 4},
}


@dataclass
class TickerDataStatus:
    """Data status for a single ticker across all intervals."""
    ticker: str
    weekly_2y: DataStatus = DataStatus.NOT_STARTED
    daily_1y: DataStatus = DataStatus.NOT_STARTED
    hourly_60m: DataStatus = DataStatus.NOT_STARTED
    hourly_30m: DataStatus = DataStatus.NOT_STARTED
    hourly_15m: DataStatus = DataStatus.NOT_STARTED
    last_update: Optional[datetime] = None
    error_message: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if all 5 intervals are completed."""
        return all([
            self.weekly_2y == DataStatus.COMPLETED,
            self.daily_1y == DataStatus.COMPLETED,
            self.hourly_60m == DataStatus.COMPLETED,
            self.hourly_30m == DataStatus.COMPLETED,
            self.hourly_15m == DataStatus.COMPLETED,
        ])

    def completion_rate(self) -> float:
        """Return completion percentage (0-100)."""
        completed = sum([
            1 if self.weekly_2y == DataStatus.COMPLETED else 0,
            1 if self.daily_1y == DataStatus.COMPLETED else 0,
            1 if self.hourly_60m == DataStatus.COMPLETED else 0,
            1 if self.hourly_30m == DataStatus.COMPLETED else 0,
            1 if self.hourly_15m == DataStatus.COMPLETED else 0,
        ])
        return (completed / 5) * 100


class APIRateLimiter:
    """
    Rate limiter for API calls using sliding window algorithm.

    Attributes:
        max_calls_per_minute: Maximum number of API calls allowed per minute
    """

    def __init__(self, max_calls_per_minute: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_calls_per_minute: Maximum API calls allowed per minute
        """
        self.max_calls = max_calls_per_minute
        self.calls: List[float] = []
        self._lock = Lock()

    def _clean_old_calls(self) -> None:
        """Remove calls older than 1 minute from the list."""
        now = time.time()
        cutoff = now - 60
        self.calls = [t for t in self.calls if t > cutoff]

    def can_call(self) -> bool:
        """
        Check if an API call can be made without exceeding the rate limit.

        Returns:
            True if a call can be made, False otherwise
        """
        with self._lock:
            self._clean_old_calls()
            return len(self.calls) < self.max_calls

    def record_call(self) -> None:
        """Record an API call."""
        with self._lock:
            self.calls.append(time.time())

    def wait_if_needed(self) -> None:
        """
        Block until an API call can be made.

        If the rate limit is reached, this method will sleep until
        at least one call slot becomes available.
        """
        while True:
            with self._lock:
                self._clean_old_calls()
                if len(self.calls) < self.max_calls:
                    return

                # Calculate how long to wait
                if self.calls:
                    oldest_call = min(self.calls)
                    wait_time = 60 - (time.time() - oldest_call)
                    if wait_time > 0:
                        logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    else:
                        return
                else:
                    return

            # Sleep outside the lock
            time.sleep(max(0.1, wait_time))

    def get_remaining_calls(self) -> int:
        """Get the number of remaining calls in the current window."""
        with self._lock:
            self._clean_old_calls()
            return max(0, self.max_calls - len(self.calls))


class StockDataManager:
    """
    Main class for managing stock data downloads, updates, and queries.

    This class handles:
    - Database initialization and management
    - Cold start data downloads
    - Incremental daily/weekly updates
    - Status tracking for all tickers
    - Data queries
    """

    # Maximum retry attempts for failed downloads
    MAX_RETRIES = 3

    def __init__(
        self,
        db_path: str = "stock_data.db",
        data_dir: str = "./data"
    ) -> None:
        """
        Initialize the Stock Data Manager.

        Args:
            db_path: Path to the SQLite database file
            data_dir: Directory for storing data files
        """
        self.db_path = db_path
        self.data_dir = data_dir
        self.rate_limiter = APIRateLimiter(max_calls_per_minute=60)
        self._db_lock = Lock()
        self._status_cache: Dict[str, TickerDataStatus] = {}

        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Initialize database
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_database(self) -> None:
        """Initialize the database with all required tables and indexes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Tickers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    exchange TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Data status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_status (
                    ticker TEXT PRIMARY KEY,
                    weekly_2y TEXT DEFAULT 'not_started',
                    daily_1y TEXT DEFAULT 'not_started',
                    hourly_60m TEXT DEFAULT 'not_started',
                    hourly_30m TEXT DEFAULT 'not_started',
                    hourly_15m TEXT DEFAULT 'not_started',
                    last_update TIMESTAMP,
                    last_check TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (ticker) REFERENCES tickers(ticker)
                )
            """)

            # Download queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS download_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    interval TEXT,
                    priority INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    UNIQUE(ticker, interval)
                )
            """)

            # Create K-line tables for each interval
            for interval_name in INTERVALS.keys():
                table_name = f"kline_{interval_name}"
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        ticker TEXT,
                        timestamp TIMESTAMP,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        adj_close REAL,
                        PRIMARY KEY (ticker, timestamp)
                    )
                """)

                # Create indexes
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{interval_name}_ticker
                    ON {table_name}(ticker)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{interval_name}_timestamp
                    ON {table_name}(timestamp)
                """)

            # API calls log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    interval TEXT,
                    call_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    rows_fetched INTEGER,
                    error_message TEXT
                )
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()

    def _add_tickers(self, tickers: List[str]) -> None:
        """
        Add tickers to the database.

        Args:
            tickers: List of ticker symbols to add
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for ticker in tickers:
                # Insert into tickers table
                cursor.execute("""
                    INSERT OR IGNORE INTO tickers (ticker)
                    VALUES (?)
                """, (ticker,))

                # Insert initial status
                cursor.execute("""
                    INSERT OR IGNORE INTO data_status (ticker)
                    VALUES (?)
                """, (ticker,))

            conn.commit()
            logger.info(f"Added {len(tickers)} tickers to database")

        finally:
            conn.close()

    def _generate_download_tasks(self, tickers: List[str]) -> None:
        """
        Generate download tasks for all tickers and intervals.

        Args:
            tickers: List of ticker symbols
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for ticker in tickers:
                for interval_name, config in INTERVALS.items():
                    # Check if task already exists
                    cursor.execute("""
                        SELECT status FROM download_queue
                        WHERE ticker = ? AND interval = ?
                    """, (ticker, interval_name))

                    existing = cursor.fetchone()

                    if existing is None:
                        # Create new task
                        cursor.execute("""
                            INSERT INTO download_queue (ticker, interval, priority, status)
                            VALUES (?, ?, ?, 'pending')
                        """, (ticker, interval_name, config['priority']))
                    elif existing[0] == 'failed':
                        # Reset failed task for retry
                        cursor.execute("""
                            UPDATE download_queue
                            SET status = 'pending', retry_count = retry_count
                            WHERE ticker = ? AND interval = ?
                        """, (ticker, interval_name))

            conn.commit()

            # Count total tasks
            cursor.execute("SELECT COUNT(*) FROM download_queue WHERE status = 'pending'")
            pending = cursor.fetchone()[0]
            logger.info(f"Generated download tasks, {pending} pending")

        finally:
            conn.close()

    def _download_data(self, ticker: str, interval: str) -> bool:
        """
        Download data for a specific ticker and interval.

        Args:
            ticker: Stock ticker symbol
            interval: Time interval (e.g., 'daily_1y')

        Returns:
            True if download succeeded, False otherwise
        """
        if yf is None:
            logger.error("yfinance not installed")
            return False

        config = INTERVALS.get(interval)
        if not config:
            logger.error(f"Invalid interval: {interval}")
            return False

        # Wait for rate limit
        self.rate_limiter.wait_if_needed()

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Update status to downloading
            cursor.execute("""
                UPDATE data_status
                SET {} = 'downloading'
                WHERE ticker = ?
            """.format(interval), (ticker,))
            conn.commit()

            # Download data from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(period=config['period'], interval=config['interval'])

            # Record the API call
            self.rate_limiter.record_call()

            if df.empty:
                # Record failed API call
                cursor.execute("""
                    INSERT INTO api_calls (ticker, interval, success, rows_fetched, error_message)
                    VALUES (?, ?, 0, 0, 'No data returned')
                """, (ticker, interval))

                cursor.execute("""
                    UPDATE data_status
                    SET {} = 'failed', error_message = 'No data returned'
                    WHERE ticker = ?
                """.format(interval), (ticker,))
                conn.commit()

                logger.warning(f"No data returned for {ticker} {interval}")
                return False

            # Save data to database
            table_name = f"kline_{interval}"

            for idx, row in df.iterrows():
                timestamp = idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx)
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (ticker, timestamp, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    timestamp,
                    float(row.get('Open', 0)),
                    float(row.get('High', 0)),
                    float(row.get('Low', 0)),
                    float(row.get('Close', 0)),
                    int(row.get('Volume', 0)),
                    float(row.get('Close', 0))  # Use Close as adj_close if not available
                ))

            # Record successful API call
            cursor.execute("""
                INSERT INTO api_calls (ticker, interval, success, rows_fetched)
                VALUES (?, ?, 1, ?)
            """, (ticker, interval, len(df)))

            # Update status to completed
            cursor.execute("""
                UPDATE data_status
                SET {} = 'completed', last_update = CURRENT_TIMESTAMP, error_message = NULL
                WHERE ticker = ?
            """.format(interval), (ticker,))

            # Update download queue
            cursor.execute("""
                UPDATE download_queue
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE ticker = ? AND interval = ?
            """, (ticker, interval))

            conn.commit()
            logger.info(f"Downloaded {len(df)} rows for {ticker} {interval}")
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error downloading {ticker} {interval}: {error_msg}")

            # Record failed API call
            cursor.execute("""
                INSERT INTO api_calls (ticker, interval, success, rows_fetched, error_message)
                VALUES (?, ?, 0, 0, ?)
            """, (ticker, interval, error_msg))

            # Update status to failed
            cursor.execute("""
                UPDATE data_status
                SET {} = 'failed', error_message = ?
                WHERE ticker = ?
            """.format(interval), (error_msg, ticker))

            # Update download queue
            cursor.execute("""
                UPDATE download_queue
                SET status = 'failed', error_message = ?, retry_count = retry_count + 1
                WHERE ticker = ? AND interval = ?
            """, (error_msg, ticker, interval))

            conn.commit()
            return False

        finally:
            conn.close()

    def _process_download_queue(self, max_tasks: int = None) -> Dict[str, int]:
        """
        Process pending download tasks.

        Args:
            max_tasks: Maximum number of tasks to process (None for all)

        Returns:
            Dictionary with completed, failed, and skipped counts
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        result = {'completed': 0, 'failed': 0, 'skipped': 0}

        try:
            # Get pending tasks ordered by priority
            query = """
                SELECT ticker, interval, retry_count FROM download_queue
                WHERE status IN ('pending', 'failed')
                AND retry_count < ?
                ORDER BY priority DESC, created_at ASC
            """
            if max_tasks:
                query += f" LIMIT {max_tasks}"

            cursor.execute(query, (self.MAX_RETRIES,))
            tasks = cursor.fetchall()

            for ticker, interval, retry_count in tasks:
                # Update status to downloading
                cursor.execute("""
                    UPDATE download_queue
                    SET status = 'downloading', started_at = CURRENT_TIMESTAMP
                    WHERE ticker = ? AND interval = ?
                """, (ticker, interval))
                conn.commit()

                # Download
                success = self._download_data(ticker, interval)

                if success:
                    result['completed'] += 1
                else:
                    result['failed'] += 1
                    # Ensure retry_count is incremented for failed downloads
                    # (in case _download_data didn't update it)
                    cursor.execute("""
                        UPDATE download_queue
                        SET retry_count = retry_count + 1, status = 'failed'
                        WHERE ticker = ? AND interval = ? AND status = 'downloading'
                    """, (ticker, interval))
                    conn.commit()

            return result

        finally:
            conn.close()

    def cold_start(
        self,
        tickers: List[str],
        resume: bool = True
    ) -> None:
        """
        Cold start: Download all historical data for the given tickers.

        Args:
            tickers: List of stock ticker symbols
            resume: If True, continue from where it left off

        Raises:
            ValueError: If tickers list is empty
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")

        logger.info(f"Starting cold start for {len(tickers)} tickers (resume={resume})")

        # Add tickers to database
        self._add_tickers(tickers)

        # Generate download tasks
        self._generate_download_tasks(tickers)

        # Process download queue
        result = self._process_download_queue()

        logger.info(
            f"Cold start complete: {result['completed']} completed, "
            f"{result['failed']} failed, {result['skipped']} skipped"
        )

    def _load_status_cache(self) -> None:
        """Load all ticker statuses into cache."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT ticker, weekly_2y, daily_1y, hourly_60m, hourly_30m, hourly_15m,
                       last_update, error_message
                FROM data_status
            """)

            self._status_cache.clear()

            for row in cursor.fetchall():
                ticker = row[0]
                self._status_cache[ticker] = TickerDataStatus(
                    ticker=ticker,
                    weekly_2y=DataStatus(row[1]) if row[1] else DataStatus.NOT_STARTED,
                    daily_1y=DataStatus(row[2]) if row[2] else DataStatus.NOT_STARTED,
                    hourly_60m=DataStatus(row[3]) if row[3] else DataStatus.NOT_STARTED,
                    hourly_30m=DataStatus(row[4]) if row[4] else DataStatus.NOT_STARTED,
                    hourly_15m=DataStatus(row[5]) if row[5] else DataStatus.NOT_STARTED,
                    last_update=datetime.fromisoformat(row[6]) if row[6] else None,
                    error_message=row[7]
                )

        finally:
            conn.close()

    def get_status(self, ticker: str) -> TickerDataStatus:
        """
        Get the data status for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            TickerDataStatus object

        Raises:
            KeyError: If ticker doesn't exist
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT ticker, weekly_2y, daily_1y, hourly_60m, hourly_30m, hourly_15m,
                       last_update, error_message
                FROM data_status
                WHERE ticker = ?
            """, (ticker,))

            row = cursor.fetchone()

            if row is None:
                raise KeyError(f"Ticker {ticker} not found")

            return TickerDataStatus(
                ticker=row[0],
                weekly_2y=DataStatus(row[1]) if row[1] else DataStatus.NOT_STARTED,
                daily_1y=DataStatus(row[2]) if row[2] else DataStatus.NOT_STARTED,
                hourly_60m=DataStatus(row[3]) if row[3] else DataStatus.NOT_STARTED,
                hourly_30m=DataStatus(row[4]) if row[4] else DataStatus.NOT_STARTED,
                hourly_15m=DataStatus(row[5]) if row[5] else DataStatus.NOT_STARTED,
                last_update=datetime.fromisoformat(row[6]) if row[6] else None,
                error_message=row[7]
            )

        finally:
            conn.close()

    def is_ready_for_analysis(self, ticker: str) -> bool:
        """
        Check if a ticker's data is ready for analysis.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if all 5 intervals are completed, False otherwise
        """
        try:
            status = self.get_status(ticker)
            return status.is_complete()
        except KeyError:
            return False

    def get_overall_progress(self) -> Dict[str, Any]:
        """
        Get overall progress statistics.

        Returns:
            Dictionary with total, completed, progress, and by_interval stats
        """
        self._load_status_cache()

        total = len(self._status_cache)
        completed = sum(1 for s in self._status_cache.values() if s.is_complete())

        # Calculate by-interval progress
        by_interval = {}
        for interval_name in INTERVALS.keys():
            interval_completed = sum(
                1 for s in self._status_cache.values()
                if getattr(s, interval_name) == DataStatus.COMPLETED
            )
            by_interval[interval_name] = {
                'completed': interval_completed,
                'total': total,
                'progress': round((interval_completed / total * 100), 2) if total > 0 else 0
            }

        return {
            'total': total,
            'completed': completed,
            'progress': round((completed / total * 100), 2) if total > 0 else 0,
            'by_interval': by_interval
        }

    def get_ready_tickers(self) -> List[str]:
        """
        Get list of tickers with complete data.

        Returns:
            List of ticker symbols with all 5 intervals completed
        """
        self._load_status_cache()

        return [
            ticker for ticker, status in self._status_cache.items()
            if status.is_complete()
        ]

    def get_kline_data(
        self,
        ticker: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Get K-line data for a specific ticker and interval.

        Args:
            ticker: Stock ticker symbol
            interval: Time interval (e.g., 'daily_1y', 'weekly_2y')

        Returns:
            DataFrame with columns: ticker, timestamp, open, high, low, close, volume

        Raises:
            ValueError: If interval is invalid
            KeyError: If ticker doesn't exist or has no data
        """
        if interval not in INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid intervals: {list(INTERVALS.keys())}")

        table_name = f"kline_{interval}"

        conn = self._get_connection()

        try:
            df = pd.read_sql_query(f"""
                SELECT ticker, timestamp, open, high, low, close, volume
                FROM {table_name}
                WHERE ticker = ?
                ORDER BY timestamp ASC
            """, conn, params=(ticker,))

            if df.empty:
                raise KeyError(f"No data found for {ticker} in {interval}")

            return df

        finally:
            conn.close()

    def _get_last_update_time(self, ticker: str, interval: str) -> Optional[datetime]:
        """Get the timestamp of the latest data for a ticker and interval."""
        table_name = f"kline_{interval}"

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(f"""
                SELECT MAX(timestamp) FROM {table_name}
                WHERE ticker = ?
            """, (ticker,))

            row = cursor.fetchone()

            if row and row[0]:
                try:
                    return datetime.fromisoformat(row[0])
                except ValueError:
                    return datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')

            return None

        finally:
            conn.close()

    def _should_update_interval(self, interval: str) -> bool:
        """
        Check if an interval should be updated today.

        Weekly data is only updated on Mondays.
        """
        today = datetime.now()

        if interval == 'weekly_2y':
            return today.weekday() == 0  # Monday

        return True

    def daily_update(self) -> Dict[str, int]:
        """
        Perform daily incremental update.

        Returns:
            Dictionary with updated, failed, and skipped counts
        """
        result = {'updated': 0, 'failed': 0, 'skipped': 0}

        self._load_status_cache()
        today = datetime.now().date()

        for ticker, status in self._status_cache.items():
            for interval_name in INTERVALS.keys():
                # Check if this interval should be updated today
                if not self._should_update_interval(interval_name):
                    result['skipped'] += 1
                    continue

                # Check if data is already up to date
                last_update = self._get_last_update_time(ticker, interval_name)

                if last_update and last_update.date() >= today:
                    result['skipped'] += 1
                    continue

                # Download updated data
                success = self._download_data(ticker, interval_name)

                if success:
                    result['updated'] += 1
                else:
                    result['failed'] += 1

        logger.info(
            f"Daily update complete: {result['updated']} updated, "
            f"{result['failed']} failed, {result['skipped']} skipped"
        )

        return result

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            stats = {}

            # Total tickers
            cursor.execute("SELECT COUNT(*) FROM tickers")
            stats['total_tickers'] = cursor.fetchone()[0]

            # K-line counts by interval
            stats['kline_counts'] = {}
            for interval_name in INTERVALS.keys():
                table_name = f"kline_{interval_name}"
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                stats['kline_counts'][interval_name] = cursor.fetchone()[0]

            # API call stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    SUM(rows_fetched) as total_rows
                FROM api_calls
            """)
            row = cursor.fetchone()
            stats['api_calls'] = {
                'total': row[0],
                'successful': row[1] or 0,
                'total_rows_fetched': row[2] or 0
            }

            return stats

        finally:
            conn.close()
