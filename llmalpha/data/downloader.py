"""
Binance Futures Data Downloader

High-performance async downloader for Binance futures data.
Supports K-lines, metrics, funding rates, and various price indices.
"""

import asyncio
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from llmalpha.config import get_settings


# Constants
API_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BASE_URL = "https://data.binance.vision/data/futures/um"
KLINE_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# Default symbols (50 major coins)
DEFAULT_SYMBOLS = [
    # Tier 1 - Major
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    # Tier 2 - Large cap
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT",
    # Tier 3 - Emerging
    "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT", "INJUSDT",
    "SEIUSDT", "TIAUSDT", "JUPUSDT", "WIFUSDT", "PENDLEUSDT",
    # Tier 4 - Active trading
    "MATICUSDT", "NEARUSDT", "FILUSDT", "ICPUSDT", "RNDRUSDT",
    "FTMUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "STXUSDT",
    # Tier 5 - Supplementary
    "TRXUSDT", "TONUSDT", "WLDUSDT", "ONDOUSDT", "ENAUSDT",
    "PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "BONKUSDT", "ORDIUSDT",
    # Tier 6 - DeFi/Infrastructure
    "CRVUSDT", "COMPUSDT", "SNXUSDT", "GMXUSDT", "DYDXUSDT",
]


class DataType(Enum):
    """Data types available for download."""
    KLINES = "klines"
    PREMIUM_INDEX = "premiumIndexKlines"
    MARK_PRICE = "markPriceKlines"
    INDEX_PRICE = "indexPriceKlines"
    METRICS = "metrics"
    FUNDING_RATE = "fundingRate"


@dataclass
class DownloadTask:
    """A single download task."""
    symbol: str
    data_type: DataType
    url: str
    date: Optional[datetime] = None
    is_monthly: bool = False
    interval: str = "1m"


@dataclass
class DownloadStats:
    """Download statistics."""
    total: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    bytes_downloaded: int = 0


def fetch_all_symbols(proxy: Optional[str] = None) -> List[str]:
    """Fetch all USDT perpetual contracts from Binance API."""
    import requests

    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        resp = requests.get(API_URL, proxies=proxies, timeout=30)
        data = resp.json()
        symbols = [
            s['symbol'] for s in data['symbols']
            if s.get('contractType') == 'PERPETUAL'
            and s.get('quoteAsset') == 'USDT'
            and s.get('status') == 'TRADING'
        ]
        return sorted(symbols)
    except Exception as e:
        print(f"Warning: Cannot fetch symbol list ({e}), using defaults")
        return DEFAULT_SYMBOLS


class BinanceDownloader:
    """
    High-performance async Binance futures data downloader.

    Features:
    - Async/await with high concurrency (100+ connections)
    - Monthly + daily data for maximum coverage
    - Multiple timeframes support
    - Complete data: K-lines + Metrics + FundingRate + Price indices
    - Smart retry with progress display
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        months: int = 12,
        output_dir: str = "data",
        concurrency: int = 100,
        proxy: Optional[str] = None,
        intervals: Optional[List[str]] = None,
        include_metrics: bool = True,
        include_funding_rate: bool = True,
        include_premium: bool = True,
        include_mark_price: bool = False,
        include_index_price: bool = False,
        retries: int = 3,
        sample_months: Optional[int] = None,
    ):
        """
        Initialize the downloader.

        Args:
            symbols: List of symbols to download (default: 50 major coins)
            months: Number of months of data to download
            output_dir: Output directory for parquet files
            concurrency: Number of concurrent connections
            proxy: HTTP proxy URL (e.g., "http://127.0.0.1:7890")
            intervals: K-line intervals (default: ["1m"])
            include_metrics: Include OI and long/short ratio data
            include_funding_rate: Include funding rate data
            include_premium: Include premium index data
            include_mark_price: Include mark price data
            include_index_price: Include index price data
            retries: Number of retry attempts
            sample_months: Sample N months evenly from the range
        """
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.months = months
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.proxy = proxy
        self.intervals = intervals or ["1m"]
        self.include_metrics = include_metrics
        self.include_funding_rate = include_funding_rate
        self.include_premium = include_premium
        self.include_mark_price = include_mark_price
        self.include_index_price = include_index_price
        self.retries = retries
        self.sample_months = sample_months

        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=months * 30)

        self.stats = DownloadStats()
        self._lock = asyncio.Lock()

    async def _update_progress(self, success: bool = True, bytes_count: int = 0):
        """Update and display download progress."""
        async with self._lock:
            self.stats.completed += 1
            if success:
                self.stats.success += 1
                self.stats.bytes_downloaded += bytes_count
            else:
                self.stats.failed += 1

            done = self.stats.completed
            total = self.stats.total
            if done % 100 == 0 or done == total:
                pct = done / total * 100 if total > 0 else 0
                mb = self.stats.bytes_downloaded / 1024 / 1024
                print(f"\r  Progress: {done}/{total} ({pct:.1f}%) | Success: {self.stats.success} | Downloaded: {mb:.1f}MB", end="", flush=True)

    def _generate_month_list(self) -> List[Tuple[int, int]]:
        """Generate list of months, with optional uniform sampling."""
        months = []
        current = self.start_date.replace(day=1)
        while current < self.end_date:
            months.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if self.sample_months and self.sample_months < len(months):
            step = len(months) / self.sample_months
            return [months[int(i * step)] for i in range(self.sample_months)]

        return months

    def _generate_day_list(self, recent_days: int = 10) -> List[datetime]:
        """Generate list of recent days for daily data."""
        days = []
        for i in range(recent_days):
            d = self.end_date - timedelta(days=i + 1)
            if d >= self.start_date:
                days.append(d)
        return days

    def _generate_klines_tasks(self, symbol: str, interval: str) -> List[DownloadTask]:
        """Generate K-line download tasks."""
        tasks = []

        # Monthly data
        for year, month in self._generate_month_list():
            url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
            tasks.append(DownloadTask(
                symbol=symbol,
                data_type=DataType.KLINES,
                url=url,
                date=datetime(year, month, 1),
                is_monthly=True,
                interval=interval
            ))

        # Recent daily data
        for d in self._generate_day_list():
            url = f"{BASE_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d.strftime('%Y-%m-%d')}.zip"
            tasks.append(DownloadTask(
                symbol=symbol,
                data_type=DataType.KLINES,
                url=url,
                date=d,
                is_monthly=False,
                interval=interval
            ))

        return tasks

    def _generate_price_klines_tasks(self, symbol: str, data_type: DataType, interval: str) -> List[DownloadTask]:
        """Generate Premium/Mark/Index Price K-line tasks."""
        tasks = []
        type_name = data_type.value

        for year, month in self._generate_month_list():
            url = f"{BASE_URL}/monthly/{type_name}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
            tasks.append(DownloadTask(
                symbol=symbol,
                data_type=data_type,
                url=url,
                date=datetime(year, month, 1),
                is_monthly=True,
                interval=interval
            ))

        for d in self._generate_day_list():
            url = f"{BASE_URL}/daily/{type_name}/{symbol}/{interval}/{symbol}-{interval}-{d.strftime('%Y-%m-%d')}.zip"
            tasks.append(DownloadTask(
                symbol=symbol,
                data_type=data_type,
                url=url,
                date=d,
                is_monthly=False,
                interval=interval
            ))

        return tasks

    def _generate_metrics_tasks(self, symbol: str) -> List[DownloadTask]:
        """Generate Metrics tasks (daily only)."""
        tasks = []
        current = self.start_date
        while current <= self.end_date:
            url = f"{BASE_URL}/daily/metrics/{symbol}/{symbol}-metrics-{current.strftime('%Y-%m-%d')}.zip"
            tasks.append(DownloadTask(
                symbol=symbol,
                data_type=DataType.METRICS,
                url=url,
                date=current,
                is_monthly=False
            ))
            current += timedelta(days=1)
        return tasks

    def _generate_funding_rate_tasks(self, symbol: str) -> List[DownloadTask]:
        """Generate FundingRate tasks (monthly only)."""
        tasks = []
        for year, month in self._generate_month_list():
            url = f"{BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
            tasks.append(DownloadTask(
                symbol=symbol,
                data_type=DataType.FUNDING_RATE,
                url=url,
                date=datetime(year, month, 1),
                is_monthly=True
            ))
        return tasks

    async def _download_file(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        """Download a single file with retries."""
        for attempt in range(self.retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    elif resp.status == 404:
                        return None
            except asyncio.TimeoutError:
                if attempt < self.retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
            except Exception:
                if attempt < self.retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return None

    def _parse_klines_zip(self, content: bytes) -> Optional[pd.DataFrame]:
        """Parse K-line ZIP file."""
        try:
            with zipfile.ZipFile(BytesIO(content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return None
                with z.open(csv_files[0]) as f:
                    first_line = f.readline().decode('utf-8').strip()
                    f.seek(0)

                    has_header = 'open_time' in first_line.lower() or not first_line[0].isdigit()
                    df = pd.read_csv(f, header=0 if has_header else None)

                    if not has_header:
                        cols = ["open_time", "open", "high", "low", "close", "volume",
                                "close_time", "quote_volume", "count", "taker_buy_volume",
                                "taker_buy_quote_volume", "ignore"]
                        df.columns = cols[:len(df.columns)]

            if len(df) < 10:
                return None

            df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
            df = df.dropna(subset=["open_time"])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.set_index("timestamp")

            result = pd.DataFrame()
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    result[col] = pd.to_numeric(df[col], errors="coerce")

            if "taker_buy_volume" in df.columns:
                result["taker_buy_volume"] = pd.to_numeric(df["taker_buy_volume"], errors="coerce")
            if "quote_volume" in df.columns:
                result["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
            if "count" in df.columns:
                result["trade_count"] = pd.to_numeric(df["count"], errors="coerce")

            return result
        except Exception:
            return None

    def _parse_price_klines_zip(self, content: bytes, col_name: str) -> Optional[pd.DataFrame]:
        """Parse Premium/Mark/Index Price K-line ZIP."""
        try:
            with zipfile.ZipFile(BytesIO(content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return None
                with z.open(csv_files[0]) as f:
                    first_line = f.readline().decode('utf-8').strip()
                    f.seek(0)

                    has_header = 'open_time' in first_line.lower() or not first_line[0].isdigit()
                    df = pd.read_csv(f, header=0 if has_header else None)

                    if not has_header:
                        cols = ["open_time", "open", "high", "low", "close",
                                "ignore1", "close_time", "ignore2", "count",
                                "ignore3", "ignore4", "ignore5"]
                        df.columns = cols[:len(df.columns)]

            if len(df) < 10:
                return None

            df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
            df = df.dropna(subset=["open_time"])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.set_index("timestamp")

            result = pd.DataFrame()
            result[col_name] = pd.to_numeric(df["close"], errors="coerce")

            return result
        except Exception:
            return None

    def _parse_metrics_zip(self, content: bytes) -> Optional[pd.DataFrame]:
        """Parse Metrics ZIP file."""
        try:
            with zipfile.ZipFile(BytesIO(content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return None
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)

            if len(df) < 1:
                return None

            df["timestamp"] = pd.to_datetime(df["create_time"])
            df = df.set_index("timestamp")

            result = pd.DataFrame()

            if "sum_open_interest" in df.columns:
                result["oi"] = pd.to_numeric(df["sum_open_interest"], errors="coerce")
            if "sum_open_interest_value" in df.columns:
                result["oi_value"] = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
            if "count_toptrader_long_short_ratio" in df.columns:
                result["top_trader_ls_ratio"] = pd.to_numeric(df["count_toptrader_long_short_ratio"], errors="coerce")
            if "sum_toptrader_long_short_ratio" in df.columns:
                result["top_trader_pos_ratio"] = pd.to_numeric(df["sum_toptrader_long_short_ratio"], errors="coerce")
            if "count_long_short_ratio" in df.columns:
                result["global_ls_ratio"] = pd.to_numeric(df["count_long_short_ratio"], errors="coerce")
            if "sum_taker_long_short_vol_ratio" in df.columns:
                result["taker_ls_vol_ratio"] = pd.to_numeric(df["sum_taker_long_short_vol_ratio"], errors="coerce")

            return result
        except Exception:
            return None

    def _parse_funding_rate_zip(self, content: bytes) -> Optional[pd.DataFrame]:
        """Parse FundingRate ZIP file."""
        try:
            with zipfile.ZipFile(BytesIO(content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return None
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)

            if len(df) < 1:
                return None

            df["timestamp"] = pd.to_datetime(df["calc_time"], unit="ms")
            df = df.set_index("timestamp")

            result = pd.DataFrame()
            result["funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")

            return result
        except Exception:
            return None

    async def _process_task(
        self,
        session: aiohttp.ClientSession,
        task: DownloadTask,
        semaphore: asyncio.Semaphore
    ) -> Tuple[DownloadTask, Optional[pd.DataFrame]]:
        """Process a single download task."""
        async with semaphore:
            content = await self._download_file(session, task.url)

            if content is None:
                await self._update_progress(success=False)
                return task, None

            df = None
            if task.data_type == DataType.KLINES:
                df = self._parse_klines_zip(content)
            elif task.data_type == DataType.PREMIUM_INDEX:
                df = self._parse_price_klines_zip(content, "premium_index")
            elif task.data_type == DataType.MARK_PRICE:
                df = self._parse_price_klines_zip(content, "mark_price")
            elif task.data_type == DataType.INDEX_PRICE:
                df = self._parse_price_klines_zip(content, "index_price")
            elif task.data_type == DataType.METRICS:
                df = self._parse_metrics_zip(content)
            elif task.data_type == DataType.FUNDING_RATE:
                df = self._parse_funding_rate_zip(content)

            await self._update_progress(success=df is not None, bytes_count=len(content))
            return task, df

    def _merge_symbol_data(
        self,
        symbol: str,
        results: List[Tuple[DownloadTask, Optional[pd.DataFrame]]],
        primary_interval: str
    ) -> Optional[pd.DataFrame]:
        """Merge all data for a single symbol."""
        klines_by_interval: Dict[str, List[pd.DataFrame]] = {}
        premium_data = []
        mark_data = []
        index_data = []
        metrics_data = []
        funding_data = []

        for task, df in results:
            if df is None:
                continue

            if task.data_type == DataType.KLINES:
                interval = task.interval
                if interval not in klines_by_interval:
                    klines_by_interval[interval] = []
                klines_by_interval[interval].append(df)
            elif task.data_type == DataType.PREMIUM_INDEX:
                premium_data.append(df)
            elif task.data_type == DataType.MARK_PRICE:
                mark_data.append(df)
            elif task.data_type == DataType.INDEX_PRICE:
                index_data.append(df)
            elif task.data_type == DataType.METRICS:
                metrics_data.append(df)
            elif task.data_type == DataType.FUNDING_RATE:
                funding_data.append(df)

        if primary_interval not in klines_by_interval or not klines_by_interval[primary_interval]:
            return None

        main_df = pd.concat(klines_by_interval[primary_interval]).sort_index()
        main_df = main_df[~main_df.index.duplicated(keep="first")]

        # Merge price indices
        for data_list, name in [
            (premium_data, None),
            (mark_data, None),
            (index_data, None)
        ]:
            if data_list:
                pdf = pd.concat(data_list).sort_index()
                pdf = pdf[~pdf.index.duplicated(keep="first")]
                main_df = main_df.join(pdf, how="left")

        # Merge metrics (5-minute granularity)
        if metrics_data:
            medf = pd.concat(metrics_data).sort_index()
            medf = medf[~medf.index.duplicated(keep="first")]
            medf = medf.resample("1min").ffill()
            main_df = main_df.join(medf, how="left")
            for col in medf.columns:
                if col in main_df.columns:
                    main_df[col] = main_df[col].ffill()

        # Merge funding rate (8-hour intervals)
        if funding_data:
            fdf = pd.concat(funding_data).sort_index()
            fdf = fdf[~fdf.index.duplicated(keep="first")]
            fdf = fdf.resample("1min").ffill()
            main_df = main_df.join(fdf, how="left")
            if "funding_rate" in main_df.columns:
                main_df["funding_rate"] = main_df["funding_rate"].ffill()

        # Fill missing values
        if "taker_buy_volume" not in main_df.columns:
            main_df["taker_buy_volume"] = main_df["volume"] * 0.5

        # Calculate derived metrics
        if "taker_buy_volume" in main_df.columns and "volume" in main_df.columns:
            main_df["taker_sell_volume"] = main_df["volume"] - main_df["taker_buy_volume"]
            main_df["net_taker_volume"] = main_df["taker_buy_volume"] - main_df["taker_sell_volume"]

        return main_df

    async def download_all(self) -> Tuple[List[str], List[str]]:
        """
        Download all data.

        Returns:
            Tuple of (successful_symbols, failed_symbols)
        """
        print("=" * 70)
        print("Binance Futures Data Downloader")
        print("=" * 70)
        print(f"Symbols: {len(self.symbols)}")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')} ({self.months} months)")
        if self.sample_months:
            print(f"Sampled months: {self.sample_months}")
        print(f"Intervals: {', '.join(self.intervals)}")
        print(f"Concurrency: {self.concurrency}")
        print(f"Proxy: {self.proxy or 'None'}")

        data_types = ["K-lines"]
        if self.include_metrics:
            data_types.append("Metrics")
        if self.include_funding_rate:
            data_types.append("FundingRate")
        if self.include_premium:
            data_types.append("Premium")
        if self.include_mark_price:
            data_types.append("MarkPrice")
        if self.include_index_price:
            data_types.append("IndexPrice")
        print(f"Data types: {', '.join(data_types)}")
        print("=" * 70)

        # Generate all tasks
        all_tasks: Dict[str, List[DownloadTask]] = {}
        primary_interval = self.intervals[0]

        for symbol in self.symbols:
            tasks = []

            for interval in self.intervals:
                tasks.extend(self._generate_klines_tasks(symbol, interval))

            if self.include_premium:
                tasks.extend(self._generate_price_klines_tasks(symbol, DataType.PREMIUM_INDEX, primary_interval))
            if self.include_mark_price:
                tasks.extend(self._generate_price_klines_tasks(symbol, DataType.MARK_PRICE, primary_interval))
            if self.include_index_price:
                tasks.extend(self._generate_price_klines_tasks(symbol, DataType.INDEX_PRICE, primary_interval))
            if self.include_metrics:
                tasks.extend(self._generate_metrics_tasks(symbol))
            if self.include_funding_rate:
                tasks.extend(self._generate_funding_rate_tasks(symbol))

            all_tasks[symbol] = tasks

        total_tasks = sum(len(tasks) for tasks in all_tasks.values())
        self.stats.total = total_tasks
        print(f"Total tasks: {total_tasks}")
        print()

        # Set proxy environment
        if self.proxy:
            os.environ['HTTP_PROXY'] = self.proxy
            os.environ['HTTPS_PROXY'] = self.proxy

        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            limit_per_host=50,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False
        )

        semaphore = asyncio.Semaphore(self.concurrency)
        start_time = time.time()

        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "Mozilla/5.0"}
        ) as session:
            all_results: Dict[str, List[Tuple[DownloadTask, Optional[pd.DataFrame]]]] = {}

            async_tasks = []
            for symbol, tasks in all_tasks.items():
                for task in tasks:
                    async_tasks.append(self._process_task(session, task, semaphore))

            results = await asyncio.gather(*async_tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, tuple) and len(r) == 2:
                    task, df = r
                    symbol = task.symbol
                    if symbol not in all_results:
                        all_results[symbol] = []
                    all_results[symbol].append((task, df))

        elapsed = time.time() - start_time
        mb = self.stats.bytes_downloaded / 1024 / 1024
        print(f"\n\nDownload complete! Time: {elapsed:.1f}s | Size: {mb:.1f}MB | Speed: {mb/elapsed:.1f}MB/s")
        print()

        # Merge and save
        print("Merging and saving data...")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        success_symbols = []
        failed_symbols = []

        for symbol in self.symbols:
            if symbol not in all_results:
                failed_symbols.append(symbol)
                continue

            df = self._merge_symbol_data(symbol, all_results[symbol], primary_interval)

            if df is not None and len(df) > 10000:
                symbol_clean = symbol.replace("USDT", "_USDT")
                filepath = Path(self.output_dir) / f"{symbol_clean}.parquet"
                df.to_parquet(filepath, compression="snappy")

                days = (df.index.max() - df.index.min()).days
                cols = [c for c in df.columns if not c.startswith('_')]
                print(f"  OK {symbol}: {len(df):,} rows ({days} days) | cols: {len(cols)}")
                success_symbols.append(symbol)
            else:
                rows = len(df) if df is not None else 0
                print(f"  FAIL {symbol}: insufficient data ({rows} rows)")
                failed_symbols.append(symbol)

        # Save additional timeframe K-lines
        if len(self.intervals) > 1:
            print(f"\nSaving additional timeframe K-lines...")
            for interval in self.intervals[1:]:
                interval_dir = Path(self.output_dir) / f"klines_{interval}"
                interval_dir.mkdir(parents=True, exist_ok=True)

                for symbol in success_symbols:
                    if symbol not in all_results:
                        continue

                    klines = [df for task, df in all_results[symbol]
                              if df is not None and task.data_type == DataType.KLINES and task.interval == interval]

                    if klines:
                        kdf = pd.concat(klines).sort_index()
                        kdf = kdf[~kdf.index.duplicated(keep="first")]

                        symbol_clean = symbol.replace("USDT", "_USDT")
                        filepath = interval_dir / f"{symbol_clean}.parquet"
                        kdf.to_parquet(filepath, compression="snappy")

                print(f"  OK {interval}: {len(success_symbols)} symbols")

        print()
        print("=" * 70)
        print(f"Complete! Time: {elapsed:.1f}s")
        print(f"Success: {len(success_symbols)}/{len(self.symbols)}")
        if failed_symbols:
            print(f"Failed: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        return success_symbols, failed_symbols

    def run(self) -> Tuple[List[str], List[str]]:
        """Synchronous wrapper for download_all."""
        return asyncio.run(self.download_all())
