"""
Data Tool for LLM Alpha Agent.

Allows the LLM to autonomously download and manage market data.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmalpha.data.downloader import BinanceDownloader, DEFAULT_SYMBOLS
from llmalpha.data.loader import DataLoader


# Commonly traded symbols by category (for LLM guidance)
SYMBOL_CATEGORIES = {
    "major": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "large_cap": ["SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"],
    "defi": ["UNIUSDT", "AAVEUSDT", "LINKUSDT", "MKRUSDT", "COMPUSDT"],
    "layer2": ["ARBUSDT", "OPUSDT", "MATICUSDT"],
    "new": ["APTUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "JUPUSDT"],
    "meme": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT"],
}


@dataclass
class DataStatus:
    """Status of available data."""
    available_symbols: List[str]
    missing_symbols: List[str]
    data_dir: str
    has_sufficient_data: bool
    message: str


@dataclass
class DownloadResult:
    """Result of data download operation."""
    success: bool
    downloaded: List[str]
    failed: List[str]
    message: str


class DataTool:
    """
    Tool for LLM to manage market data autonomously.

    This tool allows the LLM agent to:
    - Check what data is available
    - Download data for specific symbols
    - Get data recommendations based on research goals

    Example:
        tool = DataTool(data_dir="data")

        # Check available data
        status = tool.check_data(symbols=["BTC", "ETH", "SOL"])

        # Download missing data
        if status.missing_symbols:
            result = await tool.download_data(
                symbols=status.missing_symbols,
                months=3
            )
    """

    def __init__(
        self,
        data_dir: str = "data",
        proxy: Optional[str] = None,
        concurrency: int = 50,
    ):
        """
        Initialize data tool.

        Args:
            data_dir: Directory for data storage
            proxy: HTTP proxy URL (e.g., "http://127.0.0.1:7890")
            concurrency: Number of concurrent download connections
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.proxy = proxy
        self.concurrency = concurrency
        self.loader = DataLoader(data_dir)

    def check_data(
        self,
        symbols: Optional[List[str]] = None,
        min_symbols: int = 3,
    ) -> DataStatus:
        """
        Check what data is available.

        Args:
            symbols: Specific symbols to check (None = check all available)
            min_symbols: Minimum symbols needed for sufficient data

        Returns:
            DataStatus with availability information
        """
        available = self.loader.list_symbols()

        if symbols is None:
            # Just report what's available
            has_sufficient = len(available) >= min_symbols
            return DataStatus(
                available_symbols=available,
                missing_symbols=[],
                data_dir=str(self.data_dir),
                has_sufficient_data=has_sufficient,
                message=f"Found {len(available)} symbols: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}"
            )

        # Check specific symbols
        symbols_clean = [s.replace("USDT", "").upper() for s in symbols]
        missing = [s for s in symbols_clean if s not in available]
        found = [s for s in symbols_clean if s in available]

        has_sufficient = len(found) >= min_symbols

        if missing:
            message = f"Missing data for: {', '.join(missing)}. Available: {', '.join(found) if found else 'None'}"
        else:
            message = f"All requested data available: {', '.join(found)}"

        return DataStatus(
            available_symbols=found,
            missing_symbols=missing,
            data_dir=str(self.data_dir),
            has_sufficient_data=has_sufficient,
            message=message,
        )

    async def download_data(
        self,
        symbols: List[str],
        months: int = 3,
        include_metrics: bool = True,
        include_funding_rate: bool = True,
    ) -> DownloadResult:
        """
        Download market data for specified symbols.

        Args:
            symbols: List of symbols (e.g., ["BTC", "ETH"] or ["BTCUSDT", "ETHUSDT"])
            months: Number of months of historical data
            include_metrics: Include open interest and long/short ratio
            include_funding_rate: Include funding rate data

        Returns:
            DownloadResult with success/failure information
        """
        # Normalize symbol names
        symbols_normalized = []
        for s in symbols:
            s = s.upper().strip()
            if not s.endswith("USDT"):
                s = s + "USDT"
            symbols_normalized.append(s)

        try:
            downloader = BinanceDownloader(
                symbols=symbols_normalized,
                months=months,
                output_dir=str(self.data_dir),
                concurrency=self.concurrency,
                proxy=self.proxy,
                intervals=["1m"],
                include_metrics=include_metrics,
                include_funding_rate=include_funding_rate,
                include_premium=True,
            )

            success, failed = await downloader.download_all()

            # Convert back to simple names
            success_clean = [s.replace("USDT", "") for s in success]
            failed_clean = [s.replace("USDT", "") for s in failed]

            if failed:
                message = f"Downloaded {len(success)} symbols. Failed: {', '.join(failed_clean)}"
            else:
                message = f"Successfully downloaded data for {len(success)} symbols: {', '.join(success_clean)}"

            return DownloadResult(
                success=len(failed) == 0,
                downloaded=success_clean,
                failed=failed_clean,
                message=message,
            )

        except Exception as e:
            return DownloadResult(
                success=False,
                downloaded=[],
                failed=symbols,
                message=f"Download failed: {e}",
            )

    def download_data_sync(
        self,
        symbols: List[str],
        months: int = 3,
        **kwargs
    ) -> DownloadResult:
        """Synchronous wrapper for download_data."""
        return asyncio.run(self.download_data(symbols, months, **kwargs))

    def get_recommended_symbols(
        self,
        category: Optional[str] = None,
        count: int = 5,
    ) -> List[str]:
        """
        Get recommended symbols for research.

        Args:
            category: Symbol category (major, large_cap, defi, layer2, new, meme)
                      None = balanced mix from all categories
            count: Number of symbols to return

        Returns:
            List of recommended symbol names (without USDT suffix)
        """
        if category and category in SYMBOL_CATEGORIES:
            symbols = SYMBOL_CATEGORIES[category][:count]
        else:
            # Mix from categories
            symbols = []
            categories = ["major", "large_cap", "defi", "new"]
            per_category = max(1, count // len(categories))

            for cat in categories:
                symbols.extend(SYMBOL_CATEGORIES[cat][:per_category])
                if len(symbols) >= count:
                    break

            symbols = symbols[:count]

        return [s.replace("USDT", "") for s in symbols]

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data.

        Returns:
            Dictionary with data summary information
        """
        available = self.loader.list_symbols()

        # Try to get date range from a sample
        date_range = None
        rows = 0
        if available:
            try:
                sample = self.loader.load_symbol(available[0])
                if len(sample) > 0:
                    date_range = {
                        "start": str(sample.index.min()),
                        "end": str(sample.index.max()),
                    }
                    rows = len(sample)
            except Exception:
                pass

        return {
            "available_symbols": available,
            "symbol_count": len(available),
            "data_directory": str(self.data_dir),
            "date_range": date_range,
            "sample_rows": rows,
            "categories": {
                cat: [s.replace("USDT", "") for s in syms[:3]]
                for cat, syms in SYMBOL_CATEGORIES.items()
            }
        }

    def ensure_data(
        self,
        symbols: Optional[List[str]] = None,
        months: int = 3,
        min_symbols: int = 3,
    ) -> DataStatus:
        """
        Ensure data is available, downloading if necessary.

        This is a convenience method that checks data and downloads
        if needed, all in one call.

        Args:
            symbols: Symbols to ensure (None = use defaults)
            months: Months of data to download if missing
            min_symbols: Minimum symbols required

        Returns:
            Final DataStatus after any downloads
        """
        if symbols is None:
            symbols = self.get_recommended_symbols(count=min_symbols)

        # Check current status
        status = self.check_data(symbols, min_symbols)

        if status.has_sufficient_data and not status.missing_symbols:
            return status

        # Need to download
        to_download = status.missing_symbols if status.missing_symbols else symbols

        result = self.download_data_sync(to_download, months)

        # Re-check status
        return self.check_data(symbols, min_symbols)


# Tool schema for LLM function calling
DATA_TOOL_SCHEMA = {
    "name": "data_tool",
    "description": "Manage market data: check availability, download data, get recommendations",
    "functions": [
        {
            "name": "check_data",
            "description": "Check what market data is currently available",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Symbols to check (e.g., ['BTC', 'ETH']). None to list all available."
                    }
                }
            }
        },
        {
            "name": "download_data",
            "description": "Download market data for specified symbols",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Symbols to download (e.g., ['BTC', 'ETH', 'SOL'])"
                    },
                    "months": {
                        "type": "integer",
                        "description": "Months of historical data (default: 3)",
                        "default": 3
                    }
                },
                "required": ["symbols"]
            }
        },
        {
            "name": "get_recommended_symbols",
            "description": "Get recommended symbols for research based on category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["major", "large_cap", "defi", "layer2", "new", "meme"],
                        "description": "Symbol category (optional)"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of symbols to return (default: 5)",
                        "default": 5
                    }
                }
            }
        }
    ]
}
