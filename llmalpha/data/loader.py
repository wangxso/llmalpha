"""
Data Loader for LLM Alpha.

Provides unified interface for loading and resampling parquet data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class DataLoader:
    """
    Load and resample cryptocurrency data from parquet files.

    Example:
        loader = DataLoader("data/")
        data = loader.load_symbols(["BTC", "ETH"], resample="1h")
        prices = loader.get_prices(data)
    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing parquet files
        """
        self.data_dir = Path(data_dir)

    def list_symbols(self) -> List[str]:
        """List all available symbols in the data directory."""
        symbols = []
        for f in self.data_dir.glob("*.parquet"):
            symbol = f.stem.replace("_USDT", "")
            symbols.append(symbol)
        return sorted(symbols)

    def load_symbol(
        self,
        symbol: str,
        resample: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data for a single symbol.

        Args:
            symbol: Symbol name (e.g., "BTC" or "BTCUSDT")
            resample: Resample frequency (e.g., "1h", "4h", "1d")
            columns: Specific columns to load

        Returns:
            DataFrame with the symbol data
        """
        # Normalize symbol name
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}_USDT"
        else:
            symbol = symbol.replace("USDT", "_USDT")

        filepath = self.data_dir / f"{symbol}.parquet"

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_parquet(filepath, columns=columns)

        if resample:
            df = self._resample(df, resample)

        return df

    def load_symbols(
        self,
        symbols: List[str],
        resample: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of symbol names
            resample: Resample frequency
            columns: Specific columns to load

        Returns:
            Dictionary mapping symbol names to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.load_symbol(symbol, resample=resample, columns=columns)
                # Use clean symbol name
                clean_symbol = symbol.replace("USDT", "").replace("_", "")
                data[clean_symbol] = df
            except FileNotFoundError:
                print(f"Warning: Data not found for {symbol}")
        return data

    def get_prices(
        self,
        data: Dict[str, pd.DataFrame],
        align: bool = True
    ) -> pd.DataFrame:
        """
        Extract close prices from loaded data.

        Args:
            data: Dictionary of symbol DataFrames
            align: Align all symbols to common time range

        Returns:
            DataFrame with close prices for each symbol
        """
        prices = {}
        for symbol, df in data.items():
            if "close" in df.columns:
                prices[symbol] = df["close"]

        if not prices:
            return pd.DataFrame()

        price_df = pd.DataFrame(prices)

        if align:
            # Find common time range
            start = max(s.index.min() for s in prices.values())
            end = min(s.index.max() for s in prices.values())
            price_df = price_df.loc[start:end]

        return price_df

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample DataFrame to a new frequency.

        Args:
            df: Input DataFrame with datetime index
            freq: Target frequency (e.g., "1h", "4h", "1d")

        Returns:
            Resampled DataFrame
        """
        agg_dict = {}

        # OHLCV columns
        if "open" in df.columns:
            agg_dict["open"] = "first"
        if "high" in df.columns:
            agg_dict["high"] = "max"
        if "low" in df.columns:
            agg_dict["low"] = "min"
        if "close" in df.columns:
            agg_dict["close"] = "last"

        # Volume columns
        volume_cols = ["volume", "taker_buy_volume", "taker_sell_volume",
                       "quote_volume", "net_taker_volume", "trade_count"]
        for col in volume_cols:
            if col in df.columns:
                agg_dict[col] = "sum"

        # Ratio/rate columns (use last value)
        ratio_cols = ["oi", "oi_value", "funding_rate", "premium_index",
                      "mark_price", "index_price", "top_trader_ls_ratio",
                      "top_trader_pos_ratio", "global_ls_ratio", "taker_ls_vol_ratio"]
        for col in ratio_cols:
            if col in df.columns:
                agg_dict[col] = "last"

        if not agg_dict:
            return df

        resampled = df.resample(freq).agg(agg_dict)
        return resampled.dropna(subset=["close"] if "close" in agg_dict else [])

    def load_all(
        self,
        resample: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all available symbols.

        Args:
            resample: Resample frequency
            columns: Specific columns to load

        Returns:
            Dictionary mapping symbol names to DataFrames
        """
        symbols = self.list_symbols()
        return self.load_symbols(symbols, resample=resample, columns=columns)


def load_data(
    data_dir: str = "data",
    symbols: Optional[List[str]] = None,
    resample: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load data.

    Args:
        data_dir: Data directory
        symbols: Specific symbols to load (None = all)
        resample: Resample frequency

    Returns:
        Dictionary of symbol DataFrames
    """
    loader = DataLoader(data_dir)
    if symbols:
        return loader.load_symbols(symbols, resample=resample)
    return loader.load_all(resample=resample)


def prepare_prices(
    data: Dict[str, pd.DataFrame],
    align: bool = True
) -> pd.DataFrame:
    """
    Convenience function to prepare price matrix.

    Args:
        data: Dictionary of symbol DataFrames
        align: Align to common time range

    Returns:
        DataFrame with close prices
    """
    loader = DataLoader()
    return loader.get_prices(data, align=align)
