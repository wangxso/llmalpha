"""
Factor System for LLM Alpha.

Provides a base class for factors and a registry for factor discovery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd


@dataclass
class FactorMeta:
    """Metadata for a factor."""
    code: str
    name: str
    category: str  # "momentum", "mean_reversion", "volatility", "sentiment", etc.
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class Factor(ABC):
    """
    Base class for all factors.

    Factors are reusable signal generators that can be composed
    to build trading strategies.

    Example:
        class RSIFactor(Factor):
            def compute(self, df: pd.DataFrame) -> pd.Series:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                return 100 - 100 / (1 + gain / loss)
    """

    # Factor metadata (should be overridden in subclasses)
    code: str = "base"
    name: str = "Base Factor"
    category: str = "general"
    description: str = ""

    def __init__(self, **params):
        """
        Initialize the factor with parameters.

        Args:
            **params: Factor-specific parameters
        """
        self.params = params

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the factor values.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with factor values
        """
        pass

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """Allow factor to be called directly."""
        return self.compute(df)

    @property
    def meta(self) -> FactorMeta:
        """Get factor metadata."""
        return FactorMeta(
            code=self.code,
            name=self.name,
            category=self.category,
            description=self.description,
            params=self.params,
        )


class FactorRegistry:
    """
    Registry for factor discovery and management.

    Example:
        registry = FactorRegistry()
        registry.register(RSIFactor)
        rsi = registry.get("rsi")(period=14)
    """

    def __init__(self):
        """Initialize the registry."""
        self._factors: Dict[str, Type[Factor]] = {}

    def register(self, factor_class: Type[Factor]) -> None:
        """
        Register a factor class.

        Args:
            factor_class: Factor class to register
        """
        self._factors[factor_class.code] = factor_class

    def get(self, code: str) -> Optional[Type[Factor]]:
        """
        Get a factor class by code.

        Args:
            code: Factor code

        Returns:
            Factor class or None
        """
        return self._factors.get(code)

    def list(self, category: Optional[str] = None) -> List[str]:
        """
        List registered factor codes.

        Args:
            category: Filter by category

        Returns:
            List of factor codes
        """
        if category:
            return [
                code for code, cls in self._factors.items()
                if cls.category == category
            ]
        return list(self._factors.keys())

    def categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(cls.category for cls in self._factors.values()))


# Global registry instance
_registry = FactorRegistry()


def register(cls: Type[Factor]) -> Type[Factor]:
    """
    Decorator to register a factor class.

    Example:
        @register
        class RSIFactor(Factor):
            code = "rsi"
            ...
    """
    _registry.register(cls)
    return cls


def get_factor(code: str) -> Optional[Type[Factor]]:
    """Get a factor by code from the global registry."""
    return _registry.get(code)


def list_factors(category: Optional[str] = None) -> List[str]:
    """List all registered factors."""
    return _registry.list(category)


# ============ Common Factor Implementations ============

@register
class RobustZScoreFactor(Factor):
    """
    Robust Z-Score using median and MAD.

    More resistant to outliers than standard z-score.
    """
    code = "robust_zscore"
    name = "Robust Z-Score"
    category = "statistical"
    description = "Z-score using median and MAD for outlier resistance"

    def __init__(self, column: str = "close", window: int = 240):
        super().__init__(column=column, window=window)
        self.column = column
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        series = df[self.column].pct_change() if self.column == "close" else df[self.column]
        series = series.fillna(0)

        median = series.rolling(self.window, min_periods=10).median()
        mad = (series - median).abs().rolling(self.window, min_periods=10).median()
        sigma = 1.4826 * mad + 1e-10

        z = (series - median) / sigma
        return z.fillna(0)


@register
class RSIFactor(Factor):
    """Relative Strength Index factor."""
    code = "rsi"
    name = "RSI"
    category = "momentum"
    description = "Relative Strength Index"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        return rsi


@register
class ATRFactor(Factor):
    """Average True Range factor."""
    code = "atr"
    name = "ATR"
    category = "volatility"
    description = "Average True Range"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.DataFrame({
            "hl": high - low,
            "hc": (high - close.shift(1)).abs(),
            "lc": (low - close.shift(1)).abs(),
        }).max(axis=1)

        return tr.rolling(self.period).mean()


@register
class VolumeZScoreFactor(Factor):
    """Volume Z-Score factor."""
    code = "volume_zscore"
    name = "Volume Z-Score"
    category = "volume"
    description = "Z-score of log volume"

    def __init__(self, window: int = 240):
        super().__init__(window=window)
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        log_vol = np.log1p(df["volume"])
        median = log_vol.rolling(self.window, min_periods=10).median()
        mad = (log_vol - median).abs().rolling(self.window, min_periods=10).median()
        sigma = 1.4826 * mad + 1e-10
        return ((log_vol - median) / sigma).fillna(0)


@register
class OIChangeFactor(Factor):
    """Open Interest change factor."""
    code = "oi_change"
    name = "OI Change"
    category = "sentiment"
    description = "Open Interest percentage change"

    def __init__(self, period: int = 1):
        super().__init__(period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "oi" not in df.columns:
            return pd.Series(0, index=df.index)
        return df["oi"].pct_change(self.period).fillna(0)


@register
class FundingZScoreFactor(Factor):
    """Funding rate Z-Score factor."""
    code = "funding_zscore"
    name = "Funding Z-Score"
    category = "sentiment"
    description = "Z-score of funding rate"

    def __init__(self, window: int = 168):  # 7 days at hourly
        super().__init__(window=window)
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in df.columns:
            return pd.Series(0, index=df.index)

        fr = df["funding_rate"].fillna(0)
        mean = fr.rolling(self.window).mean()
        std = fr.rolling(self.window).std()
        return ((fr - mean) / (std + 1e-10)).fillna(0)


@register
class ImbalanceFactor(Factor):
    """Taker buy/sell imbalance factor."""
    code = "imbalance"
    name = "Taker Imbalance"
    category = "volume"
    description = "Taker buy vs sell volume imbalance"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "taker_buy_volume" not in df.columns:
            return pd.Series(0, index=df.index)

        tbv = df["taker_buy_volume"].fillna(0)
        vol = df["volume"] + 1e-10
        return (2 * tbv / vol) - 1


@register
class PremiumFactor(Factor):
    """Futures premium factor."""
    code = "premium"
    name = "Futures Premium"
    category = "sentiment"
    description = "Futures to spot premium"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "premium_index" not in df.columns:
            return pd.Series(0, index=df.index)
        return df["premium_index"].fillna(0)
