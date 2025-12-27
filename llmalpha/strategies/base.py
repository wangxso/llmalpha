"""
Strategy System for LLM Alpha.

Strategies combine Signals with position sizing, risk management,
and portfolio rules to produce complete trading instructions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd

from llmalpha.signals import Signal, SignalResult


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""

    # Position sizing
    position_size: float = 1.0  # Fraction of capital per trade
    max_positions: int = 1  # Maximum concurrent positions

    # Risk management
    stop_loss_pct: Optional[float] = None  # Stop loss percentage
    take_profit_pct: Optional[float] = None  # Take profit percentage
    max_holding_bars: Optional[int] = None  # Maximum holding period

    # Costs
    fee_pct: float = 0.0008  # Trading fee (0.08%)
    slippage_pct: float = 0.0007  # Slippage (0.07%)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position_size": self.position_size,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_holding_bars": self.max_holding_bars,
            "fee_pct": self.fee_pct,
            "slippage_pct": self.slippage_pct,
        }


class Strategy(ABC):
    """
    Base class for all strategies.

    A Strategy combines one or more Signals with position sizing
    and risk management rules.

    Example:
        class RSIMeanReversionStrategy(Strategy):
            code = "rsi_mr"
            name = "RSI Mean Reversion"

            def __init__(self, config: StrategyConfig = None):
                super().__init__(config)
                self.signal = RSIOversoldSignal(oversold=30, overbought=70)

            def generate_signals(self, df: pd.DataFrame) -> SignalResult:
                return self.signal.generate(df)

            def calculate_position_size(self, df: pd.DataFrame, idx: int) -> float:
                # ATR-based position sizing
                atr = ATRFactor(period=14).compute(df)
                risk_per_trade = 0.01  # 1% risk
                return risk_per_trade / (atr.iloc[idx] / df['close'].iloc[idx])
    """

    # Strategy metadata (override in subclasses)
    code: str = "base"
    name: str = "Base Strategy"
    category: str = "general"
    description: str = ""

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize the strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> SignalResult:
        """
        Generate trading signals using underlying Signal(s).

        Args:
            df: DataFrame with OHLCV and other data

        Returns:
            SignalResult with entry/exit signals
        """
        pass

    def calculate_position_size(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calculate position size for a trade.

        Override this for custom position sizing logic.

        Args:
            df: DataFrame with OHLCV data
            idx: Index of the signal bar

        Returns:
            Position size as fraction of capital (0.0 to 1.0)
        """
        return self.config.position_size

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> Optional[float]:
        """
        Calculate stop loss price.

        Args:
            df: DataFrame with OHLCV data
            idx: Index of entry bar
            direction: 1 for long, -1 for short

        Returns:
            Stop loss price or None
        """
        if self.config.stop_loss_pct is None:
            return None

        entry_price = df["close"].iloc[idx]
        if direction == 1:  # Long
            return entry_price * (1 - self.config.stop_loss_pct)
        else:  # Short
            return entry_price * (1 + self.config.stop_loss_pct)

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int) -> Optional[float]:
        """
        Calculate take profit price.

        Args:
            df: DataFrame with OHLCV data
            idx: Index of entry bar
            direction: 1 for long, -1 for short

        Returns:
            Take profit price or None
        """
        if self.config.take_profit_pct is None:
            return None

        entry_price = df["close"].iloc[idx]
        if direction == 1:  # Long
            return entry_price * (1 + self.config.take_profit_pct)
        else:  # Short
            return entry_price * (1 - self.config.take_profit_pct)

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy configuration to dictionary."""
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "config": self.config.to_dict(),
        }


class StrategyRegistry:
    """
    Registry for strategy discovery and management.

    Example:
        registry = StrategyRegistry()
        registry.register(RSIMeanReversionStrategy)
        strategy = registry.get("rsi_mr")(config=StrategyConfig())
    """

    def __init__(self):
        """Initialize the registry."""
        self._strategies: Dict[str, Type[Strategy]] = {}

    def register(self, strategy_class: Type[Strategy]) -> None:
        """Register a strategy class."""
        self._strategies[strategy_class.code] = strategy_class

    def get(self, code: str) -> Optional[Type[Strategy]]:
        """Get a strategy class by code."""
        return self._strategies.get(code)

    def list(self, category: Optional[str] = None) -> List[str]:
        """List registered strategy codes."""
        if category:
            return [
                code
                for code, cls in self._strategies.items()
                if cls.category == category
            ]
        return list(self._strategies.keys())

    def categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(cls.category for cls in self._strategies.values()))


# Global registry instance
_registry = StrategyRegistry()


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Decorator to register a strategy class."""
    _registry.register(cls)
    return cls


def get_strategy(code: str) -> Optional[Type[Strategy]]:
    """Get a strategy by code from the global registry."""
    return _registry.get(code)


def list_strategies(category: Optional[str] = None) -> List[str]:
    """List all registered strategies."""
    return _registry.list(category)
