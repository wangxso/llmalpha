"""
Signal System for LLM Alpha.

Signals transform Factor values into trading signals (entry/exit points).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from llmalpha.factors import Factor


@dataclass
class SignalResult:
    """Result of signal generation."""

    entries: pd.Series  # Long entry signals (bool)
    exits: pd.Series  # Long exit signals (bool)
    short_entries: Optional[pd.Series] = None  # Short entry signals
    short_exits: Optional[pd.Series] = None  # Short exit signals

    def __post_init__(self):
        """Ensure all signals are boolean series."""
        self.entries = self.entries.astype(bool)
        self.exits = self.exits.astype(bool)
        if self.short_entries is not None:
            self.short_entries = self.short_entries.astype(bool)
        if self.short_exits is not None:
            self.short_exits = self.short_exits.astype(bool)


class Signal(ABC):
    """
    Base class for all signals.

    A Signal takes Factor values and generates entry/exit points.

    Example:
        class RSIOversoldSignal(Signal):
            code = "rsi_oversold"
            name = "RSI Oversold"

            def __init__(self, oversold=30, overbought=70):
                self.oversold = oversold
                self.overbought = overbought

            def generate(self, df: pd.DataFrame) -> SignalResult:
                rsi = RSIFactor(period=14).compute(df)
                entries = rsi < self.oversold
                exits = rsi > self.overbought
                return SignalResult(entries=entries, exits=exits)
    """

    # Signal metadata (override in subclasses)
    code: str = "base"
    name: str = "Base Signal"
    category: str = "general"  # momentum, mean_reversion, breakout, etc.
    description: str = ""

    def __init__(self, **params):
        """
        Initialize the signal with parameters.

        Args:
            **params: Signal-specific parameters
        """
        self.params = params

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> SignalResult:
        """
        Generate trading signals.

        Args:
            df: DataFrame with OHLCV and other data

        Returns:
            SignalResult with entry/exit signals
        """
        pass

    def __call__(self, df: pd.DataFrame) -> SignalResult:
        """Allow signal to be called directly."""
        return self.generate(df)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal configuration to dictionary."""
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "params": self.params,
        }


class SignalRegistry:
    """
    Registry for signal discovery and management.

    Example:
        registry = SignalRegistry()
        registry.register(RSIOversoldSignal)
        signal = registry.get("rsi_oversold")(oversold=25)
    """

    def __init__(self):
        """Initialize the registry."""
        self._signals: Dict[str, Type[Signal]] = {}

    def register(self, signal_class: Type[Signal]) -> None:
        """Register a signal class."""
        self._signals[signal_class.code] = signal_class

    def get(self, code: str) -> Optional[Type[Signal]]:
        """Get a signal class by code."""
        return self._signals.get(code)

    def list(self, category: Optional[str] = None) -> List[str]:
        """List registered signal codes."""
        if category:
            return [
                code
                for code, cls in self._signals.items()
                if cls.category == category
            ]
        return list(self._signals.keys())

    def categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(cls.category for cls in self._signals.values()))


# Global registry instance
_registry = SignalRegistry()


def register(cls: Type[Signal]) -> Type[Signal]:
    """Decorator to register a signal class."""
    _registry.register(cls)
    return cls


def get_signal(code: str) -> Optional[Type[Signal]]:
    """Get a signal by code from the global registry."""
    return _registry.get(code)


def list_signals(category: Optional[str] = None) -> List[str]:
    """List all registered signals."""
    return _registry.list(category)
