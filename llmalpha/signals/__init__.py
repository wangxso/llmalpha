"""
Signals layer for LLM Alpha.

Signals take one or more Factors and generate entry/exit signals.

Provides:
- Signal: Base class for all signals
- SignalResult: Entry/exit signal container
- SignalValidator: Validates signal quality (win rate, profit factor, edge ratio)

Hierarchy:
    Factor (computes a value) → Signal (generates buy/sell) → Strategy (position sizing)
"""

from llmalpha.signals.base import (
    Signal,
    SignalResult,
    SignalRegistry,
    register,
    get_signal,
    list_signals,
)
from llmalpha.signals.validator import (
    SignalValidator,
    SignalValidationResult,
    PerformanceMetrics,
    DecayAnalysis,
    SignalStats,
    quick_validate_signal,
)

__all__ = [
    # Core
    "Signal",
    "SignalResult",
    "SignalRegistry",
    "register",
    "get_signal",
    "list_signals",
    # Validation
    "SignalValidator",
    "SignalValidationResult",
    "PerformanceMetrics",
    "DecayAnalysis",
    "SignalStats",
    "quick_validate_signal",
]
