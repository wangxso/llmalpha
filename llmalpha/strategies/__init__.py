"""
Strategies layer for LLM Alpha.

Strategies combine Signals with position sizing and risk management.

Hierarchy:
    Factor (computes a value) → Signal (generates buy/sell) → Strategy (position sizing)
"""

from llmalpha.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyRegistry,
    register,
    get_strategy,
    list_strategies,
)

__all__ = [
    "Strategy",
    "StrategyConfig",
    "StrategyRegistry",
    "register",
    "get_strategy",
    "list_strategies",
]
