"""
Backtest layer for LLM Alpha.

Provides:
- VBTEngine: VectorBT-based backtesting engine
- BacktestResult: Standardized result container
"""

from llmalpha.backtest.result import BacktestResult, WalkForwardResult, RollingResult
from llmalpha.backtest.vbt_engine import VBTEngine, run_vbt_backtest

__all__ = [
    "BacktestResult",
    "WalkForwardResult",
    "RollingResult",
    "VBTEngine",
    "run_vbt_backtest",
]
