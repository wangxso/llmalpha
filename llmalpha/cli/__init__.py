"""
CLI layer for LLM Alpha.

Provides Click-based command line interface for:
- data: Download and manage data
- backtest: Run backtests
- optimize: Parameter optimization
- research: Run hypothesis tests
- kb: Knowledge base operations
"""

from llmalpha.cli.main import cli

__all__ = ["cli"]
