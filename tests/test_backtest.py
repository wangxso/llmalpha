"""
Tests for the backtest module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestVBTEngine:
    """Test VBT backtest engine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.abs(np.random.randn(100) * 0.3)
        low = close - np.abs(np.random.randn(100) * 0.3)
        open_ = close + np.random.randn(100) * 0.2

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

    @pytest.fixture
    def sample_signals(self, sample_data):
        """Create sample trading signals."""
        n = len(sample_data)
        entries = pd.Series([False] * n, index=sample_data.index)
        exits = pd.Series([False] * n, index=sample_data.index)

        # Simple alternating signals
        entries.iloc[10] = True
        exits.iloc[20] = True
        entries.iloc[30] = True
        exits.iloc[40] = True

        return entries, exits

    def test_engine_import(self):
        """Test engine can be imported."""
        from llmalpha.backtest.vbt_engine import VBTEngine
        engine = VBTEngine()
        assert engine is not None

    def test_run_backtest(self, sample_data, sample_signals):
        """Test running a basic backtest."""
        from llmalpha.backtest.vbt_engine import VBTEngine

        engine = VBTEngine()
        entries, exits = sample_signals

        result = engine.run(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            strategy_name="test_strategy",
        )

        assert result is not None
        assert hasattr(result, "total_trades")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "total_return")

    def test_backtest_with_shorts(self, sample_data, sample_signals):
        """Test backtest with short positions."""
        from llmalpha.backtest.vbt_engine import VBTEngine

        engine = VBTEngine()
        entries, exits = sample_signals

        # Create short signals
        short_entries = pd.Series([False] * len(sample_data), index=sample_data.index)
        short_exits = pd.Series([False] * len(sample_data), index=sample_data.index)
        short_entries.iloc[50] = True
        short_exits.iloc[60] = True

        result = engine.run(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            strategy_name="test_long_short",
        )

        assert result is not None


class TestBacktestResult:
    """Test backtest result dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        from llmalpha.backtest.vbt_engine import BacktestResult

        result = BacktestResult(
            strategy_name="test",
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.1,
            win_rate=0.55,
            total_trades=20,
        )

        assert result.strategy_name == "test"
        assert result.total_return == 0.15
        assert result.sharpe_ratio == 1.2

    def test_result_to_dict(self):
        """Test result to dict conversion."""
        from llmalpha.backtest.vbt_engine import BacktestResult

        result = BacktestResult(
            strategy_name="test",
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.1,
            win_rate=0.55,
            total_trades=20,
        )

        d = result.to_dict()
        assert d["strategy_name"] == "test"
        assert d["sharpe_ratio"] == 1.2
