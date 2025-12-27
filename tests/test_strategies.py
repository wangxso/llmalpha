"""
Tests for the strategy module.
"""

import pytest
import pandas as pd
import numpy as np


class TestStrategyBase:
    """Test base strategy class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        high = close + np.abs(np.random.randn(200) * 0.3)
        low = close - np.abs(np.random.randn(200) * 0.3)
        open_ = close + np.random.randn(200) * 0.2

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, 200),
        }, index=dates)

    def test_strategy_import(self):
        """Test strategy base can be imported."""
        from llmalpha.strategies.base import Strategy, StrategyConfig
        assert Strategy is not None
        assert StrategyConfig is not None

    def test_signal_result_dataclass(self, sample_data):
        """Test SignalResult dataclass."""
        from llmalpha.signals.base import SignalResult

        signals = SignalResult(
            entries=pd.Series([True, False, True]),
            exits=pd.Series([False, True, False]),
        )

        assert len(signals.entries) == 3
        assert len(signals.exits) == 3
        assert signals.short_entries is None

    def test_strategy_abstract(self):
        """Test that Strategy is abstract."""
        from llmalpha.strategies.base import Strategy

        with pytest.raises(TypeError):
            Strategy()


class TestSignalBase:
    """Test base signal class."""

    def test_signal_import(self):
        """Test signal base can be imported."""
        from llmalpha.signals.base import Signal, SignalResult
        assert Signal is not None
        assert SignalResult is not None

    def test_signal_result_dataclass(self):
        """Test SignalResult dataclass."""
        from llmalpha.signals.base import SignalResult

        output = SignalResult(
            entries=pd.Series([True, False, False, True]),
            exits=pd.Series([False, True, False, False]),
        )

        assert len(output.entries) == 4
        assert output.entries.sum() == 2


class TestFactorBase:
    """Test base factor class."""

    def test_factor_import(self):
        """Test factor base can be imported."""
        from llmalpha.factors.base import Factor, FactorMeta
        assert Factor is not None
        assert FactorMeta is not None

    def test_factor_meta_dataclass(self):
        """Test FactorMeta dataclass."""
        from llmalpha.factors.base import FactorMeta

        meta = FactorMeta(
            code="test_factor",
            name="Test Factor",
            category="test",
            description="A test factor",
        )

        assert meta.code == "test_factor"
        assert meta.name == "Test Factor"
