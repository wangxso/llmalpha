"""
Hypothesis Testing Framework

Provides a base class and utilities for running hypothesis tests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llmalpha.backtest.result import BacktestResult, WalkForwardResult
from llmalpha.data.loader import DataLoader
from llmalpha.optimize.validator import run_walk_forward, run_rolling_validation


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    hypothesis_code: str
    hypothesis_name: str
    passed: bool

    # Walk-forward results
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    test_sharpe: float = 0.0
    wf_passed: bool = False

    # Rolling results
    rolling_positive_ratio: float = 0.0
    rolling_avg_sharpe: float = 0.0
    rolling_passed: bool = False

    # Additional metrics
    total_trades: int = 0
    best_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> str:
        """Generate summary string."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Hypothesis: {self.hypothesis_code} - {self.hypothesis_name}",
            f"Status: {status}",
            "",
            f"Walk-Forward: {'PASS' if self.wf_passed else 'FAIL'}",
            f"  Train Sharpe: {self.train_sharpe:.2f}",
            f"  Val Sharpe: {self.val_sharpe:.2f}",
            f"  Test Sharpe: {self.test_sharpe:.2f}",
            "",
            f"Rolling: {'PASS' if self.rolling_passed else 'FAIL'}",
            f"  Positive Ratio: {self.rolling_positive_ratio:.1%}",
            f"  Avg Sharpe: {self.rolling_avg_sharpe:.2f}",
            "",
            f"Total Trades: {self.total_trades}",
        ]
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


class HypothesisTest(ABC):
    """
    Base class for hypothesis tests.

    Subclass this to implement specific hypothesis tests.

    Example:
        class TrendFollowingTest(HypothesisTest):
            code = "H027"
            name = "Trend Following with Volume"

            def generate_signals(self, df):
                # Implement signal generation
                ...

            def backtest(self, data):
                # Implement backtest logic
                ...
    """

    # Override in subclass
    code: str = "H000"
    name: str = "Base Hypothesis"
    category: str = "general"
    description: str = ""

    def __init__(
        self,
        data_dir: str = "data",
        symbols: Optional[List[str]] = None,
        resample: str = "1h",
    ):
        """
        Initialize the hypothesis test.

        Args:
            data_dir: Data directory
            symbols: List of symbols to test
            resample: Resample frequency
        """
        self.data_dir = data_dir
        self.symbols = symbols
        self.resample = resample
        self.loader = DataLoader(data_dir)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for the test."""
        if self.symbols:
            return self.loader.load_symbols(self.symbols, resample=self.resample)
        return self.loader.load_all(resample=self.resample)

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals.

        Args:
            df: DataFrame with OHLCV data
            params: Optional parameters

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        pass

    @abstractmethod
    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        Run backtest on data.

        Args:
            data: Dictionary of symbol DataFrames
            params: Optional parameters

        Returns:
            BacktestResult
        """
        pass

    def run(
        self,
        validate_wf: bool = True,
        validate_rolling: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> HypothesisTestResult:
        """
        Run the hypothesis test.

        Args:
            validate_wf: Run walk-forward validation
            validate_rolling: Run rolling validation
            params: Optional parameters

        Returns:
            HypothesisTestResult
        """
        print(f"=" * 60)
        print(f"Testing: {self.code} - {self.name}")
        print(f"=" * 60)

        data = self.load_data()
        print(f"Loaded {len(data)} symbols")

        result = HypothesisTestResult(
            hypothesis_code=self.code,
            hypothesis_name=self.name,
        )

        # Run walk-forward validation
        if validate_wf:
            print("\nRunning Walk-Forward Validation...")
            wf_result = run_walk_forward(
                data,
                lambda d: self.backtest(d, params),
            )
            result.train_sharpe = wf_result.train_result.sharpe_ratio
            result.val_sharpe = wf_result.val_result.sharpe_ratio
            result.test_sharpe = wf_result.test_result.sharpe_ratio
            result.wf_passed = wf_result.passed
            result.total_trades = wf_result.test_result.total_trades

            print(f"  Train: {result.train_sharpe:.2f}")
            print(f"  Val: {result.val_sharpe:.2f}")
            print(f"  Test: {result.test_sharpe:.2f}")
            print(f"  WF Passed: {result.wf_passed}")

        # Run rolling validation
        if validate_rolling:
            print("\nRunning Rolling Validation...")
            rolling_result = run_rolling_validation(
                data,
                lambda d: self.backtest(d, params),
            )
            result.rolling_positive_ratio = rolling_result.positive_ratio
            result.rolling_avg_sharpe = rolling_result.avg_sharpe
            result.rolling_passed = rolling_result.passed

            print(f"  Positive Ratio: {result.rolling_positive_ratio:.1%}")
            print(f"  Avg Sharpe: {result.rolling_avg_sharpe:.2f}")
            print(f"  Rolling Passed: {result.rolling_passed}")

        # Determine overall pass/fail
        if validate_wf and validate_rolling:
            result.passed = result.wf_passed and result.rolling_passed
        elif validate_wf:
            result.passed = result.wf_passed
        elif validate_rolling:
            result.passed = result.rolling_passed
        else:
            # Just run full backtest
            bt_result = self.backtest(data, params)
            result.passed = bt_result.sharpe_ratio > 0.3
            result.total_trades = bt_result.total_trades

        result.best_params = params or {}

        print("\n" + "=" * 60)
        status = "PASSED" if result.passed else "FAILED"
        print(f"Result: {status}")
        print("=" * 60)

        return result


class RelativeValueTest(HypothesisTest):
    """
    Relative Value Strategy Test (H026).

    Trades based on relative performance across assets.
    """

    code = "H026"
    name = "Relative Value Strategy"
    category = "mean_reversion"
    description = "Long cheap assets, short expensive assets based on relative z-score"

    def __init__(
        self,
        data_dir: str = "data",
        symbols: Optional[List[str]] = None,
        resample: str = "1h",
        lookback: int = 120,
        z_threshold: float = 3.0,
        hold_bars: int = 24,
    ):
        super().__init__(data_dir, symbols, resample)
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.hold_bars = hold_bars

    def generate_signals(
        self,
        prices: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate relative value signals."""
        params = params or {}
        lookback = params.get("lookback", self.lookback)
        z_threshold = params.get("z_threshold", self.z_threshold)
        hold_bars = params.get("hold_bars", self.hold_bars)

        # Calculate relative returns
        returns = prices.pct_change(lookback)
        avg_ret = returns.mean(axis=1)
        rel_ret = returns.sub(avg_ret, axis=0)

        # Z-score of relative returns
        rel_mean = rel_ret.rolling(lookback * 2).mean()
        rel_std = rel_ret.rolling(lookback * 2).std()
        z_score = (rel_ret - rel_mean) / (rel_std + 1e-10)

        # Signals
        long_signals = z_score < -z_threshold
        short_signals = z_score > z_threshold
        long_exits = long_signals.shift(hold_bars).fillna(False)
        short_exits = short_signals.shift(hold_bars).fillna(False)

        return long_signals, long_exits, short_signals, short_exits

    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """Run relative value backtest."""
        try:
            import vectorbt as vbt
        except ImportError:
            raise ImportError("vectorbt required for this test")

        # Prepare prices
        prices = {}
        for symbol, df in data.items():
            if "close" in df.columns:
                prices[symbol] = df["close"]

        if not prices:
            return BacktestResult()

        price_df = pd.DataFrame(prices)

        # Align to common range
        start = max(s.index.min() for s in prices.values())
        end = min(s.index.max() for s in prices.values())
        price_df = price_df.loc[start:end]

        # Generate signals
        long_sigs, long_exits, short_sigs, short_exits = self.generate_signals(
            price_df, params
        )

        # Run backtests for each symbol
        all_returns = []
        total_trades = 0

        for symbol in price_df.columns:
            try:
                # Long portfolio
                pf_long = vbt.Portfolio.from_signals(
                    close=price_df[symbol].values,
                    entries=long_sigs[symbol].values,
                    exits=long_exits[symbol].values,
                    size=1.0,
                    size_type="percent",
                    init_cash=10000,
                    fees=0.0008,
                    slippage=0.0007,
                    freq="1h",
                )

                # Short portfolio
                pf_short = vbt.Portfolio.from_signals(
                    close=price_df[symbol].values,
                    short_entries=short_sigs[symbol].values,
                    short_exits=short_exits[symbol].values,
                    size=1.0,
                    size_type="percent",
                    init_cash=10000,
                    fees=0.0008,
                    slippage=0.0007,
                    freq="1h",
                )

                # Combine returns (market neutral)
                combined = (pf_long.returns() + pf_short.returns()) / 2
                all_returns.append(combined)

                total_trades += int(pf_long.stats()["Total Trades"])
                total_trades += int(pf_short.stats()["Total Trades"])

            except Exception:
                pass

        if not all_returns:
            return BacktestResult()

        # Combine all symbols
        returns_df = pd.DataFrame(all_returns).T
        combined_returns = returns_df.mean(axis=1)

        # Calculate metrics
        sharpe = (
            combined_returns.mean() / combined_returns.std() * np.sqrt(365 * 24)
            if combined_returns.std() > 0
            else 0
        )
        total_return = (1 + combined_returns).prod() - 1
        cumulative = (1 + combined_returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()

        return BacktestResult(
            total_trades=total_trades,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=abs(max_dd),
            returns_series=combined_returns,
            strategy_name=self.name,
            symbols=list(data.keys()),
        )


def run_hypothesis_test(
    test_class: type,
    data_dir: str = "data",
    symbols: Optional[List[str]] = None,
    **kwargs,
) -> HypothesisTestResult:
    """
    Convenience function to run a hypothesis test.

    Args:
        test_class: HypothesisTest subclass
        data_dir: Data directory
        symbols: Symbols to test
        **kwargs: Additional parameters

    Returns:
        HypothesisTestResult
    """
    test = test_class(data_dir=data_dir, symbols=symbols, **kwargs)
    return test.run()
