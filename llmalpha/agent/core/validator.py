"""
Hypothesis Validator for LLM Alpha.

Validates generated strategies through backtesting and walk-forward analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from llmalpha.backtest.result import BacktestResult
from llmalpha.strategies.base import Strategy


@dataclass
class ValidationOutput:
    """Complete validation output."""

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

    # Overall metrics
    total_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0

    # Failure reasons
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "train_sharpe": self.train_sharpe,
            "val_sharpe": self.val_sharpe,
            "test_sharpe": self.test_sharpe,
            "wf_passed": self.wf_passed,
            "rolling_positive_ratio": self.rolling_positive_ratio,
            "rolling_avg_sharpe": self.rolling_avg_sharpe,
            "rolling_passed": self.rolling_passed,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "failure_reasons": self.failure_reasons,
        }


class HypothesisValidator:
    """
    Validates trading hypotheses (strategies) through backtesting.

    Example:
        validator = HypothesisValidator(loader)
        result = await validator.validate(strategy, mode="full")

        if result.passed:
            print(f"Strategy validated with Sharpe: {result.test_sharpe}")
    """

    def __init__(
        self,
        data_loader,
        min_sharpe: float = 0.3,
        min_trades: int = 50,
        decay_threshold: float = 0.5,
        min_positive_ratio: float = 0.6,
    ):
        """
        Initialize validator.

        Args:
            data_loader: DataLoader instance for loading market data
            min_sharpe: Minimum Sharpe ratio required
            min_trades: Minimum number of trades required
            decay_threshold: Maximum acceptable decay between periods
            min_positive_ratio: Minimum ratio of positive windows in rolling validation
        """
        self.loader = data_loader
        self.min_sharpe = min_sharpe
        self.min_trades = min_trades
        self.decay_threshold = decay_threshold
        self.min_positive_ratio = min_positive_ratio

        # Lazy load heavy dependencies
        self._vbt_engine = None
        self._wf_validator = None
        self._rolling_validator = None

    @property
    def vbt_engine(self):
        """Lazy load VBT engine."""
        if self._vbt_engine is None:
            from llmalpha.backtest.vbt_engine import VBTEngine
            self._vbt_engine = VBTEngine()
        return self._vbt_engine

    @property
    def wf_validator(self):
        """Lazy load walk-forward validator."""
        if self._wf_validator is None:
            from llmalpha.optimize.validator import WalkForwardValidator
            self._wf_validator = WalkForwardValidator()
        return self._wf_validator

    @property
    def rolling_validator(self):
        """Lazy load rolling validator."""
        if self._rolling_validator is None:
            from llmalpha.optimize.validator import RollingWindowValidator
            self._rolling_validator = RollingWindowValidator()
        return self._rolling_validator

    def _run_single_backtest(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest on single symbol data."""
        if "close" not in df.columns:
            return BacktestResult()

        try:
            signals = strategy.generate_signals(df)
            result = self.vbt_engine.run(
                close=df["close"],
                entries=signals.entries,
                exits=signals.exits,
                short_entries=signals.short_entries,
                short_exits=signals.short_exits,
            )
            return result
        except Exception as e:
            return BacktestResult()

    def _create_backtest_func(
        self,
        strategy: Strategy,
    ) -> Callable[[Dict[str, pd.DataFrame]], BacktestResult]:
        """Create backtest function for validators."""
        def backtest_func(data: Dict[str, pd.DataFrame]) -> BacktestResult:
            all_results = []

            for symbol, df in data.items():
                result = self._run_single_backtest(strategy, df)
                if result.total_trades > 0:
                    all_results.append(result)

            if not all_results:
                return BacktestResult()

            # Combine results
            total_trades = sum(r.total_trades for r in all_results)
            avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
            avg_return = np.mean([r.total_return for r in all_results])
            max_dd = max(r.max_drawdown for r in all_results)
            avg_win_rate = np.mean([r.win_rate for r in all_results if r.win_rate > 0])
            avg_pf = np.mean([r.profit_factor for r in all_results if r.profit_factor > 0])

            return BacktestResult(
                total_trades=total_trades,
                sharpe_ratio=avg_sharpe,
                total_return=avg_return,
                max_drawdown=max_dd,
                win_rate=avg_win_rate,
                profit_factor=avg_pf,
            )

        return backtest_func

    async def validate(
        self,
        strategy: Strategy,
        mode: str = "full",
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
    ) -> ValidationOutput:
        """
        Validate a strategy.

        Args:
            strategy: Strategy instance to validate
            mode: Validation mode ("quick", "full", "wf_only", "rolling_only")
            symbols: Specific symbols to test (None = all)
            timeframe: Data timeframe

        Returns:
            ValidationOutput
        """
        output = ValidationOutput(passed=False)

        # Load data
        if symbols:
            data = self.loader.load_symbols(symbols, resample=timeframe)
        else:
            data = self.loader.load_all(resample=timeframe)

        if not data:
            output.failure_reasons.append("No data available")
            return output

        # Create backtest function
        backtest_func = self._create_backtest_func(strategy)

        # Quick mode: just run a simple backtest
        if mode == "quick":
            result = backtest_func(data)
            output.test_sharpe = result.sharpe_ratio
            output.total_trades = result.total_trades
            output.win_rate = result.win_rate
            output.total_return = result.total_return
            output.max_drawdown = result.max_drawdown
            output.profit_factor = result.profit_factor

            if result.sharpe_ratio < self.min_sharpe:
                output.failure_reasons.append(
                    f"Sharpe too low: {result.sharpe_ratio:.2f} < {self.min_sharpe}"
                )
            if result.total_trades < self.min_trades:
                output.failure_reasons.append(
                    f"Too few trades: {result.total_trades} < {self.min_trades}"
                )

            output.passed = len(output.failure_reasons) == 0
            return output

        # Walk-Forward validation
        if mode in ("full", "wf_only"):
            try:
                wf_result = self.wf_validator.validate(data, backtest_func)

                output.train_sharpe = wf_result.train_result.sharpe_ratio
                output.val_sharpe = wf_result.val_result.sharpe_ratio
                output.test_sharpe = wf_result.test_result.sharpe_ratio
                output.wf_passed = wf_result.passed
                output.total_trades = wf_result.test_result.total_trades
                output.win_rate = wf_result.test_result.win_rate
                output.total_return = wf_result.test_result.total_return
                output.max_drawdown = wf_result.test_result.max_drawdown
                output.profit_factor = wf_result.test_result.profit_factor

                if not wf_result.passed:
                    if not wf_result.all_positive_sharpe:
                        output.failure_reasons.append("Not all periods have positive Sharpe")
                    if not wf_result.decay_acceptable:
                        output.failure_reasons.append("Performance decay too severe")
                    if not wf_result.test_sharpe_above_threshold:
                        output.failure_reasons.append(
                            f"Test Sharpe below threshold: {output.test_sharpe:.2f}"
                        )
            except Exception as e:
                output.failure_reasons.append(f"Walk-forward error: {e}")

        # Rolling validation
        if mode in ("full", "rolling_only"):
            try:
                rolling_result = self.rolling_validator.validate(data, backtest_func)

                output.rolling_positive_ratio = rolling_result.positive_ratio
                output.rolling_avg_sharpe = rolling_result.avg_sharpe
                output.rolling_passed = rolling_result.passed

                if not rolling_result.passed:
                    output.failure_reasons.append(
                        f"Rolling validation failed: {rolling_result.positive_ratio:.1%} positive "
                        f"(need {self.min_positive_ratio:.1%})"
                    )
            except Exception as e:
                output.failure_reasons.append(f"Rolling validation error: {e}")

        # Determine overall pass/fail
        if mode == "full":
            output.passed = output.wf_passed and output.rolling_passed
        elif mode == "wf_only":
            output.passed = output.wf_passed
        elif mode == "rolling_only":
            output.passed = output.rolling_passed

        return output

    async def quick_validate(
        self,
        strategy: Strategy,
        symbols: Optional[List[str]] = None,
    ) -> tuple[bool, float, str]:
        """
        Quick validation for fast feedback.

        Args:
            strategy: Strategy to validate
            symbols: Symbols to test

        Returns:
            (passed, sharpe, message)
        """
        result = await self.validate(strategy, mode="quick", symbols=symbols)

        if result.passed:
            return True, result.test_sharpe, f"Quick check passed (Sharpe: {result.test_sharpe:.2f})"
        else:
            return False, result.test_sharpe, "; ".join(result.failure_reasons)
