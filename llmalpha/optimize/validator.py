"""
Walk-Forward and Rolling Window Validators

Provides validation frameworks to detect overfitting.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llmalpha.backtest.result import (
    BacktestResult,
    RollingResult,
    WalkForwardResult,
)


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    # Walk-forward split ratios
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Rolling window settings (in bars)
    train_window: int = 2160  # 3 months at hourly
    test_window: int = 720  # 1 month at hourly
    step: int = 720  # 1 month step

    # Thresholds
    min_sharpe: float = 0.3
    decay_threshold: float = 0.5
    min_positive_ratio: float = 0.6

    # Embargo period (hours)
    embargo_hours: int = 12


class WalkForwardValidator:
    """
    Walk-Forward validation for strategy testing.

    Splits data into train/val/test periods and validates
    that performance doesn't decay significantly.

    Example:
        validator = WalkForwardValidator(config)
        result = validator.validate(data, backtest_func)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def split_data(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Split data into train/val/test sets.

        Args:
            data: Dictionary of symbol DataFrames

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        train_data = {}
        val_data = {}
        test_data = {}

        for symbol, df in data.items():
            n = len(df)

            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

            train_data[symbol] = df.iloc[:train_end]
            val_data[symbol] = df.iloc[train_end:val_end]
            test_data[symbol] = df.iloc[val_end:]

        return train_data, val_data, test_data

    def validate(
        self,
        data: Dict[str, pd.DataFrame],
        backtest_func: Callable[[Dict[str, pd.DataFrame]], BacktestResult],
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: Dictionary of symbol DataFrames
            backtest_func: Function that runs backtest and returns BacktestResult

        Returns:
            WalkForwardResult with validation metrics
        """
        train_data, val_data, test_data = self.split_data(data)

        # Run backtests
        train_result = backtest_func(train_data)
        val_result = backtest_func(val_data)
        test_result = backtest_func(test_data)

        # Check conditions
        train_s = train_result.sharpe_ratio
        val_s = val_result.sharpe_ratio
        test_s = test_result.sharpe_ratio

        all_positive = train_s > 0 and val_s > 0 and test_s > 0
        decay_ok = (
            val_s >= train_s * self.config.decay_threshold
            and test_s >= train_s * self.config.decay_threshold
        )
        test_above_threshold = test_s > self.config.min_sharpe

        passed = all_positive and decay_ok and test_above_threshold

        return WalkForwardResult(
            train_result=train_result,
            val_result=val_result,
            test_result=test_result,
            all_positive_sharpe=all_positive,
            decay_acceptable=decay_ok,
            test_sharpe_above_threshold=test_above_threshold,
            passed=passed,
            min_sharpe=self.config.min_sharpe,
            decay_threshold=self.config.decay_threshold,
        )


class RollingWindowValidator:
    """
    Rolling window validation for strategy robustness testing.

    Tests strategy across multiple overlapping time windows
    to ensure consistent performance.

    Example:
        validator = RollingWindowValidator(config)
        result = validator.validate(data, backtest_func)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def validate(
        self,
        data: Dict[str, pd.DataFrame],
        backtest_func: Callable[[Dict[str, pd.DataFrame]], BacktestResult],
    ) -> RollingResult:
        """
        Run rolling window validation.

        Args:
            data: Dictionary of symbol DataFrames
            backtest_func: Function that runs backtest and returns BacktestResult

        Returns:
            RollingResult with validation metrics
        """
        # Get the minimum length across all symbols
        min_length = min(len(df) for df in data.values())

        window_results = []
        window_size = self.config.train_window
        test_size = self.config.test_window
        step = self.config.step

        i = 0
        while i + window_size + test_size <= min_length:
            # Extract window data
            window_data = {}
            for symbol, df in data.items():
                window_data[symbol] = df.iloc[i + window_size : i + window_size + test_size]

            # Run backtest on test window
            result = backtest_func(window_data)

            if result.total_trades > 0:
                window_results.append(result)

            i += step

        # Compute aggregated metrics
        if not window_results:
            return RollingResult(passed=False)

        sharpes = [r.sharpe_ratio for r in window_results]
        positive_count = sum(1 for s in sharpes if s > 0)

        result = RollingResult(
            window_results=window_results,
            total_windows=len(window_results),
            positive_sharpe_windows=positive_count,
            positive_ratio=positive_count / len(window_results),
            avg_sharpe=np.mean(sharpes),
            sharpe_std=np.std(sharpes),
            min_sharpe=min(sharpes),
            max_sharpe=max(sharpes),
            min_positive_ratio=self.config.min_positive_ratio,
        )

        result.passed = result.positive_ratio >= self.config.min_positive_ratio

        return result


def run_walk_forward(
    data: Dict[str, pd.DataFrame],
    backtest_func: Callable[[Dict[str, pd.DataFrame]], BacktestResult],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_sharpe: float = 0.3,
    decay_threshold: float = 0.5,
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward validation.

    Args:
        data: Dictionary of symbol DataFrames
        backtest_func: Backtest function
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        test_ratio: Test data ratio
        min_sharpe: Minimum required test sharpe
        decay_threshold: Maximum allowed performance decay

    Returns:
        WalkForwardResult
    """
    config = ValidationConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_sharpe=min_sharpe,
        decay_threshold=decay_threshold,
    )

    validator = WalkForwardValidator(config)
    return validator.validate(data, backtest_func)


def run_rolling_validation(
    data: Dict[str, pd.DataFrame],
    backtest_func: Callable[[Dict[str, pd.DataFrame]], BacktestResult],
    train_window: int = 2160,
    test_window: int = 720,
    step: int = 720,
    min_positive_ratio: float = 0.6,
) -> RollingResult:
    """
    Convenience function to run rolling window validation.

    Args:
        data: Dictionary of symbol DataFrames
        backtest_func: Backtest function
        train_window: Training window size in bars
        test_window: Test window size in bars
        step: Step size in bars
        min_positive_ratio: Minimum required positive sharpe ratio

    Returns:
        RollingResult
    """
    config = ValidationConfig(
        train_window=train_window,
        test_window=test_window,
        step=step,
        min_positive_ratio=min_positive_ratio,
    )

    validator = RollingWindowValidator(config)
    return validator.validate(data, backtest_func)
