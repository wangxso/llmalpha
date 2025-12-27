"""
Backtest Result Data Classes

Standardized result containers for backtesting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    pnl: float
    pnl_pct: float
    signal_type: str = ""
    symbol: str = ""


@dataclass
class BacktestResult:
    """Comprehensive backtest result."""
    # Core metrics
    total_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0  # Percentage
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0  # Percentage (positive value)
    profit_factor: float = 0.0

    # Additional metrics
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_per_day: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Trade breakdown
    long_trades: int = 0
    short_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Time info
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_days: int = 0

    # Raw data
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    returns_series: Optional[pd.Series] = None

    # Metadata
    strategy_name: str = ""
    symbols: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "trades_per_day": self.trades_per_day,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "strategy_name": self.strategy_name,
            "symbols": self.symbols,
            "params": self.params,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Strategy: {self.strategy_name or 'N/A'}",
            f"Period: {self.start_date} to {self.end_date} ({self.total_days} days)",
            f"Symbols: {', '.join(self.symbols) if self.symbols else 'N/A'}",
            "",
            f"Total Trades: {self.total_trades}",
            f"Win Rate: {self.win_rate:.1%}",
            f"Total Return: {self.total_return:.2%}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
            f"Profit Factor: {self.profit_factor:.2f}",
            "",
            f"Long/Short: {self.long_trades}/{self.short_trades}",
            f"Win/Loss: {self.winning_trades}/{self.losing_trades}",
            f"Avg Trade: {self.avg_trade_return:.2%}",
            f"Trades/Day: {self.trades_per_day:.2f}",
        ]
        return "\n".join(lines)


@dataclass
class WalkForwardResult:
    """Walk-Forward validation result."""
    train_result: BacktestResult
    val_result: BacktestResult
    test_result: BacktestResult

    # Validation checks
    all_positive_sharpe: bool = False
    decay_acceptable: bool = False
    test_sharpe_above_threshold: bool = False
    passed: bool = False

    # Thresholds used
    min_sharpe: float = 0.3
    decay_threshold: float = 0.5

    def summary(self) -> str:
        """Generate validation summary."""
        train_s = self.train_result.sharpe_ratio
        val_s = self.val_result.sharpe_ratio
        test_s = self.test_result.sharpe_ratio

        check_mark = lambda x: "V" if x else "X"

        lines = [
            "Walk-Forward Validation Results",
            "=" * 40,
            f"Train Sharpe: {train_s:.2f}",
            f"Val Sharpe: {val_s:.2f}",
            f"Test Sharpe: {test_s:.2f}",
            "",
            f"[{check_mark(self.all_positive_sharpe)}] All sharpe > 0",
            f"[{check_mark(self.decay_acceptable)}] Val/Test >= Train * {self.decay_threshold}",
            f"[{check_mark(self.test_sharpe_above_threshold)}] Test sharpe > {self.min_sharpe}",
            "",
            f"Overall: {'PASSED' if self.passed else 'FAILED'}",
        ]
        return "\n".join(lines)


@dataclass
class RollingResult:
    """Rolling window validation result."""
    window_results: List[BacktestResult] = field(default_factory=list)

    # Aggregated metrics
    total_windows: int = 0
    positive_sharpe_windows: int = 0
    positive_ratio: float = 0.0
    avg_sharpe: float = 0.0
    sharpe_std: float = 0.0
    min_sharpe: float = 0.0
    max_sharpe: float = 0.0

    # Validation
    passed: bool = False
    min_positive_ratio: float = 0.6

    def summary(self) -> str:
        """Generate rolling validation summary."""
        lines = [
            "Rolling Window Validation Results",
            "=" * 40,
            f"Total Windows: {self.total_windows}",
            f"Positive Sharpe: {self.positive_sharpe_windows}/{self.total_windows} ({self.positive_ratio:.1%})",
            f"Avg Sharpe: {self.avg_sharpe:.2f} (std: {self.sharpe_std:.2f})",
            f"Min/Max Sharpe: {self.min_sharpe:.2f} / {self.max_sharpe:.2f}",
            "",
            f"Threshold: {self.min_positive_ratio:.0%} positive required",
            f"Overall: {'PASSED' if self.passed else 'FAILED'}",
        ]
        return "\n".join(lines)
