"""
Signal Validation System for LLM Alpha.

Validates signal quality before using it in strategies.

Key metrics:
- Win Rate: Percentage of profitable signals
- Profit Factor: Gross profit / Gross loss
- Average Win/Loss: Risk-reward ratio
- Signal Frequency: Number of signals per period
- Signal Decay: Performance degradation over holding periods
- Edge Ratio: (Avg Win * Win Rate) / (Avg Loss * Loss Rate)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llmalpha.signals.base import Signal, SignalResult


@dataclass
class SignalStats:
    """Basic signal statistics."""

    total_signals: int  # Total entry signals
    signal_frequency: float  # Signals per 1000 bars
    avg_holding_bars: float  # Average bars between entry and exit
    long_signals: int  # Long entry count
    short_signals: int  # Short entry count (if applicable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_signals": self.total_signals,
            "signal_frequency": self.signal_frequency,
            "avg_holding_bars": self.avg_holding_bars,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
        }


@dataclass
class TradeResult:
    """Single trade result."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    return_pct: float
    holding_bars: int
    is_long: bool


@dataclass
class PerformanceMetrics:
    """Signal performance metrics."""

    win_rate: float  # Percentage of winning trades
    loss_rate: float  # Percentage of losing trades
    avg_win: float  # Average winning return
    avg_loss: float  # Average losing return (negative)
    profit_factor: float  # Gross profit / Gross loss
    edge_ratio: float  # Expected value per trade
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_return: float  # Average return per trade
    total_return: float  # Sum of all returns
    sharpe_ratio: float  # Risk-adjusted return

    def to_dict(self) -> Dict[str, float]:
        return {
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "edge_ratio": self.edge_ratio,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_return": self.avg_return,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
        }


@dataclass
class DecayAnalysis:
    """Signal decay analysis across holding periods."""

    holding_periods: List[int]  # [1, 2, 4, 8, 12, 24, ...]
    returns_by_period: Dict[int, float]  # Average return at each holding period
    optimal_holding: int  # Best holding period
    decay_rate: float  # How fast returns decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "holding_periods": self.holding_periods,
            "returns_by_period": self.returns_by_period,
            "optimal_holding": self.optimal_holding,
            "decay_rate": self.decay_rate,
        }


@dataclass
class SignalValidationResult:
    """Complete signal validation result."""

    signal_code: str
    signal_stats: SignalStats
    performance: PerformanceMetrics
    decay_analysis: DecayAnalysis
    trades: List[TradeResult]

    # Overall assessment
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    score: float = 0.0  # Overall signal quality score (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_code": self.signal_code,
            "is_valid": self.is_valid,
            "score": self.score,
            "rejection_reasons": self.rejection_reasons,
            "stats": self.signal_stats.to_dict(),
            "performance": self.performance.to_dict(),
            "decay": self.decay_analysis.to_dict(),
            "trade_count": len(self.trades),
        }


class SignalValidator:
    """
    Validates signal quality through trade simulation.

    Example:
        validator = SignalValidator(min_trades=30)
        result = validator.validate(signal, df)

        if result.is_valid:
            print(f"Signal passed! Win rate: {result.performance.win_rate:.1%}")
        else:
            print(f"Signal rejected: {result.rejection_reasons}")
    """

    def __init__(
        self,
        min_trades: int = 30,
        min_win_rate: float = 0.40,
        min_profit_factor: float = 1.0,
        min_edge_ratio: float = 0.001,
        max_holding_bars: int = 168,  # 7 days at hourly
        fee_pct: float = 0.0008,  # 0.08% fee
        slippage_pct: float = 0.0007,  # 0.07% slippage
    ):
        """
        Initialize the validator.

        Args:
            min_trades: Minimum number of trades for valid analysis
            min_win_rate: Minimum win rate required
            min_profit_factor: Minimum profit factor (gross profit / gross loss)
            min_edge_ratio: Minimum edge ratio (expected value per trade)
            max_holding_bars: Maximum holding period for exit
            fee_pct: Trading fee percentage
            slippage_pct: Slippage percentage
        """
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_edge_ratio = min_edge_ratio
        self.max_holding_bars = max_holding_bars
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.total_cost = fee_pct + slippage_pct

    def simulate_trades(
        self,
        signal_result: SignalResult,
        df: pd.DataFrame,
    ) -> List[TradeResult]:
        """
        Simulate trades from signal entries and exits.

        Args:
            signal_result: SignalResult with entries and exits
            df: DataFrame with OHLCV data

        Returns:
            List of TradeResult
        """
        trades = []
        close = df["close"].values
        entries = signal_result.entries.values
        exits = signal_result.exits.values

        in_position = False
        entry_idx = 0
        entry_price = 0.0

        for i in range(len(df)):
            if not in_position and entries[i]:
                # Enter position
                in_position = True
                entry_idx = i
                entry_price = close[i]

            elif in_position:
                # Check exit conditions
                should_exit = (
                    exits[i] or
                    (i - entry_idx) >= self.max_holding_bars
                )

                if should_exit:
                    exit_price = close[i]
                    return_pct = (exit_price / entry_price - 1) - self.total_cost * 2

                    trades.append(TradeResult(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=return_pct,
                        holding_bars=i - entry_idx,
                        is_long=True,
                    ))

                    in_position = False

        # Handle short signals if present
        if signal_result.short_entries is not None and signal_result.short_exits is not None:
            short_entries = signal_result.short_entries.values
            short_exits = signal_result.short_exits.values

            in_position = False
            for i in range(len(df)):
                if not in_position and short_entries[i]:
                    in_position = True
                    entry_idx = i
                    entry_price = close[i]

                elif in_position:
                    should_exit = (
                        short_exits[i] or
                        (i - entry_idx) >= self.max_holding_bars
                    )

                    if should_exit:
                        exit_price = close[i]
                        # Short: profit when price goes down
                        return_pct = (entry_price / exit_price - 1) - self.total_cost * 2

                        trades.append(TradeResult(
                            entry_idx=entry_idx,
                            exit_idx=i,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            return_pct=return_pct,
                            holding_bars=i - entry_idx,
                            is_long=False,
                        ))

                        in_position = False

        return trades

    def compute_stats(
        self,
        signal_result: SignalResult,
        trades: List[TradeResult],
        n_bars: int,
    ) -> SignalStats:
        """Compute basic signal statistics."""
        long_signals = signal_result.entries.sum()
        short_signals = 0
        if signal_result.short_entries is not None:
            short_signals = signal_result.short_entries.sum()

        total_signals = long_signals + short_signals
        avg_holding = np.mean([t.holding_bars for t in trades]) if trades else 0

        return SignalStats(
            total_signals=int(total_signals),
            signal_frequency=total_signals / n_bars * 1000,
            avg_holding_bars=avg_holding,
            long_signals=int(long_signals),
            short_signals=int(short_signals),
        )

    def compute_performance(self, trades: List[TradeResult]) -> PerformanceMetrics:
        """Compute performance metrics from trades."""
        if not trades:
            return PerformanceMetrics(
                win_rate=0.0,
                loss_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                edge_ratio=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_return=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
            )

        returns = [t.return_pct for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = len(wins) / len(returns) if returns else 0
        loss_rate = len(losses) / len(returns) if returns else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Edge ratio: expected value per trade
        edge_ratio = win_rate * avg_win + loss_rate * avg_loss

        # Consecutive wins/losses
        max_wins = max_losses = current_wins = current_losses = 0
        for r in returns:
            if r > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        avg_return = np.mean(returns)
        total_return = sum(returns)

        # Simple Sharpe (assuming ~2000 trades/year equivalent)
        std_return = np.std(returns) + 1e-10
        sharpe_ratio = avg_return / std_return * np.sqrt(252)

        return PerformanceMetrics(
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            edge_ratio=edge_ratio,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            avg_return=avg_return,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
        )

    def analyze_decay(
        self,
        signal_result: SignalResult,
        df: pd.DataFrame,
        holding_periods: Optional[List[int]] = None,
    ) -> DecayAnalysis:
        """
        Analyze how signal performance decays with holding period.

        Tests the signal at different holding periods to find optimal exit timing.
        """
        if holding_periods is None:
            holding_periods = [1, 2, 4, 8, 12, 24, 48, 72, 168]

        close = df["close"].values
        entries = signal_result.entries.values
        entry_indices = np.where(entries)[0]

        returns_by_period = {}

        for period in holding_periods:
            period_returns = []

            for entry_idx in entry_indices:
                exit_idx = entry_idx + period
                if exit_idx < len(close):
                    ret = (close[exit_idx] / close[entry_idx] - 1) - self.total_cost * 2
                    period_returns.append(ret)

            if period_returns:
                returns_by_period[period] = np.mean(period_returns)
            else:
                returns_by_period[period] = 0.0

        # Find optimal holding period
        if returns_by_period:
            optimal_holding = max(returns_by_period, key=returns_by_period.get)
        else:
            optimal_holding = holding_periods[0]

        # Calculate decay rate (how fast returns decrease after optimal)
        decay_rate = 0.0
        if len(returns_by_period) > 1:
            periods = sorted(returns_by_period.keys())
            returns_arr = [returns_by_period[p] for p in periods]
            if len(returns_arr) > 1:
                # Simple linear decay estimation
                max_return = max(returns_arr)
                if max_return > 0:
                    min_return = min(returns_arr)
                    decay_rate = (max_return - min_return) / max_return

        return DecayAnalysis(
            holding_periods=holding_periods,
            returns_by_period=returns_by_period,
            optimal_holding=optimal_holding,
            decay_rate=decay_rate,
        )

    def validate(
        self,
        signal: Signal,
        df: pd.DataFrame,
    ) -> SignalValidationResult:
        """
        Run complete signal validation.

        Args:
            signal: Signal instance to validate
            df: DataFrame with OHLCV data

        Returns:
            SignalValidationResult with all metrics and pass/fail status
        """
        # Generate signals
        signal_result = signal.generate(df)

        # Simulate trades
        trades = self.simulate_trades(signal_result, df)

        # Compute metrics
        stats = self.compute_stats(signal_result, trades, len(df))
        performance = self.compute_performance(trades)
        decay = self.analyze_decay(signal_result, df)

        # Validation checks
        rejection_reasons = []

        # Check minimum trades
        if len(trades) < self.min_trades:
            rejection_reasons.append(
                f"Not enough trades: {len(trades)} < {self.min_trades}"
            )

        # Check win rate
        if performance.win_rate < self.min_win_rate:
            rejection_reasons.append(
                f"Win rate too low: {performance.win_rate:.1%} < {self.min_win_rate:.1%}"
            )

        # Check profit factor
        if performance.profit_factor < self.min_profit_factor:
            rejection_reasons.append(
                f"Profit factor too low: {performance.profit_factor:.2f} < {self.min_profit_factor}"
            )

        # Check edge ratio
        if performance.edge_ratio < self.min_edge_ratio:
            rejection_reasons.append(
                f"Edge ratio too low: {performance.edge_ratio:.4f} < {self.min_edge_ratio}"
            )

        # Check for signal frequency (too few or too many)
        if stats.signal_frequency < 1:
            rejection_reasons.append(
                f"Signal too rare: {stats.signal_frequency:.2f} per 1000 bars"
            )
        elif stats.signal_frequency > 500:
            rejection_reasons.append(
                f"Signal too frequent: {stats.signal_frequency:.2f} per 1000 bars (likely noise)"
            )

        # Calculate score
        score = self._calculate_score(performance, stats, decay)

        is_valid = len(rejection_reasons) == 0

        return SignalValidationResult(
            signal_code=signal.code,
            signal_stats=stats,
            performance=performance,
            decay_analysis=decay,
            trades=trades,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            score=score,
        )

    def _calculate_score(
        self,
        performance: PerformanceMetrics,
        stats: SignalStats,
        decay: DecayAnalysis,
    ) -> float:
        """Calculate overall signal quality score (0-100)."""
        score = 0.0

        # Win rate contribution (max 25 points)
        win_score = min(performance.win_rate / 0.6 * 25, 25)
        score += win_score

        # Profit factor contribution (max 25 points)
        pf_score = min((performance.profit_factor - 1) / 1.5 * 25, 25)
        score += max(0, pf_score)

        # Edge ratio contribution (max 20 points)
        edge_score = min(performance.edge_ratio / 0.01 * 20, 20)
        score += max(0, edge_score)

        # Sharpe contribution (max 15 points)
        sharpe_score = min(performance.sharpe_ratio / 2.0 * 15, 15)
        score += max(0, sharpe_score)

        # Trade frequency bonus (max 10 points) - prefer moderate frequency
        freq = stats.signal_frequency
        if 10 < freq < 100:
            freq_score = 10
        elif 5 < freq <= 10 or 100 <= freq < 200:
            freq_score = 5
        else:
            freq_score = 0
        score += freq_score

        # Decay penalty (up to -15 points)
        if decay.decay_rate > 0.5:
            score -= (decay.decay_rate - 0.5) * 30

        return max(0, min(100, score))


def quick_validate_signal(
    signal: Signal,
    df: pd.DataFrame,
) -> Tuple[bool, str]:
    """
    Quick validation check for a signal.

    Args:
        signal: Signal to validate
        df: DataFrame with OHLCV data

    Returns:
        Tuple of (is_valid, summary_message)
    """
    validator = SignalValidator()
    result = validator.validate(signal, df)

    if result.is_valid:
        return True, (
            f"Signal {signal.code} passed (score: {result.score:.1f}, "
            f"win rate: {result.performance.win_rate:.1%}, "
            f"PF: {result.performance.profit_factor:.2f})"
        )
    else:
        return False, f"Signal {signal.code} failed: {', '.join(result.rejection_reasons)}"
