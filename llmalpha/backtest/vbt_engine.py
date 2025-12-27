"""
VectorBT-based Backtesting Engine

Provides a clean interface for VectorBT-based backtesting
with support for various signal types and portfolio management.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

from llmalpha.backtest.result import BacktestResult


class VBTEngine:
    """
    VectorBT-based backtesting engine.

    Provides a simplified interface for running backtests with VectorBT,
    with built-in support for:
    - Long and short positions
    - Multiple symbols
    - Walk-forward validation
    - Rolling window validation

    Example:
        engine = VBTEngine(init_cash=10000, fees=0.0008)
        result = engine.run(
            prices=price_df,
            entries=entry_signals,
            exits=exit_signals,
        )
    """

    def __init__(
        self,
        init_cash: float = 10000,
        fees: float = 0.0008,
        slippage: float = 0.0007,
        size: float = 1.0,
        size_type: str = "percent",
        freq: str = "1h",
    ):
        """
        Initialize the VBT engine.

        Args:
            init_cash: Initial capital
            fees: Trading fees as decimal (0.0008 = 0.08%)
            slippage: Slippage as decimal
            size: Position size
            size_type: Size type ("percent", "amount", "value")
            freq: Data frequency
        """
        if vbt is None:
            raise ImportError("vectorbt is required. Install with: pip install vectorbt")

        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.size = size
        self.size_type = size_type
        self.freq = freq

    def run(
        self,
        close: Union[pd.Series, pd.DataFrame],
        entries: Optional[Union[pd.Series, pd.DataFrame]] = None,
        exits: Optional[Union[pd.Series, pd.DataFrame]] = None,
        short_entries: Optional[Union[pd.Series, pd.DataFrame]] = None,
        short_exits: Optional[Union[pd.Series, pd.DataFrame]] = None,
        strategy_name: str = "",
    ) -> BacktestResult:
        """
        Run a backtest with the given signals.

        Args:
            close: Close prices (Series for single symbol, DataFrame for multiple)
            entries: Long entry signals
            exits: Long exit signals
            short_entries: Short entry signals
            short_exits: Short exit signals
            strategy_name: Name for the strategy

        Returns:
            BacktestResult with all metrics
        """
        # Convert to numpy for VBT
        close_values = close.values if hasattr(close, 'values') else close

        # Build portfolio arguments
        pf_args = {
            "close": close_values,
            "size": self.size,
            "size_type": self.size_type,
            "init_cash": self.init_cash,
            "fees": self.fees,
            "slippage": self.slippage,
            "freq": self.freq,
        }

        # Add signals
        if entries is not None:
            pf_args["entries"] = entries.values if hasattr(entries, 'values') else entries
        if exits is not None:
            pf_args["exits"] = exits.values if hasattr(exits, 'values') else exits
        if short_entries is not None:
            pf_args["short_entries"] = short_entries.values if hasattr(short_entries, 'values') else short_entries
        if short_exits is not None:
            pf_args["short_exits"] = short_exits.values if hasattr(short_exits, 'values') else short_exits

        # Run backtest
        pf = vbt.Portfolio.from_signals(**pf_args)

        # Extract stats
        stats = pf.stats()

        # Build result
        result = BacktestResult(
            total_trades=int(stats.get("Total Trades", 0)),
            win_rate=stats.get("Win Rate [%]", 0) / 100,
            total_return=stats.get("Total Return [%]", 0) / 100,
            sharpe_ratio=stats.get("Sharpe Ratio", 0),
            max_drawdown=abs(stats.get("Max Drawdown [%]", 0)) / 100,
            profit_factor=stats.get("Profit Factor", 0) if not np.isnan(stats.get("Profit Factor", 0)) else 0,
            strategy_name=strategy_name,
        )

        # Add equity curve if available
        try:
            result.equity_curve = pf.value()
            result.returns_series = pf.returns()
        except Exception:
            pass

        # Add time info
        if hasattr(close, 'index') and len(close) > 0:
            result.start_date = close.index[0]
            result.end_date = close.index[-1]
            result.total_days = (close.index[-1] - close.index[0]).days

        return result

    def run_multi_symbol(
        self,
        data: Dict[str, pd.DataFrame],
        signal_func: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
        combine_method: str = "equal_weight",
        strategy_name: str = "",
    ) -> BacktestResult:
        """
        Run backtest across multiple symbols.

        Args:
            data: Dictionary of symbol DataFrames
            signal_func: Function that takes DataFrame and returns (entries, exits)
            combine_method: How to combine results ("equal_weight", "sum")
            strategy_name: Name for the strategy

        Returns:
            Combined BacktestResult
        """
        all_returns = []
        total_trades = 0

        for symbol, df in data.items():
            if "close" not in df.columns:
                continue

            # Generate signals
            entries, exits = signal_func(df)

            # Run backtest
            result = self.run(
                close=df["close"],
                entries=entries,
                exits=exits,
                strategy_name=f"{strategy_name}_{symbol}",
            )

            if result.returns_series is not None:
                all_returns.append(result.returns_series)
            total_trades += result.total_trades

        if not all_returns:
            return BacktestResult(strategy_name=strategy_name)

        # Combine returns
        returns_df = pd.DataFrame(all_returns).T

        if combine_method == "equal_weight":
            combined_returns = returns_df.mean(axis=1)
        else:  # sum
            combined_returns = returns_df.sum(axis=1)

        # Calculate combined metrics
        sharpe = (
            combined_returns.mean() / combined_returns.std() * np.sqrt(365 * 24)
            if combined_returns.std() > 0 else 0
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
            strategy_name=strategy_name,
            symbols=list(data.keys()),
        )


def run_vbt_backtest(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    short_entries: Optional[pd.DataFrame] = None,
    short_exits: Optional[pd.DataFrame] = None,
    init_cash: float = 10000,
    fees: float = 0.0008,
    slippage: float = 0.0007,
    freq: str = "1h",
) -> BacktestResult:
    """
    Convenience function to run a VBT backtest.

    Args:
        prices: DataFrame of close prices (columns = symbols)
        entries: DataFrame of entry signals
        exits: DataFrame of exit signals
        short_entries: DataFrame of short entry signals
        short_exits: DataFrame of short exit signals
        init_cash: Initial capital
        fees: Trading fees
        slippage: Slippage
        freq: Data frequency

    Returns:
        BacktestResult
    """
    engine = VBTEngine(
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )

    return engine.run(
        close=prices,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )
