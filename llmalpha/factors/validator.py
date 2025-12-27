"""
Factor Validation System for LLM Alpha.

Provides comprehensive factor analysis to determine if a factor
has genuine predictive power before using it in signals/strategies.

Key metrics:
- IC (Information Coefficient): Correlation between factor and forward returns
- IR (Information Ratio): IC mean / IC std (stability of IC)
- Turnover: How frequently factor values change (trading cost implications)
- Quantile Returns: Monotonicity test across factor quintiles
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from llmalpha.factors.base import Factor


@dataclass
class FactorStats:
    """Statistical summary of a factor."""

    mean: float
    std: float
    skew: float
    kurtosis: float
    min: float
    max: float
    pct_nan: float  # Percentage of NaN values

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "min": self.min,
            "max": self.max,
            "pct_nan": self.pct_nan,
        }


@dataclass
class ICResult:
    """Information Coefficient analysis result."""

    ic_series: pd.Series  # IC at each time point
    ic_mean: float  # Mean IC
    ic_std: float  # IC standard deviation
    ir: float  # Information Ratio (IC_mean / IC_std)
    ic_positive_pct: float  # Percentage of positive IC
    t_stat: float  # T-statistic for IC != 0
    p_value: float  # P-value

    def is_significant(self, alpha: float = 0.05, min_ir: float = 0.5) -> bool:
        """Check if factor has significant predictive power."""
        return self.p_value < alpha and abs(self.ir) > min_ir

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ir": self.ir,
            "ic_positive_pct": self.ic_positive_pct,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
        }


@dataclass
class QuantileResult:
    """Quantile (group) backtest result."""

    n_quantiles: int
    quantile_returns: pd.DataFrame  # Returns per quantile over time
    mean_returns: pd.Series  # Mean return per quantile
    cumulative_returns: pd.DataFrame  # Cumulative returns per quantile
    long_short_return: float  # Annualized long-short return
    long_short_sharpe: float  # Long-short Sharpe ratio
    monotonicity: float  # Spearman correlation of quantile ranks vs returns
    spread: float  # Return spread between top and bottom quantile

    def is_monotonic(self, min_corr: float = 0.7) -> bool:
        """Check if returns are monotonic across quantiles."""
        return abs(self.monotonicity) >= min_corr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_quantiles": self.n_quantiles,
            "mean_returns": self.mean_returns.to_dict(),
            "long_short_return": self.long_short_return,
            "long_short_sharpe": self.long_short_sharpe,
            "monotonicity": self.monotonicity,
            "spread": self.spread,
        }


@dataclass
class TurnoverResult:
    """Factor turnover analysis result."""

    turnover_series: pd.Series  # Turnover at each rebalance
    mean_turnover: float  # Average turnover
    max_turnover: float  # Maximum turnover
    autocorr: float  # Factor autocorrelation (stability)

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_turnover": self.mean_turnover,
            "max_turnover": self.max_turnover,
            "autocorr": self.autocorr,
        }


@dataclass
class ValidationResult:
    """Complete factor validation result."""

    factor_code: str
    factor_stats: FactorStats
    ic_result: ICResult
    quantile_result: QuantileResult
    turnover_result: TurnoverResult

    # Overall assessment
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    score: float = 0.0  # Overall factor quality score (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_code": self.factor_code,
            "is_valid": self.is_valid,
            "score": self.score,
            "rejection_reasons": self.rejection_reasons,
            "stats": self.factor_stats.to_dict(),
            "ic": self.ic_result.to_dict(),
            "quantile": self.quantile_result.to_dict(),
            "turnover": self.turnover_result.to_dict(),
        }


class FactorValidator:
    """
    Validates factor predictive power through multiple tests.

    Example:
        validator = FactorValidator(forward_periods=24)  # 24h forward return
        result = validator.validate(factor, df)

        if result.is_valid:
            print(f"Factor passed! Score: {result.score}")
        else:
            print(f"Factor rejected: {result.rejection_reasons}")
    """

    def __init__(
        self,
        forward_periods: int = 24,
        n_quantiles: int = 5,
        min_ic: float = 0.02,
        min_ir: float = 0.3,
        min_monotonicity: float = 0.6,
        max_turnover: float = 0.8,
        significance_level: float = 0.05,
    ):
        """
        Initialize the validator.

        Args:
            forward_periods: Number of periods for forward return calculation
            n_quantiles: Number of quantiles for group backtest
            min_ic: Minimum absolute IC required
            min_ir: Minimum IR (IC/IC_std) required
            min_monotonicity: Minimum monotonicity score
            max_turnover: Maximum acceptable turnover
            significance_level: P-value threshold for IC significance
        """
        self.forward_periods = forward_periods
        self.n_quantiles = n_quantiles
        self.min_ic = min_ic
        self.min_ir = min_ir
        self.min_monotonicity = min_monotonicity
        self.max_turnover = max_turnover
        self.significance_level = significance_level

    def compute_forward_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute forward returns for IC calculation."""
        return df["close"].pct_change(self.forward_periods).shift(-self.forward_periods)

    def compute_stats(self, factor_values: pd.Series) -> FactorStats:
        """Compute basic factor statistics."""
        clean = factor_values.dropna()

        return FactorStats(
            mean=clean.mean() if len(clean) > 0 else 0.0,
            std=clean.std() if len(clean) > 0 else 0.0,
            skew=stats.skew(clean) if len(clean) > 2 else 0.0,
            kurtosis=stats.kurtosis(clean) if len(clean) > 3 else 0.0,
            min=clean.min() if len(clean) > 0 else 0.0,
            max=clean.max() if len(clean) > 0 else 0.0,
            pct_nan=factor_values.isna().mean() * 100,
        )

    def compute_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        window: int = 240,
    ) -> ICResult:
        """
        Compute Information Coefficient (IC).

        IC is the Spearman rank correlation between factor values
        and subsequent returns at each time point.
        """
        # Align data
        aligned = pd.DataFrame({
            "factor": factor_values,
            "returns": forward_returns,
        }).dropna()

        if len(aligned) < window * 2:
            # Not enough data - return empty result
            return ICResult(
                ic_series=pd.Series(dtype=float),
                ic_mean=0.0,
                ic_std=1.0,
                ir=0.0,
                ic_positive_pct=0.0,
                t_stat=0.0,
                p_value=1.0,
            )

        # Rolling IC calculation
        ic_list = []
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i - window:i]
            ic, _ = stats.spearmanr(window_data["factor"], window_data["returns"])
            ic_list.append(ic if not np.isnan(ic) else 0.0)

        ic_series = pd.Series(ic_list, index=aligned.index[window:])

        # Calculate statistics
        ic_mean = ic_series.mean()
        ic_std = ic_series.std() + 1e-10
        ir = ic_mean / ic_std
        ic_positive_pct = (ic_series > 0).mean() * 100

        # T-test for IC != 0
        n = len(ic_series)
        t_stat = ic_mean / (ic_std / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

        return ICResult(
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_positive_pct=ic_positive_pct,
            t_stat=t_stat,
            p_value=p_value,
        )

    def compute_quantile_returns(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: int = 5,
    ) -> QuantileResult:
        """
        Compute returns by factor quantile for monotonicity analysis.

        Divides the factor into quantiles and tracks returns for each
        group to verify that higher factor values lead to better/worse returns.
        """
        aligned = pd.DataFrame({
            "factor": factor_values,
            "returns": forward_returns,
        }).dropna()

        if len(aligned) < n_quantiles * 10:
            # Not enough data
            empty_returns = pd.Series([0.0] * n_quantiles, index=range(1, n_quantiles + 1))
            return QuantileResult(
                n_quantiles=n_quantiles,
                quantile_returns=pd.DataFrame(),
                mean_returns=empty_returns,
                cumulative_returns=pd.DataFrame(),
                long_short_return=0.0,
                long_short_sharpe=0.0,
                monotonicity=0.0,
                spread=0.0,
            )

        # Assign quantiles
        aligned["quantile"] = pd.qcut(
            aligned["factor"],
            q=n_quantiles,
            labels=range(1, n_quantiles + 1),
            duplicates="drop"
        )

        # Compute returns per quantile
        quantile_returns = aligned.groupby("quantile")["returns"].mean()

        # Long-short portfolio (top quantile - bottom quantile)
        long_short = quantile_returns.iloc[-1] - quantile_returns.iloc[0]

        # Annualized return (assuming hourly data, 8760 hours/year)
        periods_per_year = 8760 / self.forward_periods
        long_short_annual = long_short * periods_per_year

        # Long-short Sharpe (simplified)
        ls_series = aligned[aligned["quantile"] == n_quantiles]["returns"] - \
                    aligned[aligned["quantile"] == 1]["returns"]
        ls_sharpe = ls_series.mean() / (ls_series.std() + 1e-10) * np.sqrt(periods_per_year)

        # Monotonicity: Spearman correlation of quantile rank vs mean return
        monotonicity, _ = stats.spearmanr(
            range(1, len(quantile_returns) + 1),
            quantile_returns.values
        )

        # Spread
        spread = quantile_returns.iloc[-1] - quantile_returns.iloc[0]

        return QuantileResult(
            n_quantiles=n_quantiles,
            quantile_returns=pd.DataFrame(),  # Could add time series here
            mean_returns=quantile_returns,
            cumulative_returns=pd.DataFrame(),
            long_short_return=long_short_annual,
            long_short_sharpe=ls_sharpe if not np.isnan(ls_sharpe) else 0.0,
            monotonicity=monotonicity if not np.isnan(monotonicity) else 0.0,
            spread=spread,
        )

    def compute_turnover(
        self,
        factor_values: pd.Series,
        rebalance_period: int = 24,
    ) -> TurnoverResult:
        """
        Compute factor turnover.

        High turnover means the factor changes frequently, leading to
        higher trading costs when used in a strategy.
        """
        clean = factor_values.dropna()

        if len(clean) < rebalance_period * 2:
            return TurnoverResult(
                turnover_series=pd.Series(dtype=float),
                mean_turnover=0.0,
                max_turnover=0.0,
                autocorr=0.0,
            )

        # Rank at each rebalance point
        ranks = []
        indices = []
        for i in range(0, len(clean), rebalance_period):
            if i + rebalance_period <= len(clean):
                window = clean.iloc[i:i + rebalance_period]
                rank = (window - window.min()) / (window.max() - window.min() + 1e-10)
                ranks.append(rank.iloc[-1])
                indices.append(clean.index[i + rebalance_period - 1])

        if len(ranks) < 2:
            return TurnoverResult(
                turnover_series=pd.Series(dtype=float),
                mean_turnover=0.0,
                max_turnover=0.0,
                autocorr=0.0,
            )

        ranks_series = pd.Series(ranks, index=indices)

        # Turnover = absolute change in rank
        turnover = ranks_series.diff().abs().dropna()

        # Autocorrelation (stability)
        autocorr = factor_values.autocorr(lag=rebalance_period)

        return TurnoverResult(
            turnover_series=turnover,
            mean_turnover=turnover.mean(),
            max_turnover=turnover.max(),
            autocorr=autocorr if not np.isnan(autocorr) else 0.0,
        )

    def validate(
        self,
        factor: Factor,
        df: pd.DataFrame,
    ) -> ValidationResult:
        """
        Run complete factor validation.

        Args:
            factor: Factor instance to validate
            df: DataFrame with OHLCV data

        Returns:
            ValidationResult with all metrics and pass/fail status
        """
        # Compute factor values
        factor_values = factor.compute(df)

        # Compute forward returns
        forward_returns = self.compute_forward_returns(df)

        # Run all analyses
        factor_stats = self.compute_stats(factor_values)
        ic_result = self.compute_ic(factor_values, forward_returns)
        quantile_result = self.compute_quantile_returns(
            factor_values, forward_returns, self.n_quantiles
        )
        turnover_result = self.compute_turnover(factor_values)

        # Validation checks
        rejection_reasons = []

        # Check IC
        if abs(ic_result.ic_mean) < self.min_ic:
            rejection_reasons.append(
                f"IC too low: {ic_result.ic_mean:.4f} < {self.min_ic}"
            )

        # Check IR
        if abs(ic_result.ir) < self.min_ir:
            rejection_reasons.append(
                f"IR too low: {ic_result.ir:.4f} < {self.min_ir}"
            )

        # Check significance
        if ic_result.p_value > self.significance_level:
            rejection_reasons.append(
                f"IC not significant: p={ic_result.p_value:.4f} > {self.significance_level}"
            )

        # Check monotonicity
        if abs(quantile_result.monotonicity) < self.min_monotonicity:
            rejection_reasons.append(
                f"Poor monotonicity: {quantile_result.monotonicity:.4f} < {self.min_monotonicity}"
            )

        # Check turnover
        if turnover_result.mean_turnover > self.max_turnover:
            rejection_reasons.append(
                f"Turnover too high: {turnover_result.mean_turnover:.4f} > {self.max_turnover}"
            )

        # Check for too many NaNs
        if factor_stats.pct_nan > 50:
            rejection_reasons.append(
                f"Too many NaN values: {factor_stats.pct_nan:.1f}%"
            )

        # Calculate overall score (0-100)
        score = self._calculate_score(ic_result, quantile_result, turnover_result)

        is_valid = len(rejection_reasons) == 0

        return ValidationResult(
            factor_code=factor.code,
            factor_stats=factor_stats,
            ic_result=ic_result,
            quantile_result=quantile_result,
            turnover_result=turnover_result,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            score=score,
        )

    def _calculate_score(
        self,
        ic_result: ICResult,
        quantile_result: QuantileResult,
        turnover_result: TurnoverResult,
    ) -> float:
        """Calculate overall factor quality score (0-100)."""
        score = 0.0

        # IC contribution (max 30 points)
        ic_score = min(abs(ic_result.ic_mean) / 0.1 * 30, 30)
        score += ic_score

        # IR contribution (max 25 points)
        ir_score = min(abs(ic_result.ir) / 1.0 * 25, 25)
        score += ir_score

        # Monotonicity contribution (max 25 points)
        mono_score = abs(quantile_result.monotonicity) * 25
        score += mono_score

        # Turnover penalty (max -20 points, bonus up to +10)
        if turnover_result.mean_turnover > 0:
            turnover_penalty = min(turnover_result.mean_turnover * 20, 20)
            score -= turnover_penalty

        # Stability bonus from autocorrelation
        stability_bonus = max(0, turnover_result.autocorr * 10)
        score += stability_bonus

        return max(0, min(100, score))

    def validate_batch(
        self,
        factors: List[Factor],
        df: pd.DataFrame,
        min_score: float = 50.0,
    ) -> List[ValidationResult]:
        """
        Validate multiple factors and return sorted by score.

        Args:
            factors: List of factors to validate
            df: DataFrame with OHLCV data
            min_score: Minimum score to include in results

        Returns:
            List of ValidationResults sorted by score (descending)
        """
        results = []
        for factor in factors:
            result = self.validate(factor, df)
            if result.score >= min_score:
                results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results


def quick_validate(
    factor: Factor,
    df: pd.DataFrame,
    forward_periods: int = 24,
) -> Tuple[bool, str]:
    """
    Quick validation check for a factor.

    Args:
        factor: Factor to validate
        df: DataFrame with OHLCV data
        forward_periods: Forward return periods

    Returns:
        Tuple of (is_valid, summary_message)
    """
    validator = FactorValidator(forward_periods=forward_periods)
    result = validator.validate(factor, df)

    if result.is_valid:
        return True, f"Factor {factor.code} passed (score: {result.score:.1f}, IC: {result.ic_result.ic_mean:.4f}, IR: {result.ic_result.ir:.2f})"
    else:
        return False, f"Factor {factor.code} failed: {', '.join(result.rejection_reasons)}"
