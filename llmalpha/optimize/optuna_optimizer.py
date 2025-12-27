"""
Optuna-based Parameter Optimizer

Provides hyperparameter optimization with Optuna,
including walk-forward validation integration.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    TPESampler = None

from llmalpha.backtest.result import BacktestResult


@dataclass
class ParamSpace:
    """Parameter search space definition."""
    name: str
    param_type: str  # "int", "float", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    log: bool = False


@dataclass
class OptimizeResult:
    """Optimization result."""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    n_trials: int = 0
    best_train_result: Optional[BacktestResult] = None
    best_val_result: Optional[BacktestResult] = None
    study: Any = None


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer.

    Features:
    - TPE sampler for efficient search
    - Walk-forward validation integration
    - Multi-objective optimization support
    - Pruning for early stopping

    Example:
        optimizer = OptunaOptimizer(param_space)
        result = optimizer.optimize(
            data=data,
            backtest_func=backtest,
            n_trials=500,
        )
    """

    def __init__(
        self,
        param_space: List[ParamSpace],
        direction: str = "maximize",
        sampler_seed: int = 42,
    ):
        """
        Initialize the optimizer.

        Args:
            param_space: List of parameter search spaces
            direction: Optimization direction ("maximize" or "minimize")
            sampler_seed: Random seed for reproducibility
        """
        if optuna is None:
            raise ImportError("optuna is required. Install with: pip install optuna")

        self.param_space = param_space
        self.direction = direction
        self.sampler_seed = sampler_seed

    def _suggest_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        params = {}

        for p in self.param_space:
            if p.param_type == "int":
                params[p.name] = trial.suggest_int(
                    p.name, int(p.low), int(p.high), step=int(p.step) if p.step else 1
                )
            elif p.param_type == "float":
                params[p.name] = trial.suggest_float(
                    p.name, p.low, p.high, step=p.step, log=p.log
                )
            elif p.param_type == "categorical":
                params[p.name] = trial.suggest_categorical(p.name, p.choices)

        return params

    def optimize(
        self,
        train_data: Dict[str, Any],
        val_data: Dict[str, Any],
        backtest_func: Callable[[Dict[str, Any], Dict[str, Any]], BacktestResult],
        score_func: Optional[Callable[[BacktestResult, BacktestResult], float]] = None,
        n_trials: int = 500,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        show_progress: bool = True,
    ) -> OptimizeResult:
        """
        Run optimization.

        Args:
            train_data: Training data
            val_data: Validation data
            backtest_func: Function(params, data) -> BacktestResult
            score_func: Function(train_result, val_result) -> score
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            study_name: Name for the study (for persistence)
            storage: Storage URL (e.g., "sqlite:///optuna.db")
            load_if_exists: Load existing study if available
            show_progress: Show progress bar

        Returns:
            OptimizeResult with best parameters and metrics
        """
        # Default score function
        if score_func is None:
            score_func = self._default_score

        # Create study
        sampler = TPESampler(seed=self.sampler_seed)

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        # Objective function
        def objective(trial):
            params = self._suggest_params(trial)

            # Run backtests
            train_result = backtest_func(params, train_data)
            val_result = backtest_func(params, val_data)

            # Store results
            trial.set_user_attr("train_trades", train_result.total_trades)
            trial.set_user_attr("train_sharpe", train_result.sharpe_ratio)
            trial.set_user_attr("train_return", train_result.total_return)
            trial.set_user_attr("val_trades", val_result.total_trades)
            trial.set_user_attr("val_sharpe", val_result.sharpe_ratio)
            trial.set_user_attr("val_return", val_result.total_return)

            # Compute score
            return score_func(train_result, val_result)

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )

        # Build result
        best_trial = study.best_trial

        # Re-run best params to get full results
        best_params = best_trial.params
        best_train = backtest_func(best_params, train_data)
        best_val = backtest_func(best_params, val_data)

        return OptimizeResult(
            best_params=best_params,
            best_score=best_trial.value,
            n_trials=len(study.trials),
            best_train_result=best_train,
            best_val_result=best_val,
            study=study,
        )

    def _default_score(
        self,
        train_result: BacktestResult,
        val_result: BacktestResult,
    ) -> float:
        """Default scoring function."""
        # Minimum trade requirement
        if train_result.total_trades < 15 or val_result.total_trades < 5:
            return -1000.0

        # Base score from validation
        score = (
            val_result.total_return * 100
            + min(val_result.sharpe_ratio, 10) * 5
            + val_result.win_rate * 50
            + min(val_result.profit_factor, 5) * 10
            - val_result.max_drawdown * 120
        )

        # Consistency bonus
        pnl_diff = abs(train_result.total_return - val_result.total_return)
        consistency = 1.0 - min(pnl_diff, 0.3) / 0.3
        score += consistency * 20

        # Penalize inconsistent direction
        if (train_result.total_return > 0) != (val_result.total_return > 0):
            score -= 50

        return score


def create_param_space(
    param_ranges: Dict[str, Tuple[float, float]],
    param_types: Optional[Dict[str, str]] = None,
) -> List[ParamSpace]:
    """
    Create parameter space from simple dictionary.

    Args:
        param_ranges: Dict of param_name: (low, high)
        param_types: Dict of param_name: type ("int" or "float")

    Returns:
        List of ParamSpace objects
    """
    param_types = param_types or {}

    space = []
    for name, (low, high) in param_ranges.items():
        ptype = param_types.get(name, "float")
        space.append(ParamSpace(name=name, param_type=ptype, low=low, high=high))

    return space


def run_optimization(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    backtest_func: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    param_types: Optional[Dict[str, str]] = None,
    n_trials: int = 500,
    n_jobs: int = 1,
) -> OptimizeResult:
    """
    Convenience function to run optimization.

    Args:
        train_data: Training data
        val_data: Validation data
        backtest_func: Function(params, data) -> BacktestResult
        param_ranges: Dict of param_name: (low, high)
        param_types: Dict of param_name: type
        n_trials: Number of trials
        n_jobs: Parallel jobs

    Returns:
        OptimizeResult
    """
    param_space = create_param_space(param_ranges, param_types)
    optimizer = OptunaOptimizer(param_space)

    return optimizer.optimize(
        train_data=train_data,
        val_data=val_data,
        backtest_func=backtest_func,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
