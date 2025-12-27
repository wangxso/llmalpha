"""
Optimization layer for LLM Alpha.

Provides:
- OptunaOptimizer: Parameter optimization with Optuna
- WalkForwardValidator: Walk-forward validation
- RollingWindowValidator: Rolling window validation
"""

from llmalpha.optimize.validator import (
    ValidationConfig,
    WalkForwardValidator,
    RollingWindowValidator,
    run_walk_forward,
    run_rolling_validation,
)
from llmalpha.optimize.optuna_optimizer import (
    OptunaOptimizer,
    OptimizeResult,
    ParamSpace,
    create_param_space,
    run_optimization,
)

__all__ = [
    "ValidationConfig",
    "WalkForwardValidator",
    "RollingWindowValidator",
    "run_walk_forward",
    "run_rolling_validation",
    "OptunaOptimizer",
    "OptimizeResult",
    "ParamSpace",
    "create_param_space",
    "run_optimization",
]
