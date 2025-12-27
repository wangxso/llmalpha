"""
Factors layer for LLM Alpha.

Provides:
- Factor: Base class for all factors
- FactorRegistry: Registry for factor discovery
- FactorValidator: Validates factor predictive power (IC, IR, monotonicity)
- Common factors: ZScore, RSI, ATR, etc.

Hierarchy:
    Factor (computes a value) → Signal (generates buy/sell) → Strategy (position sizing)
"""

from llmalpha.factors.base import (
    Factor,
    FactorMeta,
    FactorRegistry,
    register,
    get_factor,
    list_factors,
    # Common factors
    RobustZScoreFactor,
    RSIFactor,
    ATRFactor,
    VolumeZScoreFactor,
    OIChangeFactor,
    FundingZScoreFactor,
    ImbalanceFactor,
    PremiumFactor,
)
from llmalpha.factors.validator import (
    FactorValidator,
    ValidationResult,
    ICResult,
    QuantileResult,
    TurnoverResult,
    FactorStats,
    quick_validate,
)

__all__ = [
    # Core
    "Factor",
    "FactorMeta",
    "FactorRegistry",
    "register",
    "get_factor",
    "list_factors",
    # Validation
    "FactorValidator",
    "ValidationResult",
    "ICResult",
    "QuantileResult",
    "TurnoverResult",
    "FactorStats",
    "quick_validate",
    # Common factors
    "RobustZScoreFactor",
    "RSIFactor",
    "ATRFactor",
    "VolumeZScoreFactor",
    "OIChangeFactor",
    "FundingZScoreFactor",
    "ImbalanceFactor",
    "PremiumFactor",
]
