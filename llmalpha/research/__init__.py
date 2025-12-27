"""
Research layer for LLM Alpha.

Provides:
- HypothesisTest: Base class for hypothesis testing
- HypothesisTestResult: Result container for hypothesis tests
- RelativeValueTest: Example implementation (H026)
- Research utilities and templates
"""

from llmalpha.research.hypothesis import (
    HypothesisTest,
    HypothesisTestResult,
    RelativeValueTest,
    run_hypothesis_test,
)

__all__ = [
    "HypothesisTest",
    "HypothesisTestResult",
    "RelativeValueTest",
    "run_hypothesis_test",
]
