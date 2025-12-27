"""
Core components for LLM Alpha Agent.

Contains the research loop, validation, and recovery logic.
"""

from llmalpha.agent.core.loop import (
    ResearchLoop,
    IterationResult,
    LoopResult,
)
from llmalpha.agent.core.validator import (
    HypothesisValidator,
    ValidationOutput,
)
from llmalpha.agent.core.context import (
    ContextWindow,
    IterationRecord,
    estimate_tokens,
)

__all__ = [
    "ResearchLoop",
    "IterationResult",
    "LoopResult",
    "HypothesisValidator",
    "ValidationOutput",
    "ContextWindow",
    "IterationRecord",
    "estimate_tokens",
]
