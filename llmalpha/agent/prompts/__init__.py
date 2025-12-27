"""
Prompt templates for LLM code generation.
"""

from llmalpha.agent.prompts.templates import (
    SYSTEM_PROMPT,
    FACTOR_GENERATION_PROMPT,
    SIGNAL_GENERATION_PROMPT,
    STRATEGY_GENERATION_PROMPT,
    IMPROVEMENT_PROMPT,
    HYPOTHESIS_SUMMARY_PROMPT,
    DATA_ANALYSIS_PROMPT,
    format_similar_items,
    format_failures,
    format_best_strategies,
)

__all__ = [
    "SYSTEM_PROMPT",
    "FACTOR_GENERATION_PROMPT",
    "SIGNAL_GENERATION_PROMPT",
    "STRATEGY_GENERATION_PROMPT",
    "IMPROVEMENT_PROMPT",
    "HYPOTHESIS_SUMMARY_PROMPT",
    "DATA_ANALYSIS_PROMPT",
    "format_similar_items",
    "format_failures",
    "format_best_strategies",
]
