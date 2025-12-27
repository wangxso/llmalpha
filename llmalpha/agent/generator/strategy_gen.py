"""
Strategy Code Generator for LLM Alpha.
"""

from typing import Type

from llmalpha.strategies.base import Strategy
from llmalpha.agent.generator.base import BaseGenerator
from llmalpha.agent.prompts.templates import STRATEGY_GENERATION_PROMPT


class StrategyGenerator(BaseGenerator):
    """Generator for Strategy classes."""

    def get_prompt_template(self) -> str:
        return STRATEGY_GENERATION_PROMPT

    def get_class_name(self, hypothesis_code: str) -> str:
        return f"Generated{hypothesis_code}Strategy"

    def get_expected_base_class(self) -> Type:
        return Strategy
