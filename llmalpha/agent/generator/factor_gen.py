"""
Factor Code Generator for LLM Alpha.
"""

from typing import Type

from llmalpha.factors.base import Factor
from llmalpha.agent.generator.base import BaseGenerator
from llmalpha.agent.prompts.templates import FACTOR_GENERATION_PROMPT


class FactorGenerator(BaseGenerator):
    """Generator for Factor classes."""

    def get_prompt_template(self) -> str:
        return FACTOR_GENERATION_PROMPT

    def get_class_name(self, hypothesis_code: str) -> str:
        return f"Generated{hypothesis_code}Factor"

    def get_expected_base_class(self) -> Type:
        return Factor
