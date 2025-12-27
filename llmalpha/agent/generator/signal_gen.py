"""
Signal Code Generator for LLM Alpha.
"""

from typing import Type

from llmalpha.signals.base import Signal
from llmalpha.agent.generator.base import BaseGenerator
from llmalpha.agent.prompts.templates import SIGNAL_GENERATION_PROMPT


class SignalGenerator(BaseGenerator):
    """Generator for Signal classes."""

    def get_prompt_template(self) -> str:
        return SIGNAL_GENERATION_PROMPT

    def get_class_name(self, hypothesis_code: str) -> str:
        return f"Generated{hypothesis_code}Signal"

    def get_expected_base_class(self) -> Type:
        return Signal
