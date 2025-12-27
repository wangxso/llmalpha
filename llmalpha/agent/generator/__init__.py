"""
Code Generators for LLM Alpha.

Generates Factor, Signal, and Strategy code using LLM.
"""

from llmalpha.agent.generator.base import BaseGenerator, GenerationResult
from llmalpha.agent.generator.factor_gen import FactorGenerator
from llmalpha.agent.generator.signal_gen import SignalGenerator
from llmalpha.agent.generator.strategy_gen import StrategyGenerator

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "FactorGenerator",
    "SignalGenerator",
    "StrategyGenerator",
]


def get_generator(
    generation_type: str,
    llm,
    executor,
):
    """
    Factory function to create appropriate generator.

    Args:
        generation_type: "factor", "signal", or "strategy"
        llm: LLM provider
        executor: Safe executor

    Returns:
        Generator instance
    """
    generators = {
        "factor": FactorGenerator,
        "signal": SignalGenerator,
        "strategy": StrategyGenerator,
    }

    generator_class = generators.get(generation_type)
    if not generator_class:
        raise ValueError(f"Unknown generation type: {generation_type}")

    return generator_class(llm=llm, executor=executor)
