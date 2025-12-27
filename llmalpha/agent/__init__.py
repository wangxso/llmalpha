"""
LLM Alpha Agent Module.

Provides LLM-driven autonomous research capabilities for
generating and validating trading strategies.

Usage:
    from llmalpha.agent import AlphaResearcher, create_researcher

    # Quick start
    researcher = create_researcher(provider="openai", model="gpt-5.2")
    result = await researcher.research("Create a momentum strategy using RSI")

    # With full config
    from llmalpha.agent import AgentConfig, LLMConfig, AlphaResearcher

    config = AgentConfig(
        llm=LLMConfig(provider="anthropic", model="claude-3-opus-20240229"),
        max_iterations=10,
        early_stop_sharpe=1.5,
    )
    researcher = AlphaResearcher(config)
    result = await researcher.research("Create mean reversion strategy")
"""

from llmalpha.agent.config import (
    AgentConfig,
    LLMConfig,
    ValidationConfig,
    IterationState,
)
from llmalpha.agent.researcher import (
    AlphaResearcher,
    create_researcher,
)
from llmalpha.agent.core import (
    ResearchLoop,
    IterationResult,
    LoopResult,
    HypothesisValidator,
    ValidationOutput,
    ContextWindow,
    IterationRecord,
)
from llmalpha.agent.generator import (
    BaseGenerator,
    GenerationResult,
    FactorGenerator,
    SignalGenerator,
    StrategyGenerator,
    get_generator,
)
from llmalpha.agent.providers import (
    LLMProvider,
    Message,
    LLMResponse,
    get_provider,
)
from llmalpha.agent.sandbox import (
    SafeExecutor,
    CodeValidator,
    ExecutionResult,
)
from llmalpha.agent.tools import DataTool

__all__ = [
    # Main classes
    "AlphaResearcher",
    "create_researcher",
    # Config
    "AgentConfig",
    "LLMConfig",
    "ValidationConfig",
    "IterationState",
    # Core
    "ResearchLoop",
    "IterationResult",
    "LoopResult",
    "HypothesisValidator",
    "ValidationOutput",
    "ContextWindow",
    "IterationRecord",
    # Generators
    "BaseGenerator",
    "GenerationResult",
    "FactorGenerator",
    "SignalGenerator",
    "StrategyGenerator",
    "get_generator",
    # Providers
    "LLMProvider",
    "Message",
    "LLMResponse",
    "get_provider",
    # Sandbox
    "SafeExecutor",
    "CodeValidator",
    "ExecutionResult",
    # Tools
    "DataTool",
]
