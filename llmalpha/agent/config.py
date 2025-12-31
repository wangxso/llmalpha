"""
Agent Configuration for LLM Alpha.

Defines configuration classes for the LLM research agent.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "ollama", "deepseek", "qwen"] = "openai"
    model: str = "gpt-5.2"
    temperature: float = 0.5
    max_tokens: int = 16000

    # API credentials (can be overridden by environment variables)
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama or custom endpoints


class ValidationConfig(BaseModel):
    """Validation settings."""

    mode: Literal["quick", "full", "wf_only", "rolling_only"] = "full"
    min_sharpe: float = 0.5
    min_trades: int = 50
    decay_threshold: float = 0.5
    min_positive_ratio: float = 0.6


class AgentConfig(BaseModel):
    """
    Main Agent configuration.

    Example:
        config = AgentConfig(
            llm=LLMConfig(provider="openai", model="gpt-4-turbo-preview"),
            max_iterations=10,
        )
    """

    # LLM settings
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Iteration settings
    max_iterations: int = 20
    max_consecutive_failures: int = 5
    improvement_threshold: float = 0.1  # 10% improvement threshold

    # Generation settings
    generation_type: Literal["factor", "signal", "strategy"] = "strategy"
    category: Optional[str] = None  # momentum, mean_reversion, etc.

    # Validation settings
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # Early stopping conditions
    early_stop_sharpe: float = 1.5  # Stop if Sharpe reaches this
    early_stop_success_count: int = 3  # Stop after N consecutive successes

    # Safety settings
    sandbox_timeout: int = 120  # Code execution timeout in seconds
    max_code_lines: int = 200  # Maximum lines of generated code

    # Knowledge base settings
    learn_from_failures: bool = True
    include_similar_hypotheses: int = 5  # Context for LLM

    # Paths
    data_dir: str = "data"
    db_path: str = "data/knowledge.db"

    @classmethod
    def from_yaml(cls, path: Union[Path, str]) -> "AgentConfig":
        """
        Load AgentConfig from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            AgentConfig instance with values from YAML, or defaults if file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Extract agent section from YAML
        agent_data = data.get("agent", {})
        if not agent_data:
            return cls()

        return cls(**agent_data)


@dataclass
class IterationState:
    """State tracking for research iterations."""

    iteration: int = 0
    best_sharpe: float = 0.0
    best_hypothesis_code: Optional[str] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        """Reset state for a new research session."""
        self.iteration = 0
        self.best_sharpe = 0.0
        self.best_hypothesis_code = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.history = []

    def record_iteration(
        self,
        hypothesis_code: str,
        passed: bool,
        sharpe: float,
    ):
        """Record an iteration result."""
        self.iteration += 1

        if passed:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_hypothesis_code = hypothesis_code
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        self.history.append({
            "iteration": self.iteration,
            "code": hypothesis_code,
            "passed": passed,
            "sharpe": sharpe,
        })

    def should_stop(self, config: AgentConfig) -> tuple[bool, Optional[str]]:
        """
        Check if research should stop.

        Returns:
            (should_stop, reason)
        """
        if self.iteration >= config.max_iterations:
            return True, "Reached maximum iterations"

        if self.consecutive_failures >= config.max_consecutive_failures:
            return True, "Too many consecutive failures"

        if self.best_sharpe >= config.early_stop_sharpe:
            return True, f"Achieved target Sharpe: {self.best_sharpe:.2f}"

        if self.consecutive_successes >= config.early_stop_success_count:
            return True, "Sufficient successful hypotheses found"

        return False, None
