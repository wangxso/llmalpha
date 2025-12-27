"""
Alpha Researcher - Main Agent Class for LLM Alpha.

The AlphaResearcher is the primary interface for conducting
LLM-driven strategy research.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llmalpha.agent.config import AgentConfig
from llmalpha.agent.core.loop import ResearchLoop, LoopResult
from llmalpha.agent.core.validator import HypothesisValidator
from llmalpha.agent.generator import get_generator
from llmalpha.agent.providers import get_provider
from llmalpha.agent.sandbox import SafeExecutor
from llmalpha.agent.tools.data_tool import DataTool
from llmalpha.data.loader import DataLoader

if TYPE_CHECKING:
    from llmalpha.agent.ui.exploration import ExplorationReporter


class AlphaResearcher:
    """
    Main Agent for LLM-driven alpha research.

    Orchestrates the entire research process:
    1. Initialize LLM provider and tools
    2. Run research loop to generate and validate hypotheses
    3. Record results to knowledge base
    4. Improve existing hypotheses

    Example:
        researcher = AlphaResearcher(config)
        result = await researcher.research(
            requirements="Create a momentum strategy using RSI",
            category="momentum",
        )
        print(result.summary())
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        knowledge_service=None,
        auto_download_data: bool = True,
        immersive: bool = True,
    ):
        """
        Initialize the researcher.

        Args:
            config: Agent configuration (uses defaults if not provided)
            knowledge_service: Optional knowledge base service
            auto_download_data: If True, automatically download data when needed
            immersive: If True, enable immersive exploration UI
        """
        self.config = config or AgentConfig()
        self.knowledge = knowledge_service
        self.auto_download_data = auto_download_data
        self.immersive = immersive

        # Initialize components
        self._llm = None
        self._executor = None
        self._generator = None
        self._validator = None
        self._data_loader = None
        self._data_tool = None
        self._loop = None
        self._reporter: Optional["ExplorationReporter"] = None

    def _ensure_initialized(self):
        """Lazy initialization of components."""
        if self._llm is None:
            self._llm = get_provider(
                provider_name=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

        if self._executor is None:
            self._executor = SafeExecutor(
                timeout=self.config.sandbox_timeout,
            )

        if self._generator is None:
            self._generator = get_generator(
                generation_type=self.config.generation_type,
                llm=self._llm,
                executor=self._executor,
            )

        if self._data_loader is None:
            self._data_loader = DataLoader(self.config.data_dir)

        if self._data_tool is None and self.auto_download_data:
            # Get proxy from environment if available
            proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
            self._data_tool = DataTool(
                data_dir=self.config.data_dir,
                proxy=proxy,
            )

        if self._validator is None:
            self._validator = HypothesisValidator(
                data_loader=self._data_loader,
                min_sharpe=self.config.validation.min_sharpe,
                min_trades=self.config.validation.min_trades,
                decay_threshold=self.config.validation.decay_threshold,
                min_positive_ratio=self.config.validation.min_positive_ratio,
            )

        # Create immersive reporter if enabled
        if self._reporter is None and self.immersive:
            from llmalpha.agent.ui.exploration import ExplorationReporter
            self._reporter = ExplorationReporter()

        if self._loop is None:
            self._loop = ResearchLoop(
                config=self.config,
                generator=self._generator,
                validator=self._validator,
                knowledge_service=self.knowledge,
                data_tool=self._data_tool,
                reporter=self._reporter,
            )

    async def research(
        self,
        requirements: str,
        category: Optional[str] = None,
        base_name: str = "LLM Strategy",
    ) -> LoopResult:
        """
        Run research to generate new hypotheses.

        Args:
            requirements: Research requirements/description
            category: Strategy category (momentum, mean_reversion, etc.)
            base_name: Base name for generated hypotheses

        Returns:
            LoopResult containing all iteration results
        """
        self._ensure_initialized()

        # Update category if provided
        if category:
            self.config.category = category

        return await self._loop.run(
            requirements=requirements,
            base_hypothesis_name=base_name,
        )

    async def improve_hypothesis(
        self,
        hypothesis_code: str,
        max_iterations: Optional[int] = None,
    ) -> LoopResult:
        """
        Improve an existing hypothesis.

        Args:
            hypothesis_code: Code of hypothesis to improve (e.g., "H042")
            max_iterations: Max improvement attempts (default: 3)

        Returns:
            LoopResult
        """
        self._ensure_initialized()

        if not self.knowledge:
            raise ValueError("Knowledge service required for improvement")

        # Get hypothesis from knowledge base
        hypothesis = self.knowledge.get_hypothesis(hypothesis_code)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_code} not found")

        # Get backtest results
        backtest = self.knowledge.get_latest_backtest(hypothesis_code)

        # Build improvement requirements
        requirements = f"""
Improve the existing hypothesis {hypothesis_code}: {hypothesis.name}

Original Logic:
{hypothesis.logic or 'Not available'}

Original Rationale:
{hypothesis.rationale or 'Not available'}

Previous Results:
- Status: {hypothesis.status}
- Sharpe: {backtest.sharpe_ratio if backtest else 'N/A'}
- Win Rate: {backtest.win_rate if backtest else 'N/A'}
- Failure Reasons: {hypothesis.failure_reason or 'None'}

Please improve this hypothesis to achieve better performance.
"""

        # Temporarily reduce max iterations for improvement
        original_max = self.config.max_iterations
        self.config.max_iterations = max_iterations or 3

        try:
            result = await self._loop.run(
                requirements=requirements,
                base_hypothesis_name=f"{hypothesis.name} Improved",
            )
        finally:
            self.config.max_iterations = original_max

        return result

    async def analyze_failures(
        self,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze recent failures to identify patterns.

        Args:
            limit: Number of failures to analyze

        Returns:
            Analysis results
        """
        self._ensure_initialized()

        if not self.knowledge:
            return {"error": "Knowledge service required"}

        failed = self.knowledge.list_failed_hypotheses()[:limit]

        if not failed:
            return {"message": "No failures to analyze"}

        # Categorize failures
        failure_categories = {}
        for h in failed:
            reason = h.failure_reason or "Unknown"
            if "Sharpe" in reason:
                cat = "low_sharpe"
            elif "trade" in reason.lower():
                cat = "insufficient_trades"
            elif "decay" in reason.lower():
                cat = "strategy_decay"
            elif "generation" in reason.lower():
                cat = "code_generation"
            else:
                cat = "other"

            if cat not in failure_categories:
                failure_categories[cat] = []
            failure_categories[cat].append(h.code)

        return {
            "total_failures": len(failed),
            "categories": failure_categories,
            "recent_codes": [h.code for h in failed[:5]],
            "recommendations": self._generate_recommendations(failure_categories),
        }

    def _generate_recommendations(
        self,
        failure_categories: Dict[str, List[str]],
    ) -> List[str]:
        """Generate recommendations based on failure patterns."""
        recommendations = []

        if "low_sharpe" in failure_categories:
            count = len(failure_categories["low_sharpe"])
            if count > 3:
                recommendations.append(
                    f"{count} hypotheses failed due to low Sharpe. "
                    "Consider using more robust signal combinations."
                )

        if "insufficient_trades" in failure_categories:
            count = len(failure_categories["insufficient_trades"])
            if count > 2:
                recommendations.append(
                    f"{count} hypotheses had too few trades. "
                    "Try more active signals or shorter timeframes."
                )

        if "strategy_decay" in failure_categories:
            count = len(failure_categories["strategy_decay"])
            if count > 2:
                recommendations.append(
                    f"{count} hypotheses showed performance decay. "
                    "Focus on more adaptive or regime-aware strategies."
                )

        if "code_generation" in failure_categories:
            count = len(failure_categories["code_generation"])
            if count > 3:
                recommendations.append(
                    f"{count} code generation failures. "
                    "Consider simplifying requirements or adjusting temperature."
                )

        return recommendations

    def get_statistics(self) -> Dict[str, Any]:
        """Get research statistics."""
        if not self.knowledge:
            return {"error": "Knowledge service required"}

        return self.knowledge.get_statistics()

    def list_validated(self) -> List[Any]:
        """List all validated hypotheses."""
        if not self.knowledge:
            return []

        return self.knowledge.list_validated_hypotheses()

    def list_failed(self) -> List[Any]:
        """List all failed hypotheses."""
        if not self.knowledge:
            return []

        return self.knowledge.list_failed_hypotheses()


def create_researcher(
    provider: str = "openai",
    model: str = "gpt-5.2",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    data_dir: str = "data",
    db_path: str = "data/knowledge.db",
    auto_download_data: bool = True,
    immersive: bool = True,
    **kwargs,
) -> AlphaResearcher:
    """
    Factory function to create a researcher with common settings.

    Args:
        provider: LLM provider (openai, anthropic, ollama)
        model: Model name
        api_key: API key (or use environment variable)
        base_url: Custom API base URL (for third-party proxies)
        data_dir: Data directory
        db_path: Knowledge base path
        auto_download_data: If True, automatically download data when needed
        immersive: If True, enable immersive exploration UI
        **kwargs: Additional config options

    Returns:
        Configured AlphaResearcher
    """
    from llmalpha.agent.config import AgentConfig, LLMConfig

    llm_config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    config = AgentConfig(
        llm=llm_config,
        data_dir=data_dir,
        db_path=db_path,
        **kwargs,
    )

    # Initialize knowledge service if db exists
    knowledge = None
    try:
        from llmalpha.knowledge import get_knowledge_service
        knowledge = get_knowledge_service(db_path)
    except Exception:
        pass

    return AlphaResearcher(
        config=config,
        knowledge_service=knowledge,
        auto_download_data=auto_download_data,
        immersive=immersive,
    )
