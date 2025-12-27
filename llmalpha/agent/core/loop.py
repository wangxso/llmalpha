"""
Research Loop Controller for LLM Alpha.

Manages the iterative research process: generate → validate → learn → repeat.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llmalpha.agent.config import AgentConfig, IterationState
from llmalpha.agent.generator.base import BaseGenerator, GenerationResult
from llmalpha.agent.core.validator import HypothesisValidator, ValidationOutput

if TYPE_CHECKING:
    from llmalpha.agent.ui.exploration import ExplorationReporter


@dataclass
class IterationResult:
    """Result of a single research iteration."""

    iteration: int
    hypothesis_code: str
    hypothesis_name: str

    # Generation results
    generation_success: bool
    code: Optional[str] = None
    generation_error: Optional[str] = None

    # Validation results
    validation_success: bool = False
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    wf_passed: bool = False
    rolling_passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "hypothesis_code": self.hypothesis_code,
            "hypothesis_name": self.hypothesis_name,
            "generation_success": self.generation_success,
            "validation_success": self.validation_success,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "wf_passed": self.wf_passed,
            "rolling_passed": self.rolling_passed,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class LoopResult:
    """Result of complete research loop."""

    total_iterations: int
    successful_hypotheses: int
    best_hypothesis_code: Optional[str]
    best_sharpe: float
    iterations: List[IterationResult]
    early_stopped: bool = False
    stop_reason: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return 0.0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "Research Loop Summary",
            "=" * 60,
            f"Total Iterations: {self.total_iterations}",
            f"Successful Hypotheses: {self.successful_hypotheses}",
            f"Best Hypothesis: {self.best_hypothesis_code or 'None'}",
            f"Best Sharpe: {self.best_sharpe:.2f}",
            f"Duration: {self.duration_seconds:.1f}s",
        ]
        if self.early_stopped:
            lines.append(f"Early Stopped: {self.stop_reason}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ResearchLoop:
    """
    Controls the research iteration loop.

    The loop:
    1. Generates hypothesis code using LLM
    2. Validates the hypothesis through backtesting
    3. Records results to knowledge base
    4. Decides to continue, improve, or stop
    """

    def __init__(
        self,
        config: AgentConfig,
        generator: BaseGenerator,
        validator: HypothesisValidator,
        knowledge_service=None,
        data_tool=None,
        reporter: Optional["ExplorationReporter"] = None,
    ):
        """
        Initialize research loop.

        Args:
            config: Agent configuration
            generator: Code generator
            validator: Hypothesis validator
            knowledge_service: Knowledge base service (optional)
            data_tool: DataTool for autonomous data management (optional)
            reporter: ExplorationReporter for immersive UI (optional)
        """
        self.config = config
        self.generator = generator
        self.validator = validator
        self.knowledge = knowledge_service
        self.data_tool = data_tool
        self.reporter = reporter
        self.state = IterationState()

        # Context window for managing LLM context (dynamically adapts to model limits)
        from llmalpha.agent.core.context import ContextWindow
        self.context_window = ContextWindow(
            model=config.llm.model,
        )

    async def _ensure_data_available(self, requirements: str) -> bool:
        """
        Ensure market data is available before starting research.

        If data_tool is configured, this will:
        1. Check what data is available
        2. Parse requirements to identify needed symbols
        3. Download missing data automatically

        Args:
            requirements: Research requirements text

        Returns:
            True if sufficient data is available
        """
        if not self.data_tool:
            return True  # Skip if no data tool configured

        print("\n[Data Check] Checking data availability...")

        # Check current data status
        status = self.data_tool.check_data()

        if status.has_sufficient_data:
            print(f"[Data Check] {status.message}")
            return True

        # No data available - need to download
        print("[Data Check] No data found. Downloading recommended symbols...")

        # Get recommended symbols based on category if specified
        category = self.config.category
        symbols = self.data_tool.get_recommended_symbols(
            category=category,
            count=5  # Start with 5 symbols
        )

        # Download data
        print(f"[Data Check] Downloading: {', '.join(symbols)}")
        result = await self.data_tool.download_data(
            symbols=symbols,
            months=3,  # 3 months of data for research
        )

        if result.success:
            print(f"[Data Check] ✓ {result.message}")
            return True
        else:
            print(f"[Data Check] ✗ {result.message}")
            # Try with fewer symbols
            if result.downloaded:
                print(f"[Data Check] Proceeding with available data: {', '.join(result.downloaded)}")
                return True
            return False

    def _generate_hypothesis_code(self) -> str:
        """Generate unique hypothesis code."""
        if self.knowledge:
            # Query existing hypotheses to find next number
            try:
                existing = self.knowledge.search_hypotheses()
                max_num = 0
                for h in existing:
                    if hasattr(h, "code") and h.code.startswith("H"):
                        try:
                            num = int(h.code[1:])
                            max_num = max(max_num, num)
                        except ValueError:
                            pass
                return f"H{max_num + 1:03d}"
            except Exception:
                pass

        # Fallback: use iteration number
        return f"H{self.state.iteration + 1:03d}"

    async def _get_context(self) -> Dict[str, Any]:
        """Build context for LLM from knowledge base and context window."""
        context = {}

        # Add context from sliding window (recent iterations + compressed history)
        window_context = self.context_window.get_context_for_llm()
        if window_context:
            context.update(window_context)

        # Add learning prompt from history
        learning_prompt = self.context_window.get_learning_prompt()
        if learning_prompt:
            context["learning_insights"] = learning_prompt

        # Add knowledge base context if available
        if self.knowledge:
            try:
                # Get similar validated hypotheses
                validated = self.knowledge.list_validated_hypotheses()
                if validated:
                    from llmalpha.agent.prompts.templates import format_similar_items
                    context["similar_strategies"] = format_similar_items(
                        validated[:self.config.include_similar_hypotheses]
                    )

                # Get recent failures from KB (complementary to window)
                if self.config.learn_from_failures:
                    failed = self.knowledge.list_failed_hypotheses()
                    if failed:
                        from llmalpha.agent.prompts.templates import format_failures
                        context["kb_failures"] = format_failures(failed[:3])

            except Exception:
                pass

        return context

    def _record_to_context_window(self, result: IterationResult):
        """Record iteration result to context window."""
        self.context_window.add_iteration({
            "code": result.hypothesis_code,
            "passed": result.validation_success,
            "sharpe": result.sharpe_ratio,
            "failure_reasons": result.failure_reasons,
            "strategy_code": result.code,
        })

    async def _run_iteration(
        self,
        requirements: str,
        base_name: str,
    ) -> IterationResult:
        """Run a single research iteration."""
        hypothesis_code = self._generate_hypothesis_code()
        hypothesis_name = f"{base_name} {hypothesis_code}"

        # Use reporter if available, otherwise fallback to print
        if self.reporter:
            self.reporter.start_iteration(self.state.iteration + 1, hypothesis_code)
        else:
            print(f"\n{'='*60}")
            print(f" 迭代 [{self.state.iteration + 1}/{self.config.max_iterations}] - {hypothesis_code}")
            print(f"{'='*60}")

        result = IterationResult(
            iteration=self.state.iteration + 1,
            hypothesis_code=hypothesis_code,
            hypothesis_name=hypothesis_name,
            generation_success=False,
        )

        # Get context
        context = await self._get_context()

        # Add previous failure context if available
        if self.state.history and not self.state.history[-1].get("passed", True):
            last = self.state.history[-1]
            requirements = f"""
{requirements}

NOTE: Previous attempt ({last.get('code', 'unknown')}) failed with Sharpe: {last.get('sharpe', 0):.2f}.
Try a different approach.
"""

        # Generate code
        gen_result = await self.generator.generate(
            requirements=requirements,
            hypothesis_code=hypothesis_code,
            hypothesis_name=hypothesis_name,
            category=self.config.category or "general",
            context=context,
            reporter=self.reporter,
        )

        if not gen_result.success:
            result.generation_error = gen_result.error
            self.state.record_iteration(hypothesis_code, False, 0.0)

            # Record to knowledge base
            if self.knowledge:
                try:
                    self.knowledge.record_hypothesis(
                        code=hypothesis_code,
                        name=hypothesis_name,
                        category=self.config.category or "general",
                        status="failed",
                    )
                    self.knowledge.fail_hypothesis(
                        hypothesis_code,
                        f"Generation failed: {gen_result.error}",
                    )
                except Exception:
                    pass

            return result

        result.generation_success = True
        result.code = gen_result.code

        # Record hypothesis to knowledge base
        if self.knowledge:
            try:
                self.knowledge.record_hypothesis(
                    code=hypothesis_code,
                    name=hypothesis_name,
                    category=self.config.category or "general",
                    logic=gen_result.code,
                    rationale=requirements[:500],
                )
            except Exception:
                pass

        # Validate
        if self.reporter:
            from llmalpha.agent.ui.exploration import ExplorationPhase
            self.reporter.enter_phase(ExplorationPhase.TESTING, "Running validation...")

        try:
            validation = await self.validator.validate(
                strategy=gen_result.instance,
                mode=self.config.validation.mode,
            )

            result.validation_success = validation.passed
            result.sharpe_ratio = validation.test_sharpe
            result.win_rate = validation.win_rate
            result.total_trades = validation.total_trades
            result.wf_passed = validation.wf_passed
            result.rolling_passed = validation.rolling_passed
            result.failure_reasons = validation.failure_reasons

        except Exception as e:
            result.failure_reasons = [f"Validation error: {e}"]
            validation = None

        # Learning phase
        if self.reporter:
            from llmalpha.agent.ui.exploration import ExplorationPhase
            self.reporter.enter_phase(ExplorationPhase.LEARNING, "Analyzing results...")

        # Update state
        self.state.record_iteration(
            hypothesis_code,
            result.validation_success,
            result.sharpe_ratio,
        )

        # Update knowledge base
        if self.knowledge:
            try:
                if result.validation_success:
                    self.knowledge.validate_hypothesis(hypothesis_code)
                else:
                    self.knowledge.fail_hypothesis(
                        hypothesis_code,
                        "; ".join(result.failure_reasons),
                    )

                # Record backtest result
                if validation:
                    self.knowledge.record_backtest(
                        hypothesis_code=hypothesis_code,
                        result=validation.to_dict(),
                    )
            except Exception:
                pass

        # Record to context window for next iteration
        self._record_to_context_window(result)

        # Use reporter if available for results display
        if self.reporter:
            self.reporter.end_iteration(result)

        return result

    async def run(
        self,
        requirements: str,
        base_hypothesis_name: str = "LLM Strategy",
    ) -> LoopResult:
        """
        Run the complete research loop.

        Args:
            requirements: Research requirements/description
            base_hypothesis_name: Base name for generated hypotheses

        Returns:
            LoopResult
        """
        self.state.reset()
        self.context_window.reset()  # Reset context window for new run
        iterations: List[IterationResult] = []
        started_at = datetime.now()

        # Ensure data is available (auto-download if needed)
        data_ready = await self._ensure_data_available(requirements)
        if not data_ready:
            print("\n[Error] No market data available. Cannot proceed with research.")
            return LoopResult(
                total_iterations=0,
                successful_hypotheses=0,
                best_hypothesis_code=None,
                best_sharpe=0.0,
                iterations=[],
                early_stopped=True,
                stop_reason="No market data available",
                started_at=started_at,
                ended_at=datetime.now(),
            )

        # Start session with reporter or fallback to print
        if self.reporter:
            self.reporter.start_session(
                requirements=requirements,
                max_iterations=self.config.max_iterations,
                category=self.config.category or "general",
            )
        else:
            print(f"\n{'='*60}")
            print(f"Starting Research Loop")
            print(f"Requirements: {requirements[:80]}...")
            print(f"Max Iterations: {self.config.max_iterations}")
            print(f"{'='*60}\n")

        while True:
            # Check stop conditions
            should_stop, stop_reason = self.state.should_stop(self.config)
            if should_stop:
                loop_result = LoopResult(
                    total_iterations=self.state.iteration,
                    successful_hypotheses=sum(1 for i in iterations if i.validation_success),
                    best_hypothesis_code=self.state.best_hypothesis_code,
                    best_sharpe=self.state.best_sharpe,
                    iterations=iterations,
                    early_stopped=True,
                    stop_reason=stop_reason,
                    started_at=started_at,
                    ended_at=datetime.now(),
                )
                # End session with reporter
                if self.reporter:
                    self.reporter.end_session(loop_result)
                return loop_result

            # Run iteration
            result = await self._run_iteration(requirements, base_hypothesis_name)
            iterations.append(result)

            # Print progress (only in legacy mode without reporter)
            if not self.reporter:
                status = " PASS" if result.validation_success else " FAIL"
                print(f"\n{''*60}")
                print(f" 验证结果: {status}")
                print(f"   Sharpe: {result.sharpe_ratio:.2f} | Trades: {result.total_trades} | Win Rate: {result.win_rate:.1%}")

                if not result.generation_success:
                    print(f"\n Generation Error: {result.generation_error}")
                elif not result.validation_success:
                    print("\n  失败原因:")
                    for reason in result.failure_reasons[:3]:
                        print(f"    {reason}")

                print(f"{''*60}\n")

        # Should not reach here, but just in case
        loop_result = LoopResult(
            total_iterations=self.state.iteration,
            successful_hypotheses=sum(1 for i in iterations if i.validation_success),
            best_hypothesis_code=self.state.best_hypothesis_code,
            best_sharpe=self.state.best_sharpe,
            iterations=iterations,
            started_at=started_at,
            ended_at=datetime.now(),
        )
        if self.reporter:
            self.reporter.end_session(loop_result)
        return loop_result
