"""
Code Generator Base Classes for LLM Alpha.

Provides abstract interface for generating Factor/Signal/Strategy code.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Type

from llmalpha.agent.providers.base import LLMProvider, Message
from llmalpha.agent.sandbox.executor import SafeExecutor, ExecutionResult
from llmalpha.agent.prompts.templates import SYSTEM_PROMPT

if TYPE_CHECKING:
    from llmalpha.agent.ui.exploration import ExplorationReporter, ThinkingStream


# Timeout bounds for LLM-estimated values
MIN_TIMEOUT = 10
MAX_TIMEOUT = 300
DEFAULT_TIMEOUT = 60

# Max retries for code fix
MAX_CODE_FIX_RETRIES = 3


@dataclass
class GenerationResult:
    """Result of code generation."""

    success: bool
    code: Optional[str] = None
    class_type: Optional[Type] = None
    instance: Any = None
    error: Optional[str] = None
    llm_response: Optional[str] = None
    execution_result: Optional[ExecutionResult] = None
    estimated_timeout: int = DEFAULT_TIMEOUT  # LLM-estimated execution timeout
    fix_attempts: int = 0  # Number of fix attempts made


def extract_timeout_from_response(response: str) -> int:
    """
    Extract LLM-estimated timeout from response.

    Looks for pattern: ESTIMATED_TIMEOUT: <number>
    Returns bounded value within [MIN_TIMEOUT, MAX_TIMEOUT].
    """
    match = re.search(r'ESTIMATED_TIMEOUT:\s*(\d+)', response)
    if match:
        timeout = int(match.group(1))
        return max(MIN_TIMEOUT, min(MAX_TIMEOUT, timeout))
    return DEFAULT_TIMEOUT


CODE_FIX_PROMPT = """The code you generated failed to execute. Please fix it.

## Error:
{error}

## Your Previous Code:
```python
{code}
```

## IMPORTANT REMINDERS:
1. DO NOT write any import statements - all classes (Strategy, SignalResult, pd, np) are pre-loaded
2. Just define the class directly without imports
3. Fix the specific error mentioned above

## Please provide the corrected code:
"""


class BaseGenerator(ABC):
    """
    Abstract base class for code generators.

    Subclasses implement generation of Factors, Signals, or Strategies.
    """

    def __init__(
        self,
        llm: LLMProvider,
        executor: SafeExecutor,
        max_fix_retries: int = MAX_CODE_FIX_RETRIES,
    ):
        """
        Initialize generator.

        Args:
            llm: LLM provider for code generation
            executor: Safe executor for code validation
            max_fix_retries: Maximum number of code fix attempts
        """
        self.llm = llm
        self.executor = executor
        self.max_fix_retries = max_fix_retries

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for this generator type."""
        pass

    @abstractmethod
    def get_class_name(self, hypothesis_code: str) -> str:
        """Generate class name from hypothesis code."""
        pass

    @abstractmethod
    def get_expected_base_class(self) -> Type:
        """Get the expected base class for generated code."""
        pass

    async def _try_fix_code(
        self,
        code: str,
        error: str,
        messages_history: List[Message],
    ) -> Optional[str]:
        """
        Ask LLM to fix code that failed execution.

        Args:
            code: The code that failed
            error: The error message
            messages_history: Previous conversation messages

        Returns:
            Fixed code or None if fix failed
        """
        fix_prompt = CODE_FIX_PROMPT.format(error=error, code=code)

        messages = messages_history + [
            Message(role="assistant", content=f"```python\n{code}\n```"),
            Message(role="user", content=fix_prompt),
        ]

        try:
            response = await self.llm.complete_with_retry(messages)
            fixed_code = self.llm.extract_code(response.content)
            return fixed_code
        except Exception:
            return None

    async def generate(
        self,
        requirements: str,
        hypothesis_code: str,
        hypothesis_name: str,
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
        stream: bool = True,
        reporter: Optional["ExplorationReporter"] = None,
    ) -> GenerationResult:
        """
        Generate code using LLM with automatic retry on execution failure.

        Args:
            requirements: Description of what to generate
            hypothesis_code: Unique hypothesis identifier (e.g., "H001")
            hypothesis_name: Human-readable hypothesis name
            category: Category (momentum, mean_reversion, etc.)
            context: Additional context (similar items, failures, etc.)
            stream: Whether to stream LLM output in real-time
            reporter: Optional ExplorationReporter for immersive UI

        Returns:
            GenerationResult
        """
        context = context or {}

        # Build prompt
        class_name = self.get_class_name(hypothesis_code)

        # Merge failures from different sources
        failures = context.get("failures") or context.get("kb_failures") or "None recorded"

        # Add learning insights to requirements if available
        learning = context.get("learning_insights", "")
        if learning:
            requirements = f"{requirements}\n\n## 历史学习:\n{learning}"

        prompt = self.get_prompt_template().format(
            requirements=requirements,
            category=category,
            class_name=class_name,
            code=hypothesis_code,
            name=hypothesis_name,
            similar_factors=context.get("similar_factors", "None available"),
            similar_signals=context.get("similar_signals", "None available"),
            similar_strategies=context.get("similar_strategies", "None available"),
            available_factors=context.get("available_factors", "See llmalpha.factors"),
            available_components=context.get("available_components", "See llmalpha.factors and llmalpha.signals"),
            best_strategies=context.get("best_strategies", "None validated yet"),
            failures=failures,
        )

        # Initial messages
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]

        # Call LLM with streaming
        try:
            if reporter:
                # Immersive mode with reporter
                from llmalpha.agent.ui.exploration import ThinkingStream, ExplorationPhase
                reporter.enter_phase(ExplorationPhase.THINKING, "Analyzing requirements...")

                thinking_stream = ThinkingStream(reporter)

                if stream and hasattr(self.llm, 'complete_stream'):
                    response = await self.llm.complete_stream(
                        messages,
                        print_output=False,
                        on_token=thinking_stream.on_token,
                    )
                else:
                    response = await self.llm.complete_with_retry(messages)

                thinking_stream.on_complete()
            else:
                # Legacy mode without reporter
                print("\n🤖 AI thinking...\n")
                if stream and hasattr(self.llm, 'complete_stream'):
                    response = await self.llm.complete_stream(messages, print_output=True)
                else:
                    response = await self.llm.complete_with_retry(messages)
                print("\n")
        except Exception as e:
            return GenerationResult(
                success=False,
                error=f"LLM request failed: {e}",
            )

        # Extract code
        code = self.llm.extract_code(response.content)
        if not code:
            return GenerationResult(
                success=False,
                error="Failed to extract code from LLM response",
                llm_response=response.content,
            )

        # Extract LLM-estimated timeout
        estimated_timeout = extract_timeout_from_response(response.content)

        # Try to execute, with retry loop for fixes
        expected_base = self.get_expected_base_class()
        fix_attempts = 0

        while fix_attempts <= self.max_fix_retries:
            exec_result, extracted_class = self.executor.execute_and_extract(
                code=code,
                base_class=expected_base,
                timeout=estimated_timeout,
            )

            if exec_result.success and extracted_class:
                # Success! Try to instantiate
                try:
                    instance = extracted_class()
                    return GenerationResult(
                        success=True,
                        code=code,
                        class_type=extracted_class,
                        instance=instance,
                        llm_response=response.content,
                        execution_result=exec_result,
                        estimated_timeout=estimated_timeout,
                        fix_attempts=fix_attempts,
                    )
                except Exception as e:
                    error = f"Failed to instantiate: {e}"
            else:
                error = exec_result.error or f"No valid {expected_base.__name__} subclass found"

            # If we've exhausted retries, return failure
            if fix_attempts >= self.max_fix_retries:
                return GenerationResult(
                    success=False,
                    code=code,
                    error=error,
                    llm_response=response.content,
                    execution_result=exec_result,
                    estimated_timeout=estimated_timeout,
                    fix_attempts=fix_attempts,
                )

            # Try to fix the code
            fixed_code = await self._try_fix_code(code, error, messages)
            if not fixed_code:
                return GenerationResult(
                    success=False,
                    code=code,
                    error=f"{error} (fix attempt failed)",
                    llm_response=response.content,
                    execution_result=exec_result,
                    estimated_timeout=estimated_timeout,
                    fix_attempts=fix_attempts,
                )

            code = fixed_code
            fix_attempts += 1

        # Should not reach here
        return GenerationResult(
            success=False,
            code=code,
            error="Max fix retries exceeded",
            fix_attempts=fix_attempts,
        )

    async def improve(
        self,
        previous_code: str,
        validation_results: Dict[str, Any],
        failure_reasons: List[str],
        hypothesis_code: str,
        hypothesis_name: str,
    ) -> GenerationResult:
        """
        Generate improved code based on failure analysis.

        Args:
            previous_code: Code that failed validation
            validation_results: Results from validation
            failure_reasons: List of reasons for failure
            hypothesis_code: New hypothesis code for improved version
            hypothesis_name: Name for improved version

        Returns:
            GenerationResult
        """
        from llmalpha.agent.prompts.templates import IMPROVEMENT_PROMPT

        prompt = IMPROVEMENT_PROMPT.format(
            previous_code=previous_code,
            sharpe_ratio=validation_results.get("sharpe_ratio", 0),
            win_rate=validation_results.get("win_rate", 0),
            total_trades=validation_results.get("total_trades", 0),
            wf_passed=validation_results.get("wf_passed", False),
            rolling_passed=validation_results.get("rolling_passed", False),
            failure_reasons="\n".join(f"- {r}" for r in failure_reasons),
        )

        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]

        try:
            response = await self.llm.complete_with_retry(messages)
        except Exception as e:
            return GenerationResult(
                success=False,
                error=f"LLM request failed: {e}",
            )

        # Extract and execute
        code = self.llm.extract_code(response.content)
        if not code:
            return GenerationResult(
                success=False,
                error="Failed to extract improved code",
                llm_response=response.content,
            )

        # Extract LLM-estimated timeout
        estimated_timeout = extract_timeout_from_response(response.content)

        # Try to execute with retry loop for fixes
        expected_base = self.get_expected_base_class()
        fix_attempts = 0

        while fix_attempts <= self.max_fix_retries:
            exec_result, extracted_class = self.executor.execute_and_extract(
                code=code,
                base_class=expected_base,
                timeout=estimated_timeout,
            )

            if exec_result.success and extracted_class:
                try:
                    instance = extracted_class()
                    return GenerationResult(
                        success=True,
                        code=code,
                        class_type=extracted_class,
                        instance=instance,
                        llm_response=response.content,
                        execution_result=exec_result,
                        estimated_timeout=estimated_timeout,
                        fix_attempts=fix_attempts,
                    )
                except Exception as e:
                    error = f"Failed to instantiate: {e}"
            else:
                error = exec_result.error or "No valid class found"

            if fix_attempts >= self.max_fix_retries:
                return GenerationResult(
                    success=False,
                    code=code,
                    error=error,
                    llm_response=response.content,
                    execution_result=exec_result,
                    estimated_timeout=estimated_timeout,
                    fix_attempts=fix_attempts,
                )

            # Try to fix
            fixed_code = await self._try_fix_code(code, error, messages)
            if not fixed_code:
                return GenerationResult(
                    success=False,
                    code=code,
                    error=f"{error} (fix attempt failed)",
                    llm_response=response.content,
                    execution_result=exec_result,
                    estimated_timeout=estimated_timeout,
                    fix_attempts=fix_attempts,
                )

            code = fixed_code
            fix_attempts += 1

        return GenerationResult(
            success=False,
            code=code,
            error="Max fix retries exceeded",
            fix_attempts=fix_attempts,
        )
