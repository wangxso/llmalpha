"""
Context Manager for LLM Alpha.

Handles context window management to prevent token limit issues.
Dynamically compresses context when approaching token limits.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimation.
    Rule of thumb: ~4 chars per token for English, ~2 for code.
    """
    if not text:
        return 0
    return len(text) // 3  # Conservative estimate


@dataclass
class IterationRecord:
    """Full record of an iteration."""
    code: str
    passed: bool
    sharpe: float
    failure_reasons: List[str] = field(default_factory=list)
    strategy_code: Optional[str] = None

    def to_full_dict(self) -> Dict[str, Any]:
        """Full representation."""
        return {
            "code": self.code,
            "passed": self.passed,
            "sharpe": self.sharpe,
            "failure_reasons": self.failure_reasons[:2],
            "strategy_snippet": self._truncate_code(self.strategy_code, 50),
        }

    def to_summary(self) -> str:
        """Compressed one-line summary."""
        status = "PASS" if self.passed else "FAIL"
        issue = ""
        if not self.passed and self.failure_reasons:
            issue = f" ({self.failure_reasons[0][:30]})"
        return f"{self.code}: {status} Sharpe={self.sharpe:.2f}{issue}"

    def _truncate_code(self, code: str, max_lines: int) -> Optional[str]:
        if not code:
            return None
        lines = code.split("\n")
        if len(lines) <= max_lines:
            return code
        return "\n".join(lines[:max_lines]) + "\n# ..."

    def estimate_tokens(self) -> int:
        """Estimate tokens for full representation."""
        full = str(self.to_full_dict())
        return estimate_tokens(full)


# Common model context limits
MODEL_LIMITS = {
    "gpt-5.2": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "llama2": 4096,
    "mistral": 8192,
}


@dataclass
class ContextWindow:
    """
    Dynamically manages context for LLM to stay within token limits.

    Strategy:
    - Keep all iterations in full detail by default
    - When approaching token limit, progressively compress:
      1. First: truncate strategy code
      2. Then: convert old iterations to one-line summaries
      3. Finally: aggregate summaries into statistics
    """

    model: str = "gpt-5.2"
    reserve_for_response: int = 4096
    reserve_for_prompt: int = 2000  # System prompt, requirements, etc.

    # All iterations (full records)
    iterations: List[IterationRecord] = field(default_factory=list)

    # Best result tracking
    best_code: Optional[str] = None
    best_sharpe: float = 0.0
    best_strategy_code: Optional[str] = None

    def __post_init__(self):
        self.token_limit = MODEL_LIMITS.get(self.model, 8192)
        self.available_tokens = self.token_limit - self.reserve_for_response - self.reserve_for_prompt

    def set_model(self, model: str):
        """Update model and recalculate limits."""
        self.model = model
        self.token_limit = MODEL_LIMITS.get(model, 8192)
        self.available_tokens = self.token_limit - self.reserve_for_response - self.reserve_for_prompt

    def add_iteration(self, result: Dict[str, Any]):
        """Add a new iteration result."""
        record = IterationRecord(
            code=result.get("code", "?"),
            passed=result.get("passed", False),
            sharpe=result.get("sharpe", 0),
            failure_reasons=result.get("failure_reasons", []),
            strategy_code=result.get("strategy_code"),
        )

        self.iterations.append(record)

        # Update best
        if record.sharpe > self.best_sharpe:
            self.best_sharpe = record.sharpe
            self.best_code = record.code
            self.best_strategy_code = record.strategy_code

    def get_context_for_llm(self) -> Dict[str, Any]:
        """
        Build context dict, dynamically compressing if needed.
        """
        if not self.iterations:
            return {}

        # Start with full context, compress if needed
        context = self._build_context_adaptive()

        # Add statistics
        total = len(self.iterations)
        passed = sum(1 for r in self.iterations if r.passed)
        context["stats"] = {
            "total_iterations": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "best_sharpe": self.best_sharpe,
            "best_code": self.best_code,
        }

        return context

    def _build_context_adaptive(self) -> Dict[str, Any]:
        """Build context, adapting compression level to fit token budget."""
        context = {}

        # Try full context first
        full_context = self._build_full_context()
        full_tokens = estimate_tokens(str(full_context))

        if full_tokens <= self.available_tokens:
            return full_context

        # Need compression - try progressive levels
        # Level 1: Truncate strategy code more aggressively
        compressed = self._build_compressed_context(code_lines=20)
        if estimate_tokens(str(compressed)) <= self.available_tokens:
            return compressed

        # Level 2: Keep only recent N full, rest as summaries
        for keep_recent in [5, 3, 1]:
            mixed = self._build_mixed_context(keep_recent=keep_recent)
            if estimate_tokens(str(mixed)) <= self.available_tokens:
                return mixed

        # Level 3: All summaries
        summary_only = self._build_summary_only_context()
        if estimate_tokens(str(summary_only)) <= self.available_tokens:
            return summary_only

        # Level 4: Aggregate statistics only
        return self._build_stats_only_context()

    def _build_full_context(self) -> Dict[str, Any]:
        """Full context with all details."""
        return {
            "iterations": [r.to_full_dict() for r in self.iterations],
            "best_strategy_reference": self._truncate_best_strategy(100),
        }

    def _build_compressed_context(self, code_lines: int) -> Dict[str, Any]:
        """Compressed context with shorter code snippets."""
        iterations = []
        for r in self.iterations:
            item = {
                "code": r.code,
                "passed": r.passed,
                "sharpe": r.sharpe,
            }
            if not r.passed and r.failure_reasons:
                item["issue"] = r.failure_reasons[0][:50]
            iterations.append(item)

        return {
            "iterations": iterations,
            "best_strategy_reference": self._truncate_best_strategy(code_lines),
        }

    def _build_mixed_context(self, keep_recent: int) -> Dict[str, Any]:
        """Recent iterations full, older as summaries."""
        n = len(self.iterations)

        # Old iterations as one-line summaries
        old_summaries = []
        for r in self.iterations[: max(0, n - keep_recent)]:
            old_summaries.append(r.to_summary())

        # Recent iterations with more detail
        recent = []
        for r in self.iterations[-keep_recent:]:
            item = {"code": r.code, "passed": r.passed, "sharpe": r.sharpe}
            if not r.passed and r.failure_reasons:
                item["issues"] = r.failure_reasons[:2]
            recent.append(item)

        context = {"recent_iterations": recent}
        if old_summaries:
            context["history"] = "\n".join(old_summaries)

        return context

    def _build_summary_only_context(self) -> Dict[str, Any]:
        """All iterations as one-line summaries."""
        summaries = [r.to_summary() for r in self.iterations[-20:]]  # Max 20
        return {"iteration_history": "\n".join(summaries)}

    def _build_stats_only_context(self) -> Dict[str, Any]:
        """Aggregate statistics only - minimal context."""
        total = len(self.iterations)
        passed = sum(1 for r in self.iterations if r.passed)

        # Group failure reasons
        failure_counts = {}
        for r in self.iterations:
            if not r.passed and r.failure_reasons:
                reason = r.failure_reasons[0][:30]
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        top_failures = sorted(failure_counts.items(), key=lambda x: -x[1])[:3]

        return {
            "summary": f"{total} attempts: {passed} passed, {total - passed} failed",
            "common_failures": [f"{r}: {c}" for r, c in top_failures],
            "best": f"{self.best_code} with Sharpe {self.best_sharpe:.2f}" if self.best_code else None,
        }

    def _truncate_best_strategy(self, max_lines: int) -> Optional[str]:
        """Get truncated best strategy code."""
        if not self.best_strategy_code:
            return None
        lines = self.best_strategy_code.split("\n")
        if len(lines) <= max_lines:
            return self.best_strategy_code
        return "\n".join(lines[:max_lines]) + "\n# ... (truncated)"

    def get_learning_prompt(self) -> str:
        """Generate learning insights from history."""
        if not self.iterations:
            return ""

        total = len(self.iterations)
        passed = sum(1 for r in self.iterations if r.passed)

        lines = [f"Previous attempts: {total} ({passed} passed, {total - passed} failed)"]

        # Failure pattern analysis
        failure_reasons = {}
        for r in self.iterations:
            if not r.passed:
                for reason in r.failure_reasons[:1]:
                    key = reason[:40]
                    failure_reasons[key] = failure_reasons.get(key, 0) + 1

        if failure_reasons:
            lines.append("Common issues:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"  - {reason} ({count}x)")

        if self.best_code:
            lines.append(f"Best so far: {self.best_code} (Sharpe {self.best_sharpe:.2f})")

        return "\n".join(lines)

    def reset(self):
        """Reset all context."""
        self.iterations = []
        self.best_code = None
        self.best_sharpe = 0.0
        self.best_strategy_code = None
