"""
Exploration Reporter for LLM Alpha.

Provides immersive terminal UI for AI exploration visualization.
Displays real-time thinking process, streaming LLM output,
and progress through research phases.
"""

import asyncio
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich.table import Table
from rich.box import ROUNDED, DOUBLE

if TYPE_CHECKING:
    from llmalpha.agent.core.loop import IterationResult, LoopResult


class ExplorationPhase(Enum):
    """Phases of the AI exploration process."""
    INITIALIZING = "initializing"
    THINKING = "thinking"
    HYPOTHESIZING = "hypothesizing"
    TESTING = "testing"
    LEARNING = "learning"
    COMPLETE = "complete"


@dataclass
class PhaseConfig:
    """Configuration for a phase display."""
    title: str
    icon: str
    color: str
    spinner: str
    description: str


# Phase display configurations
PHASE_CONFIGS = {
    ExplorationPhase.INITIALIZING: PhaseConfig(
        title="Initializing",
        icon="",
        color="cyan",
        spinner="dots",
        description="Preparing research environment...",
    ),
    ExplorationPhase.THINKING: PhaseConfig(
        title="Deep Thinking",
        icon="",
        color="yellow",
        spinner="moon",
        description="Analyzing requirements and patterns...",
    ),
    ExplorationPhase.HYPOTHESIZING: PhaseConfig(
        title="Generating Hypothesis",
        icon="",
        color="magenta",
        spinner="aesthetic",
        description="Formulating strategy code...",
    ),
    ExplorationPhase.TESTING: PhaseConfig(
        title="Validating",
        icon="",
        color="blue",
        spinner="dots12",
        description="Running backtests and validation...",
    ),
    ExplorationPhase.LEARNING: PhaseConfig(
        title="Learning",
        icon="",
        color="green",
        spinner="arc",
        description="Extracting insights from results...",
    ),
    ExplorationPhase.COMPLETE: PhaseConfig(
        title="Complete",
        icon="",
        color="green",
        spinner="dots",
        description="Research session finished.",
    ),
}


# ASCII Art Banner
BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════════════════════╗
║     _     _     __  __      _    _     ____  _   _    _              ║
║    | |   | |   |  \\/  |    / \\  | |   |  _ \\| | | |  / \\             ║
║    | |   | |   | |\\/| |   / _ \\ | |   | |_) | |_| | / _ \\            ║
║    | |___| |___| |  | |  / ___ \\| |___|  __/|  _  |/ ___ \\           ║
║    |_____|_____|_|  |_| /_/   \\_\\_____|_|   |_| |_/_/   \\_\\          ║
║                                                                       ║
║               [bold yellow]Autonomous Alpha Research System[/bold yellow]                    ║
╚══════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""


class ExplorationReporter:
    """
    Immersive terminal UI for AI exploration visualization.

    Displays real-time thinking process, streaming LLM output,
    and progress through research phases.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        thinking_speed: float = 0.01,
        show_code: bool = False,
    ):
        """
        Initialize the reporter.

        Args:
            console: Rich console instance (creates new one if None)
            thinking_speed: Delay between characters for typewriter effect
            show_code: Whether to display generated code
        """
        self.console = console or Console()
        self.thinking_speed = thinking_speed
        self.show_code = show_code

        self.current_phase = ExplorationPhase.INITIALIZING
        self._live: Optional[Live] = None
        self._thinking_buffer: List[str] = []
        self._iteration = 0
        self._total_iterations = 0
        self._best_sharpe = 0.0
        self._best_hypothesis = None
        self._successes = 0
        self._failures = 0
        self._current_hypothesis_code = ""
        self._in_code_block = False

    def start_session(
        self,
        requirements: str,
        max_iterations: int,
        category: str = "general",
    ):
        """Display session start banner and info."""
        self.console.print(BANNER)
        self.console.print()

        # Session info panel
        info_text = Text()
        info_text.append("  Research Target: ", style="bold")
        info_text.append(requirements[:60] + ("..." if len(requirements) > 60 else ""), style="white")
        info_text.append("\n  Max Iterations:  ", style="bold")
        info_text.append(str(max_iterations), style="cyan")
        info_text.append("\n  Category:        ", style="bold")
        info_text.append(category, style="magenta")

        self.console.print(Panel(
            info_text,
            title="[bold]Research Session[/bold]",
            border_style="cyan",
            box=ROUNDED,
        ))
        self.console.print()

        self._total_iterations = max_iterations
        self._thinking_buffer = []
        self._successes = 0
        self._failures = 0
        self._best_sharpe = 0.0
        self._best_hypothesis = None

    def start_iteration(self, iteration: int, hypothesis_code: str):
        """Display iteration start."""
        self._iteration = iteration
        self._current_hypothesis_code = hypothesis_code
        self._thinking_buffer = []
        self._in_code_block = False

        # Progress bar
        progress_width = 40
        filled = int((iteration / self._total_iterations) * progress_width)
        bar = "" * filled + "" * (progress_width - filled)

        self.console.print()
        self.console.print(f"[bold cyan]{'━' * 70}[/bold cyan]")
        self.console.print(
            f"  [bold]Iteration:[/bold] [{bar}] {iteration}/{self._total_iterations}   "
            f"[bold]Best:[/bold] {self._best_sharpe:.2f} ({self._best_hypothesis or 'N/A'})"
        )
        self.console.print(f"[bold cyan]{'━' * 70}[/bold cyan]")
        self.console.print()

        # Iteration panel
        self.console.print(Panel(
            f"[bold magenta]Hypothesis {hypothesis_code}[/bold magenta]",
            title=f"[bold] Iteration {iteration} [/bold]",
            border_style="magenta",
            box=ROUNDED,
        ))

    def enter_phase(self, phase: ExplorationPhase, message: str = ""):
        """Transition to a new phase."""
        self.current_phase = phase
        config = PHASE_CONFIGS[phase]

        self.console.print()
        phase_text = f"[bold {config.color}]{config.icon} {config.title}[/bold {config.color}]"
        if message:
            phase_text += f": {message}"
        self.console.print(phase_text)

    def stream_token(self, token: str):
        """
        Handle a streaming token from LLM.
        Displays thinking content with typewriter effect.
        """
        # Detect code block boundaries
        if "```" in token:
            if not self._in_code_block:
                self._in_code_block = True
                if not self.show_code:
                    self.console.print()
                    self.console.print("  [dim italic]Writing strategy code...[/dim italic]")
            else:
                self._in_code_block = False
            return

        # Don't display code content unless show_code is True
        if self._in_code_block and not self.show_code:
            return

        # Display thinking content
        if not self._in_code_block:
            # Buffer tokens and print complete lines
            self._thinking_buffer.append(token)
            full_text = "".join(self._thinking_buffer)

            # Print complete lines
            if "\n" in token:
                lines = full_text.split("\n")
                for line in lines[:-1]:
                    if line.strip():
                        self.console.print(f"  [yellow]> {line.strip()}[/yellow]")
                self._thinking_buffer = [lines[-1]] if lines[-1] else []

    async def stream_token_async(self, token: str):
        """Async version of stream_token with optional delay for effect."""
        self.stream_token(token)
        # Small delay for visual effect
        if not self._in_code_block and token.strip():
            await asyncio.sleep(self.thinking_speed)

    def flush_thinking_buffer(self):
        """Flush any remaining content in the thinking buffer."""
        if self._thinking_buffer:
            remaining = "".join(self._thinking_buffer).strip()
            if remaining:
                self.console.print(f"  [yellow]> {remaining}[/yellow]")
            self._thinking_buffer = []
        self._in_code_block = False

    def show_validation_progress(self, stage: str, progress: float):
        """Show validation progress."""
        width = 20
        filled = int(progress * width)
        bar = "" * filled + "" * (width - filled)
        status = "" if progress >= 1.0 else ""
        self.console.print(f"  {stage}: [{bar}] {int(progress * 100)}% {status}")

    def show_validation_result(
        self,
        passed: bool,
        sharpe: float,
        win_rate: float,
        trades: int,
        wf_passed: bool,
        rolling_passed: bool,
        failure_reasons: Optional[List[str]] = None,
        max_drawdown: float = 0.0,
    ):
        """Display validation results panel."""
        self.console.print()

        if passed:
            status_text = "[bold green] PASSED[/bold green]"
            border_style = "green"
            self._successes += 1
        else:
            status_text = "[bold red] FAILED[/bold red]"
            border_style = "red"
            self._failures += 1

        # Update best if this is better
        if sharpe > self._best_sharpe:
            self._best_sharpe = sharpe
            self._best_hypothesis = self._current_hypothesis_code

        # Build result table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Status", status_text)
        table.add_row("Sharpe Ratio", f"[{'green' if sharpe >= 0.5 else 'yellow' if sharpe >= 0.3 else 'red'}]{sharpe:.2f}[/]")
        table.add_row("Win Rate", f"{win_rate:.1%}")
        table.add_row("Total Trades", str(trades))
        table.add_row("Max Drawdown", f"{max_drawdown:.1%}")
        table.add_row("WF Test", " Passed" if wf_passed else " Failed")
        table.add_row("Rolling Test", " Passed" if rolling_passed else " Failed")

        self.console.print(Panel(
            table,
            title=f"[bold]Validation Results: {self._current_hypothesis_code}[/bold]",
            border_style=border_style,
            box=ROUNDED,
        ))

        # Show failure reasons if failed
        if not passed and failure_reasons:
            self.console.print()
            self.console.print("  [bold red]Failure Reasons:[/bold red]")
            for reason in failure_reasons[:3]:
                self.console.print(f"    [red]  {reason}[/red]")

    def show_learning(self, insights: List[str]):
        """Display learning insights from iteration."""
        if not insights:
            return

        self.console.print()
        insights_text = "\n".join(f"   {insight}" for insight in insights[:5])

        self.console.print(Panel(
            insights_text,
            title=f"[bold] Learning from {self._current_hypothesis_code}[/bold]",
            border_style="green",
            box=ROUNDED,
        ))

    def end_iteration(self, result: "IterationResult"):
        """End current iteration display."""
        self.flush_thinking_buffer()

        # Show validation results
        self.show_validation_result(
            passed=result.validation_success,
            sharpe=result.sharpe_ratio,
            win_rate=result.win_rate,
            trades=result.total_trades,
            wf_passed=result.wf_passed,
            rolling_passed=result.rolling_passed,
            failure_reasons=result.failure_reasons,
        )

        # If failed, show learning insights
        if not result.validation_success and result.failure_reasons:
            insights = [f"Issue: {result.failure_reasons[0]}"]
            if result.sharpe_ratio < 0.3:
                insights.append("Sharpe ratio too low - strategy needs stronger edge")
            if result.total_trades < 50:
                insights.append("Too few trades - parameters may be too restrictive")
            self.show_learning(insights)

    def end_session(self, result: "LoopResult"):
        """Display final session summary."""
        self.console.print()
        self.console.print(f"[bold cyan]{'═' * 70}[/bold cyan]")
        self.console.print()

        # Summary table
        table = Table(title="Research Session Summary", box=DOUBLE)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Iterations", str(result.total_iterations))
        table.add_row("Successful Hypotheses", f"[green]{result.successful_hypotheses}[/green]")
        table.add_row("Failed Hypotheses", f"[red]{result.total_iterations - result.successful_hypotheses}[/red]")
        table.add_row("Best Hypothesis", result.best_hypothesis_code or "None")
        table.add_row("Best Sharpe Ratio", f"[bold green]{result.best_sharpe:.2f}[/bold green]")
        table.add_row("Duration", f"{result.duration_seconds:.1f}s")

        if result.early_stopped:
            table.add_row("Early Stop", f"[yellow]{result.stop_reason}[/yellow]")

        self.console.print(table)
        self.console.print()

        # Final message
        if result.successful_hypotheses > 0:
            self.console.print(Panel(
                f"[bold green] Research successful! Found {result.successful_hypotheses} validated strategy/strategies.[/bold green]\n"
                f"Best: {result.best_hypothesis_code} with Sharpe {result.best_sharpe:.2f}",
                border_style="green",
            ))
        else:
            self.console.print(Panel(
                "[bold yellow] No validated strategies found in this session.[/bold yellow]\n"
                "Consider adjusting parameters or trying different approaches.",
                border_style="yellow",
            ))

        self.console.print()


class ThinkingStream:
    """
    Handles real-time streaming of LLM thinking tokens.
    Provides a callback interface for providers.
    """

    def __init__(self, reporter: ExplorationReporter):
        """
        Initialize thinking stream.

        Args:
            reporter: ExplorationReporter instance for display
        """
        self.reporter = reporter
        self._buffer = ""
        self._in_code_block = False

    async def on_token(self, token: str):
        """
        Called for each streaming token.

        Args:
            token: The token content
        """
        await self.reporter.stream_token_async(token)

    def on_token_sync(self, token: str):
        """
        Synchronous version of on_token.

        Args:
            token: The token content
        """
        self.reporter.stream_token(token)

    def on_complete(self):
        """Called when streaming completes."""
        self.reporter.flush_thinking_buffer()
