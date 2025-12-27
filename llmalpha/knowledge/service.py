"""
Knowledge Service for high-level operations.

Provides business logic and convenience methods for knowledge management.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from llmalpha.backtest.result import BacktestResult as BTResult
from llmalpha.backtest.result import WalkForwardResult
from llmalpha.knowledge.models import (
    BacktestResult,
    Factor,
    Hypothesis,
    Strategy,
    get_session,
    init_db,
)
from llmalpha.knowledge.repository import (
    BacktestResultRepository,
    FactorRepository,
    HypothesisRepository,
    ResearchNoteRepository,
    StrategyRepository,
)


@dataclass
class HypothesisSummary:
    """Summary of a hypothesis and its results."""
    code: str
    name: str
    status: str
    category: str
    n_backtests: int
    best_sharpe: Optional[float]
    best_return: Optional[float]
    wf_passed: bool


@dataclass
class StrategyComparison:
    """Comparison of two strategies."""
    strategy1: str
    strategy2: str
    sharpe_diff: float
    return_diff: float
    drawdown_diff: float
    winner: str
    comparison_notes: str


class KnowledgeService:
    """
    High-level service for knowledge base operations.

    Provides:
    - Hypothesis lifecycle management
    - Strategy tracking
    - Backtest result storage
    - Search and comparison functionality
    - Evolution tracking

    Example:
        service = KnowledgeService()
        service.record_hypothesis(
            code="H027",
            name="Trend Following with Volume",
            category="momentum",
        )
        service.record_backtest_result(...)
    """

    def __init__(self, db_path: str = "data/knowledge.db"):
        """
        Initialize the service.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._session = None
        self._init_repos()

    def _init_repos(self):
        """Initialize repositories."""
        session = get_session(self.db_path)
        self._session = session
        self.hypotheses = HypothesisRepository(session)
        self.strategies = StrategyRepository(session)
        self.backtests = BacktestResultRepository(session)
        self.factors = FactorRepository(session)
        self.notes = ResearchNoteRepository(session)

    # ============ Hypothesis Management ============

    def record_hypothesis(
        self,
        code: str,
        name: str,
        category: str = "general",
        hypothesis_type: str = "strategy",
        description: Optional[str] = None,
        logic: Optional[str] = None,
        rationale: Optional[str] = None,
        parent_code: Optional[str] = None,
        entry_logic_desc: Optional[str] = None,
        exit_logic_desc: Optional[str] = None,
        indicators_used: Optional[str] = None,
        key_params: Optional[str] = None,
    ) -> Hypothesis:
        """
        Record a new hypothesis with detailed descriptions.

        Args:
            code: Unique hypothesis code (e.g., "H027")
            name: Descriptive name
            category: Factor category
            hypothesis_type: Type of hypothesis
            description: Short one-line description
            logic: Entry/exit logic (code or description)
            rationale: Why this should work
            parent_code: Code of parent hypothesis
            entry_logic_desc: Human-readable entry condition
            exit_logic_desc: Human-readable exit condition
            indicators_used: Comma-separated indicators (e.g., "RSI,MACD,ATR")
            key_params: JSON string of key parameters

        Returns:
            Created Hypothesis object
        """
        parent_id = None
        if parent_code:
            parent = self.hypotheses.get_by_code(parent_code)
            if parent:
                parent_id = parent.id

        # Auto-extract indicators if not provided
        if not indicators_used and logic:
            indicators_used = self._extract_indicators(logic)

        return self.hypotheses.create(
            code=code,
            name=name,
            factor_category=category,
            hypothesis_type=hypothesis_type,
            description=description,
            logic=logic,
            rationale=rationale,
            parent_id=parent_id,
            entry_logic_desc=entry_logic_desc,
            exit_logic_desc=exit_logic_desc,
            indicators_used=indicators_used,
            key_params=key_params,
        )

    def _extract_indicators(self, logic: str) -> str:
        """Auto-extract indicator names from logic/code."""
        if not logic:
            return ""

        indicators = []
        logic_lower = logic.lower()

        indicator_patterns = [
            ("rsi", "RSI"),
            ("macd", "MACD"),
            ("ema", "EMA"),
            ("sma", "SMA"),
            ("bollinger", "Bollinger"),
            ("atr", "ATR"),
            ("donchian", "Donchian"),
            ("volume", "Volume"),
            ("funding", "FundingRate"),
            ("stoch", "Stochastic"),
            ("adx", "ADX"),
            ("obv", "OBV"),
            ("vwap", "VWAP"),
        ]

        for pattern, name in indicator_patterns:
            if pattern in logic_lower:
                indicators.append(name)

        return ",".join(indicators)

    def validate_hypothesis(self, code: str) -> Optional[Hypothesis]:
        """Mark hypothesis as validated."""
        return self.hypotheses.update_status(code, "validated")

    def fail_hypothesis(self, code: str, reason: str) -> Optional[Hypothesis]:
        """Mark hypothesis as failed."""
        return self.hypotheses.update_status(code, "failed", reason)

    def update_hypothesis_metrics(
        self,
        code: str,
        sharpe: float,
        win_rate: float,
        total_trades: int,
        max_drawdown: float = 0.0,
    ) -> Optional[Hypothesis]:
        """
        Update hypothesis with best performance metrics.

        Only updates if new sharpe is better than existing.
        """
        hypothesis = self.hypotheses.get_by_code(code)
        if not hypothesis:
            return None

        # Only update if this is better
        current_best = hypothesis.best_sharpe or 0
        if sharpe > current_best:
            hypothesis.best_sharpe = sharpe
            hypothesis.best_win_rate = win_rate
            hypothesis.best_total_trades = total_trades
            hypothesis.best_max_drawdown = max_drawdown
            self._session.commit()

        return hypothesis

    def get_hypothesis_summary(self, code: str) -> Optional[HypothesisSummary]:
        """Get summary of a hypothesis including backtest results."""
        hypothesis = self.hypotheses.get_by_code(code)
        if not hypothesis:
            return None

        results = self.backtests.list_by_hypothesis(code)

        best_sharpe = None
        best_return = None
        wf_passed = False

        for r in results:
            if best_sharpe is None or (r.sharpe_ratio and r.sharpe_ratio > best_sharpe):
                best_sharpe = r.sharpe_ratio
            if best_return is None or (r.total_return and r.total_return > best_return):
                best_return = r.total_return
            if r.wf_passed:
                wf_passed = True

        return HypothesisSummary(
            code=hypothesis.code,
            name=hypothesis.name,
            status=hypothesis.status,
            category=hypothesis.factor_category or "general",
            n_backtests=len(results),
            best_sharpe=best_sharpe,
            best_return=best_return,
            wf_passed=wf_passed,
        )

    # ============ Backtest Results ============

    def record_backtest(
        self,
        result: BTResult,
        hypothesis_code: Optional[str] = None,
        strategy_code: Optional[str] = None,
        params: Optional[Dict] = None,
        wf_result: Optional[WalkForwardResult] = None,
        notes: Optional[str] = None,
    ) -> BacktestResult:
        """
        Record a backtest result.

        Args:
            result: BacktestResult from engine
            hypothesis_code: Related hypothesis code
            strategy_code: Related strategy code
            params: Parameters used
            wf_result: Walk-forward validation result
            notes: Additional notes

        Returns:
            Stored BacktestResult object
        """
        hypothesis_id = None
        if hypothesis_code:
            hypothesis = self.hypotheses.get_by_code(hypothesis_code)
            if hypothesis:
                hypothesis_id = hypothesis.id

        strategy_id = None
        if strategy_code:
            strategy = self.strategies.get_by_code(strategy_code)
            if strategy:
                strategy_id = strategy.id

        # Extract WF metrics if available
        train_sharpe = None
        val_sharpe = None
        test_sharpe = None
        wf_passed = None

        if wf_result:
            train_sharpe = wf_result.train_result.sharpe_ratio
            val_sharpe = wf_result.val_result.sharpe_ratio
            test_sharpe = wf_result.test_result.sharpe_ratio
            wf_passed = wf_result.passed

        return self.backtests.create(
            strategy_id=strategy_id,
            hypothesis_id=hypothesis_id,
            symbols=result.symbols,
            start_date=result.start_date,
            end_date=result.end_date,
            params=params,
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            profit_factor=result.profit_factor,
            train_sharpe=train_sharpe,
            val_sharpe=val_sharpe,
            test_sharpe=test_sharpe,
            wf_passed=wf_passed,
            notes=notes,
        )

    # ============ Search and Query ============

    def search_hypotheses(
        self,
        query: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Hypothesis]:
        """
        Search hypotheses.

        Args:
            query: Text search query
            status: Filter by status
            category: Filter by category

        Returns:
            List of matching hypotheses
        """
        if query:
            return self.hypotheses.search(query)
        return self.hypotheses.list(status=status, category=category)

    def list_failed_hypotheses(self, category: Optional[str] = None) -> List[Hypothesis]:
        """List all failed hypotheses."""
        return self.hypotheses.list(status="failed", category=category)

    def list_validated_hypotheses(self) -> List[Hypothesis]:
        """List all validated hypotheses."""
        return self.hypotheses.list(status="validated")

    # ============ Comparison ============

    def compare_strategies(
        self,
        code1: str,
        code2: str,
    ) -> Optional[StrategyComparison]:
        """
        Compare two strategies.

        Args:
            code1: First strategy code
            code2: Second strategy code

        Returns:
            StrategyComparison object
        """
        s1 = self.strategies.get_by_code(code1)
        s2 = self.strategies.get_by_code(code2)

        if not s1 or not s2:
            return None

        sharpe1 = s1.best_sharpe or 0
        sharpe2 = s2.best_sharpe or 0
        return1 = s1.best_return or 0
        return2 = s2.best_return or 0
        dd1 = s1.best_max_dd or 0
        dd2 = s2.best_max_dd or 0

        winner = code1 if sharpe1 > sharpe2 else code2

        notes = []
        if sharpe1 > sharpe2:
            notes.append(f"{code1} has better risk-adjusted returns")
        elif sharpe2 > sharpe1:
            notes.append(f"{code2} has better risk-adjusted returns")

        if return1 > return2:
            notes.append(f"{code1} has higher total return")
        elif return2 > return1:
            notes.append(f"{code2} has higher total return")

        if dd1 < dd2:
            notes.append(f"{code1} has lower drawdown")
        elif dd2 < dd1:
            notes.append(f"{code2} has lower drawdown")

        return StrategyComparison(
            strategy1=code1,
            strategy2=code2,
            sharpe_diff=sharpe1 - sharpe2,
            return_diff=return1 - return2,
            drawdown_diff=dd1 - dd2,
            winner=winner,
            comparison_notes="; ".join(notes),
        )

    # ============ Statistics ============

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall knowledge base statistics."""
        all_hypotheses = self.hypotheses.list(limit=1000)
        all_strategies = self.strategies.list()

        total_h = len(all_hypotheses)
        validated_h = len([h for h in all_hypotheses if h.status == "validated"])
        failed_h = len([h for h in all_hypotheses if h.status == "failed"])
        pending_h = len([h for h in all_hypotheses if h.status == "pending"])

        categories = {}
        for h in all_hypotheses:
            cat = h.factor_category or "general"
            if cat not in categories:
                categories[cat] = {"total": 0, "validated": 0, "failed": 0}
            categories[cat]["total"] += 1
            if h.status == "validated":
                categories[cat]["validated"] += 1
            elif h.status == "failed":
                categories[cat]["failed"] += 1

        return {
            "hypotheses": {
                "total": total_h,
                "validated": validated_h,
                "failed": failed_h,
                "pending": pending_h,
                "success_rate": validated_h / total_h if total_h > 0 else 0,
            },
            "strategies": {
                "total": len(all_strategies),
                "production": len([s for s in all_strategies if s.status == "production"]),
            },
            "categories": categories,
        }


# Convenience function
def get_knowledge_service(db_path: str = "data/knowledge.db") -> KnowledgeService:
    """Get a knowledge service instance."""
    return KnowledgeService(db_path)
