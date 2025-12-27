"""
Repository Layer for Knowledge Base.

Provides CRUD operations for all entities.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from llmalpha.knowledge.models import (
    BacktestResult,
    Factor,
    Hypothesis,
    ResearchNote,
    Strategy,
    StrategyEvolution,
    get_session,
)


class HypothesisRepository:
    """Repository for Hypothesis operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        code: str,
        name: str,
        hypothesis_type: str = "strategy",
        factor_category: Optional[str] = None,
        description: Optional[str] = None,
        logic: Optional[str] = None,
        rationale: Optional[str] = None,
        parent_id: Optional[int] = None,
        entry_logic_desc: Optional[str] = None,
        exit_logic_desc: Optional[str] = None,
        indicators_used: Optional[str] = None,
        key_params: Optional[str] = None,
    ) -> Hypothesis:
        """Create a new hypothesis with enhanced description fields."""
        hypothesis = Hypothesis(
            code=code,
            name=name,
            hypothesis_type=hypothesis_type,
            factor_category=factor_category,
            description=description,
            logic=logic,
            rationale=rationale,
            parent_id=parent_id,
            entry_logic_desc=entry_logic_desc,
            exit_logic_desc=exit_logic_desc,
            indicators_used=indicators_used,
            key_params=key_params,
        )
        self.session.add(hypothesis)
        self.session.commit()
        return hypothesis

    def get_by_code(self, code: str) -> Optional[Hypothesis]:
        """Get hypothesis by code."""
        return self.session.query(Hypothesis).filter(Hypothesis.code == code).first()

    def get_by_id(self, id: int) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        return self.session.query(Hypothesis).filter(Hypothesis.id == id).first()

    def list(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Hypothesis]:
        """List hypotheses with optional filters."""
        query = self.session.query(Hypothesis)

        if status:
            query = query.filter(Hypothesis.status == status)
        if category:
            query = query.filter(Hypothesis.factor_category == category)

        return query.order_by(Hypothesis.created_at.desc()).limit(limit).all()

    def update_status(
        self,
        code: str,
        status: str,
        failure_reason: Optional[str] = None,
    ) -> Optional[Hypothesis]:
        """Update hypothesis status."""
        hypothesis = self.get_by_code(code)
        if hypothesis:
            hypothesis.status = status
            if failure_reason:
                hypothesis.failure_reason = failure_reason
            self.session.commit()
        return hypothesis

    def search(self, query: str) -> List[Hypothesis]:
        """Search hypotheses by name or description."""
        pattern = f"%{query}%"
        return (
            self.session.query(Hypothesis)
            .filter(
                or_(
                    Hypothesis.name.ilike(pattern),
                    Hypothesis.description.ilike(pattern),
                    Hypothesis.code.ilike(pattern),
                )
            )
            .all()
        )


class StrategyRepository:
    """Repository for Strategy operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        code: str,
        name: str,
        hypothesis_id: Optional[int] = None,
        strategy_type: str = "long_short",
        entry_logic: Optional[str] = None,
        exit_logic: Optional[str] = None,
        default_params: Optional[Dict] = None,
    ) -> Strategy:
        """Create a new strategy."""
        strategy = Strategy(
            code=code,
            name=name,
            hypothesis_id=hypothesis_id,
            strategy_type=strategy_type,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            default_params=json.dumps(default_params) if default_params else None,
        )
        self.session.add(strategy)
        self.session.commit()
        return strategy

    def get_by_code(self, code: str) -> Optional[Strategy]:
        """Get strategy by code."""
        return self.session.query(Strategy).filter(Strategy.code == code).first()

    def list(
        self,
        status: Optional[str] = None,
        strategy_type: Optional[str] = None,
    ) -> List[Strategy]:
        """List strategies with optional filters."""
        query = self.session.query(Strategy)

        if status:
            query = query.filter(Strategy.status == status)
        if strategy_type:
            query = query.filter(Strategy.strategy_type == strategy_type)

        return query.order_by(Strategy.created_at.desc()).all()

    def update_metrics(
        self,
        code: str,
        sharpe: float,
        total_return: float,
        max_dd: float,
    ) -> Optional[Strategy]:
        """Update strategy best metrics."""
        strategy = self.get_by_code(code)
        if strategy:
            # Only update if better
            if strategy.best_sharpe is None or sharpe > strategy.best_sharpe:
                strategy.best_sharpe = sharpe
                strategy.best_return = total_return
                strategy.best_max_dd = max_dd
                self.session.commit()
        return strategy


class BacktestResultRepository:
    """Repository for BacktestResult operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        strategy_id: Optional[int] = None,
        hypothesis_id: Optional[int] = None,
        symbols: List[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1h",
        params: Optional[Dict] = None,
        total_trades: int = 0,
        win_rate: float = 0.0,
        total_return: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        profit_factor: float = 0.0,
        train_sharpe: Optional[float] = None,
        val_sharpe: Optional[float] = None,
        test_sharpe: Optional[float] = None,
        wf_passed: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> BacktestResult:
        """Create a new backtest result."""
        result = BacktestResult(
            run_id=str(uuid.uuid4())[:8],
            strategy_id=strategy_id,
            hypothesis_id=hypothesis_id,
            symbols=",".join(symbols) if symbols else None,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            params=json.dumps(params) if params else None,
            total_trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            train_sharpe=train_sharpe,
            val_sharpe=val_sharpe,
            test_sharpe=test_sharpe,
            wf_passed=wf_passed,
            notes=notes,
        )
        self.session.add(result)
        self.session.commit()
        return result

    def list_by_strategy(self, strategy_code: str) -> List[BacktestResult]:
        """List results for a strategy."""
        return (
            self.session.query(BacktestResult)
            .join(Strategy)
            .filter(Strategy.code == strategy_code)
            .order_by(BacktestResult.created_at.desc())
            .all()
        )

    def list_by_hypothesis(self, hypothesis_code: str) -> List[BacktestResult]:
        """List results for a hypothesis."""
        return (
            self.session.query(BacktestResult)
            .join(Hypothesis)
            .filter(Hypothesis.code == hypothesis_code)
            .order_by(BacktestResult.created_at.desc())
            .all()
        )

    def get_best(
        self,
        strategy_code: Optional[str] = None,
        metric: str = "sharpe_ratio",
    ) -> Optional[BacktestResult]:
        """Get best backtest result by metric."""
        query = self.session.query(BacktestResult)

        if strategy_code:
            query = query.join(Strategy).filter(Strategy.code == strategy_code)

        return query.order_by(getattr(BacktestResult, metric).desc()).first()


class FactorRepository:
    """Repository for Factor operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        code: str,
        name: str,
        category: str,
        description: Optional[str] = None,
        formula: Optional[str] = None,
        default_params: Optional[Dict] = None,
    ) -> Factor:
        """Create a new factor."""
        factor = Factor(
            code=code,
            name=name,
            category=category,
            description=description,
            formula=formula,
            default_params=json.dumps(default_params) if default_params else None,
        )
        self.session.add(factor)
        self.session.commit()
        return factor

    def get_by_code(self, code: str) -> Optional[Factor]:
        """Get factor by code."""
        return self.session.query(Factor).filter(Factor.code == code).first()

    def list(self, category: Optional[str] = None) -> List[Factor]:
        """List factors with optional category filter."""
        query = self.session.query(Factor)
        if category:
            query = query.filter(Factor.category == category)
        return query.all()

    def update_metrics(
        self,
        code: str,
        avg_ic: float,
        icir: float,
        is_validated: bool = False,
    ) -> Optional[Factor]:
        """Update factor metrics."""
        factor = self.get_by_code(code)
        if factor:
            factor.avg_ic = avg_ic
            factor.icir = icir
            factor.is_validated = is_validated
            self.session.commit()
        return factor


class ResearchNoteRepository:
    """Repository for ResearchNote operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        title: str,
        content: str,
        category: str = "insight",
        hypothesis_id: Optional[int] = None,
        strategy_id: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> ResearchNote:
        """Create a new research note."""
        note = ResearchNote(
            title=title,
            content=content,
            category=category,
            hypothesis_id=hypothesis_id,
            strategy_id=strategy_id,
            tags=",".join(tags) if tags else None,
        )
        self.session.add(note)
        self.session.commit()
        return note

    def search(self, query: str) -> List[ResearchNote]:
        """Search notes by title or content."""
        pattern = f"%{query}%"
        return (
            self.session.query(ResearchNote)
            .filter(
                or_(
                    ResearchNote.title.ilike(pattern),
                    ResearchNote.content.ilike(pattern),
                )
            )
            .all()
        )
