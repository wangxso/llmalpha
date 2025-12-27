"""
Knowledge base layer for LLM Alpha.

Provides:
- SQLite database management
- SQLAlchemy ORM models
- Repository pattern for data access
- KnowledgeService for high-level operations
"""

from llmalpha.knowledge.models import (
    Base,
    Hypothesis,
    Strategy,
    BacktestResult,
    Factor,
    StrategyEvolution,
    ResearchNote,
    init_db,
    get_session,
)
from llmalpha.knowledge.repository import (
    HypothesisRepository,
    StrategyRepository,
    BacktestResultRepository,
    FactorRepository,
    ResearchNoteRepository,
)
from llmalpha.knowledge.service import (
    KnowledgeService,
    HypothesisSummary,
    StrategyComparison,
    get_knowledge_service,
)

__all__ = [
    # Models
    "Base",
    "Hypothesis",
    "Strategy",
    "BacktestResult",
    "Factor",
    "StrategyEvolution",
    "ResearchNote",
    "init_db",
    "get_session",
    # Repositories
    "HypothesisRepository",
    "StrategyRepository",
    "BacktestResultRepository",
    "FactorRepository",
    "ResearchNoteRepository",
    # Service
    "KnowledgeService",
    "HypothesisSummary",
    "StrategyComparison",
    "get_knowledge_service",
]
