"""
SQLAlchemy ORM Models for Knowledge Base.

Defines database schema for tracking:
- Hypotheses
- Strategies
- Backtest Results
- Factors
- Strategy Evolution
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Hypothesis(Base):
    """
    Hypothesis table.

    Tracks research hypotheses and their validation status.
    """
    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), unique=True, nullable=False, index=True)  # e.g., "H001"
    name = Column(String(200), nullable=False)
    hypothesis_type = Column(String(50))  # "factor", "strategy", "pattern"
    factor_category = Column(String(50))  # "momentum", "mean_reversion", "sentiment", etc.

    # Hypothesis content
    description = Column(Text)  # Short one-line description
    logic = Column(Text)  # Entry/exit logic description (can store code)
    rationale = Column(Text)  # Why this should work

    # Enhanced description fields for better LLM context
    entry_logic_desc = Column(Text)  # Human-readable entry condition description
    exit_logic_desc = Column(Text)   # Human-readable exit condition description
    indicators_used = Column(Text)   # Comma-separated list: "RSI,MACD,ATR"
    key_params = Column(Text)        # JSON: {"rsi_period": 14, "threshold": 30}

    # Status
    status = Column(String(20), default="pending")  # pending, validated, failed, deprecated
    failure_reason = Column(Text)

    # Performance metrics (best results)
    best_sharpe = Column(Float)
    best_win_rate = Column(Float)
    best_total_trades = Column(Integer)
    best_max_drawdown = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parent_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=True)
    parent = relationship("Hypothesis", remote_side=[id], backref="children")

    strategies = relationship("Strategy", back_populates="hypothesis")
    backtest_results = relationship("BacktestResult", back_populates="hypothesis")


class Strategy(Base):
    """
    Strategy table.

    Represents a validated trading strategy derived from hypotheses.
    """
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), unique=True, nullable=False, index=True)  # e.g., "S001"
    name = Column(String(200), nullable=False)

    # Link to hypothesis
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"))
    hypothesis = relationship("Hypothesis", back_populates="strategies")

    # Strategy details
    strategy_type = Column(String(50))  # "long_only", "short_only", "long_short", "market_neutral"
    entry_logic = Column(Text)
    exit_logic = Column(Text)
    default_params = Column(Text)  # JSON string

    # Status and version
    status = Column(String(20), default="development")  # development, testing, production, deprecated
    version = Column(Integer, default=1)

    # Performance metrics (from best backtest)
    best_sharpe = Column(Float)
    best_return = Column(Float)
    best_max_dd = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    backtest_results = relationship("BacktestResult", back_populates="strategy")
    evolutions = relationship("StrategyEvolution", back_populates="strategy")


class BacktestResult(Base):
    """
    Backtest Result table.

    Stores individual backtest runs with all metrics.
    """
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)  # UUID

    # Links
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    strategy = relationship("Strategy", back_populates="backtest_results")
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"))
    hypothesis = relationship("Hypothesis", back_populates="backtest_results")

    # Run parameters
    symbols = Column(Text)  # Comma-separated list
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    timeframe = Column(String(10))  # "1m", "5m", "1h", etc.
    params = Column(Text)  # JSON string

    # Core metrics
    total_trades = Column(Integer)
    win_rate = Column(Float)
    total_return = Column(Float)  # Percentage
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)  # Percentage
    profit_factor = Column(Float)

    # Walk-forward metrics
    train_sharpe = Column(Float)
    val_sharpe = Column(Float)
    test_sharpe = Column(Float)
    wf_passed = Column(Boolean)

    # Rolling validation
    rolling_positive_ratio = Column(Float)
    rolling_avg_sharpe = Column(Float)
    rolling_passed = Column(Boolean)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Notes
    notes = Column(Text)


class Factor(Base):
    """
    Factor table.

    Tracks factor definitions and their effectiveness.
    """
    __tablename__ = "factors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50))  # "momentum", "mean_reversion", "volatility", etc.

    # Factor details
    description = Column(Text)
    formula = Column(Text)  # Mathematical formula or code reference
    default_params = Column(Text)  # JSON string

    # Effectiveness metrics
    avg_ic = Column(Float)  # Information Coefficient
    icir = Column(Float)  # IC Information Ratio
    turnover = Column(Float)
    is_validated = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StrategyEvolution(Base):
    """
    Strategy Evolution table.

    Tracks how strategies evolve over time.
    """
    __tablename__ = "strategy_evolutions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    strategy = relationship("Strategy", back_populates="evolutions")

    from_version = Column(Integer)
    to_version = Column(Integer)
    change_type = Column(String(50))  # "param_tune", "logic_change", "factor_add", "factor_remove"

    # Performance comparison
    before_sharpe = Column(Float)
    after_sharpe = Column(Float)
    before_return = Column(Float)
    after_return = Column(Float)

    # Change details
    change_description = Column(Text)
    params_before = Column(Text)  # JSON
    params_after = Column(Text)  # JSON

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)


class ResearchNote(Base):
    """
    Research Notes table.

    Stores insights and observations from research.
    """
    __tablename__ = "research_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    category = Column(String(50))  # "insight", "bug", "idea", "observation"

    # Links (optional)
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)

    # Tags (comma-separated)
    tags = Column(Text)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization

def init_db(db_path: str = "data/knowledge.db") -> sessionmaker:
    """
    Initialize the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Session factory
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def get_session(db_path: str = "data/knowledge.db"):
    """
    Get a database session.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Session instance
    """
    Session = init_db(db_path)
    return Session()
