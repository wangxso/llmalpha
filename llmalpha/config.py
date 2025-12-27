"""
Configuration management for LLM Alpha.

Uses Pydantic for validation and YAML for configuration files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DownloadConfig(BaseModel):
    """Data download configuration."""
    proxy: str = "http://127.0.0.1:7890"
    use_proxy: bool = True
    concurrency: int = 100
    retries: int = 3
    months: int = 12


class DataIncludeConfig(BaseModel):
    """Data types to include in download."""
    klines: bool = True
    metrics: bool = True
    funding_rate: bool = True
    premium_index: bool = True
    mark_price: bool = False
    index_price: bool = False


class DataConfig(BaseModel):
    """Data layer configuration."""
    output_dir: str = "data"
    default_symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    )
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    include: DataIncludeConfig = Field(default_factory=DataIncludeConfig)


class BacktestConfig(BaseModel):
    """Backtest configuration."""
    init_cash: float = 10000
    fees: float = 0.0008
    slippage: float = 0.001
    freq: str = "1h"
    min_trades: int = 50


class OptimizeConfig(BaseModel):
    """Optimization configuration."""
    n_trials: int = 500
    timeout: int = 3600
    n_jobs: int = -1
    sampler: str = "tpe"
    pruner: str = "median"


class WalkForwardConfig(BaseModel):
    """Walk-Forward validation configuration."""
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    min_sharpe: float = 0.5
    decay_threshold: float = 0.5


class RollingConfig(BaseModel):
    """Rolling window validation configuration."""
    train_window: int = 2160  # 3 months in hours
    test_window: int = 720    # 1 month in hours
    step: int = 720           # 1 month step
    min_positive_ratio: float = 0.6


class KnowledgeConfig(BaseModel):
    """Knowledge base configuration."""
    db_path: str = "data/knowledge.db"
    auto_save: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/llmalpha.log"


class ResearchConfig(BaseModel):
    """Research configuration."""
    hypotheses_dir: str = "hypotheses"
    strategies_dir: str = "strategies"


class Settings(BaseSettings):
    """Main settings class."""
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    optimize: OptimizeConfig = Field(default_factory=OptimizeConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    rolling: RollingConfig = Field(default_factory=RollingConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)

    class Config:
        env_prefix = "LLMALPHA_"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Union[Path, str]) -> "Settings":
        """Load settings from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


def find_config_file() -> Optional[Path]:
    """Find configuration file in standard locations."""
    search_paths = [
        Path.cwd() / "configs" / "default.yaml",
        Path.cwd() / "config.yaml",
        Path.home() / ".llmalpha" / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    config_path = find_config_file()
    if config_path:
        return Settings.from_yaml(config_path)
    return Settings()
