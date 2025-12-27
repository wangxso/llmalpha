"""
LLM Alpha - Cryptocurrency Strategy Research Framework

A comprehensive framework for cryptocurrency trading strategy research,
backtesting, optimization, and knowledge management.
"""

__version__ = "0.1.0"
__author__ = "Noah"

from llmalpha.config import Settings, get_settings

# Lazy imports for heavy modules
def get_researcher():
    """Get the AlphaResearcher class."""
    from llmalpha.agent import AlphaResearcher
    return AlphaResearcher

def create_researcher(**kwargs):
    """Create an AlphaResearcher instance."""
    from llmalpha.agent import create_researcher as _create
    return _create(**kwargs)

__all__ = [
    "Settings",
    "get_settings",
    "get_researcher",
    "create_researcher",
    "__version__",
]
