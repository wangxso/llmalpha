"""
CLI Main Entry Point for LLM Alpha.

Provides command-line interface using Click.
"""

import os
import warnings
from pathlib import Path

import click

from llmalpha import __version__

# Suppress pandas FutureWarnings from generated code
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_dotenv():
    """Load .env file from project root if exists."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try project root
        env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load .env on import
load_dotenv()


@click.group()
@click.version_option(version=__version__)
def cli():
    """LLM Alpha - Cryptocurrency Strategy Research Framework."""
    pass


# ============ Data Commands ============

@cli.group()
def data():
    """Data download and management commands."""
    pass


@data.command("download")
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols (e.g., BTC,ETH)")
@click.option("--months", "-m", default=12, help="Number of months to download")
@click.option("--output", "-o", default="data", help="Output directory")
@click.option("--concurrency", "-c", default=100, help="Concurrent connections")
@click.option("--no-proxy", is_flag=True, help="Disable proxy")
@click.option("--only-klines", is_flag=True, help="Only download K-lines")
@click.option("--intervals", "-i", default="1m", help="K-line intervals (e.g., 1m,5m,1h)")
def download_data(symbols, months, output, concurrency, no_proxy, only_klines, intervals):
    """Download data from Binance."""
    import asyncio
    from llmalpha.data.downloader import BinanceDownloader, DEFAULT_SYMBOLS

    # Parse symbols
    if symbols:
        symbol_list = [s.strip().upper() + "USDT" if not s.endswith("USDT") else s.strip().upper()
                       for s in symbols.split(",")]
    else:
        symbol_list = DEFAULT_SYMBOLS

    # Parse intervals
    interval_list = [i.strip() for i in intervals.split(",")]

    # Get proxy from config if not disabled
    proxy = None
    if not no_proxy:
        from llmalpha.config import get_settings
        settings = get_settings()
        if settings.data.download.use_proxy:
            proxy = settings.data.download.proxy

    # Create downloader
    downloader = BinanceDownloader(
        symbols=symbol_list,
        months=months,
        output_dir=output,
        concurrency=concurrency,
        proxy=proxy,
        intervals=interval_list,
        include_metrics=not only_klines,
        include_funding_rate=not only_klines,
        include_premium=not only_klines,
    )

    # Run download
    asyncio.run(downloader.download_all())


@data.command("list")
@click.option("--path", "-p", default="data", help="Data directory")
def list_data(path):
    """List available data files."""
    from llmalpha.data.loader import DataLoader

    loader = DataLoader(path)
    symbols = loader.list_symbols()

    if not symbols:
        click.echo("No data files found.")
        return

    click.echo(f"Found {len(symbols)} symbols:")
    for symbol in symbols:
        try:
            df = loader.load_symbol(symbol)
            days = (df.index.max() - df.index.min()).days
            click.echo(f"  {symbol}: {len(df):,} rows ({days} days)")
        except Exception as e:
            click.echo(f"  {symbol}: Error - {e}")


# ============ Backtest Commands ============

@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command("run")
@click.option("--data-dir", "-d", default="data", help="Data directory")
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols")
@click.option("--timeframe", "-t", default="1h", help="Resample timeframe")
@click.option("--strategy", "-S", default=None, help="Strategy code (e.g., 'rsi_mr')")
@click.option("--days", default=None, type=int, help="Limit to last N days")
def run_backtest(data_dir, symbols, timeframe, strategy, days):
    """Run a backtest with a strategy."""
    from llmalpha.data.loader import DataLoader
    from llmalpha.backtest.vbt_engine import VBTEngine
    from llmalpha.strategies import get_strategy

    loader = DataLoader(data_dir)

    # Load data
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        data = loader.load_symbols(symbol_list, resample=timeframe)
    else:
        data = loader.load_all(resample=timeframe)

    if not data:
        click.echo("No data loaded.")
        return

    click.echo(f"Loaded {len(data)} symbols")

    # Get strategy
    if strategy:
        strategy_class = get_strategy(strategy)
        if not strategy_class:
            click.echo(f"Strategy '{strategy}' not found.")
            click.echo("Use 'llmalpha research list' to see available strategies.")
            return
        strat = strategy_class()
    else:
        click.echo("No strategy specified. Use --strategy/-S to specify a strategy code.")
        click.echo("Example: llmalpha backtest run --strategy rsi_mr")
        return

    # Run backtest for each symbol
    engine = VBTEngine()
    results = []

    for symbol, df in data.items():
        if "close" not in df.columns:
            continue

        try:
            signals = strat.generate_signals(df)
            result = engine.run(
                close=df["close"],
                entries=signals.entries,
                exits=signals.exits,
                short_entries=signals.short_entries,
                short_exits=signals.short_exits,
                strategy_name=f"{strat.code}_{symbol}",
            )
            results.append((symbol, result))
            click.echo(f"  {symbol}: {result.total_trades} trades, Sharpe: {result.sharpe_ratio:.2f}")
        except Exception as e:
            click.echo(f"  {symbol}: Error - {e}")

    if not results:
        click.echo("No results generated.")
        return

    # Display summary
    import numpy as np
    total_trades = sum(r.total_trades for _, r in results)
    avg_sharpe = np.mean([r.sharpe_ratio for _, r in results])
    avg_return = np.mean([r.total_return for _, r in results])
    avg_win_rate = np.mean([r.win_rate for _, r in results if r.win_rate > 0])

    click.echo("\n" + "=" * 50)
    click.echo(f"Backtest Results - {strat.name}")
    click.echo("=" * 50)
    click.echo(f"Symbols: {len(results)}")
    click.echo(f"Total Trades: {total_trades}")
    click.echo(f"Avg Win Rate: {avg_win_rate:.1%}")
    click.echo(f"Avg Return: {avg_return:.2%}")
    click.echo(f"Avg Sharpe: {avg_sharpe:.2f}")


# ============ Optimize Commands ============

@cli.group()
def optimize():
    """Parameter optimization commands."""
    pass


@optimize.command("run")
@click.option("--data-dir", "-d", default="data", help="Data directory")
@click.option("--trials", "-n", default=100, help="Number of trials")
@click.option("--jobs", "-j", default=1, help="Parallel jobs")
@click.option("--val-days", default=7, help="Validation days")
@click.option("--test-days", default=7, help="Test days")
def run_optimize(data_dir, trials, jobs, val_days, test_days):
    """Run parameter optimization."""
    click.echo("Optimization not yet implemented in CLI.")
    click.echo("Use the Python API directly for now.")


# ============ Knowledge Base Commands ============

@cli.group()
def kb():
    """Knowledge base commands."""
    pass


@kb.command("init")
@click.option("--db-path", default="data/knowledge.db", help="Database path")
def init_kb(db_path):
    """Initialize the knowledge base."""
    from llmalpha.knowledge import init_db

    init_db(db_path)
    click.echo(f"Knowledge base initialized at {db_path}")


@kb.command("search")
@click.option("--query", "-q", default=None, help="Search query")
@click.option("--status", "-s", default=None, help="Filter by status")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.option("--db-path", default="data/knowledge.db", help="Database path")
def search_kb(query, status, category, db_path):
    """Search the knowledge base."""
    from llmalpha.knowledge import get_knowledge_service

    service = get_knowledge_service(db_path)
    results = service.search_hypotheses(query=query, status=status, category=category)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Found {len(results)} hypotheses:")
    for h in results:
        status_icon = {"validated": "V", "failed": "X", "pending": "?"}.get(h.status, "?")
        click.echo(f"  [{status_icon}] {h.code}: {h.name}")


@kb.command("stats")
@click.option("--db-path", default="data/knowledge.db", help="Database path")
def kb_stats(db_path):
    """Show knowledge base statistics."""
    from llmalpha.knowledge import get_knowledge_service

    service = get_knowledge_service(db_path)
    stats = service.get_statistics()

    click.echo("\n" + "=" * 50)
    click.echo("Knowledge Base Statistics")
    click.echo("=" * 50)

    h = stats["hypotheses"]
    click.echo(f"\nHypotheses: {h['total']}")
    click.echo(f"  Validated: {h['validated']}")
    click.echo(f"  Failed: {h['failed']}")
    click.echo(f"  Pending: {h['pending']}")
    click.echo(f"  Success Rate: {h['success_rate']:.1%}")

    s = stats["strategies"]
    click.echo(f"\nStrategies: {s['total']}")
    click.echo(f"  Production: {s['production']}")

    click.echo("\nBy Category:")
    for cat, data in stats["categories"].items():
        click.echo(f"  {cat}: {data['total']} ({data['validated']} validated)")


@kb.command("compare")
@click.argument("strategy1")
@click.argument("strategy2")
@click.option("--db-path", default="data/knowledge.db", help="Database path")
def compare_strategies(strategy1, strategy2, db_path):
    """Compare two strategies."""
    from llmalpha.knowledge import get_knowledge_service

    service = get_knowledge_service(db_path)
    result = service.compare_strategies(strategy1, strategy2)

    if not result:
        click.echo("One or both strategies not found.")
        return

    click.echo("\n" + "=" * 50)
    click.echo(f"Comparing {strategy1} vs {strategy2}")
    click.echo("=" * 50)
    click.echo(f"Sharpe Diff: {result.sharpe_diff:+.2f}")
    click.echo(f"Return Diff: {result.return_diff:+.2%}")
    click.echo(f"Drawdown Diff: {result.drawdown_diff:+.2%}")
    click.echo(f"Winner: {result.winner}")
    click.echo(f"Notes: {result.comparison_notes}")


# ============ Research Commands ============

@cli.group()
def research():
    """Research and hypothesis testing commands."""
    pass


@research.command("list")
@click.option("--db-path", default="data/knowledge.db", help="Database path")
def list_research(db_path):
    """List all hypotheses."""
    from llmalpha.knowledge import get_knowledge_service

    service = get_knowledge_service(db_path)

    for status in ["validated", "failed", "pending"]:
        results = service.search_hypotheses(status=status)
        if results:
            click.echo(f"\n{status.upper()} ({len(results)}):")
            for h in results:
                click.echo(f"  {h.code}: {h.name}")


@research.command("record")
@click.argument("code")
@click.argument("name")
@click.option("--category", "-c", default="general", help="Category")
@click.option("--description", "-d", default=None, help="Description")
@click.option("--db-path", default="data/knowledge.db", help="Database path")
def record_hypothesis(code, name, category, description, db_path):
    """Record a new hypothesis."""
    from llmalpha.knowledge import get_knowledge_service

    service = get_knowledge_service(db_path)
    h = service.record_hypothesis(
        code=code,
        name=name,
        category=category,
        description=description,
    )
    click.echo(f"Recorded hypothesis: {h.code} - {h.name}")


# ============ Agent Commands ============

@cli.group()
def agent():
    """LLM-driven autonomous research commands."""
    pass


@agent.command("research")
@click.option("--requirements", "-r", default="Find profitable trading strategies", help="Research requirements")
@click.option("--category", "-c", default=None, help="Strategy category (momentum, mean_reversion, etc.)")
@click.option("--iterations", "-n", default=20, help="Max iterations")
@click.option("--max-failures", "-f", default=5, help="Max consecutive failures before stopping")
@click.option("--provider", "-p", default="openai", help="LLM provider (openai, anthropic, ollama, deepseek)")
@click.option("--model", "-m", default="gpt-5.2", help="Model name")
@click.option("--base-url", default=None, help="Custom API base URL (e.g., https://api.gptsapi.net)")
@click.option("--api-key", default=None, help="API key (or use env var OPENAI_API_KEY)")
@click.option("--data-dir", "-d", default="data", help="Data directory")
@click.option("--db-path", default="data/knowledge.db", help="Knowledge base path")
@click.option("--early-stop-sharpe", default=1.5, help="Stop if Sharpe reaches this value")
@click.option("--immersive/--no-immersive", default=True, help="Enable immersive exploration UI")
def run_research(requirements, category, iterations, max_failures, provider, model, base_url, api_key, data_dir, db_path, early_stop_sharpe, immersive):
    """Run LLM-driven strategy research."""
    import asyncio
    from llmalpha.agent import create_researcher, AgentConfig, LLMConfig

    # Only show basic header in non-immersive mode
    # In immersive mode, the reporter will show a nice banner
    if not immersive:
        click.echo("\n" + "=" * 60)
        click.echo("LLM Alpha - Autonomous Research")
        click.echo("=" * 60)
        click.echo(f"Provider: {provider}")
        click.echo(f"Model: {model}")
        if base_url:
            click.echo(f"Base URL: {base_url}")
        click.echo(f"Category: {category or 'general'}")
        click.echo(f"Max Iterations: {iterations}")
        click.echo(f"Max Consecutive Failures: {max_failures}")
        click.echo("=" * 60 + "\n")

    # Create researcher
    researcher = create_researcher(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        data_dir=data_dir,
        db_path=db_path,
        max_iterations=iterations,
        max_consecutive_failures=max_failures,
        category=category,
        early_stop_sharpe=early_stop_sharpe,
        immersive=immersive,
    )

    # Run research
    async def _run():
        return await researcher.research(
            requirements=requirements,
            category=category,
        )

    try:
        result = asyncio.run(_run())

        # Only print summary in non-immersive mode
        # In immersive mode, the reporter already showed the summary
        if not immersive:
            click.echo("\n" + result.summary())

        if result.best_hypothesis_code:
            click.echo(f"\nBest hypothesis saved: {result.best_hypothesis_code}")
            click.echo("Use 'llmalpha kb search' to view details.")

    except Exception as e:
        click.echo(f"\nResearch failed: {e}")
        raise click.Abort()


@agent.command("improve")
@click.argument("hypothesis_code")
@click.option("--iterations", "-n", default=3, help="Max improvement iterations")
@click.option("--provider", "-p", default="openai", help="LLM provider")
@click.option("--model", "-m", default="gpt-5.2", help="Model name")
@click.option("--data-dir", "-d", default="data", help="Data directory")
@click.option("--db-path", default="data/knowledge.db", help="Knowledge base path")
def improve_hypothesis(hypothesis_code, iterations, provider, model, data_dir, db_path):
    """Improve an existing hypothesis."""
    import asyncio
    from llmalpha.agent import create_researcher

    click.echo(f"\nImproving hypothesis: {hypothesis_code}")
    click.echo(f"Max iterations: {iterations}\n")

    researcher = create_researcher(
        provider=provider,
        model=model,
        data_dir=data_dir,
        db_path=db_path,
    )

    async def _run():
        return await researcher.improve_hypothesis(
            hypothesis_code=hypothesis_code,
            max_iterations=iterations,
        )

    try:
        result = asyncio.run(_run())
        click.echo("\n" + result.summary())
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise click.Abort()
    except Exception as e:
        click.echo(f"Improvement failed: {e}")
        raise click.Abort()


@agent.command("analyze")
@click.option("--limit", "-n", default=10, help="Number of failures to analyze")
@click.option("--db-path", default="data/knowledge.db", help="Knowledge base path")
def analyze_failures(limit, db_path):
    """Analyze recent failures to identify patterns."""
    import asyncio
    from llmalpha.agent import create_researcher

    researcher = create_researcher(db_path=db_path)

    async def _run():
        return await researcher.analyze_failures(limit=limit)

    analysis = asyncio.run(_run())

    if "error" in analysis:
        click.echo(f"Error: {analysis['error']}")
        return

    if "message" in analysis:
        click.echo(analysis["message"])
        return

    click.echo("\n" + "=" * 50)
    click.echo("Failure Analysis")
    click.echo("=" * 50)
    click.echo(f"\nTotal Failures Analyzed: {analysis['total_failures']}")

    click.echo("\nBy Category:")
    for cat, codes in analysis["categories"].items():
        click.echo(f"  {cat}: {len(codes)} ({', '.join(codes[:3])}...)")

    if analysis["recommendations"]:
        click.echo("\nRecommendations:")
        for rec in analysis["recommendations"]:
            click.echo(f"  • {rec}")


@agent.command("status")
@click.option("--db-path", default="data/knowledge.db", help="Knowledge base path")
def agent_status(db_path):
    """Show agent research status and statistics."""
    from llmalpha.agent import create_researcher

    researcher = create_researcher(db_path=db_path)
    stats = researcher.get_statistics()

    if "error" in stats:
        click.echo(f"Error: {stats['error']}")
        return

    click.echo("\n" + "=" * 50)
    click.echo("Agent Research Status")
    click.echo("=" * 50)

    h = stats.get("hypotheses", {})
    click.echo(f"\nHypotheses:")
    click.echo(f"  Total: {h.get('total', 0)}")
    click.echo(f"  Validated: {h.get('validated', 0)}")
    click.echo(f"  Failed: {h.get('failed', 0)}")
    click.echo(f"  Success Rate: {h.get('success_rate', 0):.1%}")

    # Show top validated
    validated = researcher.list_validated()
    if validated:
        click.echo("\nTop Validated Hypotheses:")
        for h in validated[:5]:
            click.echo(f"  {h.code}: {h.name}")


if __name__ == "__main__":
    cli()
