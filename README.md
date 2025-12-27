# LLM Alpha

**LLM-driven autonomous Alpha research system for cryptocurrency trading.**

An automated closed-loop system where LLM continuously generates, tests, and iterates on trading hypotheses until validated strategies emerge.

## Core Concept

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM-Driven Autonomous Research Loop                │
└─────────────────────────────────────────────────────────────────┘

                         ┌──────────────┐
                         │     LLM      │
                         │ (GPT/Claude/ │
                         │    Local)    │
                         └──────┬───────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│Market Know-  │       │    Data      │       │  Knowledge   │
│ledge (papers,│       │  Analysis    │       │Base (history │
│patterns)     │       │  (features)  │       │& experience) │
└──────────────┘       └──────────────┘       └──────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │  Generate Hypothesis │
                    │ (Factor/Signal/Strat)│
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │    Generate Code     │
                    │  (Python executable) │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Backtest (VectorBT) │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Validate (WF/Roll)  │
                    └──────────┬───────────┘
                               ▼
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       ┌───────────┐                     ┌───────────┐
       │  PASSED   │                     │  FAILED   │
       └─────┬─────┘                     └─────┬─────┘
             │                                 │
             ▼                                 ▼
  ┌─────────────────┐              ┌─────────────────────┐
  │ Save to         │              │ Analyze failure     │
  │ strategies/     │              │ Log to knowledge DB │
  └─────────────────┘              └──────────┬──────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
              ┌────────────────┐                           ┌────────────────┐
              │ Iterate/Refine │                           │ Abandon &      │
              │ (tune params)  │                           │ New Hypothesis │
              └───────┬────────┘                           └────────────────┘
                      │
                      └─────────► Back to LLM for next iteration
```

## Features

- **LLM Agent**: Autonomous hypothesis generation (supports GPT-4/Claude/Local LLMs)
- **Auto-Iteration**: Closed-loop refinement based on backtest results
- **Three-Layer Architecture**: Factor → Signal → Strategy separation
- **Knowledge Base**: SQLite-based learning from historical successes/failures
- **Data Pipeline**: Async high-concurrency Binance data downloader
- **Backtesting**: VectorBT integration for fast vectorized backtests
- **Validation**: Walk-Forward + Rolling Window anti-overfitting checks
- **Optimization**: Optuna-based parameter search
- **CLI**: Full command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llmalpha.git
cd llmalpha

# Install
pip install .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Configuration

### 1. Set API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for third-party proxies

# Or for Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. Initialize Knowledge Base

```bash
llmalpha kb init
```

## Quick Start

```bash
# Run LLM autonomous research - fully automatic!
# (Data will be downloaded automatically if not available)
llmalpha agent research

# Or with custom options
llmalpha agent research \
  -r "Create a momentum strategy using RSI and MACD" \
  -c momentum \
  -n 10

# Manual data download (optional - agent downloads automatically)
llmalpha data download -s BTC,ETH -m 3

# List available data
llmalpha data list
```

### Automatic Data Download

The agent **automatically downloads market data** when needed:

1. When you run `llmalpha agent research`, the agent checks if data is available
2. If no data exists, it downloads recommended symbols (BTC, ETH, SOL, etc.)
3. Downloads 3 months of 1-minute data with OHLCV, Open Interest, and Funding Rate
4. Research proceeds automatically after data is ready

This means you can start researching immediately without manual data preparation!

### Agent Research Options

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --requirements` | "Find profitable trading strategies" | Research goal |
| `-c, --category` | None | Strategy category (momentum, mean_reversion, etc.) |
| `-n, --iterations` | 10 | Max iterations |
| `-p, --provider` | openai | LLM provider (openai, anthropic, ollama) |
| `-m, --model` | gpt-5.2 | Model name |
| `--base-url` | None | Custom API URL (for proxies) |
| `--api-key` | From .env | API key |
| `--early-stop-sharpe` | 1.5 | Stop if Sharpe reaches this |

### Other Commands

```bash
# Improve an existing hypothesis
llmalpha agent improve H001 -n 3

# View research status
llmalpha agent status

# Analyze failure patterns
llmalpha agent analyze

# Manual backtest with a strategy
llmalpha backtest run -s BTC -S rsi_mr

# Query knowledge base
llmalpha kb search --status validated
llmalpha kb stats
```

## Python API

```python
import asyncio
from llmalpha import create_researcher

async def main():
    researcher = create_researcher(
        provider="openai",           # or "anthropic", "ollama"
        model="gpt-5.2",
        max_iterations=10,
    )

    result = await researcher.research(
        requirements="Create a volatility breakout strategy using ATR",
        category="breakout",
    )

    print(result.summary())
    print(f"Best: {result.best_hypothesis_code}, Sharpe: {result.best_sharpe:.2f}")

asyncio.run(main())
```

## How It Works

### 1. LLM Hypothesis Generation

LLM analyzes three sources to generate hypotheses:
- **Market Knowledge**: Trading patterns, academic papers, market microstructure
- **Data Analysis**: Statistical features, anomalies, correlations in the data
- **Knowledge Base**: Past hypotheses (both successful and failed) to learn from

### 2. Three-Layer Architecture: Factor → Signal → Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│          Factor → (Validate) → Signal → Strategy                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Factor    │ ──> │  Validate   │ ──> │   Signal    │ ──> │  Strategy   │
│             │     │             │     │             │     │             │
│ Computes a  │     │ IC, IR,     │     │ Generates   │     │ Position    │
│ numeric     │     │ Quantile,   │     │ entry/exit  │     │ sizing +    │
│ feature     │     │ Turnover    │     │ points      │     │ risk mgmt   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
  pd.Series          Pass/Fail           SignalResult          Backtest
  [0.5, 1.2,         + Score             (entries,             Ready
   -0.3, ...]                             exits)
```

**Factor**: Computes a market feature (e.g., RSI, funding rate z-score, OI change)
```python
class FundingZScoreFactor(Factor):
    def compute(self, df) -> pd.Series:
        return zscore(df["funding_rate"], window=168)
```

**Factor Validation**: Before using a factor, validate its predictive power
```python
validator = FactorValidator(forward_periods=24)  # 24h forward return
result = validator.validate(factor, df)

# Key metrics:
# - IC (Information Coefficient): correlation with future returns
# - IR (Information Ratio): IC stability (IC_mean / IC_std)
# - Quantile monotonicity: do higher factor values → higher returns?
# - Turnover: trading cost implications

if result.is_valid:
    print(f"Factor passed! Score: {result.score}, IC: {result.ic_result.ic_mean:.4f}")
else:
    print(f"Factor rejected: {result.rejection_reasons}")
```

**Signal**: Transforms validated factor values into buy/sell signals
```python
class FundingReversalSignal(Signal):
    def generate(self, df) -> SignalResult:
        z = FundingZScoreFactor().compute(df)
        entries = z < -2.0  # Extreme negative funding → long
        exits = z > 0
        return SignalResult(entries=entries, exits=exits)
```

**Strategy**: Combines signals with position sizing and risk management
```python
class FundingStrategy(Strategy):
    def generate_signals(self, df) -> SignalResult:
        return FundingReversalSignal().generate(df)

    def calculate_position_size(self, df, idx) -> float:
        return 0.1  # 10% of capital per trade
```

### 3. Factor Validation Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **IC** | Spearman correlation between factor and forward returns | \|IC\| > 0.02 |
| **IR** | IC stability (IC_mean / IC_std) | \|IR\| > 0.3 |
| **Monotonicity** | Quantile returns correlation | \|corr\| > 0.6 |
| **Turnover** | Factor value change frequency | < 0.8 |
| **p-value** | Statistical significance of IC | < 0.05 |

### 4. Signal Validation Metrics

```python
validator = SignalValidator(min_trades=30)
result = validator.validate(signal, df)

if result.is_valid:
    print(f"Win rate: {result.performance.win_rate:.1%}")
    print(f"Profit factor: {result.performance.profit_factor:.2f}")
```

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Win Rate** | Percentage of profitable trades | > 40% |
| **Profit Factor** | Gross profit / Gross loss | > 1.0 |
| **Edge Ratio** | Expected value per trade | > 0.1% |
| **Signal Frequency** | Signals per 1000 bars | 1-500 |
| **Decay Analysis** | Performance at different holding periods | Low decay |

### 5. Strategy Validation

Each strategy goes through rigorous validation:
- **Walk-Forward**: Train(60%) → Validate(20%) → Test(20%)
- **Rolling Window**: Test across multiple time periods for consistency
- **Minimum Requirements**: Sharpe > 0.3, no severe decay between periods

### 6. Closed-Loop Learning

```
IF validation passed:
    → Save strategy to strategies/
    → Record success factors in knowledge base

IF validation failed:
    → Analyze WHY it failed (overfitting? bad logic? market regime?)
    → Record failure patterns in knowledge base
    → LLM decides: refine hypothesis OR abandon and try new direction
```

## Architecture

```
llmalpha/
├── llmalpha/
│   ├── agent/            # LLM agent (hypothesis generation, iteration)
│   ├── data/             # Data download and loading
│   ├── factors/          # Factor: computes numeric features + validation (IC/IR)
│   ├── signals/          # Signal: generates entry/exit points + validation
│   ├── strategies/       # Strategy: position sizing + risk management
│   ├── backtest/         # VBTEngine (vectorbt-based)
│   ├── optimize/         # Optuna optimizer + validators
│   ├── research/         # Hypothesis testing framework
│   ├── knowledge/        # SQLite knowledge base
│   ├── cli/              # CLI commands
│   └── utils/            # Utilities
├── hypotheses/           # Generated hypothesis files
├── strategies/           # Validated strategies (production-ready)
├── configs/              # Configuration files
└── data/                 # Data storage (parquet files)
```

## Configuration

Edit `configs/default.yaml`:

```yaml
# Agent settings
agent:
  llm:
    provider: "openai"  # openai, anthropic, ollama
    model: "gpt-5.2"
    temperature: 0.7

  max_iterations: 10
  max_consecutive_failures: 3

  validation:
    mode: "full"  # quick, full, wf_only, rolling_only
    min_sharpe: 0.3
    min_trades: 50

  early_stop_sharpe: 1.5
  sandbox_timeout: 60
  learn_from_failures: true

# Walk-Forward validation
walk_forward:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  min_sharpe: 0.3
  decay_threshold: 0.5

# Rolling window validation
rolling:
  train_window: 2160  # 3 months in hours
  test_window: 720    # 1 month
  min_positive_ratio: 0.6
```

## License

MIT
