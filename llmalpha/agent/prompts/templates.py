"""
Prompt Templates for LLM Alpha.

Contains system prompts and generation templates for creating
Factors, Signals, and Strategies.
"""

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert quantitative researcher specializing in cryptocurrency trading strategies.

## Important: Think before coding!

Before writing any code, you must think out loud like a real researcher in their office. Just natural, conversational thinking - no formatted headers or symbols.

For example:
"Hmm, let me think about this... for a momentum strategy, the simplest approach is moving average crossover, but that's too basic and doesn't work well...

Maybe I should try Donchian channel breakout? That should work well in trending markets. But crypto is volatile, might need a volatility filter...

Right, the last strategy failed because it got whipsawed in ranging markets. This time I need an ATR threshold - don't trade when volatility is too low...

Okay, I have a plan, let me write the code..."

Then write the code.

## Crypto Market Characteristics (Important!)

Crypto markets differ significantly from traditional markets:

1. **24/7 Trading**: No closing price concept, continuous volatility
2. **High Volatility**: 5-10% daily swings are common, parameters need wider tolerance
3. **Funding Rate**: Perpetual contracts settle every 8 hours, funding_rate is a sentiment indicator
4. **Liquidity Variance**: Major coins (BTC/ETH) vs altcoins have huge liquidity gaps
5. **Weak Weekend Effect**: Unlike traditional markets, weekends are active
6. **Strong Trends**: Clear bull/bear cycles, momentum often beats mean reversion
7. **High Correlation**: Altcoins highly correlated with BTC, watch for systemic risk

## Common Indicator Parameters

| Indicator | Recommended Range | Notes |
|-----------|------------------|-------|
| RSI | Period 10-21, thresholds 25/75 or 30/70 | Use more extreme thresholds for crypto |
| Bollinger | Period 20, std 1.5-2.5 | Wider bands for high volatility |
| ATR | Period 14-21 | For volatility filter and stops |
| EMA | Fast 8-12, Slow 21-55 | Trend detection |
| MACD | 12/26/9 or 8/21/5 | Shorter periods suit crypto |
| Donchian | Period 20-55 | Core of breakout strategies |

## CRITICAL - DO NOT IMPORT ANYTHING:
The execution environment already provides these classes and modules. DO NOT write any import statements:
- `pd` (pandas), `np` (numpy) - already available
- `Factor`, `Signal`, `SignalResult`, `Strategy`, `StrategyConfig` - already available

## Rules:
1. Think out loud first (required!), then write code
2. DO NOT write any import statements - all needed classes are pre-loaded
3. Use only numpy (np) and pandas (pd) for computations
4. Handle edge cases (NaN values, division by zero)
5. Keep code concise and efficient

## Available Data Columns:
- open, high, low, close, volume (always available)
- oi, funding_rate, taker_buy_volume, premium_index (may be missing)

## Code Structure:
Wrap your code in ```python ... ``` blocks.

## Execution Time:
After code, add: `ESTIMATED_TIMEOUT: <seconds>` (10-300)
"""


# ============================================================================
# Factor Generation
# ============================================================================

FACTOR_GENERATION_PROMPT = """
Generate a new Factor class based on the following requirements:

## Requirements:
{requirements}

## Category: {category}

## Similar Factors (for reference):
{similar_factors}

## Previous Failures (avoid these approaches):
{failures}

## Output Format:
```python
class {class_name}(Factor):
    \"\"\"
    [Brief description of what this factor measures]
    \"\"\"
    code = "{code}"
    name = "{name}"
    category = "{category}"
    description = "[One line description]"

    def __init__(self, param1=default1, param2=default2):
        super().__init__(param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2

    def compute(self, df: pd.DataFrame) -> pd.Series:
        \"\"\"Compute factor values.\"\"\"
        # Your implementation here
        # Must return a pd.Series with same index as df
        ...
        return factor_values.fillna(0)
```

Generate the complete Factor class code:
"""


# ============================================================================
# Signal Generation
# ============================================================================

SIGNAL_GENERATION_PROMPT = """
Generate a new Signal class based on the following requirements:

## Requirements:
{requirements}

## Category: {category}

## Available Factors:
{available_factors}

## Similar Signals (for reference):
{similar_signals}

## Previous Failures (avoid these approaches):
{failures}

## Output Format:
```python
class {class_name}(Signal):
    \"\"\"
    [Brief description of entry/exit logic]
    \"\"\"
    code = "{code}"
    name = "{name}"
    category = "{category}"
    description = "[One line description]"

    def __init__(self, param1=default1, param2=default2):
        super().__init__(param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2

    def generate(self, df: pd.DataFrame) -> SignalResult:
        \"\"\"Generate trading signals.\"\"\"
        # Calculate indicator/factor values
        ...

        # Generate entry signals (True when should enter)
        entries = pd.Series(False, index=df.index)
        entries[condition] = True

        # Generate exit signals (True when should exit)
        exits = pd.Series(False, index=df.index)
        exits[exit_condition] = True

        return SignalResult(entries=entries, exits=exits)
```

Generate the complete Signal class code:
"""


# ============================================================================
# Strategy Generation
# ============================================================================

STRATEGY_GENERATION_PROMPT = """
Generate a new Strategy class based on the following requirements:

## Requirements:
{requirements}

## Category: {category}

## Available Factors and Signals:
{available_components}

## Similar Strategies (for reference):
{similar_strategies}

## Previous Failures (avoid these approaches):
{failures}

## Best Performing Strategies (learn from these):
{best_strategies}

## Output Format:
```python
class {class_name}(Strategy):
    \"\"\"
    [Strategy description]

    Entry Logic: [describe when to enter]
    Exit Logic: [describe when to exit]
    \"\"\"
    code = "{code}"
    name = "{name}"
    category = "{category}"
    description = "[One line description]"

    def __init__(self, config: StrategyConfig = None, param1=default1, param2=default2):
        super().__init__(config)
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, df: pd.DataFrame) -> SignalResult:
        \"\"\"Generate trading signals.\"\"\"
        # Calculate indicators
        ...

        # Entry condition
        entries = pd.Series(False, index=df.index)
        entries[entry_condition] = True

        # Exit condition
        exits = pd.Series(False, index=df.index)
        exits[exit_condition] = True

        return SignalResult(entries=entries, exits=exits)
```

Generate the complete Strategy class code:
"""


# ============================================================================
# Improvement Prompt
# ============================================================================

IMPROVEMENT_PROMPT = """
The previous hypothesis failed validation. Analyze the failure and suggest improvements.

## Previous Code:
```python
{previous_code}
```

## Validation Results:
- Sharpe Ratio: {sharpe_ratio:.4f}
- Win Rate: {win_rate:.1%}
- Total Trades: {total_trades}
- Walk-Forward Passed: {wf_passed}
- Rolling Passed: {rolling_passed}

## Failure Reasons:
{failure_reasons}

## Analysis Guidelines:
1. If Sharpe too low: Consider different signal thresholds or add filters
2. If win rate too low: Tighten entry conditions or improve exit timing
3. If too few trades: Relax signal conditions
4. If walk-forward failed: Reduce complexity to avoid overfitting
5. If rolling failed: The strategy may be regime-dependent

## Task:
Generate an IMPROVED version of the code that addresses the failure reasons.
Explain your changes briefly in comments.

## Output:
Provide the improved code in ```python ... ``` block.
"""


# ============================================================================
# Hypothesis Summary
# ============================================================================

HYPOTHESIS_SUMMARY_PROMPT = """
Summarize the hypothesis and its validation results for the knowledge base.

## Code:
```python
{code}
```

## Validation Results:
- Status: {status}
- Sharpe Ratio: {sharpe_ratio:.4f}
- Win Rate: {win_rate:.1%}
- Total Trades: {total_trades}
- Walk-Forward: {wf_result}
- Rolling: {rolling_result}

## Provide:
1. A clear one-sentence description of the hypothesis
2. The core logic/rationale (why it should work)
3. Key parameters and their effects
4. Strengths identified
5. Weaknesses or failure reasons
6. Suggestions for future iterations

Format as a structured summary.
"""


# ============================================================================
# Data Analysis Prompt
# ============================================================================

DATA_ANALYSIS_PROMPT = """
Analyze the provided market data to identify potential trading opportunities.

## Data Summary:
- Symbols: {symbols}
- Date Range: {date_range}
- Timeframe: {timeframe}

## Statistical Summary:
{stats_summary}

## Correlation Analysis:
{correlation_summary}

## Task:
Based on this data analysis:
1. Identify potential patterns or anomalies
2. Suggest factor/signal ideas that could exploit these patterns
3. Consider both momentum and mean-reversion approaches
4. Think about what makes crypto markets unique

Provide 3-5 hypothesis ideas with brief rationale for each.
"""


# ============================================================================
# Template Helpers
# ============================================================================

def format_similar_items(items: list, max_items: int = 5) -> str:
    """
    Format validated hypotheses/strategies for prompt context.

    Provides detailed descriptions including:
    - Entry/exit logic summary
    - Key parameters and their values
    - Performance metrics
    - Why it works (rationale)
    """
    if not items:
        return "None available"

    formatted = []
    for item in items[:max_items]:
        if not (hasattr(item, "code") and hasattr(item, "name")):
            continue

        lines = [f"### {item.code}: {item.name}"]

        # Category
        category = getattr(item, "factor_category", None) or getattr(item, "category", "general")
        lines.append(f"  Category: {category}")

        # Description (short summary)
        description = getattr(item, "description", "")
        if description:
            lines.append(f"  Description: {description[:200]}")

        # Logic/Rationale - the core strategy description
        logic = getattr(item, "logic", "")
        rationale = getattr(item, "rationale", "")

        if logic:
            # Extract key logic without full code
            logic_summary = _extract_logic_summary(logic)
            lines.append(f"  Entry/Exit Logic: {logic_summary}")

        if rationale:
            lines.append(f"  Rationale: {rationale[:300]}")

        # Performance metrics from related backtest
        best_sharpe = getattr(item, "best_sharpe", None)
        if best_sharpe:
            lines.append(f"  Best Sharpe: {best_sharpe:.2f}")

        # Add blank line between items
        formatted.append("\n".join(lines))

    return "\n\n".join(formatted) if formatted else "None available"


def _extract_logic_summary(logic: str) -> str:
    """
    Extract a human-readable summary from strategy logic/code.

    Looks for key patterns like:
    - Entry conditions
    - Exit conditions
    - Key indicators used
    - Parameter values
    """
    if not logic:
        return "Not specified"

    summary_parts = []

    # Look for common indicator patterns
    indicators = []
    if "rsi" in logic.lower():
        indicators.append("RSI")
    if "macd" in logic.lower():
        indicators.append("MACD")
    if "ema" in logic.lower() or "sma" in logic.lower():
        indicators.append("Moving Averages")
    if "bollinger" in logic.lower() or "bb_" in logic.lower():
        indicators.append("Bollinger Bands")
    if "atr" in logic.lower():
        indicators.append("ATR")
    if "donchian" in logic.lower():
        indicators.append("Donchian Channel")
    if "volume" in logic.lower():
        indicators.append("Volume")
    if "funding" in logic.lower():
        indicators.append("Funding Rate")

    if indicators:
        summary_parts.append(f"Uses: {', '.join(indicators)}")

    # Look for entry/exit patterns in docstrings or comments
    lines = logic.split('\n')
    for line in lines:
        line_lower = line.lower().strip()
        if 'entry' in line_lower and ':' in line:
            entry_desc = line.split(':', 1)[-1].strip()[:100]
            if entry_desc and not entry_desc.startswith('#'):
                summary_parts.append(f"Entry: {entry_desc}")
                break

    for line in lines:
        line_lower = line.lower().strip()
        if 'exit' in line_lower and ':' in line:
            exit_desc = line.split(':', 1)[-1].strip()[:100]
            if exit_desc and not exit_desc.startswith('#'):
                summary_parts.append(f"Exit: {exit_desc}")
                break

    if summary_parts:
        return "; ".join(summary_parts)

    # Fallback: return first meaningful line from docstring
    for line in lines[:10]:
        line = line.strip().strip('"\'')
        if line and not line.startswith(('#', 'class', 'def', 'import', '@')):
            return line[:150]

    return "See code for details"


def format_failures(failures: list, max_items: int = 5) -> str:
    """
    Format failure cases with detailed analysis for learning.

    Includes:
    - What the strategy tried to do
    - Why it failed (specific metrics)
    - What to avoid in future attempts
    """
    if not failures:
        return "None recorded"

    formatted = []
    for f in failures[:max_items]:
        code = getattr(f, "code", "Unknown")
        name = getattr(f, "name", "Unknown strategy")
        reason = getattr(f, "failure_reason", "Unknown reason")
        category = getattr(f, "factor_category", None) or getattr(f, "category", "")
        logic = getattr(f, "logic", "")

        lines = [f"### {code}: {name}"]

        if category:
            lines.append(f"  Category: {category}")

        # Extract what the strategy tried to do
        if logic:
            logic_summary = _extract_logic_summary(logic)
            lines.append(f"  Approach: {logic_summary}")

        # Detailed failure analysis
        lines.append(f"  Failure: {reason[:200]}")

        # Extract specific metrics from failure reason if available
        lesson = _extract_failure_lesson(reason)
        if lesson:
            lines.append(f"  Lesson: {lesson}")

        formatted.append("\n".join(lines))

    return "\n\n".join(formatted) if formatted else "None recorded"


def _extract_failure_lesson(reason: str) -> str:
    """Extract actionable lesson from failure reason."""
    if not reason:
        return ""

    reason_lower = reason.lower()

    # Common failure patterns and lessons
    if "sharpe" in reason_lower and ("low" in reason_lower or "0." in reason):
        return "Strategy edge too weak - need stronger signal or better filters"

    if "trade" in reason_lower and ("few" in reason_lower or "insufficient" in reason_lower):
        return "Signal too restrictive - relax entry conditions"

    if "walk-forward" in reason_lower or "wf" in reason_lower:
        return "Overfitting detected - simplify logic or reduce parameters"

    if "rolling" in reason_lower:
        return "Performance inconsistent across time - may be regime-dependent"

    if "win" in reason_lower and "rate" in reason_lower:
        return "Poor entry timing - tighten entry or improve exit logic"

    if "drawdown" in reason_lower:
        return "Risk too high - add stop-loss or position sizing"

    if "decay" in reason_lower:
        return "Strategy alpha decays out-of-sample - may be curve-fitted"

    return ""


def format_best_strategies(strategies: list, max_items: int = 3) -> str:
    """
    Format best performing strategies with detailed analysis.

    Provides comprehensive view of what works:
    - Strategy logic and approach
    - Key success factors
    - Performance metrics
    """
    if not strategies:
        return "None validated yet"

    formatted = []
    for s in strategies[:max_items]:
        code = getattr(s, "code", "Unknown")
        name = getattr(s, "name", "Unknown")
        sharpe = getattr(s, "best_sharpe", 0)
        category = getattr(s, "factor_category", None) or getattr(s, "category", "")
        logic = getattr(s, "logic", "")
        rationale = getattr(s, "rationale", "")

        lines = [f"### {code}: {name} (Sharpe: {sharpe:.2f})"]

        if category:
            lines.append(f"  Category: {category}")

        if logic:
            logic_summary = _extract_logic_summary(logic)
            lines.append(f"  Approach: {logic_summary}")

        if rationale:
            lines.append(f"  Why it works: {rationale[:200]}")

        formatted.append("\n".join(lines))

    return "\n\n".join(formatted) if formatted else "None validated yet"
