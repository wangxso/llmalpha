# LLM Alpha

**基于大语言模型驱动的加密货币自主 Alpha 研究系统**

一个自动化闭环系统，LLM 持续生成、测试和迭代交易假设，直到产生经过验证的策略。

## 核心概念

```
┌─────────────────────────────────────────────────────────────────┐
│                  LLM 驱动的自主研究循环                           │
└─────────────────────────────────────────────────────────────────┘

                         ┌──────────────┐
                         │     LLM      │
                         │ (GPT/Claude/ │
                         │    本地模型)  │
                         └──────┬───────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ 市场知识     │       │    数据      │       │   知识库     │
│ (论文、模式) │       │    分析      │       │ (历史经验)   │
└──────────────┘       └──────────────┘       └──────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │      生成假设        │
                    │ (因子/信号/策略)     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │      生成代码        │
                    │  (可执行 Python)     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  回测 (VectorBT)     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  验证 (WF/滚动)      │
                    └──────────┬───────────┘
                               ▼
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       ┌───────────┐                     ┌───────────┐
       │   通过    │                     │   失败    │
       └─────┬─────┘                     └─────┬─────┘
             │                                 │
             ▼                                 ▼
  ┌─────────────────┐              ┌─────────────────────┐
  │ 保存到          │              │ 分析失败原因        │
  │ strategies/     │              │ 记录到知识库        │
  └─────────────────┘              └──────────┬──────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
              ┌────────────────┐                           ┌────────────────┐
              │ 迭代/优化      │                           │ 放弃 &         │
              │ (调整参数)     │                           │ 新假设         │
              └───────┬────────┘                           └────────────────┘
                      │
                      └─────────► 返回 LLM 进行下一轮迭代
```

## 功能特性

- **LLM 智能体**: 自主假设生成（支持 GPT-4/Claude/本地 LLM）
- **自动迭代**: 基于回测结果的闭环优化
- **三层架构**: 因子 → 信号 → 策略 分离设计
- **知识库**: 基于 SQLite 的历史成功/失败学习
- **数据管道**: 异步高并发 Binance 数据下载器
- **回测引擎**: VectorBT 集成，快速向量化回测
- **验证系统**: Walk-Forward + 滚动窗口 防过拟合检验
- **参数优化**: 基于 Optuna 的参数搜索
- **命令行界面**: 完整的 CLI 接口

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/llmalpha.git
cd llmalpha

# 安装
pip install .

# 或安装开发依赖
pip install -e ".[dev]"
```

## 配置

### 1. 设置 API 密钥

在项目根目录创建 `.env` 文件：

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选：用于第三方代理

# 或使用 Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. 初始化知识库

```bash
llmalpha kb init
```

## 快速开始

```bash
# 运行 LLM 自主研究 - 全自动！
# （如果数据不可用，将自动下载）
llmalpha agent research

# 或使用自定义选项
llmalpha agent research \
  -r "创建一个使用 RSI 和 MACD 的动量策略" \
  -c momentum \
  -n 10

# 手动下载数据（可选 - 智能体会自动下载）
llmalpha data download -s BTC,ETH -m 3

# 列出可用数据
llmalpha data list
```


### 自动数据下载

智能体在需要时**自动下载市场数据**：

1. 运行 `llmalpha agent research` 时，智能体会检查数据是否可用
2. 如果没有数据，会下载推荐的交易对（BTC、ETH、SOL 等）
3. 下载 3 个月的 1 分钟数据，包含 OHLCV、持仓量和资金费率
4. 数据准备好后自动继续研究

这意味着你可以立即开始研究，无需手动准备数据！

### 智能体研究选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `-r, --requirements` | "寻找盈利的交易策略" | 研究目标 |
| `-c, --category` | None | 策略类别（momentum, mean_reversion 等） |
| `-n, --iterations` | 10 | 最大迭代次数 |
| `-p, --provider` | openai | LLM 提供商（openai, anthropic, ollama） |
| `-m, --model` | gpt-5.2 | 模型名称 |
| `--base-url` | None | 自定义 API URL（用于代理） |
| `--api-key` | 来自 .env | API 密钥 |
| `--early-stop-sharpe` | 1.5 | 达到此夏普比率时停止 |

### 其他命令

```bash
# 改进现有假设
llmalpha agent improve H001 -n 3

# 查看研究状态
llmalpha agent status

# 分析失败模式
llmalpha agent analyze

# 手动回测策略
llmalpha backtest run -s BTC -S rsi_mr

# 查询知识库
llmalpha kb search --status validated
llmalpha kb stats
```

## Python API

```python
import asyncio
from llmalpha import create_researcher

async def main():
    researcher = create_researcher(
        provider="openai",           # 或 "anthropic", "ollama"
        model="gpt-5.2",
        max_iterations=10,
    )

    result = await researcher.research(
        requirements="创建一个使用 ATR 的波动率突破策略",
        category="breakout",
    )

    print(result.summary())
    print(f"最佳: {result.best_hypothesis_code}, 夏普: {result.best_sharpe:.2f}")

asyncio.run(main())
```

## 工作原理

### 1. LLM 假设生成

LLM 分析三个来源生成假设：
- **市场知识**: 交易模式、学术论文、市场微观结构
- **数据分析**: 数据中的统计特征、异常、相关性
- **知识库**: 过去的假设（成功和失败的）用于学习

### 2. 三层架构：因子 → 信号 → 策略

```
┌─────────────────────────────────────────────────────────────────┐
│          因子 → (验证) → 信号 → 策略                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    因子     │ ──> │    验证     │ ──> │    信号     │ ──> │    策略     │
│             │     │             │     │             │     │             │
│ 计算数值    │     │ IC, IR,     │     │ 生成入场/   │     │ 仓位管理 + │
│ 特征        │     │ 分位数,     │     │ 出场点      │     │ 风险管理   │
│             │     │ 换手率      │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
  pd.Series          通过/失败           SignalResult          回测
  [0.5, 1.2,         + 评分             (entries,             就绪
   -0.3, ...]                            exits)
```

**因子**: 计算市场特征（如 RSI、资金费率 z-score、持仓量变化）
```python
class FundingZScoreFactor(Factor):
    def compute(self, df) -> pd.Series:
        return zscore(df["funding_rate"], window=168)
```

**因子验证**: 使用因子前，验证其预测能力
```python
validator = FactorValidator(forward_periods=24)  # 24小时前瞻收益
result = validator.validate(factor, df)

# 关键指标：
# - IC（信息系数）：与未来收益的相关性
# - IR（信息比率）：IC 稳定性（IC_mean / IC_std）
# - 分位数单调性：更高的因子值是否对应更高的收益？
# - 换手率：交易成本影响

if result.is_valid:
    print(f"因子通过！评分: {result.score}, IC: {result.ic_result.ic_mean:.4f}")
else:
    print(f"因子被拒绝: {result.rejection_reasons}")
```

**信号**: 将验证过的因子值转换为买卖信号
```python
class FundingReversalSignal(Signal):
    def generate(self, df) -> SignalResult:
        z = FundingZScoreFactor().compute(df)
        entries = z < -2.0  # 极端负资金费率 → 做多
        exits = z > 0
        return SignalResult(entries=entries, exits=exits)
```

**策略**: 结合信号与仓位管理和风险管理
```python
class FundingStrategy(Strategy):
    def generate_signals(self, df) -> SignalResult:
        return FundingReversalSignal().generate(df)

    def calculate_position_size(self, df, idx) -> float:
        return 0.1  # 每笔交易使用 10% 资金
```


### 3. 因子验证指标

| 指标 | 描述 | 阈值 |
|------|------|------|
| **IC** | 因子与前瞻收益的 Spearman 相关性 | \|IC\| > 0.02 |
| **IR** | IC 稳定性（IC_mean / IC_std） | \|IR\| > 0.3 |
| **单调性** | 分位数收益相关性 | \|corr\| > 0.6 |
| **换手率** | 因子值变化频率 | < 0.8 |
| **p 值** | IC 的统计显著性 | < 0.05 |

### 4. 信号验证指标

```python
validator = SignalValidator(min_trades=30)
result = validator.validate(signal, df)

if result.is_valid:
    print(f"胜率: {result.performance.win_rate:.1%}")
    print(f"盈亏比: {result.performance.profit_factor:.2f}")
```

| 指标 | 描述 | 阈值 |
|------|------|------|
| **胜率** | 盈利交易的百分比 | > 40% |
| **盈亏比** | 总盈利 / 总亏损 | > 1.0 |
| **边际比率** | 每笔交易的期望值 | > 0.1% |
| **信号频率** | 每 1000 根 K 线的信号数 | 1-500 |
| **衰减分析** | 不同持仓周期的表现 | 低衰减 |

### 5. 策略验证

每个策略都经过严格验证：
- **Walk-Forward**: 训练(60%) → 验证(20%) → 测试(20%)
- **滚动窗口**: 在多个时间段测试一致性
- **最低要求**: 夏普 > 0.3，各周期间无严重衰减

### 6. 闭环学习

```
如果验证通过:
    → 保存策略到 strategies/
    → 在知识库中记录成功因素

如果验证失败:
    → 分析失败原因（过拟合？逻辑错误？市场环境？）
    → 在知识库中记录失败模式
    → LLM 决定：优化假设 或 放弃并尝试新方向
```

## 项目架构

```
llmalpha/
├── llmalpha/
│   ├── agent/            # LLM 智能体（假设生成、迭代）
│   ├── data/             # 数据下载和加载
│   ├── factors/          # 因子：计算数值特征 + 验证（IC/IR）
│   ├── signals/          # 信号：生成入场/出场点 + 验证
│   ├── strategies/       # 策略：仓位管理 + 风险管理
│   ├── backtest/         # VBTEngine（基于 vectorbt）
│   ├── optimize/         # Optuna 优化器 + 验证器
│   ├── research/         # 假设测试框架
│   ├── knowledge/        # SQLite 知识库
│   ├── cli/              # CLI 命令
│   └── utils/            # 工具函数
├── hypotheses/           # 生成的假设文件
├── strategies/           # 验证通过的策略（生产就绪）
├── configs/              # 配置文件
└── data/                 # 数据存储（parquet 文件）
```

## 配置说明

编辑 `configs/default.yaml`：

```yaml
# 智能体设置
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

# Walk-Forward 验证
walk_forward:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  min_sharpe: 0.3
  decay_threshold: 0.5

# 滚动窗口验证
rolling:
  train_window: 2160  # 3 个月（小时）
  test_window: 720    # 1 个月
  min_positive_ratio: 0.6
```

## 许可证

MIT
