# QLab - A 股量化研究平台

[![CI](https://github.com/HongRZ123/QLab/actions/workflows/ci.yml/badge.svg)](https://github.com/HongRZ123/QLab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/HongRZ123/QLab)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

> **QLab** is an A-share quantitative research platform built on Chan's *Algorithmic Trading* methodology. It provides a full pipeline from data loading, statistical testing, signal extraction, and strategy development to a backtest engine with realistic A-share constraints (T+1, price limits, lot sizing, commissions) and walk-forward optimization.

基于 Chan《Algorithmic Trading》方法论构建的 A 股量化研究框架。
覆盖从数据加载、统计检验、信号提取、策略开发、回测引擎到 Walk-Forward 优化的完整管道。

回测引擎内置 A 股交易约束（T+1、涨跌停、停牌、手数取整、佣金/印花税/滑点），并通过 Protocol 协议支持自定义约束与成本模型。

## 快速开始

### 环境要求

- Python 3.10+
- numpy, pandas, statsmodels
- 通达信盘后数据（设置环境变量 `QLAB_TDX_ROOT` 指向 `vipdoc` 目录，默认 `D:\new_tdx64\vipdoc`）

### 运行示例

每个策略有专属的端到端入口 `run/run_*.py`，参数硬编码在顶部：

```bash
cd D:\EtLab\QLab

python run/run_linear_mr.py        # S4 线性均值回归
python run/run_bollinger_mr.py     # S8 布林带均值回归
python run/run_ma_crossover.py     # 均线金叉死叉
python run/run_vpa_trend.py        # VPA 量价趋势跟踪
python run/run_vpa_reversal.py     # VPA 止损量反转
python run/run_vpa_breakout.py     # VPA 放量突破
python run/run_kalman_hedge.py     # S9 卡尔曼动态对冲
python run/run_linear_portfolio.py # S7 线性组合均值回归
python run/run_walk_forward.py     # Walk-Forward 滚动重估

# 全市场扫描
python explore/scan_stationarity.py
python explore/scan_ma_crossover.py
```

复制脚本改参数即可：

```bash
cp run/run_vpa_trend.py my_run.py
python my_run.py
```

---

## 项目架构

```
QLab/
├── data/              # 数据层：TDX 日线读取、除权除息、交易规则、数据源抽象
│   ├── interface.py       # OHLCVSource Protocol（数据源抽象）
│   ├── sources/           # 数据源实现
│   │   └── tdx.py             # TDXSource（包装 fetcher）
│   ├── fetcher.py         # 通达信盘后数据读取
│   ├── rules.py           # A 股交易规则（涨跌停、手数、停牌）
│   └── dividend.py        # 除权除息检测与复权
│
├── signals/           # 信号层：从市场数据中提取的、独立于交易策略的信息
│   ├── vpa.py             # 量价信号（effort_vs_result / stopping_volume / ...）
│   ├── pivot.py           # 价格结构信号（支点、震荡区间、突破检测）
│   ├── trend.py           # 趋势健康度信号
│   ├── kalman.py          # 卡尔曼信号（compute_kalman_spread）
│   ├── stats.py           # 统计性质信号 (ADF / Hurst / 半衰期)
│   └── stats_cointegration.py  # 协整性质信号 (CADF / Johansen)
│
├── strategies/        # 策略层：按 alpha 类型分类，消费信号生成 num_units
│   ├── MR/                 # 均值回归策略
│   │   ├── s4_linear.py        # S4 线性均值回归
│   │   ├── s7_linear_portfolio.py  # S7 组合线性均值回归
│   │   ├── s8_bollinger.py     # S8 布林带均值回归
│   │   └── s9_kalman_hedge.py  # S9 卡尔曼动态对冲（委托 signals/kalman）
│   ├── MM/                 # 做市策略
│   │   └── s10_kalman_mm.py    # S10 卡尔曼做市
│   ├── Tech/               # 技术分析策略
│   │   ├── ma_crossover.py     # 均线交叉
│   │   ├── vpa_trend.py        # VPA 量价趋势跟踪
│   │   ├── vpa_reversal.py     # VPA 反转形态
│   │   └── vpa_breakout.py     # VPA 放量突破
│   ├── experimental/       # 实验策略
│   │   ├── s11_rsi_draft.py    # RSI 草稿
│   │   └── s12_vpa_draft.py    # VPA 量价策略（消费 signals/vpa）
│   └── registry.py         # 策略注册表
│
├── backtest/          # 回测层：三层拆分（约束 ← 纯PnL ← 薄封装）
│   ├── core.py             # run_core() 纯 Chan PnL 循环（零 A 股逻辑）
│   ├── constraints.py      # Constraint/CostModel 协议 + PriceLimits/SuspensionCheck/AShareCost
│   ├── engine.py           # run_backtest() 薄封装（创建约束+成本 -> 委托 run_core）
│   ├── metrics.py          # 绩效指标（Sharpe, 最大回撤等）
│   └── walk_forward.py     # Walk-Forward 分析
│
├── tests/             # 检验层：统计检验 + 单元测试
│   ├── s1_adf.py ~ s6_johansen.py   # ADF / Hurst / 半衰期 / CADF / Johansen
│   ├── test_engine_snapshot.py      # 引擎快照测试（6 场景）
│   ├── test_backtest_core.py        # run_core vs run_backtest 等价性
│   ├── test_constraints.py          # 约束 + 成本模型单元测试
│   ├── test_kalman_spread.py        # 信号提取数值一致性
│   ├── test_vpa.py                  # VPA 信号单元测试
│   ├── test_vpa_strategy.py         # VPA 策略单元测试（草案）
│   ├── test_vpa_strategies.py       # VPA 策略单元测试（vpa_trend/reversal/breakout）
│   ├── test_pivot.py                # 价格结构信号单元测试
│   ├── test_trend.py                # 趋势健康度信号单元测试
│   ├── test_data_interface.py       # OHLCVSource 协议测试
│   └── test_tdx_source.py           # TDXSource 测试
│
├── explore/           # 探索层：全市场扫描脚本
├── experiments/       # 实验层：完全独立的研究脚本
├── run/               # 执行层：每个策略的端到端入口
├── alpha/             # 标的选取层：ETF 宇宙、平稳性筛选
```

### 模块依赖（单向，无循环）

```
data  ->  signals  ->  strategies  ->  backtest  ->  explore/
       ->  tests    ->  explore/                ->  run/
       ->  alpha   ->  run/
```

导入方向（回测层内部）：`constraints.py ← core.py ← engine.py`

---

## 已有模块清单

### 策略

| 策略 | 模块 | 类型 | 信号来源 | num_units | 状态 |
|------|------|------|----------|-----------|------|
| S4 线性均值回归 | `strategies.MR.s4_linear` | 单资产 | 价格序列 | >= 0 (连续) | ✅ 已注册 `linear_mr` |
| S8 布林带 MR | `strategies.MR.s8_bollinger` | 单资产 | 价格序列 | {0, 1} | ✅ 已注册 `bollinger_mr` |
| MA Crossover | `strategies.Tech.ma_crossover` | 单资产 | 价格序列 | {0, 1} | ✅ 已注册 `ma_crossover` |
| VPA 趋势跟踪 | `strategies.Tech.vpa_trend` | 单资产 | `signals.vpa` + `signals.trend` | {0, 0.5, 1} | ✅ 已注册 `vpa_trend` |
| VPA 反转形态 | `strategies.Tech.vpa_reversal` | 单资产 | `signals.vpa` | {0, 1} | ✅ 已注册 `vpa_reversal` |
| VPA 放量突破 | `strategies.Tech.vpa_breakout` | 单资产 | `signals.pivot` | {0, 1} | ✅ 已注册 `vpa_breakout` |
| S7 线性组合 MR | `strategies.MR.s7_linear_portfolio` | 多资产 | 价格矩阵 | >= 0 (连续) | 📦 库代码 |
| S8 布林带组合 | `strategies.MR.s8_bollinger` | 组合 | 组合净值 | {0, 1} | 📦 库代码 |
| S9 卡尔曼对冲 | `strategies.MR.s9_kalman_hedge` | 配对 | `signals.kalman` | {0, 1} | 📦 库代码 |
| S10 卡尔曼做市 | `strategies.MM.s10_kalman_mm` | 做市 | 价格+成交量 | 无 num_units | 📦 库代码 |
| RSI 草案 | `strategies.experimental.s11_rsi_draft` | 单资产 | 价格序列 | {0, 1} | 🧪 实验草稿 |
| VPA 草案 | `strategies.experimental.s12_vpa_draft` | 单资产 | `signals.vpa` | {0, 1} | 🧪 实验草稿 |

**状态说明：**

- ✅ **已注册** -- 已收录到 registry 目录索引，有 `run_validation()`，有 pytest，有 `run/run_*.py` 端到端入口
- 📦 **库代码** -- 有 `run_validation()`，可直接 import 调用
- 🧪 **实验草稿** -- 原型阶段

> 所有策略仅做多。策略内置 PnL 为理论值（无成本、无 T+1），生产回测请用 `backtest.run_backtest()`。

### 信号

| 信号 | 模块 | 输出 | 消费者 |
|------|------|------|--------|
| 卡尔曼 spread | `signals.kalman` | beta_slope, e, sqrt_Q, spread | S9 |
| 量价确认 | `signals.vpa` | +2/+1/-1/-2/0 编码 | S12 |
| K 线影线比例 | `signals.vpa` | body_ratio, signal | S12 |
| 量价背离序列 | `signals.vpa` | +1/-1/0 编码 | S12 |
| 实体强度分位 | `signals.vpa` | 0~1 百分位 | VPA-T1 |
| 成交量分位 | `signals.vpa` | 0~1 百分位 | VPA-T1/T2 |
| 量价确认矩阵 | `signals.vpa` | confirmed/trap/anomaly/neutral | VPA-T1 |
| 孤立支点 | `signals.pivot` | pivot_high, pivot_low | VPA-T3 |
| 震荡区间 | `signals.pivot` | in_range/breakout_up/breakout_down | VPA-T3 |
| 突破检测 | `signals.pivot` | breakout_confirmed/false_breakout | VPA-T3 |
| 趋势健康度 | `signals.trend` | +1/-1/0 | VPA-T1 |

> 信号只做信息提取，不输出 `num_units`。Z-score 是策略逻辑，不是信号。

### 统计检验

| 检验 | 模块 | 用途 |
|------|------|------|
| ADF | `tests.s1_adf` | 平稳性检验 |
| Hurst | `tests.s2_hurst` | 均值回归 vs 趋势判定 |
| 半衰期 | `tests.s3_half_life` | 均值回归速度 -> lookback 参数 |
| CADF | `tests.s5_cadf` | 配对交易对冲比率 |
| Johansen | `tests.s6_johansen` | 多资产协整组合 |

---

## 核心概念

### 1. Signal vs Strategy

**信号（Signal）** = 市场给定的信息，从数据中提取，有独立含义，不依赖任何交易决策。

**策略（Strategy）** = 决策逻辑。以信号（或价格）为输入，通过 Z-score、阈值等规则生成 `num_units`。

策略按 **alpha 寻找视角** 分类（MR/MM/Tech），信号按 **技术概念** 组织（vpa/pivot/trend/kalman），两者正交。一个 `signals/kalman` 信号可被 `strategies/MM/` 策略消费。

分界原则：Z-score 是策略逻辑（留在策略内部）；卡尔曼 spread 是信号（提取到 signals 层）。

### 2. `num_units`：仓位单元

所有策略的输出核心，不是股数，是**仓位比例**：

| 值 | 含义 |
|----|------|
| `0` | 空仓 |
| `1` | 满仓 |
| `0.5` | 半仓 |

策略只负责 "想做多少"，回测引擎负责转成实际股数、取整、扣成本。

### 3. 理论 PnL vs 回测 PnL

策略函数（如 `linear_mr`）内部也算 `pnl` 和 `ret`，但那是**理论值**：无成本、无 T+1、无手数取整。

**真实绩效必须走 `backtest.run_backtest()`**，它会：
- T+1 延迟执行（今天信号，明天成交）
- 100 股整数手取整
- 扣佣金（万2.5）、印花税（万5）、滑点（千1）
- 检查涨跌停、停牌

### 4. 回测引擎三层架构

```
constraints.py       core.py            engine.py
  Constraint Protocol   run_core()          run_backtest()
  CostModel Protocol    纯 Chan PnL 循环     薄封装
  PriceLimits           T+1 shift           创建 AShareCost
  SuspensionCheck       约束链式调用         创建约束列表
  AShareCost            PnL + 成本 + 权益    委托 run_core
```

- `run_core()` 是纯数学循环，零 A 股逻辑，可通过 Constraint/CostModel 协议注入任意约束和成本模型
- `run_backtest()` 是 A 股薄封装，创建 `AShareCost` + `default_a_share_constraints()` 后委托 `run_core()`
- `run_backtest()` 签名不变，所有现有策略代码无需修改

### 5. `run_validation()` 自检协议

每个核心模块底部都有 `run_validation()`，做三件事：
- **正控**：均值回归序列上策略应该赚钱
- **负控**：随机游走上策略应该不赚钱
- **不变式**：`num_units >= 0`、仓位是 100 的倍数等

```bash
python -m backtest.engine
python -m backtest.core
python -m signals.vpa
python -m signals.pivot
python -m signals.trend
python -m strategies.Tech.vpa_trend
python -m strategies.MR.s4_linear
python -m strategies.MR.s9_kalman_hedge
python -m tests.s6_johansen
```

目前 21/21 全部通过：

| 模块 | `python -m` 命令 |
|------|------------------|
| `backtest.engine` | `python -m backtest.engine` |
| `backtest.core` | `python -m backtest.core` |
| `backtest.constraints` | `python -m backtest.constraints` |
| `backtest.metrics` | `python -m backtest.metrics` |
| `backtest.walk_forward` | `python -m backtest.walk_forward` |
| `data.dividend` | `python -m data.dividend` |
| `signals.vpa` | `python -m signals.vpa` |
| `signals.pivot` | `python -m signals.pivot` |
| `signals.trend` | `python -m signals.trend` |
| `strategies.MR.s4_linear` | `python -m strategies.MR.s4_linear` |
| `strategies.MR.s7_linear_portfolio` | `python -m strategies.MR.s7_linear_portfolio` |
| `strategies.MR.s8_bollinger` | `python -m strategies.MR.s8_bollinger` |
| `strategies.MR.s9_kalman_hedge` | `python -m strategies.MR.s9_kalman_hedge` |
| `strategies.MM.s10_kalman_mm` | `python -m strategies.MM.s10_kalman_mm` |
| `strategies.Tech.ma_crossover` | `python -m strategies.Tech.ma_crossover` |
| `strategies.Tech.vpa_trend` | `python -m strategies.Tech.vpa_trend` |
| `strategies.Tech.vpa_reversal` | `python -m strategies.Tech.vpa_reversal` |
| `strategies.Tech.vpa_breakout` | `python -m strategies.Tech.vpa_breakout` |
| `strategies.experimental.s12_vpa_draft` | `python -m strategies.experimental.s12_vpa_draft` |
| `tests.s3_half_life` | `python -m tests.s3_half_life` |
| `tests.s6_johansen` | `python -m tests.s6_johansen` |

---

## 使用方式

### 方式一：直接跑端到端脚本（推荐）

```bash
python run/run_linear_mr.py
python run/run_vpa_trend.py
# 等等，见 run/ 目录下全部 run_*.py
```

### 方式二：Python API 直接 import

```python
from data import read_day
from backtest import run_backtest, performance_summary
from strategies.MR.s4_linear import linear_mr

prices = read_day("sh512670")["close"]
result = linear_mr(prices, lookback=20)
bt = run_backtest(prices, result["num_units"], dynamic_sizing=True)
stats = performance_summary(bt["ret"])
```

### 方式三：Python 脚本调用库代码策略

库代码策略（S7/S9/S10）未注册到 registry，需直接 import 调用：

```python
# S9 配对交易
from strategies.MR.s9_kalman_hedge import kalman_hedge
result = kalman_hedge(x_prices, y_prices, burn_in=60)

# S7 组合均值回归
from strategies.MR.s7_linear_portfolio import linear_portfolio
from tests.s6_johansen import johansen_test
joh = johansen_test(prices_df)
result = linear_portfolio(prices_df, joh["eigenvectors"][:, 0])

# S10 卡尔曼做市（注意：不产出 num_units，只输出 fair_value/deviation）
from strategies.MM.s10_kalman_mm import kalman_mm
result = kalman_mm(prices, volumes)
```

### 方式四：复制端到端脚本研究

`run/` 目录下每个策略有独立的端到端脚本，复制改参数即可：

```bash
cp run/run_vpa_trend.py my_research.py
python my_research.py
```

### 方式五：研究实验策略

```bash
python -m strategies.experimental.s11_rsi_draft
```

---

## API 速查

### data（数据层）

```python
from data import read_day, read_symbols, list_symbols
from data import detect_ex_dividend, adjust_close_prices
from data import round_to_lot, transaction_cost
from data import OHLCVSource, TDXSource

df = read_day("sh512670")           # -> DataFrame[date,open,high,low,close,amount,volume]
codes = list_symbols("sh")          # -> 所有上海标的代码
dfs = read_symbols(["sh512670", "sz159915"])  # -> dict[str, DataFrame]

# 数据源抽象（Protocol）
source = TDXSource()                # 实现 OHLCVSource 协议
df = source.get_ohlcv("sh512670")   # 同 read_day
codes = source.list_symbols("sh")   # 同 list_symbols
```

### signals（信号层）

```python
from signals.kalman import compute_kalman_spread
from signals.vpa import (
    volume_confirmation, wick_body_ratio, volume_anomaly_sequence,
    body_strength_percentile, volume_percentile, vpa_confirmation_matrix,
)
from signals.pivot import detect_isolated_pivots, detect_consolidation, detect_breakout
from signals.trend import trend_health

# 卡尔曼 spread 信号
sig = compute_kalman_spread(x, y, delta=0.0001, ve=0.001)
# -> {beta_slope, beta_intercept, e, Q, sqrt_Q, spread}

# VPA 量价信号
vc = volume_confirmation(prices, volume, lookback=20)       # -> int Series (+2/+1/-1/-2/0)
wbr = wick_body_ratio(open, high, low, close)               # -> DataFrame[body_ratio, upper/lower_wick_ratio, signal]
vas = volume_anomaly_sequence(prices, volume, lookback=3)   # -> int Series (+1/-1/0)
bsp = body_strength_percentile(open, close, lookback=20)    # -> float Series (0~1)
vps = volume_percentile(volume, lookback=20)                # -> float Series (0~1)
vcm = vpa_confirmation_matrix(open, high, low, close, volume, lookback=20)
# -> DataFrame[confirmed, trap, anomaly, neutral]

# 价格结构信号
pivots = detect_isolated_pivots(high, low, left=5, right=5) # -> DataFrame[pivot_high, pivot_low]
cons = detect_consolidation(high, low, lookback=20)         # -> Series[bool]
brk = detect_breakout(close, high, low, lookback=20)        # -> DataFrame[breakout_confirmed, false_breakout]

# 趋势健康度
th = trend_health(close, ma_window=20, slope_window=10)     # -> int Series (+1/-1/0)
```

### tests（统计检验层）

```python
from tests import run_adf, hurst_exponent, estimate_half_life
from tests import cadf_test, cadf_test_both_orders, johansen_test, construct_portfolio

adf = run_adf(prices)               # -> {adf_stat, p_value_aic, p_value_bic, ...}
h = hurst_exponent(prices)          # -> {hurst, r_squared, ...}  H<0.5=均值回归
hl = estimate_half_life(prices)     # -> {half_life, lambda, ...}
cadf = cadf_test(y, x)              # -> {hedge_ratio, spread, p_value, ...}
joh = johansen_test(prices_df)      # -> {eigenvectors, yport, rank, ...}
```

### strategies（策略层）

```python
from strategies import linear_mr, bollinger_mr, ma_crossover
from strategies import linear_portfolio, bollinger_portfolio, kalman_hedge, kalman_mm
from strategies.registry import list_names, get_strategy

# 直接调用
result = linear_mr(prices, lookback=20)
result = bollinger_mr(prices, lookback=20, entry_z=1.0, exit_z=0.0)
result = kalman_hedge(y, x, burn_in=60)

# 通过注册表查目录（仅供发现）
print(list_names())
result = linear_mr(prices, lookback=20)
```

> 各子模块详细文档: [MR/README.md](strategies/MR/README.md) | [MM/README.md](strategies/MM/README.md) | [Tech/README.md](strategies/Tech/README.md)

### backtest（回测层）

```python
from backtest import run_backtest, run_backtest_long_only, performance_summary
from backtest import walk_forward_linear_mr, walk_forward_bollinger, walk_forward_portfolio

# A 股回测（含 T+1、涨跌停、停牌、佣金、印花税、滑点）
bt = run_backtest(prices, num_units, dynamic_sizing=True)
stats = performance_summary(bt["ret"])
# stats = {apr, sharpe, maxdd, win_rate, trade_count, avg_holding, n_days}

# Walk-Forward
wf = walk_forward_linear_mr(prices, reest_interval=63, min_warmup=252)
bt_wf = run_backtest(prices, wf["num_units"])
```

### backtest.core + backtest.constraints（高级：自定义回测）

```python
from backtest.core import run_core
from backtest.constraints import (
    Constraint, CostModel,           # Protocol
    PriceLimits, SuspensionCheck,    # 内置约束
    default_a_share_constraints,     # 约束工厂
    AShareCost,                      # A 股成本模型
)

# 纯 PnL 循环（无约束、无成本）
result = run_core(prices, num_units)

# 完整 A 股回测（等价于 run_backtest）
result = run_core(
    prices, num_units,
    constraints=default_a_share_constraints(check_limits=True, board="main"),
    cost_model=AShareCost(commission_rate=0.00025, stamp_tax_rate=0.0005, slippage_rate=0.001),
)
```

---

## A 股交易约束

| 约束 | 参数 | 说明 |
|------|------|------|
| T+1 | - | 信号 t 日生成，t+1 日执行（`num_units.shift(1)`） |
| 手数 | `lot_size=100` | 下单量取整到 100 股 |
| 佣金 | `commission_rate=0.00025` | 万2.5，最低5元，买卖双向 |
| 印花税 | `stamp_tax_rate=0.0005` | 万5，仅卖出 |
| 滑点 | `slippage_rate=0.001` | 千1，买入加价、卖出降价 |
| 涨跌停 | `board` | 主板±10%, 创业板/科创板±20%, ST±5%, 北交所±30% |
| 停牌 | `price_data` | 停牌日/零成交日不执行交易，保持前一日持仓 |
| Long-only | - | 不融券，-1 信号 -> 0 |

约束通过 `Constraint` Protocol 注入，成本通过 `CostModel` Protocol 注入，两者独立。T+1 不是 Constraint，在 `run_core()` 内部通过 `shift(1)` 处理。

---

## 扩展指南

### 新增信号

#### 1. 放在哪里

按**技术概念**命名，放在 `signals/` 下一个 `.py` 文件中：

| 文件 | 内容 | 示例 |
|------|------|------|
| `signals/vpa.py` | 量价分析信号 | `effort_vs_result`, `stopping_volume` |
| `signals/pivot.py` | 价格结构信号 | `detect_isolated_pivots`, `detect_breakout` |
| `signals/trend.py` | 趋势健康度信号 | `trend_health`, `trend_direction` |
| `signals/kalman.py` | 卡尔曼滤波信号 | `compute_kalman_spread` |
| `signals/stats.py` | 统计性质信号 | `run_adf`, `hurst_exponent`, `estimate_half_life` |
| `signals/stats_cointegration.py` | 协整性质信号 | `cadf_test`, `johansen_test` |

如果是新的技术概念，新建 `signals/<概念名>.py`。一个文件内可放多个相关信号函数。

#### 2. 函数签名要求

```python
def my_signal(data: ..., **params) -> dict | pd.Series:
    """
    信号函数只做信息提取，不做交易决策。
    不输出 num_units，不计算 PnL。
    返回 dict（多值）或 pd.Series（单值）。
    """
    ...
    return {"key1": series1, "key2": series2}
```

**关键约束：**
- 不输出 `num_units` -- 那是策略的职责
- 不计算 PnL / ret -- 那是策略和回测引擎的职责
- 返回值有独立含义 -- 即使没人用它做策略，信号本身也有解释力

#### 3. 编写 `run_validation()`

在信号模块底部加 `run_validation()`，验证信号在已知数据上的行为：

```python
def run_validation() -> bool:
    """验证信号在合成数据上的行为符合预期。"""
    # 1. 构造已知特征的合成数据
    # 2. 提取信号
    # 3. 断言信号值/分布/方向符合预期
    # 4. 边界检查：空输入、单元素、NaN
    return all_passed
```

运行方式：`python -m signals.my_signal`

#### 4. 更新文档

- 在本 README [已有模块清单 > 信号](#信号) 表格中添加一行
- 在 `signals/README.md` 中添加信号说明（公式、接口、返回值、用法）

---

### 新增策略

#### 1. 放在哪里

按 **alpha 寻找视角** 分类，放在 `strategies/` 对应子目录：

| 视角 | 目录 | 示例 |
|------|------|------|
| 均值回归 | `strategies/MR/` | `s4_linear.py` |
| 做市 | `strategies/MM/` | `s10_kalman_mm.py` |
| 技术分析 | `strategies/Tech/` | `ma_crossover.py` |
| 实验 | `strategies/experimental/` | `s11_rsi_draft.py` |

#### 2. 函数签名要求

```python
def my_strategy(prices: pd.Series, **kwargs) -> dict:
    """
    策略函数以价格（或信号）为输入，输出含 num_units 的 dict。
    """
    ...
    return {
        "num_units": pd.Series,   # 必须：仓位比例 (>=0, 仅做多)
        "pnl": pd.Series,         # 理论 PnL（无成本）
        "ret": pd.Series,         # 理论收益率
        # ... 其他策略特有字段
    }
```

**关键约束：**
- 返回值**必须**包含 `num_units` 键
- `num_units >= 0`（仅做多，-1 信号被回测引擎截断为 0）
- `num_units` 是仓位比例（0=空仓, 1=满仓），不是股数
- 内置 `pnl`/`ret` 是理论值（无成本、无 T+1），生产回测走 `run_backtest()`

#### 3. 编写 `run_validation()`

在策略模块底部加 `run_validation()`，包含三个层次：

```python
def run_validation() -> bool:
    """策略自检：正控 + 负控 + 不变式。"""
    # 1. 正控：在均值回归序列（OU 过程）上，策略应该赚钱
    #    -> 累计 PnL > 0, Sharpe > 0
    # 2. 负控：在随机游走（GBM）上，策略应该不赚钱
    #    -> Sharpe < 0.5
    # 3. 不变式：num_units >= 0, 取值范围符合预期
    return all_passed
```

可参考 `strategies/MR/s4_linear.py` 的 `run_validation()` 实现。合成数据生成器在 `tests/s3_half_life.py`（`generate_ou_paths`, `generate_gbm_paths`）。

运行方式：`python -m strategies.MR.my_strategy`

#### 4. 注册到 Registry（可选）

策略成熟后，注册到 `strategies/registry.py`：

```python
from strategies.registry import Strategy, register

register(Strategy(
    name="my_strategy",
    fn=my_strategy_fn,
    description="我的策略",
    default_kwargs={"lookback": 20},
))
```

注册后可通过 `run/` 端到端脚本直接 import 调用。

**判断是否应该注册：**
- 策略有 `run_validation()` 且通过 -> 可以注册
- 策略还在迭代中，参数不稳定 -> 放 `experimental/`，不注册
- 策略需要多资产输入（如 S7/S9）-> 使用 `run/run_linear_portfolio.py` 或 `run/run_kalman_hedge.py`

#### 5. 更新文档

- 在本 README [已有模块清单 > 策略](#策略) 表格中添加一行，标注状态
- 在对应子目录的 `README.md` 中添加策略说明（公式、接口、返回值、用法、验证命令）

---

### 自定义回测约束

实现 `Constraint` Protocol 的 `apply(target_shares, prev_shares, i, prices) -> int` 方法，传入 `run_core()` 的 `constraints` 列表即可。约束按列表顺序链式调用。

```python
class MyConstraint:
    def apply(self, target_shares: int, prev_shares: int, i: int, prices: pd.Series) -> int:
        # 修改并返回实际股数
        return target_shares
```

### 自定义成本模型

实现 `CostModel` Protocol 的 `compute(shares_delta, price, direction) -> float` 方法，传入 `run_core()` 的 `cost_model` 参数即可。

```python
class MyCost:
    def compute(self, shares_delta: int, price: float, direction: str) -> float:
        return abs(shares_delta * price) * 0.001
```

---

## 各模块详细文档

- [data/README.md](data/README.md)
- [tests/README.md](tests/README.md)
- [strategies/README.md](strategies/README.md) -- [MR/](strategies/MR/README.md) | [MM/](strategies/MM/README.md) | [Tech/](strategies/Tech/README.md)
- [signals/README.md](signals/README.md)
- [backtest/README.md](backtest/README.md)
- [explore/README.md](explore/README.md)
- [experiments/README.md](experiments/README.md)
- [run/README.md](run/README.md)

---

## 验证状态

- `ruff check .`：0 errors
- `basedpyright`：0 errors, 1180 warnings（主要来自 pandas-stubs 类型桩不完整）
- 21/21 `run_validation()` 通过
- 92/92 pytest 测试全部通过
