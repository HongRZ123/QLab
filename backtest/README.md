# backtest — 回测引擎模块

提供含 A 股约束的回测引擎、绩效指标计算、Walk-Forward 滚动重估三个子模块。

## 依赖关系

```
backtest/
├── 内部依赖: data.rules (round_to_lot — 手数取整)
├── 内部依赖: strategies (linear_mr, bollinger_mr, linear_portfolio — Walk-Forward 调用)
├── 内部依赖: signals (estimate_half_life, johansen_test — Walk-Forward 重估)
└── 外部依赖: numpy, pandas
```

```
data.rules ──→ backtest.engine (round_to_lot)
strategies ──→ backtest.walk_forward (策略信号生成)
tests ──→ backtest.walk_forward (半衰期/Johansen 重估)
```

## 文件结构

```
backtest/
├── __init__.py              # 公开 API
├── engine.py                # 回测引擎 (T+1, 手数, 成本)
├── metrics.py               # 绩效指标
└── walk_forward.py          # Walk-Forward 滚动重估
```

## 公开 API

```python
from backtest import (
    run_backtest,
    run_backtest_long_only,
    performance_summary,
    walk_forward_linear_mr,
    walk_forward_bollinger,
    walk_forward_portfolio,
)
```

---

## engine.py — 回测引擎

### `run_backtest`

含 A 股约束的回测引擎。接收策略输出的 `num_units`，计算含成本的权益曲线。

```python
run_backtest(
    prices: pd.Series,              # 日价格序列 (index 为日期, 收盘价)
    num_units: pd.Series,           # 仓位单元序列 (与 prices 同 index, ≥0)
    initial_capital: float = 1_000_000.0,
    lot_size: int = 100,            # A股 100 股/手
    commission_rate: float = 0.00025,  # 万2.5
    stamp_tax_rate: float = 0.0005,    # 万5 (仅卖出)
    slippage_rate: float = 0.001,      # 千1 (双边)
    dynamic_sizing: bool = True,       # True=复利, False=固定初始资金
    board: str = "main",              # 板块: main/chinext/star/st
    check_limits: bool = True,        # 是否检查涨跌停
    check_suspension: bool = False,   # 是否检查停牌
    price_data: pd.DataFrame | None = None,  # check_suspension=True 时必填
) -> dict
```

**核心逻辑**:
1. **T+1 执行**: `num_units(t-1)` 决定 `positions(t)`
2. **整数手**: `shares = round_to_lot(num_units × capital / price, lot_size)`
3. **涨跌停**: 涨停日不可买入, 跌停日不可卖出
4. **停牌**: 停牌日/零成交日不执行交易, 沿用前一日持仓
5. **PnL**: `pnl(t) = shares(t-1) × (price(t) - price(t-1))`
6. **成本**: 每次持仓变化时扣除佣金 + 印花税(卖出) + 滑点

涨跌停板块参数:

| board | 含义 | 涨跌幅限制 |
|-------|------|-----------|
| `"main"` | 主板 | ±10% |
| `"chinext"` | 创业板 | ±20% |
| `"star"` | 科创板 | ±20% |
| `"st"` | ST股 | ±5% |
| `"bse"` | 北交所 | ±30% |

**返回**:

| 键 | 类型 | 说明 |
|----|------|------|
| `positions` | pd.Series | 每日持仓股数 (int, lot_size 的倍数) |
| `pnl` | pd.Series | 每日盈亏 (含成本前) |
| `ret` | pd.Series | 每日收益率 = (equity(t) - equity(t-1)) / equity(t-1) (已扣成本) |
| `equity_curve` | pd.Series | 权益曲线 (扣除所有成本后) |
| `n_trades` | int | 交易次数 (shares 变化的天数) |
| `total_cost` | float | 总交易成本 |

### `run_backtest_long_only`

仅做多回测的便捷封装，`signals ∈ {0, 1}`。

```python
run_backtest_long_only(
    prices: pd.Series,
    signals: pd.Series,             # 仅含 0 和 1
    initial_capital: float = 1_000_000.0,
    lot_size: int = 100,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.0005,
    slippage_rate: float = 0.001,
    dynamic_sizing: bool = True,
    board: str = "main",             # 同 run_backtest
    check_limits: bool = True,        # 同 run_backtest
    check_suspension: bool = False,   # 同 run_backtest
    price_data: pd.DataFrame | None = None,  # 同 run_backtest
) -> dict                           # 同 run_backtest 返回
```

`signal=1` → 满仓买入 (`shares = round_to_lot(capital / price)`)
`signal=0` → 空仓

---

## metrics.py — 绩效指标

### `performance_summary`

```python
performance_summary(
    returns: pd.Series,             # 日收益率序列
    signals: pd.Series | None = None,  # 交易信号 (可选, 0/1)
) -> dict
```

**返回**:

| 键 | 说明 | 公式 |
|----|------|------|
| `apr` | 年化收益率 | (1+mean)^252 - 1 |
| `sharpe` | 夏普比率 | mean/std × √252 |
| `maxdd` | 最大回撤 | 从累计净值曲线计算 |
| `win_rate` | 胜率 | 正收益天数占比 |
| `trade_count` | 交易次数 | 信号变化次数 (需传 signals) |
| `avg_holding` | 平均持仓天数 | 需传 signals |
| `n_days` | 总交易日数 | len(returns) |

### 独立指标函数

```python
from backtest.metrics import (
    annualized_return,   # (returns, periods_per_year=252) -> float
    sharpe_ratio,        # (returns, periods_per_year=252, rf=0.0) -> float
    max_drawdown,        # (returns) -> float
    win_rate,            # (returns) -> float
    trade_count,         # (signals) -> int
    avg_holding_days,    # (signals) -> float
)
```

---

## walk_forward.py — Walk-Forward 滚动重估

> **注意**: Walk-Forward 模块输出的是 `num_units` 信号序列，**不是回测结果**。需要再传给 `run_backtest()` 做含成本的回测。

### 通用流程

```
|←── min_warmup ──→|←── reest_interval ──→|←── reest_interval ──→|
                   ↑ 重估参数              ↑ 重估参数
                   应用策略 →              应用策略 →
```

每 `reest_interval` 天用历史数据重估参数（半衰期 → lookback），在下一个窗口应用策略。

### `walk_forward_linear_mr`

```python
walk_forward_linear_mr(
    close: pd.Series,              # 收盘价序列
    reest_interval: int = 63,      # 重估间隔 (天)
    min_warmup: int = 252,         # 最小预热期 (天)
) -> dict
```

**返回**:

| 键 | 类型 | 说明 |
|----|------|------|
| `num_units` | pd.Series | 全序列仓位 (预热期为 0) |
| `lookback_log` | list[dict] | 每次重估记录 {date, lookback, half_life} |

### `walk_forward_bollinger`

```python
walk_forward_bollinger(
    close: pd.Series,
    entry_z_candidates: list[float] | None = None,  # 默认 [1.0, 1.5, 2.0]
    reest_interval: int = 63,
    min_warmup: int = 252,
) -> dict
```

每窗口: 估计半衰期 → lookback, 在训练期评估各 `entry_z` 的 Sharpe → 选最佳。

**返回**:

| 键 | 类型 | 说明 |
|----|------|------|
| `num_units` | pd.Series | 全序列仓位 |
| `param_log` | list[dict] | 每次重估记录 {date, lookback, entry_z, train_sharpe} |

### `walk_forward_portfolio`

```python
walk_forward_portfolio(
    prices_df: pd.DataFrame,       # 价格矩阵 (T × n)
    reest_interval: int = 63,
    min_warmup: int = 252,
    lag: int = 1,                  # Johansen 差分滞后阶数
) -> dict
```

每窗口: Johansen 检验 → 第一特征向量 v₁ + 组合半衰期 → lookback → 应用 S7。

**返回**:

| 键 | 类型 | 说明 |
|----|------|------|
| `num_units` | pd.Series | 全序列仓位 |
| `ret` | pd.Series | 策略理论每日收益率 (无成本) |
| `param_log` | list[dict] | 每次重估记录 |

---

## A 股交易约束

| 约束 | 实现位置 | 说明 |
|------|----------|------|
| T+1 | `engine.py` | num_units(t-1) → positions(t) |
| 100 股/手 | `engine.py` → `data.rules.round_to_lot` | 下单量取整 |
| 佣金 万2.5 | `engine.py` | 买卖双向, 最低 5 元 |
| 印花税 万5 | `engine.py` | 仅卖出 |
| 滑点 千1 | `engine.py` | 双边 |
| Long-only | `run_backtest_long_only` | signals ∈ {0, 1} |

## 典型用法

```python
from data import read_day
from strategies import linear_mr
from backtest import run_backtest, performance_summary

df = read_day("sh512670")

# 1. 策略生成信号
s4 = linear_mr(df["close"])

# 2. ETF 回测 (默认主板 ±10%, ETF 实际也适用)
bt = run_backtest(df["close"], s4["num_units"])

# 3. 绩效评估
stats = performance_summary(bt["ret"], signals=(s4["num_units"] > 0).astype(int))
print(f"APR={stats['apr']:.2%}, Sharpe={stats['sharpe']:.2f}, MaxDD={stats['maxdd']:.2%}")
```

### 个股回测 (创业板 ±20% + 停牌检查)

```python
from backtest import run_backtest

# price_data 需包含 volume 列
df = read_day("sz300750")  # 宁德时代, 创业板
bt = run_backtest(
    df["close"],
    num_units,
    board="chinext",           # 创业板 ±20%
    check_limits=True,           # 启用涨跌停检查
    check_suspension=True,       # 启用停牌检查
    price_data=df,               # 传入完整 OHLCV
)
```

### Walk-Forward 滚动重估

```python
from backtest import walk_forward_linear_mr

wf = walk_forward_linear_mr(df["close"])          # 输出 num_units
bt_wf = run_backtest(df["close"], wf["num_units"])  # 再回测
```

## 验证协议

```bash
python -m backtest.engine       # 恒定价格 PnL=0, T+1 验证, 整数手验证
python -m backtest.metrics      # 已知收益率序列 → APR/Sharpe/MaxDD 手算对比
python -m backtest.walk_forward # 预热期 num_units=0 验证
```
