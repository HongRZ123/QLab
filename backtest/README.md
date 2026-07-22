# backtest/ — 回测引擎

接收策略输出的 `num_units`，按 A 股约束计算含成本的权益曲线。自身不调统计函数、不选参数——只做执行和绩效计算。

> Walk-Forward 参数估计框架已迁至 `run/run_walk_forward.py`。

## 文件

| 文件 | 职责 |
|------|------|
| `engine.py` | `run_backtest()` — T+1, 整数手, 成本, 涨跌停, 停牌 |
| `core.py` | `run_core()` — 纯 PnL 循环 (零 A 股逻辑) |
| `constraints.py` | Constraint/CostModel Protocol + A 股实现 |
| `metrics.py` | `performance_summary()` — Sharpe/APR/MaxDD/Win% |

## 依赖

```
data/rules.py ──→ backtest/engine.py (round_to_lot)
strategies/  ──→ (仅通过 num_units 参数传入，非 import)
```

## 公开 API

```python
from backtest import run_backtest, performance_summary

bt = run_backtest(prices, num_units, dynamic_sizing=True)
stats = performance_summary(bt["ret"])
# stats = {apr, sharpe, maxdd, win_rate, trade_count, avg_holding, n_days}
```

## A 股约束

| 约束 | 说明 |
|------|------|
| T+1 | num_units(t-1) → positions(t) |
| 手数 | lot_size=100, round_to_lot |
| 佣金 | 万2.5, 最低5元, 双向 |
| 印花税 | 万5, 仅卖出 |
| 滑点 | 千1, 双向 |
| 涨跌停 | 主板±10%, 创业/科创±20% |
| Long-only | num_units >= 0 |

## 验证

```bash
python -m backtest.engine
python -m backtest.core
python -m backtest.constraints
python -m backtest.metrics
```
