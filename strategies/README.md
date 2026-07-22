# strategies — 策略层

按 alpha 寻找视角分类的策略集合。所有策略为纯函数，输入价格或信号，输出含 `num_units` 的 dict。

> 策略内置的 `pnl`/`ret` 为理论值（无成本、无 T+1、无手数取整）。生产回测请使用 `backtest.run_backtest()`。

## 目录结构

```
strategies/
├── MR/                     # 均值回归策略
│   ├── s4_linear.py            # S4: 线性均值回归 (单资产)
│   ├── s7_linear_portfolio.py  # S7: 组合均值回归
│   ├── s8_bollinger.py         # S8: 布林带均值回归
│   └── s9_kalman_hedge.py      # S9: 卡尔曼动态对冲
├── Tech/                   # 技术分析策略
│   ├── ma_crossover.py         # MA 金叉死叉
│   ├── vpa_trend.py            # VPA 量价趋势跟踪
│   ├── vpa_reversal.py         # VPA 止损量反转
│   └── vpa_breakout.py         # VPA 放量突破
├── MM/                     # 做市策略
│   └── s10_kalman_mm.py        # S10: 卡尔曼做市
├── experimental/           # 实验策略
│   ├── s11_rsi_draft.py        # RSI 草案
│   ├── s12_vpa_draft.py        # VPA 草案
│   └── custom_strategy.py      # 自定义策略开发模板
├── registry.py             # 策略目录索引
└── __init__.py             # 公开 API
```

## 分类原则

子目录按 **alpha 寻找视角** 分类，与 signals/ 按技术概念组织正交：

| 目录 | alpha 类型 | 含义 |
|------|-----------|------|
| `MR/` | 均值回归 | 价格偏离均衡后回归 |
| `Tech/` | 技术分析 | 量价指标驱动 |
| `MM/` | 做市 | 提供流动性赚价差 |

## 策略总览

| 策略 | 文件 | 类型 | 信号来源 | num_units | 入口 |
|------|------|------|----------|-----------|------|
| S4 线性 MR | `MR/s4_linear.py` | 单资产 | 价格序列 | ≥0 (连续) | `run/run_linear_mr.py` |
| S7 组合 MR | `MR/s7_linear_portfolio.py` | 组合 | 价格矩阵 + Johansen | ≥0 (连续) | `run/run_linear_portfolio.py` |
| S8 布林带 MR | `MR/s8_bollinger.py` | 单资产 | 价格序列 | {0,1} | `run/run_bollinger_mr.py` |
| S9 卡尔曼对冲 | `MR/s9_kalman_hedge.py` | 配对 | `signals/kalman` | {0,1} | `run/run_kalman_hedge.py` |
| MA 均线交叉 | `Tech/ma_crossover.py` | 单资产 | 价格序列 | {0,1} | `run/run_ma_crossover.py` |
| VPA 趋势跟踪 | `Tech/vpa_trend.py` | 单资产 | `signals/vpa`+`trend` | {0,0.5,1} | `run/run_vpa_trend.py` |
| VPA 止损量反转 | `Tech/vpa_reversal.py` | 单资产 | `signals/vpa` | {0,1} | `run/run_vpa_reversal.py` |
| VPA 放量突破 | `Tech/vpa_breakout.py` | 单资产 | `signals/vpa` | {0,1} | `run/run_vpa_breakout.py` |
| S10 卡尔曼做市 | `MM/s10_kalman_mm.py` | 做市 | 价格+成交量 | 无 num_units | (手写脚本) |
| RSI 草案 | `experimental/s11_rsi_draft.py` | 单资产 | 价格序列 | {0,1} | 实验 |
| VPA 草案 | `experimental/s12_vpa_draft.py` | 单资产 | `signals/vpa` | {0,1} | 实验 |

---

# 均值回归策略 (MR/)

从价格偏离均值的回归行为中寻找 alpha。

## S4 — 线性均值回归 `MR/s4_linear.py`

Chan Ch.2 核心策略。计算价格的 Z-score，仅做多：Z 为负时建仓，Z 归零时平仓。

```
Z(t) = (y(t) - MA(y, L)) / Std(y, L)
num_units(t) = max(0, -Z(t))
```

```python
from strategies.MR.s4_linear import linear_mr
result = linear_mr(prices, lookback=20)
# returns: z_score, num_units, mkt_val, pnl, ret, lookback_used
```

## S7 — 线性组合均值回归 `MR/s7_linear_portfolio.py`

S4 的多资产版本。用 Johansen 特征向量构造协整组合 yport，对 yport 做 Z-score MR。

```
yport = Y · v₁
Z(t) = (yport(t) - MA(yport, L)) / Std(yport, L)
num_units(t) = max(0, -Z(t))
```

```python
from strategies.MR.s7_linear_portfolio import linear_portfolio
result = linear_portfolio(prices_df, eigenvector, lookback=20)
# returns: yport, z_score, num_units, positions, pnl, ret
```

## S8 — 布林带均值回归 `MR/s8_bollinger.py`

离散信号版 MR。价格跌破下轨买入，回到均值卖出。

```
Z(t) = (y(t) - MA(y, L)) / Std(y, L)
Z < -entry_z  -> num_units = 1
Z >= -exit_z  -> num_units = 0
```

```python
from strategies.MR.s8_bollinger import bollinger_mr
result = bollinger_mr(prices, lookback=20, entry_z=1.0, exit_z=0.0)
# returns: z_score, num_units, signals, pnl, ret, n_trades, avg_holding
```

## S9 — 卡尔曼动态对冲 `MR/s9_kalman_hedge.py`

Chan Ch.3 Box 3.1。卡尔曼滤波估计动态对冲比率 β(t)，以预测误差 e(t) 构造信号。

```
e(t) < -√Q(t) -> num_units = 1 (买入)
e(t) > -√Q(t) -> num_units = 0 (卖出)
```

信号提取委托给 `signals/kalman.py`，策略层只保留交易规则和 PnL。

```python
from strategies.MR.s9_kalman_hedge import kalman_hedge
result = kalman_hedge(y, x, burn_in=60)
# returns: beta_slope, e, Q, sqrt_Q, spread, num_units, pnl, ret
```

---

# 技术分析策略 (Tech/)

不依赖统计假设，直接从量价数据中提取交易信号。

## MA Crossover — 均线金叉死叉 `Tech/ma_crossover.py`

```
SMA_short 上穿 SMA_long -> num_units = 1
SMA_short 下穿 SMA_long -> num_units = 0
```

```python
from strategies.Tech.ma_crossover import ma_crossover
result = ma_crossover(prices, short_window=5, long_window=20)
```

## VPA Trend — 量价趋势跟踪 `Tech/vpa_trend.py`

基于 Anna Coulling 方法论。`effort_vs_result`（成交量/振幅比）判断量价和谐度，配合上下文感知的 `trend_health` 决定仓位。

```
effort_vs_result ≈ 1 (确认) + trend_health = +1 -> full (1.0)
effort_vs_result ≈ 1 + trend_health = -1          -> half (0.5)
anomaly or trap                                    -> empty (0.0)
```

```python
from strategies.Tech.vpa_trend import vpa_trend
result = vpa_trend(ohlcv, lookback=20, confirm_low=0.7, confirm_high=1.5)
# returns: num_units, effort_vs_result, trend_direction, trend_health
```

## VPA Reversal — 止损量反转 `Tech/vpa_reversal.py`

捕捉底部反转。下跌趋势中出现高量锤头线（stopping volume）入场，forward-fill 持仓，
买入高潮或跌破入场低点出场。

```python
from strategies.Tech.vpa_reversal import vpa_reversal
result = vpa_reversal(ohlcv, lookback=20)
# returns: num_units (forward-filled), stopping_volume, buying_climax
```

## VPA Breakout — 放量突破 `Tech/vpa_breakout.py`

捕捉盘整区真突破。收盘突破近期高点 + 高量 + 大振幅入场，向量化实现。

```python
from strategies.Tech.vpa_breakout import vpa_breakout
result = vpa_breakout(ohlcv, lookback=20, vol_threshold=1.5, spread_threshold=1.5)
# returns: num_units (forward-filled), breakout_up, volume_relative, spread_relative
```

---

# 做市策略 (MM/)

## S10 — 卡尔曼做市 `MM/s10_kalman_mm.py`

用卡尔曼滤波估计公允价值，在公允价值附近挂单提供流动性。不产出 `num_units`，
输出 `fair_value` 和 `deviation`。

```python
from strategies.MM.s10_kalman_mm import kalman_mm
result = kalman_mm(prices, volumes)
# returns: fair_value, deviation (not num_units)
```

---

## 理论 PnL vs 生产回测

| | 策略内置 PnL | backtest 引擎 PnL |
|---|---|---|
| T+1 执行 | ❌ | ✅ |
| 手数取整 | ❌ | ✅ round_to_lot |
| 佣金/印花税/滑点 | ❌ | ✅ |

**正确用法**: 策略输出 `num_units` → 传给 `backtest.run_backtest()`。

## 验证

```bash
python -m strategies.MR.s4_linear
python -m strategies.MR.s7_linear_portfolio
python -m strategies.MR.s8_bollinger
python -m strategies.MR.s9_kalman_hedge
python -m strategies.MM.s10_kalman_mm
python -m strategies.Tech.ma_crossover
python -m strategies.Tech.vpa_trend
python -m strategies.Tech.vpa_reversal
python -m strategies.Tech.vpa_breakout
```
