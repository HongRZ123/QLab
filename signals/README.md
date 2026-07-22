# signals/ — 信号层

从市场数据中提取的实时信息，供策略做交易决策。信号只做信息提取，不输出 `num_units`，不计算 PnL。

> 统计性质信号 (ADF/Hurst/CADF/Johansen) 已迁移到 `stats/` 模块。`signals/__init__.py` 仍从 `stats/` 重导出以保持兼容。

## 设计原则

- 信号函数是纯函数，无副作用
- 信号有独立含义，可被不同类型策略消费
- 按技术概念组织文件

## 文件

| 文件 | 概念 | 核心函数 |
|------|------|----------|
| `vpa.py` | 量价分析 (VPA) | `effort_vs_result`, `stopping_volume`, `buying_climax`, `no_demand`, `no_supply` |
| `pivot.py` | 价格结构 | `detect_isolated_pivots`, `detect_consolidation`, `detect_breakout` |
| `trend.py` | 趋势状态 | `trend_direction`, `trend_health` |
| `kalman.py` | 动态对冲 | `compute_kalman_spread` |

## vpa.py — 量价分析 (Anna Coulling)

### P0 基础信号

```python
from signals.vpa import volume_relative, spread, spread_relative, upper_wick, lower_wick, wick_ratio
```

### P1 核心信号（上下文感知）

```python
from signals.vpa import effort_vs_result, stopping_volume, buying_climax, no_demand, no_supply

evr = effort_vs_result(ohlcv)   # 投入产出比 (≈1=确认, >>1=异常)
sv = stopping_volume(ohlcv)     # 止损量: 下跌+锤头线+高量 → 底部反转
bc = buying_climax(ohlcv)       # 买入高潮: 上涨+射击十字星+高量 → 顶部反转
nd = no_demand(ohlcv)           # 无需求: 上涨+低量+小振幅 → 买方衰竭
ns = no_supply(ohlcv)           # 无供应: 下跌+低量+小振幅 → 卖方衰竭
```

### 兼容信号

```python
from signals.vpa import volume_confirmation, wick_body_ratio, volume_percentile, vpa_confirmation_matrix
```

## pivot.py — 价格结构

```python
from signals.pivot import detect_isolated_pivots, detect_consolidation, detect_breakout
```

## trend.py — 趋势状态

```python
from signals.trend import trend_direction, trend_health

td = trend_direction(close)         # +1 上涨 / -1 下跌 / 0 盘整
th = trend_health(close, volume)    # 上下文感知: 同根K线不同趋势含义不同
```

## kalman.py — 动态对冲

```python
from signals.kalman import compute_kalman_spread
sig = compute_kalman_spread(x, y)
# -> beta_slope, beta_intercept, e, Q, sqrt_Q, spread
```

## 验证

```bash
python -m signals.vpa
python -m signals.pivot
python -m signals.trend
```
