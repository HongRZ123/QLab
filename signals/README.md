# signals/ -- 信号层

从市场数据中提取的、独立于交易策略的信息。信号只做信息提取，不输出 `num_units`，不计算 PnL。

## 设计原则

- 信号函数是**纯函数**，无副作用，相同输入始终相同输出
- 信号有**独立含义**，可以被不同类型的策略消费
- 按技术概念组织文件，不按 alpha 类型分目录

## 信号清单

| 信号 | 文件 | 函数 | 输出 | 消费者 |
|------|------|------|------|--------|
| 量价确认 | `vpa.py` | `volume_confirmation` | int Series (+2/+1/-1/-2/0) | S12 |
| K 线影线比例 | `vpa.py` | `wick_body_ratio` | DataFrame[body_ratio, wick ratios, signal] | S12 |
| 量价背离序列 | `vpa.py` | `volume_anomaly_sequence` | int Series (+1/-1/0) | -- |
| 实体强度分位 | `vpa.py` | `body_strength_percentile` | float Series (0~1) | VPA-T1 |
| 成交量分位 | `vpa.py` | `volume_percentile` | float Series (0~1) | VPA-T1/T2 |
| 量价确认矩阵 | `vpa.py` | `vpa_confirmation_matrix` | str Series (confirmed/trap/anomaly/neutral) | VPA-T1 |
| 孤立支点 | `pivot.py` | `detect_isolated_pivots` | DataFrame[pivot_high, pivot_low] | VPA-T3 |
| 震荡区间 | `pivot.py` | `detect_consolidation` | str Series (in_range/breakout_up/breakout_down) | VPA-T3 |
| 突破检测 | `pivot.py` | `detect_breakout` | str Series (breakout_confirmed/false_breakout) | VPA-T3 |
| 趋势健康度 | `trend.py` | `trend_health` | int Series (+1/-1/0) | VPA-T1 |
| 卡尔曼 spread | `kalman.py` | `compute_kalman_spread` | dict[beta_slope, e, sqrt_Q, spread, ...] | S9 |

---

## vpa.py -- 量价分析信号

### volume_confirmation

量价确认信号，判断当日价格变动与成交量之间的关系。

| 编码 | 条件 | 含义 |
|------|------|------|
| +2 | 价格上涨 + 成交量高于均值 | 看涨确认 |
| +1 | 价格上涨 + 成交量低于均值 | 看涨异常 |
| -1 | 价格下跌 + 成交量低于均值 | 看跌异常 |
| -2 | 价格下跌 + 成交量高于均值 | 看跌确认 |
| 0 | 价格不变或成交量等于均值 | 中性 |

```python
from signals.vpa import volume_confirmation

vc = volume_confirmation(prices, volume, lookback=20)
```

### wick_body_ratio

K 线实体与上下影线的比例关系，识别反转形态。

```python
from signals.vpa import wick_body_ratio

wbr = wick_body_ratio(open, high, low, close)
# -> DataFrame[body_ratio, upper_wick_ratio, lower_wick_ratio, signal]
# signal: +1=下影线主导(看涨反转), -1=上影线主导(看跌反转), 0=其他
```

### volume_anomaly_sequence

多 bar 量价背离检测。

| 编码 | 含义 |
|------|------|
| +1 | 看涨耗尽（价格上涨但成交量下降） |
| -1 | 看跌吸收（价格下跌但成交量增加） |
| 0 | 其他 |

```python
from signals.vpa import volume_anomaly_sequence

vas = volume_anomaly_sequence(prices, volume, lookback=3)
```

---

## kalman.py -- 卡尔曼滤波信号

### compute_kalman_spread

从两个资产价格序列中，用卡尔曼滤波估计动态对冲比率 β(t)，提取预测误差 e(t) 及其动态标准差 √Q(t)。

基于 Chan (2013) Box 3.1 迭代公式。

```python
from signals.kalman import compute_kalman_spread

sig = compute_kalman_spread(x, y, delta=0.0001, ve=0.001)
# -> dict[beta_slope, beta_intercept, e, Q, sqrt_Q, spread]
```

| 键 | 说明 |
|----|------|
| `beta_slope` | β₁(t), 动态斜率（对冲比率） |
| `beta_intercept` | β₂(t), 动态截距 |
| `e` | 预测误差 e(t) = y(t) - ŷ(t) |
| `Q` | 预测误差方差 |
| `sqrt_Q` | √Q(t), 动态标准差 |
| `spread` | 价差 y - β₁·x |

---

## pivot.py -- 价格结构信号

### detect_isolated_pivots

孤立支点检测（Ch7）。

```python
from signals.pivot import detect_isolated_pivots

pivots = detect_isolated_pivots(high, low)
# -> DataFrame[pivot_high, pivot_low]
```

### detect_consolidation

震荡区间识别（Ch7）。

```python
from signals.pivot import detect_consolidation

consol = detect_consolidation(high, low, pivots, tolerance=0.02)
# "in_range" / "breakout_up" / "breakout_down"
```

### detect_breakout

突破与伪突破检测（Ch7）。

```python
from signals.pivot import detect_breakout

sig = detect_breakout(close, volume, range_bound=(upper, lower), lookback=20)
# "breakout_confirmed" / "false_breakout"
```

---

## trend.py -- 趋势健康度信号

### trend_health

趋势健康度（Ch8）。

```python
from signals.trend import trend_health

th = trend_health(close, volume, lookback=20)
# +1=健康, -1=走弱, 0=中性
```

---

## 验证

```bash
python -m signals.vpa       # VPA 信号
python -m signals.pivot     # 结构信号
python -m signals.trend     # 趋势健康度
```
