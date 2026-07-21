# signals/Tech -- 技术分析信号

从量价数据中提取技术分析维度的市场信息。这些信号描述"量价关系的特征"，独立于任何交易决策。

## 设计原则

- 信号只做**信息提取**，不做交易决策（不输出 `num_units`）
- 信号有**独立含义**，可以被不同类型的策略消费
- 信号函数是**纯函数**，无副作用

## 信号清单

| 信号 | 函数 | 输出类型 | 消费者 |
|------|------|----------|--------|
| 量价确认 | `volume_confirmation` | int Series (+2/+1/-1/-2/0) | `strategies/experimental/s12_vpa_draft.py` |
| K 线影线比例 | `wick_body_ratio` | DataFrame | `strategies/experimental/s12_vpa_draft.py` |
| 量价背离序列 | `volume_anomaly_sequence` | int Series (+1/-1/0) | （暂无） |

---

## 1. 量价确认 `volume_confirmation`

### 含义

判断当日价格变动与成交量之间的关系，区分"确认"与"异常"：

| 编码 | 条件 | 含义 |
|------|------|------|
| +2 | 价格上涨 + 成交量高于均值 | 看涨确认：放量上涨，趋势健康 |
| +1 | 价格上涨 + 成交量低于均值 | 看涨异常：缩量上涨，动力不足 |
| -1 | 价格下跌 + 成交量低于均值 | 看跌异常：缩量下跌，抛压不大 |
| -2 | 价格下跌 + 成交量高于均值 | 看跌确认：放量下跌，趋势健康 |
| 0 | 价格不变或成交量恰好等于均值 | 中性 |

### 接口

```python
from signals.Tech.vpa import volume_confirmation

vc = volume_confirmation(
    prices: pd.Series,      # 价格序列
    volume: pd.Series,      # 成交量序列
    lookback: int = 20,     # 滚动均值窗口
) -> pd.Series              # 整数编码, index 同 prices
```

### 用法

```python
vc = volume_confirmation(prices, volume, lookback=20)

# 看涨确认
buy_signals = vc == 2

# 看跌确认
sell_signals = vc == -2
```

---

## 2. K 线影线比例 `wick_body_ratio`

### 含义

分析 K 线实体与上下影线的比例关系，识别反转形态：

| 指标 | 计算 | 含义 |
|------|------|------|
| `body_ratio` | abs(close-open) / (high-low) | 实体占比，大=趋势明确 |
| `upper_wick_ratio` | (high-max(open,close)) / (high-low) | 上影线占比，大=上方压力大 |
| `lower_wick_ratio` | (min(open,close)-low) / (high-low) | 下影线占比，大=下方支撑强 |
| `signal` | +1/-1/0 | +1=下影线主导（看涨反转），-1=上影线主导（看跌反转） |

signal 触发条件：影线比例 > 0.5 时标记为反转信号。

### 接口

```python
from signals.Tech.vpa import wick_body_ratio

wbr = wick_body_ratio(
    open: pd.Series,    # 开盘价
    high: pd.Series,    # 最高价
    low: pd.Series,     # 最低价
    close: pd.Series,   # 收盘价
) -> pd.DataFrame       # 包含 body_ratio, upper_wick_ratio, lower_wick_ratio, signal
```

### 用法

```python
wbr = wick_body_ratio(open, high, low, close)

# 看涨反转 K 线
bullish_reversal = wbr["signal"] == 1
```

---

## 3. 量价背离序列 `volume_anomaly_sequence`

### 含义

检测过去 lookback 个 bar 内的量价背离模式：

| 编码 | 条件 | 含义 |
|------|------|------|
| +1 | 价格上涨但成交量下降 | 看涨耗尽：上涨缺乏成交量支撑，可能见顶 |
| -1 | 价格下跌但成交量增加 | 看跌吸收：下跌伴随放量，可能是主力吸筹 |
| 0 | 其他 | 无背离 |

### 接口

```python
from signals.Tech.vpa import volume_anomaly_sequence

vas = volume_anomaly_sequence(
    prices: pd.Series,     # 价格序列
    volume: pd.Series,     # 成交量序列
    lookback: int = 3,     # 回看窗口（bar 数）
) -> pd.Series             # 整数编码, index 同 prices
```

### 用法

```python
vas = volume_anomaly_sequence(prices, volume, lookback=3)

# 看涨耗尽 -> 可能减仓
exhaustion = vas == 1

# 看跌吸收 -> 可能加仓
absorption = vas == -1
```

---

## 验证

```bash
python -m signals.Tech.vpa    # 正控（趋势+放量）+ 负控（随机游走+独立成交量）
```
