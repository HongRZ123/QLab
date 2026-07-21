# strategies/MM -- 做市策略

通过提供流动性赚取买卖价差和库存回归 alpha。与均值回归策略不同，做市策略不押注方向，而是从价格波动中获利。

## 策略清单

| 策略 | 模块 | 信号来源 | num_units | 注册表 |
|------|------|----------|-----------|--------|
| S10 卡尔曼做市 | `s10_kalman_mm.py` | 价格+成交量（内联 Kalman） | 无信号 | ❌ |

---

## S10 - 卡尔曼滤波做市模型 `s10_kalman_mm.py`

### 含义

Chan Ch.3 成交量加权卡尔曼滤波。估计资产的动态公允价格 m(t)，用成交量加权调整卡尔曼增益：大单时 K→1（公允价格跳到成交价），小单时 K→0（公允价格几乎不动）。

> 本模块**不产生交易信号**，仅估计动态公允价格。本质是 VWAP 的动态升级版。做市策略可基于 `deviation`（成交价与公允价的偏离）构造挂单逻辑。

### 公式

```
V_ω = δ/(1-δ)
V_e(t) = R(t|t-1) × (T_max / T(t) - 1)    # 成交量加权观测噪声
K(t) = R(t|t-1) / (R(t|t-1) + V_e(t))
m(t|t) = m(t|t-1) + K(t) × (y(t) - m(t|t-1))
R(t|t) = (1 - K(t)) × R(t|t-1)

T=T_max -> K=1 (公允价格跳到成交价)
T<<T_max -> K≈0 (公允价格几乎不动)
```

### 接口

```python
from strategies.MM.s10_kalman_mm import kalman_mm

result = kalman_mm(
    prices: pd.Series,          # 成交价格序列
    volumes: pd.Series,         # 成交量序列 (与 prices 等长)
    t_max: float | None = None, # 基准成交量. None 时用扩展窗口 max
    delta: float = 0.0001,      # 过程噪声参数
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `fair_value` | pd.Series | 动态公允价格 m(t\|t) |
| `kalman_gain` | pd.Series | 卡尔曼增益 K(t) |
| `R` | pd.Series | 状态方差 R(t\|t) |
| `deviation` | pd.Series | 偏离 y(t) - m(t\|t) |

### 用法

```python
from data import read_day
from strategies.MM.s10_kalman_mm import kalman_mm

df = read_day("sh512670")
result = kalman_mm(df["close"], df["volume"])

# deviation > 0 -> 成交价高于公允价，可挂卖单
# deviation < 0 -> 成交价低于公允价，可挂买单
```

### 验证

```bash
python -m strategies.MM.s10_kalman_mm    # 恒定价格收敛 + 大单 K=1 + 小单 K≈0
```
