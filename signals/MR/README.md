# signals/MR -- 均值回归信号

从价格序列中提取均值回归属性的市场信息。这些信号描述"价格偏离均衡中心的程度"，独立于任何交易决策。

## 设计原则

- 信号只做**信息提取**，不做交易决策（不输出 `num_units`）
- 信号有**独立含义**，可以被不同类型的策略消费（MR 策略、MM 策略等）
- 信号函数是**纯函数**，无副作用，相同输入始终相同输出

## 信号清单

| 信号 | 模块 | 消费者 |
|------|------|--------|
| Kalman spread | `kalman_spread.py` | `strategies/MR/s9_kalman_hedge.py` |

---

## Kalman spread `kalman_spread.py`

### 含义

从两个资产 (x, y) 的价格序列中，用卡尔曼滤波估计动态对冲比率 β(t)，并提取预测误差 e(t) 及其动态标准差 √Q(t)。

核心思想：y 与 x 存在线性关系 `y = β₁·x + β₂ + ε`，但 β 随时间缓慢变化。卡尔曼滤波逐日更新 β，预测误差 e = y - ŷ 反映 y 相对于 x 的瞬时偏离，√Q 是该偏离的动态标准差。

### 算法

基于 Chan (2013) Box 3.1 迭代公式 (3.7)-(3.13)：

```
观测方程:  y(t) = x_aug(t) · β(t) + ε(t)        # x_aug = [x, 1]
状态转移:  β(t) = β(t-1) + ω(t-1)

每日迭代:
  预测:   β̂(t|t-1) = β̂(t-1|t-1)                  # 状态保持
          R(t|t-1) = R(t-1|t-1) + V_ω              # 协方差增长
  更新:   Q(t) = x_aug(t) · R(t|t-1) · x_aug(t)' + V_ε
          e(t) = y(t) - x_aug(t) · β̂(t|t-1)
          K(t) = R(t|t-1) · x_aug(t)' / Q(t)
          β̂(t|t) = β̂(t|t-1) + K(t) · e(t)
          R(t|t) = R(t|t-1) - K(t) · x_aug(t) · R(t|t-1)
```

参数：
- `delta` (δ): 状态变化速率，V_ω = δ/(1-δ) · I。越大 β 变化越快。默认 0.0001
- `ve` (V_ε): 观测噪声方差。越大更新越保守。默认 0.001

### 接口

```python
from signals.MR.kalman_spread import compute_kalman_spread

sig = compute_kalman_spread(
    x: np.ndarray,          # 资产 x 的日价格序列, shape (T,)
    y: np.ndarray,          # 资产 y 的日价格序列, shape (T,)
    delta: float = 0.0001,  # 状态变化速率
    ve: float = 0.001,      # 观测噪声方差
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `beta_slope` | np.ndarray | β₁(t), 动态斜率（对冲比率） |
| `beta_intercept` | np.ndarray | β₂(t), 动态截距 |
| `e` | np.ndarray | 预测误差 e(t) = y(t) - ŷ(t) |
| `Q` | np.ndarray | 预测误差方差 Q(t) |
| `sqrt_Q` | np.ndarray | √Q(t), 动态标准差 |
| `spread` | np.ndarray | 价差 y - β₁·x |

### 用法

```python
import numpy as np
from signals.MR.kalman_spread import compute_kalman_spread

x = np.array([...])  # 资产 x 价格
y = np.array([...])  # 资产 y 价格

sig = compute_kalman_spread(x, y)

# e 和 sqrt_Q 可直接用于策略决策：
# e < -sqrt_Q -> y 相对 x 被低估（买入信号）
# e > sqrt_Q  -> y 相对 x 被高估（卖出信号）
```

### 消费者

- `strategies/MR/s9_kalman_hedge.py`: 用 `e < -√Q` 做仅做多均值回归。策略层负责交易规则（阈值、burn_in），信号层只提供 e 和 √Q。
