# strategies/MR -- 均值回归策略

从价格偏离均值的回归行为中寻找 alpha。核心假设：价格偏离均衡后会回归，偏离越大回归越快。

## 策略清单

| 策略 | 模块 | 信号来源 | num_units | 注册表 |
|------|------|----------|-----------|--------|
| S4 线性均值回归 | `s4_linear.py` | 价格序列（内联 Z-score） | ≥ 0 (连续) | ✅ `linear_mr` |
| S7 线性组合 MR | `s7_linear_portfolio.py` | 价格矩阵（内联 Z-score） | ≥ 0 (连续) | ❌ |
| S8 布林带 MR | `s8_bollinger.py` | 价格序列（内联 Z-score） | {0, 1} | ✅ `bollinger_mr` |
| S9 卡尔曼对冲 | `s9_kalman_hedge.py` | `signals/MR/kalman_spread` | {0, 1} | ❌ |

> **重要**: 策略内置的 `pnl`/`ret` 为**理论值**（无成本、无 T+1、无手数取整）。生产回测请使用 `backtest.run_backtest()`。

---

## S4 - 线性均值回归 `s4_linear.py`

### 含义

Chan Ch.2 核心策略。计算价格的 Z-score（偏离滚动均值的标准差倍数），做空高估、做多低估。仅做多版本：Z 为负（价格低于均值）时建仓。

### 公式

```
Z(t) = (y(t) - MA(y, L)) / Std(y, L)
num_units(t) = max(0, -Z(t))       ← 仅做多
mkt_val(t) = num_units(t) × y(t)
pnl(t) = mkt_val(t-1) × (y(t) - y(t-1)) / y(t-1)
```

回望期 L 自动确定: `lookback` 显式传入 > `half_life` -> `round(half_life)` > 自动估计半衰期。

### 接口

```python
from strategies.MR.s4_linear import linear_mr

result = linear_mr(
    prices: pd.Series,              # 日价格序列
    lookback: int | None = None,    # 回望期 L (天). None 时自动确定
    half_life: float | None = None, # 半衰期. lookback=None 时 L=round(half_life)
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `z_score` | pd.Series | Z(t), 前 L-1 天为 NaN |
| `num_units` | pd.Series | 仓位单元 (≥0, 仅做多) |
| `mkt_val` | pd.Series | 市值 = num_units × price |
| `pnl` | pd.Series | 理论每日盈亏 |
| `ret` | pd.Series | 理论每日收益率 |
| `lookback_used` | int | 实际使用的回望期 L |

### 用法

```python
from data import read_day
from strategies.MR.s4_linear import linear_mr
from backtest import run_backtest, performance_summary

df = read_day("sh512670")
result = linear_mr(df["close"], half_life=14.0)
bt = run_backtest(df["close"], result["num_units"])
stats = performance_summary(bt["ret"])
```

### 验证

```bash
python -m strategies.MR.s4_linear    # OU 正控 + GBM 负控
```

---

## S7 - 线性均值回归（组合）`s7_linear_portfolio.py`

### 含义

S4 的多资产组合版本。用 Johansen 特征向量构造协整组合 yport，对 yport 做 Z-score 均值回归。适用于多个资产存在长期协整关系的场景。

### 公式

```
yport(t) = Σ vᵢ · yᵢ(t)                    # 组合净值
Z(t) = (yport(t) - MA(yport, L)) / Std(yport, L)
num_units(t) = max(0, -Z(t))                 ← 仅做多
positions_i(t) = num_units(t) × vᵢ × yᵢ(t)  # 各资产市值
pnl(t) = Σ positions_i(t-1) × (yᵢ(t) - yᵢ(t-1)) / yᵢ(t-1)
```

### 接口

```python
from strategies.MR.s7_linear_portfolio import linear_portfolio

result = linear_portfolio(
    prices_df: pd.DataFrame,      # 价格矩阵 (T × n), 列=资产
    eigenvector: np.ndarray,      # 特征向量 (长度 n), 来自 Johansen 检验
    lookback: int | None = None,  # 回望期. None 时自动估计
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `yport` | pd.Series | 组合净值 |
| `z_score` | pd.Series | Z(t) |
| `num_units` | pd.Series | 仓位倍数 (≥0) |
| `positions` | pd.DataFrame | 各资产市值 (T × n) |
| `pnl` | pd.Series | 理论每日盈亏 |
| `ret` | pd.Series | 理论每日收益率 |
| `lookback_used` | int | 实际回望期 |

### 验证

```bash
python -m strategies.MR.s7_linear_portfolio    # 协整组合正控
```

---

## S8 - 布林带均值回归 `s8_bollinger.py`

### 含义

Chan Ch.3 策略。用布林带（滚动均值 ± N 倍标准差）做均值回归。价格跌破下轨时买入，回到均值时卖出。与 S4 的区别：S8 用离散信号（0/1），S4 用连续仓位。

### 公式

```
Z(t) = (y(t) - MA(y, L)) / Std(y, L)
Z < -entry_z  -> num_units = 1 (买入)
Z >= -exit_z  -> num_units = 0 (卖出)
无信号日 -> forward-fill 沿用前一日仓位
num_units ∈ {0, 1}
```

### 接口

```python
from strategies.MR.s8_bollinger import bollinger_mr, bollinger_portfolio

result = bollinger_mr(
    prices: pd.Series,       # 日价格序列
    lookback: int,           # 回望期 L (天)
    entry_z: float = 1.0,    # 入场 Z-Score 阈值
    exit_z: float = 0.0,     # 出场 Z-Score 阈值, entry_z > exit_z
) -> dict

# 组合便捷封装
result = bollinger_portfolio(
    yport: pd.Series,        # 组合净值序列
    lookback: int,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `z_score` | pd.Series | Z(t) |
| `num_units` | pd.Series | 持仓 ∈ {0, 1} |
| `signals` | pd.Series | 原始信号 (NaN=无信号日) |
| `pnl` | pd.Series | 理论每日盈亏 |
| `ret` | pd.Series | 理论每日收益率 |
| `n_trades` | int | 往返交易次数 |
| `avg_holding` | float | 平均持仓天数 |

### 验证

```bash
python -m strategies.MR.s8_bollinger    # OU 正控 + GBM 负控
```

---

## S9 - 卡尔曼滤波动态对冲 `s9_kalman_hedge.py`

### 含义

Chan Ch.3 Box 3.1 策略。用卡尔曼滤波估计两个资产间的动态对冲比率 β(t)，以预测误差 e(t) 和动态标准差 √Q(t) 构造布林带信号。当 y 相对 x 被低估（e < -√Q）时买入。

**信号提取委托给 `signals/MR/kalman_spread.py`**，策略层只保留交易规则和 PnL 计算。

### 公式

```
# 信号层 (signals/MR/kalman_spread.py):
卡尔曼滤波 -> β(t), e(t), Q(t), √Q(t)

# 策略层 (本文件):
e(t) < -√Q(t) -> num_units = 1 (买入)
e(t) > -√Q(t) -> num_units = 0 (卖出)
前 burn_in 天信号强制为 0

# PnL (动态对冲):
spread(t) = y(t) - β₁(t) · x(t)
d_spread(t) = (y(t)-y(t-1)) - β₁(t-1)·(x(t)-x(t-1))
pnl(t) = num_units(t-1) · d_spread(t)
```

### 接口

```python
from strategies.MR.s9_kalman_hedge import kalman_hedge

result = kalman_hedge(
    x: np.ndarray,             # 资产 x 价格序列, shape (T,)
    y: np.ndarray,             # 资产 y 价格序列, shape (T,)
    delta: float = 0.0001,     # 状态变化速率
    ve: float = 0.001,         # 观测噪声方差
    burn_in: int = 50,         # 预热期 (天)
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `beta_slope` | np.ndarray | β₁(t), 动态斜率 |
| `beta_intercept` | np.ndarray | β₂(t), 动态截距 |
| `e` | np.ndarray | 预测误差 |
| `Q` | np.ndarray | 预测误差方差 |
| `sqrt_Q` | np.ndarray | √Q, 动态标准差 |
| `signals` | np.ndarray | 原始信号 (1=买入, 0=卖出) |
| `num_units` | np.ndarray | 仓位 ∈ {0, 1} |
| `pnl` | np.ndarray | 理论每日盈亏 |
| `ret` | np.ndarray | 理论每日收益率 |
| `spread` | np.ndarray | 价差 y - β₁·x |
| `burn_in` | int | 预热期参数 |

### 验证

```bash
python -m strategies.MR.s9_kalman_hedge    # 线性关系 β 收敛 + 独立 GBM 负控
```
