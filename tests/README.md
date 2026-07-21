# 统计检验模块 (tests/)

均值回归策略的统计检验工具集。基于 Ernest P. Chan《Algorithmic Trading》Ch.2-3。

每个检验模块均包含**验证协议**（正控 + 负控），可直接运行 `python -m tests.sX_xxx` 验证。

---

## 模块总览

| 模块 | 编号 | 功能 | 输入 | 输出 |
|------|------|------|------|------|
| `s1_adf.py` | S1 | ADF 单位根检验 | 单资产价格序列 | 平稳性判定 (p-value) |
| `s2_hurst.py` | S2 | Hurst 指数估计 | 单资产价格序列 | 扩散特征 H 值 |
| `s3_half_life.py` | S3 | 半衰期估计 | 价格序列 | 回归速度 t₁/₂ |
| `s5_cadf.py` | S5 | CADF 协整检验 | 两资产价格 | 协整判定 + 对冲比率 |
| `s6_johansen.py` | S6 | Johansen 检验 + 组合构建 | 多资产价格矩阵 | 对冲比率 + 平稳组合 |

---

## S1: ADF 单位根检验 (`s1_adf.py`)

### 公式

$$\Delta y(t) = \lambda \, y(t-1) + \mu + \sum_{i=1}^{k} \alpha_i \, \Delta y(t-i) + \epsilon_t$$

- 检验 H₀: λ = 0（存在单位根 → 非平稳）
- t_ADF = λ / SE(λ)，与临界值比较
- 仅截距，无漂移（Chan: 日频 βt ≈ 0）

### 接口

```python
from tests.s1_adf import run_adf

result = run_adf(prices)  # prices: pd.Series
# 返回 dict:
#   adf_stat_aic, p_value_aic    — AIC 选阶
#   adf_stat_bic, p_value_bic    — BIC 选阶
#   lambda_adf                   — y(t-1) 的系数 γ
#   used_lag_aic, used_lag_bic   — 最优滞后阶数
#   n_obs                        — 实际使用的样本数
#   critical_1pct/5pct/10pct     — 临界值
```

### 判定

| p-value | 判定 |
|---------|------|
| < 0.01 | 强烈拒绝 H₀，序列平稳 |
| < 0.05 | 拒绝 H₀，序列平稳 |
| < 0.10 | 边际拒绝，弱平稳 |
| ≥ 0.10 | 不能拒绝 H₀，非平稳 |

### 依赖

`statsmodels >= 0.14.6`

---

## S2: Hurst 指数 (`s2_hurst.py`)

### 公式

$$\langle |z(t+\tau) - z(t)|^2 \rangle \sim \tau^{2H}, \quad z = \ln(y)$$

对多尺度 τ 计算增量方差，log-log 线性回归，H = slope / 2。

### 接口

```python
from tests.s2_hurst import hurst_exponent

result = hurst_exponent(prices, max_lag=100)  # prices: pd.Series
# 返回 dict:
#   hurst      — Hurst 指数
#   r_squared  — 拟合 R²
#   lags_used  — 使用的滞后值
#   variances  — 各滞后对应的增量方差
```

### 判定

| H 值 | 含义 |
|------|------|
| H < 0.5 | 均值回归（可交易） |
| H ≈ 0.5 | 随机游走 |
| H > 0.5 | 趋势（动量领域） |

### 依赖

`numpy`, `pandas`

---

## S3: 半衰期估计 (`s3_half_life.py`)

### 公式

OLS 回归（无滞后差分项）：

$$\Delta y = \lambda \cdot y_{t-1} + \mu + \epsilon$$

离散精确半衰期（Metis B1）：

$$t_{1/2} = -\frac{\ln 2}{\ln(1 + \lambda)}$$

> 注：Chan 原文使用连续近似 -ln(2)/λ，在短半衰期 (2-60天) 误差可达 23%。本模块使用离散精确公式。

### 接口

```python
from tests.s3_half_life import estimate_half_life

result = estimate_half_life(prices, use_log=True)  # prices: pd.Series
# 返回 dict:
#   lambda            — OLS 回归系数 (负值 = 均值回归)
#   half_life         — 半衰期 (天), 非均值回归时为 inf
#   r_squared         — 回归拟合度
#   is_mean_reverting — λ < 0
#   n_obs             — 有效观测数
```

### 辅助函数

```python
from tests.s3_half_life import generate_ou_paths, generate_gbm_paths

# 精确离散化 OU 过程 (正控)
paths = generate_ou_paths(n_paths=100, n_steps=1000, theta=0.05, mu=0, sigma=1)

# 几何布朗运动 (负控)
paths = generate_gbm_paths(n_paths=100, n_steps=1000, sigma=0.01)
```

### 依赖

`numpy`, `pandas`

---

## S5: CADF 协整检验 (`s5_cadf.py`)

### 公式（Engle-Granger 两步法）

1. OLS 回归：$y = h \cdot x + c + \epsilon$
2. 构造价差：$e = y - h \cdot x - c$
3. 对价差 e 做 ADF 检验（同 S1）

> 顺序依赖：y~x 和 x~y 给出不同的 h，且 h_yx ≠ 1/h_xy。`cadf_test_both_orders` 两种顺序都试，取 ADF 统计量更负者。

### 接口

```python
from tests.s5_cadf import cadf_test, cadf_test_both_orders

# 单顺序
result = cadf_test(y, x, lag=1)  # y, x: pd.Series
# 返回 dict:
#   hedge_ratio      — 对冲比率 h
#   intercept        — 回归截距 c
#   spread           — 残差序列 e = y - h·x - c
#   adf_stat, p_value — ADF 检验结果
#   lambda_spread    — 价差的均值回归系数
#   half_life_spread — 价差的半衰期

# 双顺序 (推荐)
result = cadf_test_both_orders(y, x)
```

### 依赖

`tests.s1_adf`, `tests.s3_half_life`

---

## S6: Johansen 检验 + 组合构建 (`s6_johansen.py`)

### 公式

$$\Delta \mathbf{Y}(t) = \boldsymbol{\Lambda} \, \mathbf{Y}(t-1) + \mathbf{M} + \sum_{i=1}^{k} \mathbf{A}_i \, \Delta \mathbf{Y}(t-i) + \boldsymbol{\epsilon}_t$$

- Λ 的秩 r = 独立协整关系数
- 特征向量按特征值降序排列 → v₁ 对应最短半衰期
- 组合净值：$y_{\text{port}} = \mathbf{Y} \cdot \mathbf{v}_1$

### 接口

```python
from tests.s6_johansen import johansen_test, construct_portfolio

# Johansen 检验
result = johansen_test(prices_df, lag=1)  # prices_df: pd.DataFrame (T × n)
# 返回 dict:
#   eigenvalues     — 特征值 (降序)
#   eigenvectors    — 特征向量矩阵 (n × n)
#   trace_stats     — trace 统计量
#   trace_crit      — 95% 临界值
#   rank            — 协整秩
#   yport           — 第一特征向量构造的组合净值
#   half_life       — yport 的半衰期
#   is_cointegrated — rank ≥ 1

# 手动构造组合
yport = construct_portfolio(prices_df, eigenvector)
```

### 依赖

`statsmodels` (`coint_johansen`), `tests.s3_half_life`

---

## 验证协议

每个模块均可独立运行验证：

```bash
python -m tests.s1_adf        # ADF: sh000001 应为非平稳
python -m tests.s2_hurst      # Hurst: GBM H≈0.5, OU H<0.5
python -m tests.s3_half_life  # 半衰期: OU 正控 + GBM 负控 (10,000 paths)
python -m tests.s5_cadf       # CADF: y=2x+OU 正控 + 独立GBM 负控
python -m tests.s6_johansen   # Johansen: 协整序列 rank≥1 + 独立GBM rank=0
```

---

## 模块依赖关系

```
tests/
├── s1_adf.py         ← statsmodels
├── s2_hurst.py       ← numpy, pandas
├── s3_half_life.py   ← numpy, pandas
├── s5_cadf.py        ← s1_adf, s3_half_life
└── s6_johansen.py    ← statsmodels, s3_half_life
```

`tests/` 不依赖 `strategies/`、`backtest/`、`data/`（验证协议中 `s1_adf` 的 `__main__` 除外）。
