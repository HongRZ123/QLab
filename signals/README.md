# signals/ -- 信号层

从市场数据中提取的、独立于交易策略的信息。信号只做信息提取，不输出 `num_units`，不计算 PnL。

## 设计原则

- 信号函数是**纯函数**，无副作用，相同输入始终相同输出
- 信号有**独立含义**，可以被不同类型的策略消费
- 按技术概念组织文件，不按 alpha 类型分目录
- statistics signals (stats.py, stats_cointegration.py) 是从 tests/ 迁移而来 —— 这些本身就是信号，而非单元测试

## 文件清单

| 文件 | 技术概念 | 核心函数 |
|------|----------|----------|
| `vpa.py` | 量价分析 (VPA) | `effort_vs_result`, `stopping_volume`, `buying_climax`, `no_demand`, `no_supply`, `volume_relative`, `spread`, ... |
| `pivot.py` | 价格结构 | `detect_isolated_pivots`, `detect_consolidation`, `detect_breakout` |
| `trend.py` | 趋势健康度 | `trend_direction`, `trend_health` |
| `kalman.py` | 卡尔曼滤波 | `compute_kalman_spread` |
| `stats.py` | 统计性质 | `run_adf`, `hurst_exponent`, `estimate_half_life`, `generate_ou_paths`, `generate_gbm_paths` |
| `stats_cointegration.py` | 协整性质 | `cadf_test`, `cadf_test_both_orders`, `johansen_test`, `construct_portfolio` |

## 信号清单

### 量价信号 (vpa.py)

| 信号 | 函数 | 输出 | 消费者 |
|------|------|------|--------|
| 成交量相对值 | `volume_relative(ohlcv)` | float Series (0~2+) | alpha, VPA strategies |
| 振幅 | `spread(ohlcv)` | float Series (high-low) | VPA strategies |
| 振幅相对值 | `spread_relative(ohlcv)` | float Series (0~2+) | VPA strategies |
| 上影线 | `upper_wick(ohlcv)` | float Series (≥0) | VPA strategies |
| 下影线 | `lower_wick(ohlcv)` | float Series (≥0) | VPA strategies |
| 影线比率 | `wick_ratio(ohlcv)` | float Series (≥0) | VPA strategies |
| 投入产出比 | `effort_vs_result(ohlcv)` | float Series (≈1=确认, >>1=异常) | VPA-T1 |
| 止损量 | `stopping_volume(ohlcv)` | bool Series | VPA-T2 |
| 买入高潮 | `buying_climax(ohlcv)` | bool Series | VPA-T2 |
| 无需求 | `no_demand(ohlcv)` | bool Series | VPA-T1 |
| 无供应 | `no_supply(ohlcv)` | bool Series | VPA-T1 |
| 量价确认 | `volume_confirmation(prices, vol)` | int Series (+2/+1/-1/-2/0) | S12 |
| K线影线比例 | `wick_body_ratio(o,h,l,c)` | DataFrame[4列] | S12 |
| 量价背离序列 | `volume_anomaly_sequence(prices, vol)` | int Series (+1/-1/0) | — |
| 振幅强度百分位 | `spread_strength_percentile(ohlcv)` | float Series (0~1) | VPA-T1 |
| 成交量百分位 | `volume_percentile(volume)` | float Series (0~1) | VPA-T1 |
| 量价确认矩阵 | `vpa_confirmation_matrix(ohlcv)` | str Series (confirmed/trap/anomaly/neutral) | VPA-T1 |

### 价格结构信号 (pivot.py)

| 信号 | 函数 | 输出 | 消费者 |
|------|------|------|--------|
| 孤立支点 | `detect_isolated_pivots` | DataFrame[pivot_high, pivot_low] | VPA-T3 |
| 震荡区间 | `detect_consolidation` | str Series | VPA-T3 |
| 突破检测 | `detect_breakout` | str Series | VPA-T3 |

### 趋势信号 (trend.py)

| 信号 | 函数 | 输出 | 消费者 |
|------|------|------|--------|
| 趋势方向 | `trend_direction(close)` | int Series (+1/-1/0) | VPA-T1, alpha |
| 趋势健康度 | `trend_health(close, volume)` | int Series (+1/-1/0) | VPA-T1 |

### 统计性质信号 (stats.py)

| 信号 | 函数 | 输出 | 消费者 |
|------|------|------|--------|
| ADF 检验 | `run_adf(prices)` | dict | alpha, templates |
| Hurst 指数 | `hurst_exponent(prices)` | dict | alpha, templates |
| 半衰期 | `estimate_half_life(prices)` | dict | alpha, strategies, backtest |
| OU 路径生成 | `generate_ou_paths(...)` | ndarray | strategies (run_validation) |
| GBM 路径生成 | `generate_gbm_paths(...)` | ndarray | strategies (run_validation) |

### 协整性质信号 (stats_cointegration.py)

| 信号 | 函数 | 输出 | 消费者 |
|------|------|------|--------|
| CADF 检验 | `cadf_test(y, x)` | dict | templates, alpha |
| 双顺序 CADF | `cadf_test_both_orders(y, x)` | dict | templates, alpha |
| Johansen 检验 | `johansen_test(prices_df)` | dict | templates, backtest |
| 组合构造 | `construct_portfolio(df, v)` | Series | alpha, strategies |

### 卡尔曼信号 (kalman.py)

| 信号 | 函数 | 输出 | 消费者 |
|------|------|------|--------|
| 卡尔曼 spread | `compute_kalman_spread(x, y)` | dict | S9 |

---

## vpa.py -- 量价分析信号 (VPA)

基于 Anna Coulling《量价分析》。核心原则：Effort=volume, Result=spread (high-low, 不是 body)。

### P0 基础信号

```python
from signals.vpa import volume_relative, spread, spread_relative, upper_wick, lower_wick, wick_ratio

vr = volume_relative(ohlcv, lookback=20)     # volume / rolling_mean(volume)
sp = spread(ohlcv)                           # high - low
sr = spread_relative(ohlcv, lookback=20)     # spread / rolling_mean(spread)
uw = upper_wick(ohlcv)                       # high - max(open, close)
lw = lower_wick(ohlcv)                       # min(open, close) - low
wr = wick_ratio(ohlcv)                       # max_wick / body
```

### P1 核心信号（上下文感知）

```python
from signals.vpa import effort_vs_result, stopping_volume, buying_climax, no_demand, no_supply

evr = effort_vs_result(ohlcv, lookback=20)   # volume_relative / spread_relative (≈1=确认, >>1=异常)
sv = stopping_volume(ohlcv, lookback=20)     # 下跌+锤头线+高量 → 底部反转信号 (bool)
bc = buying_climax(ohlcv, lookback=20)       # 上涨+射击十字星+高量 → 顶部反转信号 (bool)
nd = no_demand(ohlcv, lookback=20)           # 上涨+低量+小振幅 → 买方衰竭 (bool)
ns = no_supply(ohlcv, lookback=20)           # 下跌+低量+小振幅 → 卖方衰竭 (bool)
```

### 兼容信号

```python
from signals.vpa import volume_confirmation, wick_body_ratio, volume_anomaly_sequence
from signals.vpa import volume_percentile, vpa_confirmation_matrix

vc = volume_confirmation(prices, volume, lookback=20)     # int Series
wbr = wick_body_ratio(open, high, low, close)             # DataFrame
vas = volume_anomaly_sequence(prices, volume, lookback=3)  # int Series (方向已修正)
vps = volume_percentile(volume, lookback=20)              # float Series (0~1)
vcm = vpa_confirmation_matrix(ohlcv, lookback=20)         # str Series (使用 spread)
```

---

## stats.py -- 统计性质信号 (从 tests/ 迁移)

单资产价格序列的统计特征：平稳性（ADF）、均值回归性（Hurst）、回归速度（半衰期）。
原位于 tests/s1~s3，迁移到 signals/ 以体现"信号，而非测试"的架构定位。

```python
from signals.stats import run_adf, hurst_exponent, estimate_half_life
from signals.stats import generate_ou_paths, generate_gbm_paths

# ADF 单位根检验
adf = run_adf(prices)
print(adf["p_value_aic"], adf["adf_stat_aic"])

# Hurst 指数: H<0.5 均值回归, H≈0.5 随机, H>0.5 趋势
h = hurst_exponent(prices, max_lag=100)
print(h["hurst"], h["r_squared"])

# 半衰期 (离散精确公式)
hl = estimate_half_life(prices, use_log=True)
print(hl["half_life"], hl["lambda"], hl["is_mean_reverting"])

# 路径生成器 (run_validation() 用)
ou = generate_ou_paths(n_paths=100, n_steps=1000, theta=0.05, mu=0, sigma=1)
gbm = generate_gbm_paths(n_paths=100, n_steps=1000, sigma=0.01)
```

> **迁移说明**: 旧路径 `from tests.s1_adf import run_adf` 仍可用（tests/__init__.py 向后兼容），新代码推荐从 `signals.stats` 导入。

---

## stats_cointegration.py -- 协整性质信号 (从 tests/ 迁移)

多资产间协整关系的统计检验。原位于 tests/s5~s6。

```python
from signals.stats_cointegration import cadf_test, cadf_test_both_orders
from signals.stats_cointegration import johansen_test, construct_portfolio

# Engle-Granger 两步法 CADF
result = cadf_test(y_prices, x_prices)
print(result["hedge_ratio"], result["p_value"])

# 双顺序 (推荐)
result = cadf_test_both_orders(y, x)

# Johansen 多变量协整
result = johansen_test(prices_df, lag=1)
print(result["rank"], result["half_life"])

# 构造组合净值
yport = construct_portfolio(prices_df, result["eigenvectors"][:, 0])
```

---

## 验证

```bash
python -m signals.vpa       # VPA 信号
python -m signals.pivot     # 结构信号
python -m signals.trend     # 趋势健康度
# stats/stats_cointegration 暂无 python -m 入口（原 tests/s* 仍可运行）
```