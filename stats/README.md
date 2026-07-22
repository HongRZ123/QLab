# stats/ — 统计分析与规律发现

对价格序列做离线统计分析，用于发现规律、筛选标的、检验假设。

与 `signals/` 的区别：
- `signals/` — 实时交易信号，供策略做决策（"现在发生什么？"）
- `stats/` — 离线统计分析，供研究者做发现（"这个序列有什么统计特征？"）

## 文件

| 文件 | 函数 | 用途 |
|------|------|------|
| `univariate.py` | `run_adf`, `hurst_exponent`, `estimate_half_life`, `generate_ou_paths`, `generate_gbm_paths` | 单资产统计检验 |
| `cointegration.py` | `cadf_test`, `cadf_test_both_orders`, `johansen_test`, `construct_portfolio`, `generate_cointegrated_paths`, `generate_gbm_matrix` | 多资产协整检验 |
| `scan.py` | `scan_all` | 全市场平稳性扫描 + CSV 输出 |

## univariate.py — 单资产统计检验

```python
from stats.univariate import run_adf, hurst_exponent, estimate_half_life

adf = run_adf(prices)              # ADF 单位根检验
h = hurst_exponent(prices)         # Hurst 指数: H<0.5 MR, H>0.5 趋势
hl = estimate_half_life(prices)    # 半衰期 (离散精确公式)
```

## cointegration.py — 协整检验

```python
from stats.cointegration import cadf_test, johansen_test

c = cadf_test(y, x)                # Engle-Granger CADF
j = johansen_test(prices_df)       # Johansen 多变量协整
```

## scan.py — 全市场平稳性扫描

```python
python stats/scan.py               # 输出 output/stationarity_report.csv
```

## 从旧路径迁移

| 旧路径 | 新路径 |
|--------|--------|
| `from signals.stats import` | `from stats.univariate import` |
| `from signals.stats_cointegration import` | `from stats.cointegration import` |
| `from tests.s1_adf import run_adf` | `from stats.univariate import run_adf` |
| `python explore/scan_stationarity.py` | `python stats/scan.py` |

`signals/__init__.py` 和 `tests/__init__.py` 均从 `stats/` 重导出以保持向后兼容。
