# alpha/ — 标的选取模块

按统计性质 / 因子得分筛选候选标的。alpha 做截面筛选（输出标的列表），strategy 做时序决策（输出 num_units）。

## 文件

| 文件 | 功能 |
|------|------|
| `stationarity.py` | 平稳性评分 + 批量筛选 |
| `momentum.py` | 趋势动量评分 + 批量筛选 |
| `cointegration.py` | 协整配对筛选 + 组合筛选 |
| `defaults.py` | ETF 分类宇宙 + 按类别取标的 |

## stationarity.py

```python
from alpha.stationarity import screen_stationarity, score_stationarity
candidates = screen_stationarity(universe, top_n=3)
```

## momentum.py

```python
from alpha.momentum import screen_momentum, score_momentum
candidates = screen_momentum(universe, top_n=3)
```

## cointegration.py

```python
from alpha.cointegration import screen_pairs, screen_portfolio
pairs = screen_pairs(universe, top_n=3)
groups = screen_portfolio(universe, top_n=3)
```

## defaults.py

```python
from alpha.defaults import get_universe, ETF_UNIVERSE
symbols = get_universe("broad_etf")   # ["sh510050", "sh510300", ...]
```
