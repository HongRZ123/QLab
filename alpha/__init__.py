"""
alpha — 标的选取模块 (cross-section screening)

按统计性质 / 因子得分 / 市场状态筛选候选标的。
alpha 做截面筛选 (输出标的列表), strategy 做时序决策 (输出 num_units)。

Functions:
    score_stationarity(close) -> dict           单标的平稳性评分
    screen_stationarity(universe, top_n) -> list[dict]  批量平稳性筛选
    get_mr_candidates() -> list[str]            均值回归默认候选
    get_trend_candidates() -> list[str]         趋势跟踪默认候选
    get_pair_candidates() -> list[tuple]        配对交易默认候选
    get_portfolio_candidates() -> list[list]    组合策略默认候选
    get_universe(category) -> list[str]         按类别取标的宇宙

详见 alpha/README.md
"""
from alpha.defaults import get_mr_candidates as get_mr_candidates
from alpha.defaults import get_pair_candidates as get_pair_candidates
from alpha.defaults import get_portfolio_candidates as get_portfolio_candidates
from alpha.defaults import get_trend_candidates as get_trend_candidates
from alpha.defaults import get_universe as get_universe
from alpha.stationarity import score_stationarity as score_stationarity
from alpha.stationarity import screen_stationarity as screen_stationarity
