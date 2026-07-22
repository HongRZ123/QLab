"""
alpha — 标的选取模块 (cross-section screening)

按统计性质 / 因子得分 / 市场状态筛选候选标的。
alpha 做截面筛选 (输出标的列表), strategy 做时序决策 (输出 num_units)。

Functions:
    score_stationarity(close) -> dict             单标的平稳性评分
    screen_stationarity(universe, top_n) -> list   批量平稳性筛选
    score_momentum(close, volume) -> dict          单标的趋势动量评分
    screen_momentum(universe, top_n) -> list       批量趋势动量筛选
    screen_pairs(universe, top_n) -> list          批量配对筛选 (CADF)
    screen_portfolio(universe, top_n) -> list      批量组合筛选 (Johansen)
    get_mr_candidates() -> list[str]              均值回归默认候选
    get_trend_candidates() -> list[str]           趋势跟踪默认候选
    get_pair_candidates() -> list[tuple]           配对交易默认候选
    get_portfolio_candidates() -> list[list]       组合策略默认候选
    get_universe(category) -> list[str]            按类别取标的宇宙

详见 alpha/README.md
"""
from alpha.cointegration import screen_pairs as screen_pairs
from alpha.cointegration import screen_portfolio as screen_portfolio
from alpha.defaults import get_mr_candidates as get_mr_candidates
from alpha.defaults import get_pair_candidates as get_pair_candidates
from alpha.defaults import get_portfolio_candidates as get_portfolio_candidates
from alpha.defaults import get_trend_candidates as get_trend_candidates
from alpha.defaults import get_universe as get_universe
from alpha.momentum import score_momentum as score_momentum
from alpha.momentum import screen_momentum as screen_momentum
from alpha.stationarity import score_stationarity as score_stationarity
from alpha.stationarity import screen_stationarity as screen_stationarity
