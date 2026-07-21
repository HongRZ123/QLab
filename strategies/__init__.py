"""
strategies — 量化策略模块
========================
提供多种均值回归与统计套利策略实现。

用法:
    from strategies import linear_mr, bollinger_mr
    from strategies import kalman_hedge, kalman_mm
    from strategies.registry import run_strategy, list_names
    from strategies.experimental.s11_rsi_draft import rsi_mean_reversion
"""

__all__ = [
    "linear_mr",
    "bollinger_mr",
    "bollinger_portfolio",
    "linear_portfolio",
    "kalman_hedge",
    "kalman_mm",
    "ma_crossover",
    "registry",
    "experimental",
]

from strategies import (
    experimental,  # noqa: F401
    registry,  # noqa: F401
)
from strategies.MM.s10_kalman_mm import kalman_mm
from strategies.MR.s4_linear import linear_mr
from strategies.MR.s7_linear_portfolio import linear_portfolio
from strategies.MR.s8_bollinger import bollinger_mr, bollinger_portfolio
from strategies.MR.s9_kalman_hedge import kalman_hedge
from strategies.Tech.ma_crossover import ma_crossover
