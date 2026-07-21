"""
backtest — 回测引擎模块
=======================
提供信号驱动的资金曲线回测、Walk-Forward 参数估计和绩效评价指标。

用法:
    from backtest import run_backtest, performance_summary
    from backtest import walk_forward_linear_mr, walk_forward_bollinger
"""

__all__ = [
    "run_backtest",
    "run_backtest_long_only",
    "performance_summary",
    "walk_forward_linear_mr",
    "walk_forward_bollinger",
    "walk_forward_portfolio",
]

from backtest.engine import run_backtest, run_backtest_long_only
from backtest.metrics import performance_summary
from backtest.walk_forward import (
    walk_forward_bollinger,
    walk_forward_linear_mr,
    walk_forward_portfolio,
)
