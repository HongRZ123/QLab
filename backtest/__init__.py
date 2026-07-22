"""
backtest — 回测引擎模块
=======================
提供信号驱动的资金曲线回测、Walk-Forward 参数估计和绩效评价指标。

用法:
    from backtest import run_backtest, performance_summary
"""

__all__ = [
    "run_backtest",
    "run_backtest_long_only",
    "performance_summary",
]

from backtest.engine import run_backtest, run_backtest_long_only
from backtest.metrics import performance_summary
