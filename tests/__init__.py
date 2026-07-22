"""
tests 模块 — 单元测试

pytest 测试用例。统计检验原语 (ADF/Hurst/CADF/Johansen) 已迁移到 stat/ 模块。
"""
from signals.stats import estimate_half_life as estimate_half_life
from signals.stats import hurst_exponent as hurst_exponent
from signals.stats import run_adf as run_adf
from signals.stats_cointegration import cadf_test as cadf_test
from signals.stats_cointegration import cadf_test_both_orders as cadf_test_both_orders
from signals.stats_cointegration import construct_portfolio as construct_portfolio
from signals.stats_cointegration import johansen_test as johansen_test

__all__ = [
    "run_adf", "hurst_exponent", "estimate_half_life",
    "cadf_test", "cadf_test_both_orders",
    "johansen_test", "construct_portfolio",
]