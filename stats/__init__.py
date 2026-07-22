"""
stat — 统计分析与规律发现模块

对价格序列做离线统计分析，用于发现规律、筛选标的、检验假设。
与 signals/ 的区别：
    signals/ — 实时信号，供策略做交易决策（"现在发生什么？"）
    stat/    — 离线分析，供研究者做规律发现（"这个序列有什么统计特征？"）

组成：
    stat/univariate.py     ADF 单位根、Hurst 指数、半衰期估计 + 路径生成器
    stat/cointegration.py  CADF 配对协整、Johansen 多变量协整
    stat/scan.py           平稳性全市场扫描 (从 explore/ 迁移)
"""
from stats.cointegration import cadf_test as cadf_test
from stats.cointegration import cadf_test_both_orders as cadf_test_both_orders
from stats.cointegration import construct_portfolio as construct_portfolio
from stats.cointegration import generate_cointegrated_paths as generate_cointegrated_paths
from stats.cointegration import generate_gbm_matrix as generate_gbm_matrix
from stats.cointegration import johansen_test as johansen_test
from stats.univariate import estimate_half_life as estimate_half_life
from stats.univariate import generate_gbm_paths as generate_gbm_paths
from stats.univariate import generate_ou_paths as generate_ou_paths
from stats.univariate import hurst_exponent as hurst_exponent
from stats.univariate import run_adf as run_adf
