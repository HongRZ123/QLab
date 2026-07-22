"""
QLab signals layer -- market-given information extracted from data

信号层：从市场数据中提取的、独立于交易策略的信息。

按技术概念组织，每个文件一组相关信号：
    vpa.py                  -- 量价分析信号
    kalman.py               -- 卡尔曼滤波信号
    pivot.py                -- 价格结构信号
    trend.py                -- 趋势健康度信号
    stats.py                -- 统计信号 (ADF/Hurst/半衰期)
    stats_cointegration.py  -- 协整信号 (CADF/Johansen)
"""
from signals.kalman import compute_kalman_spread as compute_kalman_spread
from signals.pivot import detect_breakout as detect_breakout
from signals.pivot import detect_consolidation as detect_consolidation
from signals.pivot import detect_isolated_pivots as detect_isolated_pivots
from signals.stats import estimate_half_life as estimate_half_life
from signals.stats import generate_gbm_paths as generate_gbm_paths
from signals.stats import generate_ou_paths as generate_ou_paths
from signals.stats import hurst_exponent as hurst_exponent
from signals.stats import run_adf as run_adf
from signals.stats_cointegration import cadf_test as cadf_test
from signals.stats_cointegration import cadf_test_both_orders as cadf_test_both_orders
from signals.stats_cointegration import construct_portfolio as construct_portfolio
from signals.stats_cointegration import generate_cointegrated_paths as generate_cointegrated_paths
from signals.stats_cointegration import generate_gbm_matrix as generate_gbm_matrix
from signals.stats_cointegration import johansen_test as johansen_test
from signals.trend import trend_direction as trend_direction
from signals.trend import trend_health as trend_health
from signals.vpa import buying_climax as buying_climax
from signals.vpa import effort_vs_result as effort_vs_result
from signals.vpa import lower_wick as lower_wick
from signals.vpa import no_demand as no_demand
from signals.vpa import no_supply as no_supply
from signals.vpa import spread as spread
from signals.vpa import spread_relative as spread_relative
from signals.vpa import spread_strength_percentile as spread_strength_percentile
from signals.vpa import stopping_volume as stopping_volume
from signals.vpa import upper_wick as upper_wick
from signals.vpa import volume_anomaly_sequence as volume_anomaly_sequence
from signals.vpa import volume_confirmation as volume_confirmation
from signals.vpa import volume_percentile as volume_percentile
from signals.vpa import volume_relative as volume_relative
from signals.vpa import vpa_confirmation_matrix as vpa_confirmation_matrix
from signals.vpa import wick_body_ratio as wick_body_ratio
from signals.vpa import wick_ratio as wick_ratio
