"""
QLab signals layer -- market-given information extracted from data

信号层：从市场数据中提取的、独立于交易策略的信息。

按技术概念组织，每个文件一组相关信号：
    vpa.py    -- 量价分析信号（volume_confirmation, wick_body_ratio, ...）
    kalman.py -- 卡尔曼滤波信号（compute_kalman_spread）
    pivot.py  -- 价格结构信号（支点、区间、突破）
    trend.py  -- 趋势健康度信号
"""
from signals.kalman import compute_kalman_spread as compute_kalman_spread
from signals.pivot import detect_breakout as detect_breakout
from signals.pivot import detect_consolidation as detect_consolidation
from signals.pivot import detect_isolated_pivots as detect_isolated_pivots
from signals.trend import trend_health as trend_health
from signals.vpa import body_strength_percentile as body_strength_percentile
from signals.vpa import volume_anomaly_sequence as volume_anomaly_sequence
from signals.vpa import volume_confirmation as volume_confirmation
from signals.vpa import volume_percentile as volume_percentile
from signals.vpa import vpa_confirmation_matrix as vpa_confirmation_matrix
from signals.vpa import wick_body_ratio as wick_body_ratio
