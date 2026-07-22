"""
QLab signals layer -- market-given information extracted from data

信号层：从市场数据中提取的、独立于交易策略的信息。

按技术概念组织，每个文件一组相关信号：
    vpa.py    -- 量价分析信号（volume_confirmation, wick_body_ratio, ...）
    kalman.py -- 卡尔曼滤波信号（compute_kalman_spread）
"""
from signals.vpa import volume_anomaly_sequence as volume_anomaly_sequence
from signals.vpa import volume_confirmation as volume_confirmation
from signals.vpa import wick_body_ratio as wick_body_ratio
