"""VPA 信号单元测试"""
import numpy as np
import pandas as pd

from signals.vpa import (
    volume_anomaly_sequence,
    volume_confirmation,
    wick_body_ratio,
)

# ── volume_confirmation ──────────────────────────────────────


def test_volume_confirmation_output_shape():
    """(a) volume_confirmation 输出形状与输入一致"""
    prices = pd.Series(np.linspace(10, 15, 100))
    volume = pd.Series(np.linspace(1000, 2000, 100))

    result = volume_confirmation(prices, volume, lookback=20)

    assert isinstance(result, pd.Series)
    assert result.shape == (100,)
    assert result.index.equals(prices.index)


def test_volume_confirmation_trend_nonzero():
    """(b) 线性上涨 + 递增成交量 → 非零信号占比 >= 60%"""
    prices = pd.Series(np.linspace(10, 15, 500))
    volume = pd.Series(np.linspace(1000, 2000, 500))

    result = volume_confirmation(prices, volume, lookback=20)
    non_zero = (result != 0).sum() / len(result)

    assert non_zero >= 0.60, f"非零信号占比 {non_zero:.2%} < 60%"


# ── wick_body_ratio ──────────────────────────────────────────


def test_wick_body_ratio_output_columns():
    """(c) wick_body_ratio 输出包含 4 列"""
    n = 50
    o = pd.Series(np.full(n, 10.0))
    h = pd.Series(np.full(n, 11.0))
    lo = pd.Series(np.full(n, 9.0))
    c = pd.Series(np.full(n, 10.5))

    result = wick_body_ratio(o, h, lo, c)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"body_ratio", "upper_wick_ratio",
                                   "lower_wick_ratio", "signal"}
    assert result.shape[0] == n


def test_wick_body_ratio_zero_range():
    """(d) high == low 时比例应返回 0（避免除零）"""
    o = pd.Series([10.0])
    h = pd.Series([10.0])
    lo = pd.Series([10.0])
    c = pd.Series([10.0])

    result = wick_body_ratio(o, h, lo, c)

    assert result.iloc[0]["body_ratio"] == 0.0
    assert result.iloc[0]["upper_wick_ratio"] == 0.0
    assert result.iloc[0]["lower_wick_ratio"] == 0.0
    assert result.iloc[0]["signal"] == 0


# ── volume_anomaly_sequence ──────────────────────────────────


def test_volume_anomaly_sequence_output_shape():
    """(e) volume_anomaly_sequence 输出形状与输入一致"""
    prices = pd.Series(np.linspace(10, 15, 100))
    volume = pd.Series(np.linspace(1000, 2000, 100))

    result = volume_anomaly_sequence(prices, volume, lookback=3)

    assert isinstance(result, pd.Series)
    assert result.shape == (100,)


def test_volume_anomaly_sequence_bullish_exhaustion():
    """(f) 价格上涨 + 成交量下降 → 检测到看涨耗尽 (+1)"""
    # 构造模式：价格单调上涨，成交量单调下降
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    volume = pd.Series([5000.0, 4000.0, 3000.0, 2000.0, 1000.0])

    result = volume_anomaly_sequence(prices, volume, lookback=3)

    # 从 index=3 开始应有看涨耗尽信号
    assert result.iloc[3] == 1
    assert result.iloc[4] == 1
