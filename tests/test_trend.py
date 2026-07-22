"""trend 信号单元测试"""
import numpy as np
import pandas as pd

from signals.trend import trend_health


def test_trend_health_output_shape():
    """输出形状与输入一致"""
    close = pd.Series(np.cumsum(np.random.randn(100)) + 10)
    volume = pd.Series(np.random.randint(500, 1500, 100))

    result = trend_health(close, volume, lookback=20)

    assert isinstance(result, pd.Series)
    assert result.shape == (100,)
    assert set(result.unique()).issubset({1, -1, 0})


def test_trend_health_uptrend_with_volume():
    """上涨+放量 -> 趋势健康 (+1)"""
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    volume = pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

    result = trend_health(close, volume, lookback=5)

    # 上涨+放量 = 健康
    assert result.iloc[1] == 1


def test_trend_health_uptrend_no_volume():
    """上涨+缩量 -> 趋势走弱 (-1)"""
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    volume = pd.Series([5000.0, 4000.0, 3000.0, 2000.0, 1000.0])

    result = trend_health(close, volume, lookback=5)

    # 上涨+缩量 = 走弱
    assert result.iloc[1] == -1
