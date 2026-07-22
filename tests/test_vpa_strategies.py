"""VPA 策略单元测试"""
import numpy as np
import pandas as pd

from strategies.Tech.vpa_breakout import vpa_breakout
from strategies.Tech.vpa_reversal import vpa_reversal
from strategies.Tech.vpa_trend import vpa_trend


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(n)) + 10)
    open_s = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({
        "open": open_s,
        "high": np.maximum(open_s, close) + 0.1,
        "low": np.minimum(open_s, close) - 0.1,
        "close": close,
        "volume": np.random.randint(500, 1500, n),
    })


def test_vpa_trend_output():
    """vpa_trend 返回含 num_units 的 dict"""
    ohlcv = _make_ohlcv()
    result = vpa_trend(ohlcv, lookback=20)

    assert "num_units" in result
    assert isinstance(result["num_units"], pd.Series)
    assert len(result["num_units"]) == len(ohlcv)
    assert (result["num_units"] >= 0).all()
    assert (result["num_units"] <= 1).all()


def test_vpa_reversal_output():
    """vpa_reversal 返回含 num_units 的 dict"""
    ohlcv = _make_ohlcv()
    result = vpa_reversal(ohlcv, lookback=20)

    assert "num_units" in result
    assert isinstance(result["num_units"], pd.Series)
    assert len(result["num_units"]) == len(ohlcv)
    assert (result["num_units"] >= 0).all()
    assert (result["num_units"] <= 1).all()


def test_vpa_breakout_output():
    """vpa_breakout 返回含 num_units 的 dict"""
    ohlcv = _make_ohlcv()
    result = vpa_breakout(ohlcv, lookback=20)

    assert "num_units" in result
    assert isinstance(result["num_units"], pd.Series)
    assert len(result["num_units"]) == len(ohlcv)
    assert (result["num_units"] >= 0).all()
    assert (result["num_units"] <= 1).all()
