"""
test_vpa_reversal_signals.py - 验证信号列表版 VPA 反转与原版本行为一致
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.experimental.vpa_reversal_signals import vpa_reversal_signals
from strategies.Tech.vpa_reversal import vpa_reversal


@pytest.fixture
def ohlcv_fixture() -> pd.DataFrame:
    """合成 OHLCV：前段下跌，末段出现止损量反转形态。"""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(20, 10, n), index=idx)
    open_s = close.shift(1).fillna(close.iloc[0])

    # 人为制造几根长下影线 + 高量
    for i in [80, 90, 95]:
        open_s.iloc[i] = close.iloc[i] + 0.3
        close.iloc[i] = close.iloc[i] - 0.1

    high = pd.concat([open_s, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_s, close], axis=1).min(axis=1) - 0.05
    for i in [80, 90, 95]:
        low.iloc[i] = close.iloc[i] - 0.5

    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[[80, 90, 95]] = 5000.0

    return pd.DataFrame(
        {
            "open": open_s,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_signal_version_matches_original(ohlcv_fixture: pd.DataFrame) -> None:
    original = vpa_reversal(ohlcv_fixture, lookback=20)
    signal_version = vpa_reversal_signals(ohlcv_fixture, lookback=20)

    pd.testing.assert_series_equal(
        original["num_units"],
        signal_version["num_units"],
        check_names=False,
    )


def test_signal_version_has_signals(ohlcv_fixture: pd.DataFrame) -> None:
    result = vpa_reversal_signals(ohlcv_fixture, lookback=20)
    assert isinstance(result["signals"], list)
    assert all(s.action in {"BUY", "CLOSE"} for s in result["signals"])


def test_signal_version_num_units_bounds(ohlcv_fixture: pd.DataFrame) -> None:
    result = vpa_reversal_signals(ohlcv_fixture, lookback=20)
    assert (result["num_units"] >= 0).all()
    assert (result["num_units"] <= 1).all()


def test_signal_version_uptrend_has_few_signals() -> None:
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(10, 20, n), index=idx)
    open_r = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_r, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_r, close], axis=1).min(axis=1) - 0.05

    ohlcv = pd.DataFrame(
        {
            "open": open_r,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(np.full(n, 1000.0), index=idx),
        }
    )

    result = vpa_reversal_signals(ohlcv, lookback=20)
    pos_ratio = (result["num_units"] > 0).sum() / len(result["num_units"])
    assert pos_ratio <= 0.10
