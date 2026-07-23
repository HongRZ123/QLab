"""
test_strategy_signals.py - 信号版策略与原版策略的 parity 测试

验证所有已迁移到 ``Signal`` 列表接口的策略，其 ``num_units``
与原策略输出完全一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.experimental.ma_crossover_signals import ma_crossover_signals
from strategies.experimental.s4_linear_signals import linear_mr_signals
from strategies.experimental.s8_bollinger_signals import bollinger_mr_signals
from strategies.experimental.vpa_breakout_signals import vpa_breakout_signals
from strategies.experimental.vpa_trend_signals import vpa_trend_signals
from strategies.MR.s4_linear import linear_mr
from strategies.MR.s8_bollinger import bollinger_mr
from strategies.Tech.ma_crossover import ma_crossover
from strategies.Tech.vpa_breakout import vpa_breakout
from strategies.Tech.vpa_trend import vpa_trend


@pytest.fixture
def close() -> pd.Series:
    n = 200
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n)
    ou_raw = np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.Series(ou_raw, index=idx)


@pytest.fixture
def ohlcv(close: pd.Series) -> pd.DataFrame:
    open_s = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_s, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_s, close], axis=1).min(axis=1) - 0.05
    volume = pd.Series(np.random.randint(1000, 5000, len(close)), index=close.index)
    return pd.DataFrame(
        {"open": open_s, "high": high, "low": low, "close": close, "volume": volume}
    )


def _assert_units_parity(actual: pd.Series, expected: pd.Series) -> None:
    """比较 num_units 数值，忽略 Series 名称差异。"""
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_s4_linear_signals_parity(close: pd.Series) -> None:
    orig = linear_mr(close, lookback=20)
    sig = linear_mr_signals(close, lookback=20)
    assert sig["signals"]
    _assert_units_parity(sig["num_units"], orig["num_units"])


def test_s8_bollinger_signals_parity(close: pd.Series) -> None:
    orig = bollinger_mr(close, lookback=20, entry_z=1.0, exit_z=0.0)
    sig = bollinger_mr_signals(close, lookback=20, entry_z=1.0, exit_z=0.0)
    assert all(s.action in {"SET", "CLOSE"} for s in sig["signals"])
    _assert_units_parity(sig["num_units"], orig["num_units"])


def test_ma_crossover_signals_parity(close: pd.Series) -> None:
    orig = ma_crossover(close, short_window=5, long_window=20)
    sig = ma_crossover_signals(close, short_window=5, long_window=20)
    assert all(s.action in {"SET", "CLOSE"} for s in sig["signals"])
    _assert_units_parity(sig["num_units"], orig["num_units"])


def test_vpa_trend_signals_parity(ohlcv: pd.DataFrame) -> None:
    orig = vpa_trend(ohlcv, lookback=20)
    sig = vpa_trend_signals(ohlcv, lookback=20)
    assert all(s.action in {"SET", "CLOSE"} for s in sig["signals"])
    _assert_units_parity(sig["num_units"], orig["num_units"])


def test_vpa_breakout_signals_parity(ohlcv: pd.DataFrame) -> None:
    orig = vpa_breakout(
        ohlcv,
        lookback=20,
        breakout_lookback=20,
        vol_threshold=1.5,
        spread_threshold=1.5,
    )
    sig = vpa_breakout_signals(
        ohlcv,
        lookback=20,
        breakout_lookback=20,
        vol_threshold=1.5,
        spread_threshold=1.5,
    )
    assert all(s.action in {"SET", "CLOSE"} for s in sig["signals"])
    _assert_units_parity(sig["num_units"], orig["num_units"])


def test_all_signal_units_in_valid_range(close: pd.Series, ohlcv: pd.DataFrame) -> None:
    # S4 为连续仓位，允许 >1（由 max(0, -Z) 决定）；其余策略限制在 [0, 1]
    s4 = linear_mr_signals(close, lookback=20)["num_units"]
    assert (s4 >= 0).all()

    discrete_sigs = [
        bollinger_mr_signals(close, lookback=20)["num_units"],
        ma_crossover_signals(close, short_window=5, long_window=20)["num_units"],
        vpa_trend_signals(ohlcv, lookback=20)["num_units"],
        vpa_breakout_signals(ohlcv, lookback=20)["num_units"],
    ]
    for s in discrete_sigs:
        assert (s >= 0).all()
        assert (s <= 1).all()
