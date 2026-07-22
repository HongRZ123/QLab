"""pivot 信号单元测试"""
import numpy as np
import pandas as pd

from signals.pivot import (
    detect_breakout,
    detect_consolidation,
    detect_isolated_pivots,
)

# ── detect_isolated_pivots ───────────────────────────────────


def test_pivot_high_detection():
    """高位孤立支点被正确检测"""
    high = pd.Series([10.0, 12.0, 10.0])
    low = pd.Series([9.0, 11.0, 9.0])

    pivots = detect_isolated_pivots(high, low)

    assert not np.isnan(pivots["pivot_high"].iloc[1])
    assert pivots["pivot_high"].iloc[1] == 12.0


def test_pivot_low_detection():
    """低位孤立支点被正确检测"""
    high = pd.Series([10.0, 8.0, 10.0])
    low = pd.Series([9.0, 7.0, 9.0])

    pivots = detect_isolated_pivots(high, low)

    assert not np.isnan(pivots["pivot_low"].iloc[1])
    assert pivots["pivot_low"].iloc[1] == 7.0


def test_pivot_no_false_positive():
    """非支点位置应返回 NaN"""
    high = pd.Series([10.0, 10.0, 10.0])
    low = pd.Series([9.0, 9.0, 9.0])

    pivots = detect_isolated_pivots(high, low)

    assert np.isnan(pivots["pivot_high"].iloc[1])
    assert np.isnan(pivots["pivot_low"].iloc[1])


# ── detect_consolidation ─────────────────────────────────────


def test_consolidation_in_range():
    """价格在区间内反复 -> in_range"""
    np.random.seed(42)
    n = 100
    prices = pd.Series(np.random.uniform(9.5, 10.5, n))
    high = prices + 0.1
    low = prices - 0.1

    result = detect_consolidation(high, low, tolerance=0.05)

    valid = result.dropna()
    in_range_ratio = (valid == "in_range").sum() / len(valid)
    assert in_range_ratio >= 0.50


# ── detect_breakout ──────────────────────────────────────────


def test_breakout_confirmed_high_volume():
    """放量突破 -> breakout_confirmed"""
    close = pd.Series(np.concatenate([
        np.full(50, 10.0),
        np.linspace(10, 15, 50),
    ]))
    volume = pd.Series(np.concatenate([
        np.full(50, 500.0),
        np.linspace(500, 5000, 50),
    ]))

    result = detect_breakout(close, volume, range_bound=(11, 9), lookback=20)

    assert (result == "breakout_confirmed").sum() > 0


def test_breakout_false_low_volume():
    """缩量突破 -> false_breakout"""
    close = pd.Series(np.concatenate([
        np.full(50, 10.0),
        np.full(50, 12.0),
    ]))
    volume = pd.Series(np.full(100, 500.0))  # 恒定低量

    result = detect_breakout(close, volume, range_bound=(11, 9), lookback=20)

    # 突破到 12 但成交量恒定 -> false_breakout
    assert (result == "false_breakout").sum() > 0
