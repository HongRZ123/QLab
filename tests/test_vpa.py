"""VPA 信号单元测试"""
import numpy as np
import pandas as pd

from signals.vpa import (
    body_strength_percentile,
    volume_anomaly_sequence,
    volume_confirmation,
    volume_percentile,
    vpa_confirmation_matrix,
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


# ── body_strength_percentile ─────────────────────────────────


def test_body_strength_percentile_range():
    """body_strength_percentile 输出在 [0, 1] 范围内"""
    n = 100
    open_s = pd.Series(np.random.uniform(9, 10, n))
    close_s = pd.Series(np.random.uniform(9, 11, n))

    result = body_strength_percentile(open_s, close_s, lookback=20)

    assert isinstance(result, pd.Series)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_body_strength_percentile_large_body_high_rank():
    """大实体K线应有较高百分位"""
    open_s = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                        10.0])
    close_s = pd.Series([10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1,
                         10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1,
                         11.0])  # 最后一根大实体

    result = body_strength_percentile(open_s, close_s, lookback=20)
    assert result.iloc[-1] > 0.9  # 最大实体应有高百分位


# ── volume_percentile ────────────────────────────────────────


def test_volume_percentile_range():
    """volume_percentile 输出在 [0, 1] 范围内"""
    volume = pd.Series(np.random.randint(500, 1500, 100))

    result = volume_percentile(volume, lookback=20)

    assert isinstance(result, pd.Series)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_volume_percentile_max_volume_high_rank():
    """最大成交量应有最高百分位"""
    volume = pd.Series([1000.0] * 20 + [5000.0])

    result = volume_percentile(volume, lookback=20)
    assert result.iloc[-1] > 0.9


# ── vpa_confirmation_matrix ──────────────────────────────────


def test_vpa_confirmation_matrix_categories():
    """输出只包含 4 种合法类别"""
    n = 100
    open_s = pd.Series(np.random.uniform(9, 10, n))
    close_s = pd.Series(np.random.uniform(9, 11, n))
    volume = pd.Series(np.random.randint(500, 1500, n))

    result = vpa_confirmation_matrix(open_s, close_s, volume, lookback=20)

    assert isinstance(result, pd.Series)
    valid_cats = {"confirmed", "trap", "anomaly", "neutral"}
    assert set(result.unique()).issubset(valid_cats)


def test_vpa_confirmation_matrix_trend_confirmed():
    """趋势 + 放量 -> confirmed 占比较高"""
    np.random.seed(42)
    n = 500
    # 价格有波动，实体大小自然变化；成交量与实体大小正相关
    close_s = pd.Series(np.cumsum(np.random.randn(n) * 0.3) + 10)
    open_s = close_s.shift(1).fillna(close_s.iloc[0])
    body = (close_s - open_s).abs()
    volume = (body * 10000 + np.random.randint(100, 500, n)).clip(lower=100)

    result = vpa_confirmation_matrix(open_s, close_s, volume, lookback=20)
    confirmed_ratio = (result == "confirmed").sum() / len(result)

    assert confirmed_ratio >= 0.30, f"confirmed 占比 {confirmed_ratio:.2%} < 30%"
