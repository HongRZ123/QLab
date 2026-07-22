"""VPA 信号单元测试 (重写版)"""
import numpy as np
import pandas as pd

from signals.vpa import (
    buying_climax,
    effort_vs_result,
    lower_wick,
    no_demand,
    no_supply,
    spread,
    spread_relative,
    spread_strength_percentile,
    stopping_volume,
    upper_wick,
    volume_anomaly_sequence,
    volume_confirmation,
    volume_percentile,
    volume_relative,
    vpa_confirmation_matrix,
    wick_body_ratio,
    wick_ratio,
)


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """构造合成 OHLCV 数据"""
    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(n)) + 10)
    open_s = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_s, close], axis=1).max(axis=1) + np.abs(np.random.randn(n)) * 0.2
    low = pd.concat([open_s, close], axis=1).min(axis=1) - np.abs(np.random.randn(n)) * 0.2
    volume = pd.Series(np.random.randint(500, 1500, n).astype(float))
    return pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })


# ── P0: 基础信号 ──────────────────────────────────────────────


def test_volume_relative_positive():
    """volume_relative 全正且无 NaN"""
    ohlcv = _make_ohlcv()
    vr = volume_relative(ohlcv, lookback=20)
    assert isinstance(vr, pd.Series)
    assert (vr > 0).all()
    assert vr.notna().all()


def test_spread_nonneg():
    """spread = high - low，全非负"""
    ohlcv = _make_ohlcv()
    sp = spread(ohlcv)
    assert isinstance(sp, pd.Series)
    assert (sp >= 0).all()


def test_spread_relative_positive():
    """spread_relative 全正且无 NaN"""
    ohlcv = _make_ohlcv()
    sr = spread_relative(ohlcv, lookback=20)
    assert (sr > 0).all()
    assert sr.notna().all()


def test_upper_wick_nonneg():
    """upper_wick 全非负"""
    ohlcv = _make_ohlcv()
    uw = upper_wick(ohlcv)
    assert (uw >= 0).all()


def test_lower_wick_nonneg():
    """lower_wick 全非负"""
    ohlcv = _make_ohlcv()
    lw = lower_wick(ohlcv)
    assert (lw >= 0).all()


def test_wick_ratio_nonneg():
    """wick_ratio 全非负且无 NaN"""
    ohlcv = _make_ohlcv()
    wr = wick_ratio(ohlcv)
    assert (wr >= 0).all()
    assert wr.notna().all()


# ── P1: 核心信号 ──────────────────────────────────────────────


def test_effort_vs_result_positive():
    """effort_vs_result 全正且无 NaN"""
    ohlcv = _make_ohlcv()
    evr = effort_vs_result(ohlcv, lookback=20)
    assert isinstance(evr, pd.Series)
    assert (evr > 0).all()
    assert evr.notna().all()


def test_stopping_volume_bool():
    """stopping_volume 返回 bool Series"""
    ohlcv = _make_ohlcv()
    sv = stopping_volume(ohlcv, lookback=20)
    assert isinstance(sv, pd.Series)
    assert sv.dtype == bool


def test_stopping_volume_detects_hammer_in_downtrend():
    """下跌趋势 + 锤头线 + 高量 -> 止损量"""
    n = 200
    close = pd.Series(np.linspace(20, 10, n))
    open_s = close.shift(1).fillna(close.iloc[0])
    open_s.iloc[-1] = 10.2
    close.iloc[-1] = 10.5
    high = pd.concat([open_s, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_s, close], axis=1).min(axis=1) - 0.05
    low.iloc[-1] = 9.0
    volume = pd.Series(np.full(n, 1000.0))
    volume.iloc[-1] = 5000.0
    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    sv = stopping_volume(ohlcv, lookback=20)
    assert sv.iloc[-1] == True  # noqa: E712


def test_buying_climax_bool():
    """buying_climax 返回 bool Series"""
    ohlcv = _make_ohlcv()
    bc = buying_climax(ohlcv, lookback=20)
    assert isinstance(bc, pd.Series)
    assert bc.dtype == bool


def test_buying_climax_detects_shooting_star_in_uptrend():
    """上涨趋势 + 射击十字星 + 高量 -> 买入高潮"""
    n = 200
    close = pd.Series(np.linspace(10, 20, n))
    open_s = close.shift(1).fillna(close.iloc[0])
    open_s.iloc[-1] = 19.8
    close.iloc[-1] = 19.5
    high = pd.concat([open_s, close], axis=1).max(axis=1) + 0.05
    high.iloc[-1] = 21.0
    low = pd.concat([open_s, close], axis=1).min(axis=1) - 0.05
    volume = pd.Series(np.full(n, 1000.0))
    volume.iloc[-1] = 5000.0
    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    bc = buying_climax(ohlcv, lookback=20)
    assert bc.iloc[-1] == True  # noqa: E712


def test_no_demand_bool():
    """no_demand 返回 bool Series"""
    ohlcv = _make_ohlcv()
    nd = no_demand(ohlcv, lookback=20)
    assert isinstance(nd, pd.Series)
    assert nd.dtype == bool


def test_no_supply_bool():
    """no_supply 返回 bool Series"""
    ohlcv = _make_ohlcv()
    ns = no_supply(ohlcv, lookback=20)
    assert isinstance(ns, pd.Series)
    assert ns.dtype == bool


def test_no_demand_no_supply_mutually_exclusive():
    """no_demand 和 no_supply 不应同时为 True"""
    ohlcv = _make_ohlcv()
    nd = no_demand(ohlcv, lookback=20)
    ns = no_supply(ohlcv, lookback=20)
    assert not (nd & ns).any()


# ── 兼容信号 (已修正方向) ──────────────────────────────────────


def test_volume_confirmation_output_shape():
    """volume_confirmation 输出形状与输入一致"""
    prices = pd.Series(np.linspace(10, 15, 100))
    volume = pd.Series(np.linspace(1000, 2000, 100))
    result = volume_confirmation(prices, volume, lookback=20)
    assert isinstance(result, pd.Series)
    assert result.shape == (100,)
    assert result.index.equals(prices.index)


def test_volume_confirmation_trend_nonzero():
    """线性上涨 + 递增成交量 → 非零信号占比 >= 60%"""
    prices = pd.Series(np.linspace(10, 15, 500))
    volume = pd.Series(np.linspace(1000, 2000, 500))
    result = volume_confirmation(prices, volume, lookback=20)
    non_zero = (result != 0).sum() / len(result)
    assert non_zero >= 0.60, f"非零信号占比 {non_zero:.2%} < 60%"


def test_volume_confirmation_no_demand_direction():
    """上涨 + 低量 = -1 (No Demand, 买方衰竭) -- 修正方向"""
    prices = pd.Series([10.0, 11.0, 12.0, 13.0])
    volume = pd.Series([5000.0, 1000.0, 500.0, 200.0])
    result = volume_confirmation(prices, volume, lookback=3)
    assert result.iloc[-1] == -1, f"期望 -1 (No Demand), 得到 {result.iloc[-1]}"


def test_volume_confirmation_no_supply_direction():
    """下跌 + 低量 = +1 (No Supply, 卖方衰竭) -- 修正方向"""
    prices = pd.Series([13.0, 12.0, 11.0, 10.0])
    volume = pd.Series([5000.0, 1000.0, 500.0, 200.0])
    result = volume_confirmation(prices, volume, lookback=3)
    assert result.iloc[-1] == 1, f"期望 +1 (No Supply), 得到 {result.iloc[-1]}"


def test_wick_body_ratio_output_columns():
    """wick_body_ratio 输出包含 4 列"""
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
    """high == low 时比例应返回 0（避免除零）"""
    o = pd.Series([10.0])
    h = pd.Series([10.0])
    lo = pd.Series([10.0])
    c = pd.Series([10.0])
    result = wick_body_ratio(o, h, lo, c)
    assert result.iloc[0]["body_ratio"] == 0.0
    assert result.iloc[0]["upper_wick_ratio"] == 0.0
    assert result.iloc[0]["lower_wick_ratio"] == 0.0
    assert result.iloc[0]["signal"] == 0


def test_volume_anomaly_sequence_output_shape():
    """volume_anomaly_sequence 输出形状与输入一致"""
    prices = pd.Series(np.linspace(10, 15, 100))
    volume = pd.Series(np.linspace(1000, 2000, 100))
    result = volume_anomaly_sequence(prices, volume, lookback=3)
    assert isinstance(result, pd.Series)
    assert result.shape == (100,)


def test_volume_anomaly_sequence_no_demand_direction():
    """价格上涨 + 成交量下降 → -1 (No Demand) -- 修正方向"""
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    volume = pd.Series([5000.0, 4000.0, 3000.0, 2000.0, 1000.0])
    result = volume_anomaly_sequence(prices, volume, lookback=3)
    assert result.iloc[-1] == -1, f"期望 -1 (No Demand), 得到 {result.iloc[-1]}"


def test_volume_anomaly_sequence_no_supply_direction():
    """价格下跌 + 成交量上升 → +1 (卖方吸收) -- 修正方向"""
    prices = pd.Series([14.0, 13.0, 12.0, 11.0, 10.0])
    volume = pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    result = volume_anomaly_sequence(prices, volume, lookback=3)
    assert result.iloc[-1] == 1, f"期望 +1 (卖方吸收), 得到 {result.iloc[-1]}"


def test_spread_strength_percentile_range():
    """spread_strength_percentile 输出在 [0, 1] 范围内"""
    ohlcv = _make_ohlcv()
    result = spread_strength_percentile(ohlcv, lookback=20)
    assert isinstance(result, pd.Series)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_spread_strength_percentile_large_spread_high_rank():
    """大振幅K线应有较高百分位"""
    n = 21
    close = pd.Series(np.full(n, 10.0))
    open_s = pd.Series(np.full(n, 10.0))
    high = pd.Series(np.full(n, 10.1))
    low = pd.Series(np.full(n, 9.9))
    high.iloc[-1] = 12.0
    low.iloc[-1] = 8.0
    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": pd.Series(np.full(n, 1000.0)),
    })
    result = spread_strength_percentile(ohlcv, lookback=20)
    assert result.iloc[-1] > 0.9


def test_volume_percentile_range():
    """volume_percentile 输出在 [0, 1] 范围内"""
    volume = pd.Series(np.random.randint(500, 1500, 100).astype(float))
    result = volume_percentile(volume, lookback=20)
    assert isinstance(result, pd.Series)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_volume_percentile_max_volume_high_rank():
    """最大成交量应有最高百分位"""
    volume = pd.Series([1000.0] * 20 + [5000.0])
    result = volume_percentile(volume, lookback=20)
    assert result.iloc[-1] > 0.9


def test_vpa_confirmation_matrix_categories():
    """输出只包含 4 种合法类别"""
    ohlcv = _make_ohlcv()
    result = vpa_confirmation_matrix(ohlcv, lookback=20)
    assert isinstance(result, pd.Series)
    valid_cats = {"confirmed", "trap", "anomaly", "neutral"}
    assert set(result.unique()).issubset(valid_cats)


def test_vpa_confirmation_matrix_trend_confirmed():
    """趋势 + 放量 -> confirmed 占比较高"""
    np.random.seed(42)
    n = 500
    close = pd.Series(np.cumsum(np.random.randn(n) * 0.3) + 10)
    open_s = close.shift(1).fillna(close.iloc[0])
    body = (close - open_s).abs()
    high = pd.concat([open_s, close], axis=1).max(axis=1) + body * 0.3
    low = pd.concat([open_s, close], axis=1).min(axis=1) - body * 0.3
    volume = (body * 10000 + np.random.randint(100, 500, n)).clip(lower=100)
    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    result = vpa_confirmation_matrix(ohlcv, lookback=20)
    confirmed_ratio = (result == "confirmed").sum() / len(result)
    assert confirmed_ratio >= 0.25, f"confirmed 占比 {confirmed_ratio:.2%} < 25%"