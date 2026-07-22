"""
VPA (Volume Price Analysis) signals -- rewritten per Anna Coulling's book.

系统性重写，修正 5 个根本性错误：
    1. spread = high - low（不是 body = |close-open|）
    2. 信号需要趋势上下文（不能孤立分析单根K线）
    3. volume_anomaly_sequence 方向修正：价格上涨+量缩 = 看跌 (No Demand)
    4. vpa_confirmation_matrix 使用 spread 代替 body
    5. 所有核心信号接受 OHLCV DataFrame，使用多列信息

P0 基础信号:
    volume_relative:    成交量相对值 (volume / rolling_mean)
    spread:             K线振幅 (high - low)
    spread_relative:    振幅相对值 (spread / rolling_mean)
    upper_wick:         上影线长度
    lower_wick:         下影线长度
    wick_ratio:         影线/实体比率

P1 核心信号 (上下文感知):
    effort_vs_result:   投入产出比 (Wyckoff 第三定律)
    stopping_volume:    止损量 / 卖出高潮 (底部反转)
    buying_climax:      买入高潮 (顶部反转)
    no_demand:          无需求 (弱势反弹)
    no_supply:          无供应 (强势回撤)

兼容信号 (已修正):
    volume_confirmation:       量价确认 (修正方向)
    wick_body_ratio:           K线影线比例
    volume_anomaly_sequence:   量价背离序列 (修正方向)
    spread_strength_percentile: 振幅强度百分位 (替代 body_strength_percentile)
    volume_percentile:         成交量百分位
    vpa_confirmation_matrix:   量价确认矩阵 (使用 spread)

Reference: docs/VPA_完整提取.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# P0 -- 基础信号
# ============================================================

def volume_relative(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """成交量相对值

    当日成交量 / 近 lookback 日成交量均值。
    >1 = 高量, <1 = 低量, >1.5 = 极高量, <0.5 = 极低量。

    书中原则：成交量永远是一个相对概念。
    """
    vol = ohlcv["volume"]
    return vol / vol.rolling(window=lookback, min_periods=1).mean()


def spread(ohlcv: pd.DataFrame) -> pd.Series:
    """K线振幅 = high - low

    书中核心概念：Result（产出）= spread，不是 body。
    Effort = volume, Result = spread。两者匹配 = 确认，不匹配 = 异常。
    """
    return ohlcv["high"] - ohlcv["low"]


def spread_relative(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """振幅相对值 = spread / rolling_mean(spread)

    >1 = 大振幅, <1 = 小振幅, >1.5 = 极大振幅, <0.7 = 极小振幅。
    """
    s = spread(ohlcv)
    return s / s.rolling(window=lookback, min_periods=1).mean()


def upper_wick(ohlcv: pd.DataFrame) -> pd.Series:
    """上影线 = high - max(open, close)

    上影线长 = 弱势信号（价格被推高后回落，卖方介入）。
    """
    oc_max = pd.concat([ohlcv["open"], ohlcv["close"]], axis=1).max(axis=1)
    return ohlcv["high"] - oc_max


def lower_wick(ohlcv: pd.DataFrame) -> pd.Series:
    """下影线 = min(open, close) - low

    下影线长 = 强势信号（价格被压低后回升，买方吸收卖压）。
    """
    oc_min = pd.concat([ohlcv["open"], ohlcv["close"]], axis=1).min(axis=1)
    return oc_min - ohlcv["low"]


def wick_ratio(ohlcv: pd.DataFrame) -> pd.Series:
    """影线/实体比率 = max(upper_wick, lower_wick) / max(body, eps)

    >1 = 影线主导（反转形态），<1 = 实体主导（趋势形态）。
    """
    body = (ohlcv["close"] - ohlcv["open"]).abs()
    uw = upper_wick(ohlcv)
    lw = lower_wick(ohlcv)
    max_wick = pd.concat([uw, lw], axis=1).max(axis=1)
    return max_wick / body.clip(lower=1e-10)


# ============================================================
# P1 -- 核心信号（上下文感知）
# ============================================================

def effort_vs_result(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """投入产出比 = volume_relative / spread_relative

    Wyckoff 第三定律：投入（成交量）应与产出（振幅）匹配。

    返回值含义：
        ≈1.0  = 确认（量价和谐）
        >>1.0 = 异常（高量小振幅 = 吸收/派发，潜在反转）
        <<1.0 = 陷阱（低量大振幅 = 假动作，缺少成交量支撑）
    """
    vr = volume_relative(ohlcv, lookback)
    sr = spread_relative(ohlcv, lookback)
    return vr / sr.clip(lower=1e-10)


def stopping_volume(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """止损量 / 卖出高潮

    下跌趋势末期，卖方全力抛售但被买方吸收，是底部反转信号。

    条件（全部满足）：
    1. 处于下跌趋势（close < rolling_mean(close)）
    2. 下影线 > body * 2（锤头线形态）
    3. volume_relative > 1.5（极高量）

    返回 bool Series: True = 止损量出现
    """
    close = ohlcv["close"]
    body = (close - ohlcv["open"]).abs()
    lw = lower_wick(ohlcv)
    vr = volume_relative(ohlcv, lookback)
    ma = close.rolling(window=lookback, min_periods=1).mean()
    downtrend = close < ma

    return downtrend & (lw > body * 2) & (vr > 1.5)


def buying_climax(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """买入高潮

    上涨趋势末期，买方全力追高但被卖方满足，是顶部反转信号。

    条件（全部满足）：
    1. 处于上涨趋势（close > rolling_mean(close)）
    2. 上影线 > body * 2（射击十字星形态）
    3. volume_relative > 1.5（极高量）

    返回 bool Series: True = 买入高潮出现
    """
    close = ohlcv["close"]
    body = (close - ohlcv["open"]).abs()
    uw = upper_wick(ohlcv)
    vr = volume_relative(ohlcv, lookback)
    ma = close.rolling(window=lookback, min_periods=1).mean()
    uptrend = close > ma

    return uptrend & (uw > body * 2) & (vr > 1.5)


def no_demand(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """无需求

    价格上涨但买方不愿追高，成交量极低且振幅极小。弱势信号。

    条件（全部满足）：
    1. close > open（价格上涨）
    2. volume_relative < 0.5（极低量）
    3. spread_relative < 0.7（小振幅）

    返回 bool Series: True = 无需求
    """
    price_up = ohlcv["close"] > ohlcv["open"]
    vr = volume_relative(ohlcv, lookback)
    sr = spread_relative(ohlcv, lookback)

    return price_up & (vr < 0.5) & (sr < 0.7)


def no_supply(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """无供应

    价格下跌但卖方不愿杀跌，成交量极低且振幅极小。强势信号。

    条件（全部满足）：
    1. close < open（价格下跌）
    2. volume_relative < 0.5（极低量）
    3. spread_relative < 0.7（小振幅）

    返回 bool Series: True = 无供应
    """
    price_down = ohlcv["close"] < ohlcv["open"]
    vr = volume_relative(ohlcv, lookback)
    sr = spread_relative(ohlcv, lookback)

    return price_down & (vr < 0.5) & (sr < 0.7)


# ============================================================
# 兼容信号（已修正方向）
# ============================================================

def volume_confirmation(
    prices: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """量价确认信号（已修正方向）

    修正说明：原版将"价格上涨+低量"标记为 +1 "看涨异常"，但按书中
    方法论这是 No Demand（买方衰竭），应为看跌。方向已修正。

    返回整数编码 Series:
        +2 = 看涨确认（价格上涨 + 高量）
        +1 = 看涨异常（价格下跌 + 低量 = No Supply，卖方衰竭）
        -1 = 看跌异常（价格上涨 + 低量 = No Demand，买方衰竭）
        -2 = 看跌确认（价格下跌 + 高量）
         0 = 中性
    """
    price_change = prices.diff()
    volume_mean = volume.rolling(window=lookback, min_periods=1).mean()

    up = price_change > 0
    down = price_change < 0
    vol_high = volume > volume_mean
    vol_low = volume < volume_mean

    result = pd.Series(0, index=prices.index, dtype=int)
    result[up & vol_high] = 2     # 看涨确认
    result[down & vol_low] = 1    # No Supply (卖方衰竭) -- 修正
    result[up & vol_low] = -1     # No Demand (买方衰竭) -- 修正
    result[down & vol_high] = -2  # 看跌确认

    return result


def wick_body_ratio(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """K线影线与实体比例分析

    Returns:
        DataFrame[body_ratio, upper_wick_ratio, lower_wick_ratio, signal]
        signal: +1=下影线主导(看涨), -1=上影线主导(看跌), 0=其他
    """
    total_range = high - low
    mask = total_range > 0

    body_ratio = pd.Series(0.0, index=open.index)
    upper_wick_ratio = pd.Series(0.0, index=open.index)
    lower_wick_ratio = pd.Series(0.0, index=open.index)
    signal = pd.Series(0, index=open.index, dtype=int)

    if mask.any():
        oc_max = pd.concat([open[mask], close[mask]], axis=1).max(axis=1)
        oc_min = pd.concat([open[mask], close[mask]], axis=1).min(axis=1)
        tr = total_range[mask]

        body_ratio[mask] = (close[mask] - open[mask]).abs() / tr
        upper_wick_ratio[mask] = (high[mask] - oc_max) / tr
        lower_wick_ratio[mask] = (oc_min - low[mask]) / tr

    signal[lower_wick_ratio > 0.5] = 1
    signal[upper_wick_ratio > 0.5] = -1

    return pd.DataFrame({
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "signal": signal,
    })


def volume_anomaly_sequence(
    prices: pd.Series,
    volume: pd.Series,
    lookback: int = 3,
) -> pd.Series:
    """多bar量价背离检测（已修正方向）

    修正说明：原版将"价格上涨+量缩"标记为 +1 "看涨耗尽"，但按书中
    方法论这是 No Demand（买方衰竭），应为看跌。方向已修正。

    返回整数编码 Series:
        +1 = 看涨信号（价格下跌但成交量增加 = 卖方吸收，潜在反转向上）
        -1 = 看跌信号（价格上涨但成交量下降 = No Demand，买方衰竭）
         0 = 其他
    """
    result = pd.Series(0, index=prices.index, dtype=int)

    for i in range(lookback, len(prices)):
        price_start = prices.iloc[i - lookback]
        price_end = prices.iloc[i - 1]
        vol_start = volume.iloc[i - lookback]
        vol_end = volume.iloc[i - 1]

        if price_end > price_start and vol_end < vol_start:
            result.iloc[i] = -1  # No Demand -- 修正
        elif price_end < price_start and vol_end > vol_start:
            result.iloc[i] = 1   # 卖方吸收 -- 修正

    return result


def spread_strength_percentile(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """振幅强度百分位

    K线振幅（high-low）相对于过去 lookback 根K线的百分位。
    值域 0.0~1.0，高振幅（>0.7）/ 中振幅（0.3~0.7）/ 低振幅（<0.3）。

    替代旧版 body_strength_percentile（使用 body 而非 spread 的错误版本）。
    """
    s = spread(ohlcv)
    return s.rolling(window=lookback, min_periods=1).rank(pct=True)


def volume_percentile(
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """成交量百分位

    当日成交量相对于过去 lookback 根K线成交量的百分位。
    值域 0.0~1.0，高量（>0.7）/ 中量（0.3~0.7）/ 低量（<0.3）/ 极高量（>0.9）。
    """
    return volume.rolling(window=lookback, min_periods=1).rank(pct=True)


def vpa_confirmation_matrix(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """量价确认/异常矩阵（使用 spread 代替 body）

    将振幅强度 × 成交量分位组合，输出每根K线的量价关系分类。

    分类规则（Wyckoff 投入产出定律，使用 spread=high-low）：
        confirmed: 大产出+大投入 / 小产出+小投入（量价和谐）
        trap:      大产出+小投入（虚假移动，如假突破）
        anomaly:   小产出+大投入（阻力显现/走弱信号）
        neutral:   其他
    """
    sp_pct = spread_strength_percentile(ohlcv, lookback)
    vol_pct = volume_percentile(ohlcv["volume"], lookback)

    high_spread = sp_pct > 0.7
    low_spread = sp_pct < 0.3
    high_vol = vol_pct > 0.7
    low_vol = vol_pct < 0.3

    result = pd.Series("neutral", index=ohlcv.index, dtype="object")
    result[high_spread & high_vol] = "confirmed"
    result[low_spread & low_vol] = "confirmed"
    result[high_spread & low_vol] = "trap"
    result[low_spread & high_vol] = "anomaly"
    return result


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """VPA 信号验证协议"""
    print("=" * 60)
    print("VPA 信号验证协议 (vpa.py 重写版)")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    T = 500

    # ── P0: 基础信号 ──
    print("\n【P0】基础信号")
    print("-" * 60)

    # 构造合成 OHLCV
    close = pd.Series(np.cumsum(np.random.randn(T) * 0.3) + 10)
    open_s = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_s, close], axis=1).max(axis=1) + np.abs(np.random.randn(T)) * 0.2
    low = pd.concat([open_s, close], axis=1).min(axis=1) - np.abs(np.random.randn(T)) * 0.2
    volume = pd.Series(np.random.randint(500, 1500, T).astype(float))
    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    vr = volume_relative(ohlcv, lookback=20)
    sp = spread(ohlcv)
    sr = spread_relative(ohlcv, lookback=20)
    uw = upper_wick(ohlcv)
    lw = lower_wick(ohlcv)
    wr = wick_ratio(ohlcv)

    vr_ok = (vr > 0).all() and vr.notna().all()
    sp_ok = (sp >= 0).all()
    sr_ok = (sr > 0).all() and sr.notna().all()
    uw_ok = (uw >= 0).all()
    lw_ok = (lw >= 0).all()
    wr_ok = (wr >= 0).all() and wr.notna().all()

    print(f"  volume_relative: 全正且无NaN = {'PASS' if vr_ok else 'FAIL'}")
    print(f"  spread:          全非负 = {'PASS' if sp_ok else 'FAIL'}")
    print(f"  spread_relative: 全正且无NaN = {'PASS' if sr_ok else 'FAIL'}")
    print(f"  upper_wick:      全非负 = {'PASS' if uw_ok else 'FAIL'}")
    print(f"  lower_wick:      全非负 = {'PASS' if lw_ok else 'FAIL'}")
    print(f"  wick_ratio:      全非负且无NaN = {'PASS' if wr_ok else 'FAIL'}")
    if not all([vr_ok, sp_ok, sr_ok, uw_ok, lw_ok, wr_ok]):
        all_pass = False

    # ── P1: effort_vs_result ──
    print("\n【P1】effort_vs_result")
    print("-" * 60)

    evr = effort_vs_result(ohlcv, lookback=20)
    evr_ok = evr.notna().all() and (evr > 0).all()
    print(f"  全正且无NaN = {'PASS' if evr_ok else 'FAIL'}")
    if not evr_ok:
        all_pass = False

    # 正控：高量+大振幅 -> evr ≈ 1
    np.random.seed(44)
    close_trend = pd.Series(np.cumsum(np.random.randn(T) * 0.3) + 10)
    open_trend = close_trend.shift(1).fillna(close_trend.iloc[0])
    body_trend = (close_trend - open_trend).abs()
    high_trend = pd.concat([open_trend, close_trend], axis=1).max(axis=1) + body_trend * 0.3
    low_trend = pd.concat([open_trend, close_trend], axis=1).min(axis=1) - body_trend * 0.3
    vol_trend = (body_trend * 10000 + np.random.randint(100, 500, T)).clip(lower=100)
    ohlcv_trend = pd.DataFrame({
        "open": open_trend, "high": high_trend, "low": low_trend,
        "close": close_trend, "volume": vol_trend,
    })
    evr_trend = effort_vs_result(ohlcv_trend, lookback=20)
    evr_median = evr_trend.median()
    evr_confirmed_ok = 0.5 < evr_median < 2.0
    print(f"  量价正相关 evr 中位数: {evr_median:.3f} (要求 0.5~2.0)  "
          f"[{'PASS' if evr_confirmed_ok else 'FAIL'}]")
    if not evr_confirmed_ok:
        all_pass = False

    # ── P1: stopping_volume ──
    print("\n【P1】stopping_volume")
    print("-" * 60)

    # 正控：构造下跌趋势 + 最后一根锤头线 + 高量
    n = 200
    close_down = pd.Series(np.linspace(20, 10, n))
    open_down = close_down.shift(1).fillna(close_down.iloc[0])
    # 最后一根锤头线
    open_down.iloc[-1] = 10.2
    close_down.iloc[-1] = 10.5
    high_down = pd.concat([open_down, close_down], axis=1).max(axis=1) + 0.05
    low_down = pd.concat([open_down, close_down], axis=1).min(axis=1) - 0.05
    low_down.iloc[-1] = 9.0  # 长下影线
    vol_down = pd.Series(np.full(n, 1000.0))
    vol_down.iloc[-1] = 5000.0  # 高量
    ohlcv_sv = pd.DataFrame({
        "open": open_down, "high": high_down, "low": low_down,
        "close": close_down, "volume": vol_down,
    })

    sv = stopping_volume(ohlcv_sv, lookback=20)
    sv_detected = sv.iloc[-1]
    print(f"  止损量检测 (最后一根) = {'PASS' if sv_detected else 'FAIL'}")
    if not sv_detected:
        all_pass = False

    # 负控：上涨趋势不应出现止损量
    close_up = pd.Series(np.linspace(10, 20, n))
    open_up = close_up.shift(1).fillna(close_up.iloc[0])
    high_up = pd.concat([open_up, close_up], axis=1).max(axis=1) + 0.05
    low_up = pd.concat([open_up, close_up], axis=1).min(axis=1) - 0.05
    vol_up = pd.Series(np.full(n, 1000.0))
    ohlcv_up = pd.DataFrame({
        "open": open_up, "high": high_up, "low": low_up,
        "close": close_up, "volume": vol_up,
    })
    sv_up = stopping_volume(ohlcv_up, lookback=20)
    sv_none = not sv_up.any()
    print(f"  上涨趋势无止损量 = {'PASS' if sv_none else 'FAIL'}")
    if not sv_none:
        all_pass = False

    # ── P1: buying_climax ──
    print("\n【P1】buying_climax")
    print("-" * 60)

    # 正控：构造上涨趋势 + 最后一根射击十字星 + 高量
    close_bc = pd.Series(np.linspace(10, 20, n))
    open_bc = close_bc.shift(1).fillna(close_bc.iloc[0])
    open_bc.iloc[-1] = 19.8
    close_bc.iloc[-1] = 19.5
    high_bc = pd.concat([open_bc, close_bc], axis=1).max(axis=1) + 0.05
    high_bc.iloc[-1] = 21.0  # 长上影线
    low_bc = pd.concat([open_bc, close_bc], axis=1).min(axis=1) - 0.05
    vol_bc = pd.Series(np.full(n, 1000.0))
    vol_bc.iloc[-1] = 5000.0
    ohlcv_bc = pd.DataFrame({
        "open": open_bc, "high": high_bc, "low": low_bc,
        "close": close_bc, "volume": vol_bc,
    })

    bc = buying_climax(ohlcv_bc, lookback=20)
    bc_detected = bc.iloc[-1]
    print(f"  买入高潮检测 (最后一根) = {'PASS' if bc_detected else 'FAIL'}")
    if not bc_detected:
        all_pass = False

    # ── P1: no_demand / no_supply ──
    print("\n【P1】no_demand / no_supply")
    print("-" * 60)

    nd = no_demand(ohlcv, lookback=20)
    ns = no_supply(ohlcv, lookback=20)
    nd_ok = isinstance(nd, pd.Series) and nd.dtype == bool
    ns_ok = isinstance(ns, pd.Series) and ns.dtype == bool
    print(f"  no_demand 返回 bool Series = {'PASS' if nd_ok else 'FAIL'}")
    print(f"  no_supply  返回 bool Series = {'PASS' if ns_ok else 'FAIL'}")
    if not nd_ok or not ns_ok:
        all_pass = False

    # no_demand 和 no_supply 不应同时为 True
    both_true = (nd & ns).any()
    ndns_ok = not both_true
    print(f"  no_demand & no_supply 互斥 = {'PASS' if ndns_ok else 'FAIL'}")
    if not ndns_ok:
        all_pass = False

    # ── 兼容: volume_confirmation (修正方向) ──
    print("\n【兼容】volume_confirmation (修正方向)")
    print("-" * 60)

    vc = volume_confirmation(close, volume, lookback=20)
    non_zero_ratio = (vc != 0).sum() / len(vc)
    vc_ok = non_zero_ratio >= 0.60
    print(f"  非零信号占比: {non_zero_ratio:.2%} (要求 >= 60%)  "
          f"[{'PASS' if vc_ok else 'FAIL'}]")
    if not vc_ok:
        all_pass = False

    # 验证方向修正：上涨+低量 = -1 (No Demand)
    prices_up = pd.Series([10.0, 11.0, 12.0, 13.0])
    vol_decline = pd.Series([5000.0, 1000.0, 500.0, 200.0])
    vc_fix = volume_confirmation(prices_up, vol_decline, lookback=3)
    # 最后两根：价格上涨 + 低量 -> 应为 -1 (No Demand)
    vc_direction_ok = vc_fix.iloc[-1] == -1
    print(f"  上涨+低量 = -1 (No Demand) = {'PASS' if vc_direction_ok else 'FAIL'}")
    if not vc_direction_ok:
        all_pass = False

    # ── 兼容: volume_anomaly_sequence (修正方向) ──
    print("\n【兼容】volume_anomaly_sequence (修正方向)")
    print("-" * 60)

    prices_rise = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    vol_fall = pd.Series([5000.0, 4000.0, 3000.0, 2000.0, 1000.0])
    vas = volume_anomaly_sequence(prices_rise, vol_fall, lookback=3)
    vas_ok = vas.iloc[-1] == -1  # No Demand -> -1
    print(f"  上涨+量缩 = -1 (No Demand) = {'PASS' if vas_ok else 'FAIL'}")
    if not vas_ok:
        all_pass = False

    # ── 兼容: vpa_confirmation_matrix (使用 spread) ──
    print("\n【兼容】vpa_confirmation_matrix (使用 spread)")
    print("-" * 60)

    matrix = vpa_confirmation_matrix(ohlcv_trend, lookback=20)
    vc_matrix = matrix.value_counts(normalize=True)
    confirmed_ratio = vc_matrix.get("confirmed", 0.0)
    confirmed_ok = confirmed_ratio >= 0.25
    print(f"  confirmed 占比: {confirmed_ratio:.2%} (要求 >= 25%)  "
          f"[{'PASS' if confirmed_ok else 'FAIL'}]")
    if not confirmed_ok:
        all_pass = False

    valid_cats = {"confirmed", "trap", "anomaly", "neutral"}
    cats_ok = set(matrix.unique()).issubset(valid_cats)
    print(f"  类别合法 = {'PASS' if cats_ok else 'FAIL'}")
    if not cats_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] VPA 信号验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys
    ok = run_validation()
    sys.exit(0 if ok else 1)
