"""
pivot.py -- 价格结构信号（支点、震荡区间、突破检测）

基于《量价分析》Ch7 支撑位和阻力位。

Functions:
    detect_isolated_pivots: 孤立支点检测（VPA-S4）
    detect_consolidation: 震荡区间识别（VPA-S5）
    detect_breakout: 突破与伪突破检测（VPA-S6）
    run_validation: 结构信号验证协议
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_isolated_pivots(
    high: pd.Series,
    low: pd.Series,
) -> pd.DataFrame:
    """孤立支点检测（VPA-S4, Ch7）

    高位孤立支点：中间K线的 H 和 L 都高于左右两根。
    低位孤立支点：中间K线的 H 和 L 都低于左右两根。

    Args:
        high: 最高价序列
        low: 最低价序列

    Returns:
        DataFrame[pivot_high, pivot_low]，值为支点价格，NaN = 非支点
    """
    pivot_high = pd.Series(np.nan, index=high.index, dtype=float)
    pivot_low = pd.Series(np.nan, index=low.index, dtype=float)

    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)

    for i in range(1, len(high) - 1):
        # 高位支点：H[i] > H[i-1] AND H[i] > H[i+1] AND L[i] > L[i-1] AND L[i] > L[i+1]
        if h[i] > h[i - 1] and h[i] > h[i + 1] and lo[i] > lo[i - 1] and lo[i] > lo[i + 1]:
            pivot_high.iloc[i] = h[i]
        # 低位支点：L[i] < L[i-1] AND L[i] < L[i+1] AND H[i] < H[i-1] AND H[i] < H[i+1]
        if lo[i] < lo[i - 1] and lo[i] < lo[i + 1] and h[i] < h[i - 1] and h[i] < h[i + 1]:
            pivot_low.iloc[i] = lo[i]

    return pd.DataFrame({"pivot_high": pivot_high, "pivot_low": pivot_low})


def detect_consolidation(
    high: pd.Series,
    low: pd.Series,
    pivots: pd.DataFrame | None = None,
    tolerance: float = 0.02,
) -> pd.Series:
    """震荡区间识别（VPA-S5, Ch7）

    从最近的支点对定义震荡区间，检测价格是否在区间内。

    Args:
        high: 最高价序列
        low: 最低价序列
        pivots: 支点 DataFrame（来自 detect_isolated_pivots）。None 则自动计算。
        tolerance: 区间边界容差（百分比），用于判断是否突破

    Returns:
        字符串 Series: "in_range" / "breakout_up" / "breakout_down" / NaN
    """
    if pivots is None:
        pivots = detect_isolated_pivots(high, low)

    result = pd.Series(np.nan, index=high.index, dtype="object")

    # 找到最近的有效支点对
    pivot_high_vals = pivots["pivot_high"].dropna()
    pivot_low_vals = pivots["pivot_low"].dropna()

    if pivot_high_vals.empty or pivot_low_vals.empty:
        return result

    upper = pivot_high_vals.iloc[-1]
    lower = pivot_low_vals.iloc[-1]
    upper_idx = pivot_high_vals.index[-1]
    lower_idx = pivot_low_vals.index[-1]

    # 区间起点 = 两个支点中较早的一个
    range_start = min(upper_idx, lower_idx)
    if range_start >= len(high) - 1:
        return result

    upper_band = upper * (1 + tolerance)
    lower_band = lower * (1 - tolerance)

    close_proxy = (high + low) / 2  # 使用高低均价作为收盘代理

    for i in range(range_start + 1, len(high)):
        if close_proxy.iloc[i] > upper_band:
            result.iloc[i] = "breakout_up"
        elif close_proxy.iloc[i] < lower_band:
            result.iloc[i] = "breakout_down"
        else:
            result.iloc[i] = "in_range"

    return result


def detect_breakout(
    close: pd.Series,
    volume: pd.Series,
    range_bound: tuple[float, float],
    lookback: int = 20,
    vol_threshold: float = 0.6,
) -> pd.Series:
    """突破与伪突破检测（VPA-S6, Ch7）

    收盘价突破区间上/下界 + 成交量分位判定。
    高量突破（>vol_threshold 分位）= 真实突破；低量突破 = 伪突破。

    Args:
        close: 收盘价序列
        volume: 成交量序列
        range_bound: (upper, lower) 区间上下界
        lookback: 成交量百分位窗口
        vol_threshold: 成交量分位阈值（0~1），低于此 = 低量

    Returns:
        字符串 Series: "breakout_confirmed" / "false_breakout" / NaN
    """
    upper, lower = range_bound
    vol_pct = volume.rolling(window=lookback, min_periods=1).rank(pct=True)

    result = pd.Series(np.nan, index=close.index, dtype="object")

    breakout_up = close > upper
    breakout_down = close < lower
    high_vol = vol_pct >= vol_threshold

    result[breakout_up & high_vol] = "breakout_confirmed"
    result[breakout_up & ~high_vol] = "false_breakout"
    result[breakout_down & high_vol] = "breakout_confirmed"
    result[breakout_down & ~high_vol] = "false_breakout"

    return result


def run_validation() -> bool:
    """结构信号验证协议"""
    print("=" * 60)
    print("结构信号验证协议（pivot.py）")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 200

    # ── VPA-S4: 孤立支点检测 ──
    print("\n【VPA-S4】孤立支点检测")
    print("-" * 60)

    # 构造明确的支点：第50根是高位支点，第100根是低位支点
    high = pd.Series(np.full(n, 10.0))
    low = pd.Series(np.full(n, 9.0))
    high.iloc[50] = 12.0
    low.iloc[50] = 11.0
    low.iloc[100] = 7.0
    high.iloc[100] = 8.0

    pivots = detect_isolated_pivots(high, low)

    s4_high_ok = not np.isnan(pivots["pivot_high"].iloc[50])
    s4_low_ok = not np.isnan(pivots["pivot_low"].iloc[100])
    print(f"  高位支点检测 (idx=50): {'PASS' if s4_high_ok else 'FAIL'}")
    print(f"  低位支点检测 (idx=100): {'PASS' if s4_low_ok else 'FAIL'}")
    if not s4_high_ok or not s4_low_ok:
        all_pass = False

    # ── VPA-S5: 震荡区间识别 ──
    print("\n【VPA-S5】震荡区间识别")
    print("-" * 60)

    # 构造区间数据：价格在 9~11 之间震荡
    np.random.seed(42)
    prices = pd.Series(np.random.uniform(9, 11, n))
    highs = prices + 0.2
    lows = prices - 0.2

    consol = detect_consolidation(highs, lows, tolerance=0.05)
    in_range_ratio = (consol == "in_range").sum() / len(consol.dropna())
    s5_ok = in_range_ratio >= 0.50
    print(f"  in_range 占比: {in_range_ratio:.2%} (要求 >= 50%)  "
          f"[{'PASS' if s5_ok else 'FAIL'}]")
    if not s5_ok:
        all_pass = False

    # ── VPA-S6: 突破与伪突破检测 ──
    print("\n【VPA-S6】突破与伪突破检测")
    print("-" * 60)

    # 构造突破数据：价格从 10 突破到 15，配高量
    close = pd.Series(np.concatenate([
        np.full(50, 10.0),
        np.linspace(10, 15, 50),
    ]))
    volume = pd.Series(np.concatenate([
        np.full(50, 500),
        np.linspace(500, 5000, 50),
    ]))

    breakout = detect_breakout(close, volume, range_bound=(11, 9), lookback=20)
    confirmed_count = (breakout == "breakout_confirmed").sum()
    s6_ok = confirmed_count > 0
    print(f"  breakout_confirmed 数量: {confirmed_count}  "
          f"[{'PASS' if s6_ok else 'FAIL'}]")
    if not s6_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 结构信号验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
