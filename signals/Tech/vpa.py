"""
VPA (Volume Price Analysis) signals

量价分析信号函数集合，从量价关系角度提供交易信号。

Functions:
    volume_confirmation: 量价确认信号（+2/+1/-1/-2 编码）
    wick_body_ratio: K线影线与实体比例分析
    volume_anomaly_sequence: 多bar异常序列检测
    run_validation: VPA信号验证协议
"""
from __future__ import annotations

import pandas as pd


def volume_confirmation(
    prices: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """量价确认信号

    返回整数编码 Series:
        +2 = 看涨确认（价格上涨 + 成交量高于滚动均值）
        +1 = 看涨异常（价格上涨 + 成交量低于滚动均值）
        -1 = 看跌异常（价格下跌 + 成交量低于滚动均值）
        -2 = 看跌确认（价格下跌 + 成交量高于滚动均值）
         0 = 中性（价格不变或成交量恰好等于均值）

    Args:
        prices: 价格序列
        volume: 成交量序列
        lookback: 滚动均值窗口

    Returns:
        整数编码的确认信号 Series
    """
    price_change = prices.diff()
    volume_mean = volume.rolling(window=lookback, min_periods=1).mean()

    result = pd.Series(0, index=prices.index, dtype=int)

    for i in range(1, len(prices)):
        pc = price_change.iloc[i]
        vol = volume.iloc[i]
        vol_mean = volume_mean.iloc[i]

        if pc > 0 and vol > vol_mean:
            result.iloc[i] = 2   # 看涨确认
        elif pc > 0 and vol < vol_mean:
            result.iloc[i] = 1   # 看涨异常
        elif pc < 0 and vol < vol_mean:
            result.iloc[i] = -1  # 看跌异常
        elif pc < 0 and vol > vol_mean:
            result.iloc[i] = -2  # 看跌确认
        else:
            result.iloc[i] = 0   # 中性

    return result


def wick_body_ratio(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """K线影线与实体比例分析

    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列

    Returns:
        DataFrame，包含列:
            - body_ratio: 实体比例 abs(close-open) / (high-low)
            - upper_wick_ratio: 上影线比例 (high-max(open,close)) / (high-low)
            - lower_wick_ratio: 下影线比例 (min(open,close)-low) / (high-low)
            - signal: +1 看涨反转（下影线主导），-1 看跌反转（上影线主导），0 其他
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

    # 生成信号：下影线主导 → 看涨反转，上影线主导 → 看跌反转
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
    """多bar异常序列检测

    检测过去 lookback 个 bar 内的量价背离模式。

    Args:
        prices: 价格序列
        volume: 成交量序列
        lookback: 回看窗口

    Returns:
        +1 = 看涨耗尽（过去 lookback 个 bar 价格上涨但成交量下降）
        -1 = 看跌吸收（过去 lookback 个 bar 价格下跌但成交量增加）
         0 = 其他
    """
    result = pd.Series(0, index=prices.index, dtype=int)

    for i in range(lookback, len(prices)):
        price_start = prices.iloc[i - lookback]
        price_end = prices.iloc[i - 1]
        vol_start = volume.iloc[i - lookback]
        vol_end = volume.iloc[i - 1]

        if price_end > price_start and vol_end < vol_start:
            result.iloc[i] = 1   # 看涨耗尽
        elif price_end < price_start and vol_end > vol_start:
            result.iloc[i] = -1  # 看跌吸收

    return result


def run_validation() -> bool:
    """VPA 信号验证协议"""
    import numpy as np

    print("=" * 60)
    print("VPA 信号验证协议")
    print("=" * 60)

    all_pass = True

    # ── 正控：线性趋势 + 递增成交量 ──
    print("\n【正控】线性趋势 + 递增成交量")
    print("-" * 60)

    np.random.seed(42)
    T = 500
    prices_trend = pd.Series(np.linspace(10, 15, T))
    volume_trend = pd.Series(np.linspace(1000, 2000, T))

    vc = volume_confirmation(prices_trend, volume_trend, lookback=20)
    non_zero_ratio = (vc != 0).sum() / len(vc)

    pos_ok = non_zero_ratio >= 0.60
    print(f"  非零信号占比: {non_zero_ratio:.2%} (要求 >= 60%)  "
          f"[{'PASS' if pos_ok else 'FAIL'}]")
    if not pos_ok:
        all_pass = False

    # ── 负控：随机游走 + 独立随机成交量 ──
    print("\n【负控】随机游走 + 独立随机成交量")
    print("-" * 60)

    np.random.seed(42)
    prices_random = pd.Series(np.cumsum(np.random.randn(500)) + 10)

    np.random.seed(43)
    volume_random = pd.Series(np.random.randint(500, 1500, 500))

    vc_random = volume_confirmation(prices_random, volume_random, lookback=20)
    counts = vc_random.value_counts(normalize=True)

    cat_ok = True
    for cat in [2, 1, -1, -2]:
        ratio = counts.get(cat, 0.0)
        in_range = 0.15 <= ratio <= 0.35
        print(f"  类别 {cat:+d}: {ratio:.2%} (要求 15%-35%)  "
              f"[{'PASS' if in_range else 'FAIL'}]")
        if not in_range:
            cat_ok = False

    zero_ratio = counts.get(0, 0.0)
    zero_ok = zero_ratio < 0.05
    print(f"  类别 0: {zero_ratio:.2%} (要求 < 5%)  "
          f"[{'PASS' if zero_ok else 'FAIL'}]")
    if not zero_ok:
        cat_ok = False

    if not cat_ok or not zero_ok:
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
