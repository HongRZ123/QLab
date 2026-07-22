"""
VPA (Volume Price Analysis) signals

量价分析信号函数集合，从量价关系角度提供交易信号。

Functions:
    volume_confirmation: 量价确认信号（+2/+1/-1/-2 编码）
    wick_body_ratio: K线影线与实体比例分析
    volume_anomaly_sequence: 多bar异常序列检测
    body_strength_percentile: 实体强度百分位（VPA-S1）
    volume_percentile: 成交量百分位（VPA-S2）
    vpa_confirmation_matrix: 量价确认/异常矩阵（VPA-S3）
    run_validation: VPA信号验证协议
"""
from __future__ import annotations

import numpy as np
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


def body_strength_percentile(
    open: pd.Series,
    close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """实体强度百分位（VPA-S1, Ch3/Ch4）

    K 线实体高度（|close-open|）相对于过去 lookback 根 K 线的百分位。
    值域 0.0~1.0，高实体（>0.7）/ 中实体（0.3~0.7）/ 低实体（<0.3）。

    Args:
        open: 开盘价序列
        close: 收盘价序列
        lookback: 滚动窗口长度

    Returns:
        百分位 Series（0.0~1.0），前 lookback-1 个为 NaN
    """
    body = (close - open).abs()
    return body.rolling(window=lookback, min_periods=1).rank(pct=True)


def volume_percentile(
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """成交量百分位（VPA-S2, Ch2/Ch4）

    当日成交量相对于过去 lookback 根 K 线成交量的百分位。
    值域 0.0~1.0，高量（>0.7）/ 中量（0.3~0.7）/ 低量（<0.3）/ 极高量（>0.9）。

    Args:
        volume: 成交量序列
        lookback: 滚动窗口长度

    Returns:
        百分位 Series（0.0~1.0），前 lookback-1 个为 NaN
    """
    return volume.rolling(window=lookback, min_periods=1).rank(pct=True)


def vpa_confirmation_matrix(
    open: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """量价确认/异常矩阵（VPA-S3, Ch4 核心方法论）

    将实体强度 × 成交量分位组合，输出每根 K 线的量价关系分类。

    分类规则（威科夫投入产出定律）：
        confirmed: 大产出+大投入 / 小产出+小投入（量价和谐）
        trap:      大产出+小投入（虚假移动，如假突破）
        anomaly:   小产出+大投入（阻力显现/走弱信号）
        neutral:   其他

    Args:
        open: 开盘价序列
        close: 收盘价序列
        volume: 成交量序列
        lookback: 百分位滚动窗口

    Returns:
        字符串 Series: "confirmed" / "trap" / "anomaly" / "neutral"
    """
    body_pct = body_strength_percentile(open, close, lookback)
    vol_pct = volume_percentile(volume, lookback)

    high_body = body_pct > 0.7
    low_body = body_pct < 0.3
    high_vol = vol_pct > 0.7
    low_vol = vol_pct < 0.3

    result = pd.Series("neutral", index=close.index, dtype="object")
    result[high_body & high_vol] = "confirmed"
    result[low_body & low_vol] = "confirmed"
    result[high_body & low_vol] = "trap"
    result[low_body & high_vol] = "anomaly"
    return result


def run_validation() -> bool:
    """VPA 信号验证协议"""

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

    # ── VPA-S1/S2: 实体强度百分位 + 成交量百分位 ──
    print("\n【VPA-S1/S2】实体强度百分位 + 成交量百分位")
    print("-" * 60)

    # 正控：随机游走 + 成交量与实体正相关 -> confirmed 占比较高
    np.random.seed(44)
    close_trend = pd.Series(np.cumsum(np.random.randn(T) * 0.3) + 10)
    open_trend = close_trend.shift(1).fillna(close_trend.iloc[0])
    body_trend = (close_trend - open_trend).abs()
    volume_trend = (body_trend * 10000 + np.random.randint(100, 500, T)).clip(lower=100)
    body_pct = body_strength_percentile(open_trend, close_trend, lookback=20)
    vol_pct = volume_percentile(volume_trend, lookback=20)

    body_range_ok = 0.0 <= body_pct.min() <= body_pct.max() <= 1.0
    vol_range_ok = 0.0 <= vol_pct.min() <= vol_pct.max() <= 1.0
    print(f"  body_pct 范围: [{body_pct.min():.3f}, {body_pct.max():.3f}]  "
          f"[{'PASS' if body_range_ok else 'FAIL'}]")
    print(f"  vol_pct 范围:  [{vol_pct.min():.3f}, {vol_pct.max():.3f}]  "
          f"[{'PASS' if vol_range_ok else 'FAIL'}]")
    if not body_range_ok or not vol_range_ok:
        all_pass = False

    # ── VPA-S3: 量价确认/异常矩阵 ──
    print("\n【VPA-S3】量价确认/异常矩阵")
    print("-" * 60)

    # 正控：量价正相关 -> confirmed 占比较高
    matrix = vpa_confirmation_matrix(open_trend, close_trend, volume_trend, lookback=20)
    vc_matrix = matrix.value_counts(normalize=True)
    confirmed_ratio = vc_matrix.get("confirmed", 0.0)
    confirmed_ok = confirmed_ratio >= 0.30
    print(f"  confirmed 占比: {confirmed_ratio:.2%} (要求 >= 30%)  "
          f"[{'PASS' if confirmed_ok else 'FAIL'}]")
    if not confirmed_ok:
        all_pass = False

    # 负控：随机数据 -> 各类别分散，无单一类别主导
    matrix_random = vpa_confirmation_matrix(
        pd.Series(np.random.randint(9, 11, 500)),
        pd.Series(np.cumsum(np.random.randn(500)) + 10),
        pd.Series(np.random.randint(500, 1500, 500)),
        lookback=20,
    )
    vc_random_matrix = matrix_random.value_counts(normalize=True)
    neutral_ratio = vc_random_matrix.get("neutral", 0.0)
    neutral_ok = neutral_ratio >= 0.30
    print(f"  随机数据 neutral 占比: {neutral_ratio:.2%} (要求 >= 30%)  "
          f"[{'PASS' if neutral_ok else 'FAIL'}]")
    if not neutral_ok:
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
