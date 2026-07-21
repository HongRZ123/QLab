"""
dividend.py — 除权除息检测与价格调整
====================================

通达信 .day 数据为不复权价格, 分红除权会导致价格跳空,
影响均值回归策略的 Z-Score 和收益率计算。

核心函数:
    detect_ex_dividend         — 检测除权除息日 (隔夜跳空法)
    adjust_close_prices        — 后复权价格调整
    filter_ex_dividend_returns — 将除权日收益率置零

原理:
    除权除息日的特征: 开盘价显著低于前一日收盘价 (隔夜跳空)
    正常隔夜跳空通常在 ±2% 以内, 除权跳空通常 > 3%
    阈值 threshold=-0.03 可检测大多数 ETF 分红 (年化分红率 1-5%)

验证协议:
    - 已知除权日 → 检测正确
    - 无除权 → 无误检
    - 调整后价格连续 (无跳空)
    - 除权日收益率置零

用法:
    python -m data.dividend
"""

import pandas as pd

# ============================================================
# 核心函数
# ============================================================

def detect_ex_dividend(
    close: pd.Series,
    open_prices: pd.Series,
    threshold: float = -0.03,
) -> pd.Series:
    """
    检测除权除息日。

    通过隔夜跳空 (open[t] / close[t-1] - 1) 判断:
    跳空幅度 < threshold 时判定为除权除息日。

    参数:
        close:       收盘价序列
        open_prices: 开盘价序列 (与 close 同 index)
        threshold:   跳空阈值 (负数), 默认 -0.03 (-3%)

    返回:
        pd.Series[bool], True = 除权除息日

    示例:
        >>> close = pd.Series([10.0, 10.1, 9.5])
        >>> open_p = pd.Series([10.0, 10.05, 9.0])
        >>> detect_ex_dividend(close, open_p)
        0    False
        1    False
        2     True
        dtype: bool
    """
    overnight_gap = open_prices / close.shift(1) - 1.0
    return overnight_gap < threshold


def adjust_close_prices(
    close: pd.Series,
    open_prices: pd.Series,
    ex_div_mask: pd.Series,
) -> pd.Series:
    """
    后复权价格调整: 消除除权除息导致的价格跳空。

    对每个除权日 t, 计算调整因子 ratio = open[t] / close[t-1],
    将 t 之前的所有价格乘以 ratio, 使价格序列连续。

    参数:
        close:       收盘价序列
        open_prices: 开盘价序列
        ex_div_mask: 除权除息日掩码 (detect_ex_dividend 的输出)

    返回:
        调整后的收盘价序列

    示例:
        >>> close = pd.Series([10.0, 10.4, 9.8])
        >>> open_p = pd.Series([10.0, 10.3, 9.5])
        >>> mask = pd.Series([False, False, True])
        >>> adjust_close_prices(close, open_p, mask)
    """
    adjusted = close.values.copy().astype(float)
    ratios = (open_prices / close.shift(1)).values

    for t in range(len(close) - 1, 0, -1):
        if ex_div_mask.iloc[t]:
            adjusted[:t] *= ratios[t]

    return pd.Series(adjusted, index=close.index, name=close.name)


def filter_ex_dividend_returns(
    returns: pd.Series,
    ex_div_mask: pd.Series,
) -> pd.Series:
    """
    将除权除息日的收益率置零。

    参数:
        returns:     日收益率序列
        ex_div_mask: 除权除息日掩码

    返回:
        过滤后的收益率序列

    示例:
        >>> ret = pd.Series([0.0, 0.01, -0.06])
        >>> mask = pd.Series([False, False, True])
        >>> filter_ex_dividend_returns(ret, mask)
        0    0.00
        1    0.01
        2    0.00
        dtype: float64
    """
    filtered = returns.copy()
    filtered[ex_div_mask] = 0.0
    return filtered


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """除权除息检测与调整验证协议。"""
    print("=" * 60)
    print("除权除息检测与调整验证协议")
    print("=" * 60)

    all_pass = True

    # ── 测试 1: 已知除权日检测 ──
    print("\n【正控】已知除权日 → 检测正确")
    print("-" * 60)

    dates = pd.date_range("2024-01-01", periods=10)
    close = pd.Series(
        [10.0, 10.1, 10.2, 10.3, 10.4, 9.8, 9.9, 10.0, 10.1, 10.2],
        index=dates,
    )
    open_p = pd.Series(
        [10.0, 10.05, 10.15, 10.25, 10.35, 9.5, 9.85, 9.95, 10.05, 10.15],
        index=dates,
    )
    # Day 5: open=9.5, prev_close=10.4 → gap = 9.5/10.4 - 1 ≈ -8.65%

    mask = detect_ex_dividend(close, open_p, threshold=-0.03)
    expected = pd.Series(
        [False, False, False, False, False, True, False, False, False, False],
        index=dates,
    )

    test1_ok = mask.equals(expected)
    if not test1_ok:
        all_pass = False
    print(f"  检测结果: {list(mask.values)}")
    print(f"  预期结果: {list(expected.values)}")
    print(f"  [{'PASS' if test1_ok else 'FAIL'}]")

    # ── 测试 2: 无除权 → 无误检 ──
    print("\n【负控】无除权 → 无误检")
    print("-" * 60)

    close2 = pd.Series([10.0, 10.1, 10.05, 10.15, 10.1], index=dates[:5])
    open2 = pd.Series([10.0, 10.05, 10.08, 10.06, 10.12], index=dates[:5])

    mask2 = detect_ex_dividend(close2, open2, threshold=-0.03)
    test2_ok = not mask2.any()
    if not test2_ok:
        all_pass = False
    print(f"  检测结果: {list(mask2.values)} (应全为 False)")
    print(f"  [{'PASS' if test2_ok else 'FAIL'}]")

    # ── 测试 3: 价格调整后连续性 ──
    print("\n【正控】调整后价格连续 (跳空缩小)")
    print("-" * 60)

    adjusted = adjust_close_prices(close, open_p, mask)
    raw_gap = close.iloc[5] / close.iloc[4] - 1
    adj_gap = adjusted.iloc[5] / adjusted.iloc[4] - 1

    test3_ok = abs(adj_gap) < abs(raw_gap)
    if not test3_ok:
        all_pass = False
    print(f"  调整前跳空: {raw_gap:.4%}")
    print(f"  调整后跳空: {adj_gap:.4%}")
    print(f"  调整因子:   {open_p.iloc[5] / close.iloc[4]:.6f}")
    print(f"  [{'PASS' if test3_ok else 'FAIL'}]")

    # ── 测试 4: 收益率过滤 ──
    print("\n【正控】除权日收益率置零")
    print("-" * 60)

    ret = close.pct_change().fillna(0.0)
    filtered = filter_ex_dividend_returns(ret, mask)

    test4_ok = filtered.iloc[5] == 0.0 and filtered.iloc[4] == ret.iloc[4]
    if not test4_ok:
        all_pass = False
    print(f"  除权日收益率: {ret.iloc[5]:.4%} → {filtered.iloc[5]:.4%}")
    print(f"  非除权日:     {ret.iloc[4]:.4%} → {filtered.iloc[4]:.4%} (不变)")
    print(f"  [{'PASS' if test4_ok else 'FAIL'}]")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 除权除息验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    success = run_validation()
    if not success:
        raise SystemExit(1)
