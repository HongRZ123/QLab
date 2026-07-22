"""
vpa_trend.py -- VPA 量价确认趋势跟踪策略 (重写版)

基于《量价分析》Ch4 确认/异常方法论 + Ch8 趋势健康度。
对应书中策略七：趋势跟踪 (Trend Following with Volume Confirmation)。

重写改进：
    1. 使用 effort_vs_result (spread-based) 代替 vpa_confirmation_matrix (body-based)
    2. 使用上下文感知的 trend_health
    3. 使用 trend_direction 确保只在趋势中持仓

策略逻辑（仅做多）：
    - effort_vs_result ≈ 1（量价确认）+ 趋势健康 + 上涨趋势 -> 满仓
    - effort_vs_result ≈ 1 + 趋势衰竭 + 上涨趋势 -> 半仓
    - effort_vs_result 异常（>>1 或 <<1）或趋势下跌 -> 空仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.trend import trend_direction, trend_health
from signals.vpa import effort_vs_result


def vpa_trend(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    confirm_low: float = 0.7,
    confirm_high: float = 1.5,
) -> dict:
    """VPA 量价确认趋势跟踪策略

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 信号滚动窗口
        confirm_low: effort_vs_result 确认下界（在此与 confirm_high 之间 = 确认）
        confirm_high: effort_vs_result 确认上界

    返回:
        dict: {
            "num_units": pd.Series,       # 仓位比例 (0~1)
            "effort_vs_result": pd.Series, # 投入产出比
            "trend_direction": pd.Series,  # 趋势方向
            "trend_health": pd.Series,     # 趋势健康度
        }
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    evr = effort_vs_result(ohlcv, lookback=lookback)
    td = trend_direction(close, lookback=lookback)
    th = trend_health(close, volume, lookback=lookback)

    # 量价确认：evr 在 [confirm_low, confirm_high] 之间
    confirmed = (evr >= confirm_low) & (evr <= confirm_high)

    # 仅在上涨趋势中做多
    uptrend = td == 1

    num_units = pd.Series(0.0, index=close.index)

    # 满仓：确认 + 健康 + 上涨趋势
    full_long = confirmed & (th == 1) & uptrend
    # 半仓：确认 + 衰竭 + 上涨趋势（减仓但不退出）
    half_long = confirmed & (th == -1) & uptrend

    num_units[full_long] = 1.0
    num_units[half_long] = 0.5

    return {
        "num_units": num_units,
        "effort_vs_result": evr,
        "trend_direction": td,
        "trend_health": th,
    }


def run_validation() -> bool:
    """VPA 趋势策略验证协议"""
    print("=" * 60)
    print("VPA 趋势策略验证协议 (vpa_trend.py 重写版)")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 500

    # ── 正控：健康上涨趋势 -> 应有较多持仓 ──
    print("\n【正控】健康上涨趋势 + 量价确认")
    print("-" * 60)

    base_body = np.linspace(0.05, 0.3, n)
    body = np.maximum(base_body + np.random.randn(n) * 0.05, 0.01)
    close = pd.Series(np.cumsum(body) + 10)
    open_s = close.shift(1).fillna(close.iloc[0])
    # 构造 high/low 使 spread 与 body 正相关
    high = pd.concat([open_s, close], axis=1).max(axis=1) + body * 0.3
    low = pd.concat([open_s, close], axis=1).min(axis=1) - body * 0.3
    volume = (body * 50000 + np.random.randint(100, 500, n)).clip(min=100)

    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    result = vpa_trend(ohlcv, lookback=20)
    pos_ratio = (result["num_units"] > 0).sum() / len(result["num_units"])
    pos_ok = pos_ratio >= 0.20
    print(f"  持仓天数占比: {pos_ratio:.2%} (要求 >= 20%)  "
          f"[{'PASS' if pos_ok else 'FAIL'}]")
    if not pos_ok:
        all_pass = False

    # ── 负控：随机游走 -> 仓位应较低 ──
    print("\n【负控】随机游走")
    print("-" * 60)

    np.random.seed(43)
    close_r = pd.Series(np.cumsum(np.random.randn(n)) + 10)
    open_r = close_r.shift(1).fillna(close_r.iloc[0])
    high_r = pd.concat([open_r, close_r], axis=1).max(axis=1) + 0.1
    low_r = pd.concat([open_r, close_r], axis=1).min(axis=1) - 0.1
    volume_r = pd.Series(np.random.randint(500, 1500, n).astype(float))

    ohlcv_r = pd.DataFrame({
        "open": open_r, "high": high_r, "low": low_r,
        "close": close_r, "volume": volume_r,
    })

    result_r = vpa_trend(ohlcv_r, lookback=20)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.50
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 50%)  "
          f"[{'PASS' if neg_ok else 'FAIL'}]")
    if not neg_ok:
        all_pass = False

    # ── 不变式：num_units >= 0 且 <= 1 ──
    print("\n【不变式】num_units ∈ [0, 1]")
    print("-" * 60)
    nonneg_ok = (result["num_units"] >= 0).all()
    nonpos_ok = (result["num_units"] <= 1).all()
    print(f"  num_units >= 0: {'PASS' if nonneg_ok else 'FAIL'}")
    print(f"  num_units <= 1: {'PASS' if nonpos_ok else 'FAIL'}")
    if not nonneg_ok or not nonpos_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] VPA 趋势策略验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
