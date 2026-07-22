"""
vpa_reversal.py -- VPA 止损量反转策略 (重写版)

基于《量价分析》Ch6 K线形态 + Ch5 市场阶段。
对应书中策略一：止损量反转 (Stopping Volume Reversal)。

重写改进：
    1. 使用 stopping_volume 信号代替手动 wick_body_ratio + volume_percentile
    2. 使用 buying_climax 作为退出信号
    3. forward-fill positions（修正 bug #3：持仓只持续1天）
    4. 添加止损逻辑：价格跌破止损量K线最低价

策略逻辑（仅做多）：
    入场：stopping_volume 信号出现（下跌趋势 + 锤头线 + 高量）
    持仓：forward-fill，直到退出信号出现
    退出：buying_climax 信号 或 价格跌破入场点最低价
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.vpa import buying_climax, stopping_volume


def vpa_reversal(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> dict:
    """VPA 止损量反转策略

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 信号滚动窗口

    返回:
        dict: {
            "num_units": pd.Series,        # 仓位比例 (0~1, forward-filled)
            "stopping_volume": pd.Series,  # 止损量信号
            "buying_climax": pd.Series,    # 买入高潮信号
        }
    """
    close = ohlcv["close"]
    low = ohlcv["low"]

    sv = stopping_volume(ohlcv, lookback=lookback)
    bc = buying_climax(ohlcv, lookback=lookback)

    num_units = pd.Series(0.0, index=close.index)

    # 止损位跟踪
    current_stop = np.nan

    for i in range(len(close)):
        # 检查止损：价格跌破止损位
        if not np.isnan(current_stop) and close.iloc[i] < current_stop:
            num_units.iloc[i] = 0.0
            current_stop = np.nan
            continue

        # 退出信号：买入高潮
        if bc.iloc[i] and num_units.iloc[i - 1] > 0 if i > 0 else False:
            num_units.iloc[i] = 0.0
            current_stop = np.nan
            continue

        # 入场信号：止损量
        if sv.iloc[i]:
            num_units.iloc[i] = 1.0
            current_stop = low.iloc[i]  # 止损位 = 止损量K线最低价
            continue

        # forward-fill：保持前一日仓位
        if i > 0:
            num_units.iloc[i] = num_units.iloc[i - 1]

    return {
        "num_units": num_units,
        "stopping_volume": sv,
        "buying_climax": bc,
    }


def run_validation() -> bool:
    """VPA 反转策略验证协议"""
    print("=" * 60)
    print("VPA 反转策略验证协议 (vpa_reversal.py 重写版)")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 200

    # ── 正控：下跌趋势底部出现止损量 -> 应产生买入信号 ──
    print("\n【正控】下跌趋势 + 止损量")
    print("-" * 60)

    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(20, 10, n), index=idx)
    open_s = close.shift(1).fillna(close.iloc[0])

    # 最后一根锤头线 + 高量
    open_s.iloc[-1] = 10.2
    close.iloc[-1] = 10.5
    high = pd.concat([open_s, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_s, close], axis=1).min(axis=1) - 0.05
    low.iloc[-1] = 9.0  # 长下影线

    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[-1] = 5000.0  # 高量

    ohlcv = pd.DataFrame({
        "open": open_s, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    result = vpa_reversal(ohlcv, lookback=20)
    has_buy = (result["num_units"] > 0).any()
    print(f"  存在买入信号: {'PASS' if has_buy else 'FAIL'}")
    if not has_buy:
        all_pass = False

    # 验证 forward-fill：入场后应持续持仓
    if has_buy:
        # 验证 forward-fill：入场后应持续持仓
        last_held = result["num_units"].iloc[-1] > 0
        print(f"  最后一根仍持仓 (forward-fill): {'PASS' if last_held else 'FAIL'}")
        if not last_held:
            all_pass = False

    # ── 负控：上涨趋势无反转形态 -> 信号稀疏 ──
    print("\n【负控】上涨趋势无反转形态")
    print("-" * 60)

    close_r = pd.Series(np.linspace(10, 20, n), index=idx)
    open_r = close_r.shift(1).fillna(close_r.iloc[0])
    high_r = pd.concat([open_r, close_r], axis=1).max(axis=1) + 0.05
    low_r = pd.concat([open_r, close_r], axis=1).min(axis=1) - 0.05

    ohlcv_r = pd.DataFrame({
        "open": open_r, "high": high_r, "low": low_r,
        "close": close_r,
        "volume": pd.Series(np.full(n, 1000.0), index=idx),
    })

    result_r = vpa_reversal(ohlcv_r, lookback=20)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.10
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 10%)  "
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
        print("[PASS] VPA 反转策略验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
