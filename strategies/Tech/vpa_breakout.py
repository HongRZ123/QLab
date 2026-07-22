"""
vpa_breakout.py -- VPA 放量突破策略 (重写版)

基于《量价分析》Ch7 支撑位和阻力位。
对应书中策略五：真突破 (True Breakout)。

重写改进：
    1. 使用 spread + volume_relative 代替 detect_breakout（消除 O(n²) 循环）
    2. 向量化实现，大幅提升性能
    3. 使用 spread_relative 判断大振幅突破
    4. forward-fill positions

策略逻辑（仅做多）：
    入场：收盘价突破近期高点 + 大振幅 + 高量（真突破）
    持仓：forward-fill，直到价格回到区间内
    退出：价格跌破近期低点（回到区间内）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.vpa import spread_relative, volume_relative


def vpa_breakout(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    breakout_lookback: int = 20,
    vol_threshold: float = 1.5,
    spread_threshold: float = 1.5,
) -> dict:
    """VPA 放量突破策略

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 成交量/振幅相对值窗口
        breakout_lookback: 区间上界/下界计算窗口
        vol_threshold: 成交量相对值阈值（>此值为高量）
        spread_threshold: 振幅相对值阈值（>此值为大振幅）

    返回:
        dict: {
            "num_units": pd.Series,        # 仓位比例 (0~1, forward-filled)
            "breakout_up": pd.Series,      # 向上突破信号
            "volume_relative": pd.Series,  # 成交量相对值
            "spread_relative": pd.Series,  # 振幅相对值
        }
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # 区间边界（不包含当前bar，避免前视偏差）
    upper = high.shift(1).rolling(window=breakout_lookback, min_periods=1).max()
    lower = low.shift(1).rolling(window=breakout_lookback, min_periods=1).min()

    vr = volume_relative(ohlcv, lookback=lookback)
    sr = spread_relative(ohlcv, lookback=lookback)

    # 真突破：收盘价突破上界 + 高量 + 大振幅
    breakout_up = (close > upper) & (vr > vol_threshold) & (sr > spread_threshold)

    # 退出：价格回到区间内（跌破下界）
    exit_signal = close < lower

    # Forward-fill positions
    num_units = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if exit_signal.iloc[i]:
            num_units.iloc[i] = 0.0
        elif breakout_up.iloc[i]:
            num_units.iloc[i] = 1.0
        elif i > 0:
            num_units.iloc[i] = num_units.iloc[i - 1]

    return {
        "num_units": num_units,
        "breakout_up": breakout_up,
        "volume_relative": vr,
        "spread_relative": sr,
    }


def run_validation() -> bool:
    """VPA 突破策略验证协议"""
    print("=" * 60)
    print("VPA 突破策略验证协议 (vpa_breakout.py 重写版)")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    # ── 正控：放量突破区间上界 -> 应产生买入信号 ──
    print("\n【正控】放量突破区间上界")
    print("-" * 60)

    # 前100天在 9~11 区间盘整（小振幅），第101天突然放量突破到15
    np.random.seed(42)
    close_vals = np.concatenate([
        np.random.uniform(9, 11, 100),    # 盘整
        [15.0],                            # 突破日
        np.full(99, 15.5),                 # 突破后维持
    ])

    open_vals = np.empty(n)
    open_vals[0] = close_vals[0]
    open_vals[1:] = close_vals[:-1]

    # 振幅：盘整期小（0.1），突破日大（2.0），突破后正常（0.2）
    spread_vals = np.full(n, 0.1)
    spread_vals[100] = 2.0    # 突破日大振幅
    spread_vals[101:] = 0.2   # 突破后正常振幅
    high_vals = np.maximum(open_vals, close_vals) + spread_vals / 2
    low_vals = np.minimum(open_vals, close_vals) - spread_vals / 2

    close = pd.Series(close_vals, index=idx)
    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[100] = 10000.0  # 突破日高量

    ohlcv = pd.DataFrame({
        "open": pd.Series(open_vals, index=idx),
        "high": pd.Series(high_vals, index=idx),
        "low": pd.Series(low_vals, index=idx),
        "close": close, "volume": volume,
    })

    result = vpa_breakout(ohlcv, lookback=20, breakout_lookback=20,
                          vol_threshold=1.5, spread_threshold=1.5)
    has_buy = (result["num_units"] > 0).any()
    print(f"  存在买入信号: {'PASS' if has_buy else 'FAIL'}")
    if not has_buy:
        all_pass = False

    # ── 负控：缩量突破 -> 应无买入信号 ──
    print("\n【负控】缩量突破")
    print("-" * 60)

    volume_r = pd.Series(np.full(n, 1000.0), index=idx)

    ohlcv_r = pd.DataFrame({
        "open": pd.Series(open_vals, index=idx),
        "high": pd.Series(high_vals, index=idx),
        "low": pd.Series(low_vals, index=idx),
        "close": close, "volume": volume_r,
    })

    result_r = vpa_breakout(ohlcv_r, lookback=20, breakout_lookback=20,
                            vol_threshold=1.5, spread_threshold=1.5)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.20
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 20%)  "
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
        print("[PASS] VPA 突破策略验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
