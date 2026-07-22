"""
vpa_breakout.py -- VPA-T3 突破策略

基于《量价分析》Ch7 支撑位和阻力位。

策略逻辑（仅做多）：
    - 收盘价突破近期震荡区间上界 + 成交量放大 -> 满仓
    - 缩量突破 -> 不入场（伪突破）
    - 价格回到区间内 -> 平仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.pivot import detect_breakout


def vpa_breakout(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    breakout_lookback: int = 20,
    vol_threshold: float = 0.6,
) -> dict:
    """
    VPA 突破策略。

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 成交量百分位窗口
        breakout_lookback: 区间上界/下界计算窗口（取近期 high/low）
        vol_threshold: 成交量分位阈值

    返回:
        dict: {
            "num_units": pd.Series,
            "breakout_signal": pd.Series,
        }
    """
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    # 用近期高低点作为震荡区间边界（不包含当前 bar）
    upper = high.shift(1).rolling(window=breakout_lookback, min_periods=1).max()
    lower = low.shift(1).rolling(window=breakout_lookback, min_periods=1).min()

    # 检测突破
    signals = pd.Series(np.nan, index=close.index, dtype="object")
    for i in range(len(close)):
        sig = detect_breakout(
            close.iloc[: i + 1],
            volume.iloc[: i + 1],
            range_bound=(upper.iloc[i], lower.iloc[i]),
            lookback=lookback,
            vol_threshold=vol_threshold,
        )
        signals.iloc[i] = sig.iloc[i]

    num_units = pd.Series(0.0, index=close.index)
    num_units[signals == "breakout_confirmed"] = 1.0

    # 价格回到区间内 -> 平仓
    in_range = (close >= lower) & (close <= upper)
    num_units[in_range] = 0.0

    return {
        "num_units": num_units,
        "breakout_signal": signals,
    }


def run_validation() -> bool:
    """VPA 突破策略验证协议"""
    print("=" * 60)
    print("VPA 突破策略验证协议（vpa_breakout.py）")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    # ── 正控：放量突破区间上界 -> 应产生买入信号 ──
    print("\n【正控】放量突破区间上界")
    print("-" * 60)

    # 前 100 天区间震荡，后 50 天放量突破
    phase1 = pd.Series(np.random.uniform(9, 11, 100), index=idx[:100])
    phase2 = pd.Series(np.linspace(11, 16, 50), index=idx[100:150])
    phase3 = pd.Series(np.full(50, 16.0), index=idx[150:])
    close = pd.concat([phase1, phase2, phase3])

    open_s = close.shift(1).fillna(close.iloc[0])
    high = close + 0.1
    low = close - 0.1

    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[100:150] = np.linspace(1000, 10000, 50).tolist()

    ohlcv = pd.DataFrame({
        "open": open_s,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    result = vpa_breakout(ohlcv, lookback=20, breakout_lookback=20, vol_threshold=0.6)
    has_buy = (result["num_units"] > 0).any()
    print(f"  存在买入信号: {'PASS' if has_buy else 'FAIL'}")
    if not has_buy:
        all_pass = False

    # ── 负控：缩量突破 -> 应无买入信号 ──
    print("\n【负控】缩量突破")
    print("-" * 60)

    close_r = pd.concat([phase1, phase2, phase3])
    volume_r = pd.Series(np.full(n, 1000.0), index=idx)

    ohlcv_r = pd.DataFrame({
        "open": close_r.shift(1).fillna(close_r.iloc[0]),
        "high": close_r + 0.1,
        "low": close_r - 0.1,
        "close": close_r,
        "volume": volume_r,
    })

    result_r = vpa_breakout(ohlcv_r, lookback=20, breakout_lookback=20, vol_threshold=0.6)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.20
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 20%)  "
          f"[{'PASS' if neg_ok else 'FAIL'}]")
    if not neg_ok:
        all_pass = False

    # ── 不变式：num_units >= 0 ──
    print("\n【不变式】num_units >= 0")
    print("-" * 60)
    nonneg_ok = (result["num_units"] >= 0).all()
    print(f"  num_units 全部 >= 0: {'PASS' if nonneg_ok else 'FAIL'}")
    if not nonneg_ok:
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
