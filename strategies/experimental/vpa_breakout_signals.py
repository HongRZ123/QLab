"""
vpa_breakout_signals.py — VPA 放量突破策略（信号列表版）
=========================================================

把原 ``strategies.Tech.vpa_breakout.vpa_breakout`` 改写成信号列表形式。
原策略 forward-fill 持仓，信号版在出现/退出突破时发出 SET 1.0 / CLOSE。
"""

from __future__ import annotations

import pandas as pd

from backtest import interpret_signals, num_units_to_signals
from strategies.Tech.vpa_breakout import vpa_breakout


def vpa_breakout_signals(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    breakout_lookback: int = 20,
    vol_threshold: float = 1.5,
    spread_threshold: float = 1.5,
) -> dict:
    """
    VPA 放量突破策略（信号列表版）。

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 成交量/振幅相对值窗口
        breakout_lookback: 区间上界/下界计算窗口
        vol_threshold: 成交量相对值阈值
        spread_threshold: 振幅相对值阈值

    返回:
        dict: {
            "num_units": pd.Series,        # 可直接传给 run_backtest
            "signals": list[Signal],       # SET / CLOSE
            "breakout_up": pd.Series,
            "volume_relative": pd.Series,
            "spread_relative": pd.Series,
        }
    """
    result = vpa_breakout(
        ohlcv,
        lookback=lookback,
        breakout_lookback=breakout_lookback,
        vol_threshold=vol_threshold,
        spread_threshold=spread_threshold,
    )
    close = ohlcv["close"]
    signals = num_units_to_signals(result["num_units"])
    num_units = interpret_signals(close, signals)

    return {
        "num_units": num_units,
        "signals": signals,
        "breakout_up": result["breakout_up"],
        "volume_relative": result["volume_relative"],
        "spread_relative": result["spread_relative"],
    }


def run_validation() -> bool:
    """与原策略做 parity 校验。"""
    import numpy as np

    print("=" * 60)
    print("VPA 放量突破（信号列表版）— 验证协议")
    print("=" * 60)

    all_pass = True
    n = 200
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n)

    close_vals = np.concatenate([
        np.random.uniform(9, 11, 100),
        [15.0],
        np.full(99, 15.5),
    ])
    open_vals = np.empty(n)
    open_vals[0] = close_vals[0]
    open_vals[1:] = close_vals[:-1]

    spread_vals = np.full(n, 0.1)
    spread_vals[100] = 2.0
    spread_vals[101:] = 0.2
    high_vals = np.maximum(open_vals, close_vals) + spread_vals / 2
    low_vals = np.minimum(open_vals, close_vals) - spread_vals / 2

    close = pd.Series(close_vals, index=idx)
    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[100] = 10000.0

    ohlcv = pd.DataFrame({
        "open": pd.Series(open_vals, index=idx),
        "high": pd.Series(high_vals, index=idx),
        "low": pd.Series(low_vals, index=idx),
        "close": close,
        "volume": volume,
    })

    result = vpa_breakout_signals(
        ohlcv,
        lookback=20,
        breakout_lookback=20,
        vol_threshold=1.5,
        spread_threshold=1.5,
    )
    orig = vpa_breakout(
        ohlcv,
        lookback=20,
        breakout_lookback=20,
        vol_threshold=1.5,
        spread_threshold=1.5,
    )

    parity = result["num_units"].equals(orig["num_units"])
    if not parity:
        all_pass = False
    print(f"  与原策略 num_units 完全一致: {'PASS' if parity else 'FAIL'}")

    has_buy = any(s.action == "SET" and s.qty > 0 for s in result["signals"])
    if not has_buy:
        all_pass = False
    print(f"  存在 SET 买入信号: {'PASS' if has_buy else 'FAIL'}")

    bounds_ok = (result["num_units"] >= 0).all() and (result["num_units"] <= 1).all()
    if not bounds_ok:
        all_pass = False
    print(f"  num_units ∈ [0, 1]: {'PASS' if bounds_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] VPA 突破信号版验证")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
