"""
vpa_trend_signals.py — VPA 量价趋势跟踪策略（信号列表版）
===========================================================

把原 ``strategies.Tech.vpa_trend.vpa_trend`` 改写成信号列表形式。
原策略根据 effort_vs_result、trend_direction、trend_health 决定
目标仓位 {0, 0.5, 1}，信号版在目标仓位变化时发出 SET / CLOSE。
"""

from __future__ import annotations

import pandas as pd

from backtest import interpret_signals, num_units_to_signals
from strategies.Tech.vpa_trend import vpa_trend


def vpa_trend_signals(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    confirm_low: float = 0.7,
    confirm_high: float = 1.5,
) -> dict:
    """
    VPA 量价趋势跟踪策略（信号列表版）。

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 信号滚动窗口
        confirm_low: effort_vs_result 确认下界
        confirm_high: effort_vs_result 确认上界

    返回:
        dict: {
            "num_units": pd.Series,         # 可直接传给 run_backtest
            "signals": list[Signal],        # SET / CLOSE
            "effort_vs_result": pd.Series,
            "trend_direction": pd.Series,
            "trend_health": pd.Series,
        }
    """
    result = vpa_trend(
        ohlcv,
        lookback=lookback,
        confirm_low=confirm_low,
        confirm_high=confirm_high,
    )
    close = ohlcv["close"]
    signals = num_units_to_signals(result["num_units"])
    num_units = interpret_signals(close, signals)

    return {
        "num_units": num_units,
        "signals": signals,
        "effort_vs_result": result["effort_vs_result"],
        "trend_direction": result["trend_direction"],
        "trend_health": result["trend_health"],
    }


def run_validation() -> bool:
    """与原策略做 parity 校验。"""
    import numpy as np

    print("=" * 60)
    print("VPA 趋势跟踪（信号列表版）— 验证协议")
    print("=" * 60)

    all_pass = True
    n = 200
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n)

    base_body = np.linspace(0.05, 0.3, n)
    body = np.maximum(base_body + np.random.randn(n) * 0.05, 0.01)
    close = pd.Series(np.cumsum(body) + 10, index=idx)
    open_s = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_s, close], axis=1).max(axis=1) + body * 0.3
    low = pd.concat([open_s, close], axis=1).min(axis=1) - body * 0.3
    volume = (body * 50000 + np.random.randint(100, 500, n)).clip(min=100)

    ohlcv = pd.DataFrame({
        "open": open_s,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    result = vpa_trend_signals(ohlcv, lookback=20)
    orig = vpa_trend(ohlcv, lookback=20)

    parity = result["num_units"].equals(orig["num_units"])
    if not parity:
        all_pass = False
    print(f"  与原策略 num_units 完全一致: {'PASS' if parity else 'FAIL'}")

    bounds_ok = (result["num_units"] >= 0).all() and (result["num_units"] <= 1).all()
    if not bounds_ok:
        all_pass = False
    print(f"  num_units ∈ [0, 1]: {'PASS' if bounds_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] VPA 趋势信号版验证")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
