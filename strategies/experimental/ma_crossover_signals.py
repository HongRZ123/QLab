"""
ma_crossover_signals.py — 均线金叉死叉策略（信号列表版）
=========================================================

把原 ``strategies.Tech.ma_crossover.ma_crossover`` 改写成信号列表形式。
与原策略保持行为一致：金叉时 SET 1.0，死叉时 CLOSE。
"""

from __future__ import annotations

import pandas as pd

from backtest import interpret_signals, num_units_to_signals
from strategies.Tech.ma_crossover import ma_crossover


def ma_crossover_signals(
    prices: pd.Series,
    short_window: int = 5,
    long_window: int = 20,
) -> dict:
    """
    均线金叉死叉策略（信号列表版）。

    参数:
        prices:       日收盘价序列
        short_window: 短期均线窗口
        long_window:  长期均线窗口

    返回:
        dict: {
            "num_units": pd.Series,   # 可直接传给 run_backtest
            "signals": list[Signal],  # SET 1.0 / CLOSE
            "sma_short": pd.Series,
            "sma_long": pd.Series,
        }
    """
    result = ma_crossover(prices, short_window=short_window, long_window=long_window)
    signals = num_units_to_signals(result["num_units"])
    num_units = interpret_signals(prices, signals)

    return {
        "num_units": num_units,
        "signals": signals,
        "sma_short": result["sma_short"],
        "sma_long": result["sma_long"],
    }


def run_validation() -> bool:
    """与原策略做 parity 校验。"""
    import numpy as np

    print("=" * 60)
    print("MA Crossover（信号列表版）— 验证协议")
    print("=" * 60)

    all_pass = True
    n = 200
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n)
    trend = np.concatenate([
        np.linspace(10, 15, 50),
        np.linspace(15, 10, 50),
        np.linspace(10, 15, 50),
        np.linspace(15, 10, 50),
    ])
    prices = pd.Series(trend + np.random.randn(n) * 0.1, index=idx)

    result = ma_crossover_signals(prices, short_window=5, long_window=20)
    orig = ma_crossover(prices, short_window=5, long_window=20)

    parity = result["num_units"].equals(orig["num_units"])
    if not parity:
        all_pass = False
    print(f"  与原策略 num_units 完全一致: {'PASS' if parity else 'FAIL'}")

    trades_ok = len(result["signals"]) >= 2
    if not trades_ok:
        all_pass = False
    print(f"  信号数量 >= 2: {'PASS' if trades_ok else 'FAIL'}")

    bounds_ok = result["num_units"].isin({0.0, 1.0}).all()
    if not bounds_ok:
        all_pass = False
    print(f"  num_units ∈ {{0, 1}}: {'PASS' if bounds_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] MA Crossover 信号版验证")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
