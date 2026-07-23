"""
s8_bollinger_signals.py — S8 布林带均值回归策略（信号列表版）
===============================================================

把原 ``strategies.MR.s8_bollinger.bollinger_mr`` 改写成信号列表形式。
与原策略保持行为一致：Z < -entry_z 时满仓，Z >= -exit_z 时空仓，
无信号日仓位不变。
"""

from __future__ import annotations

import pandas as pd

from backtest import interpret_signals, num_units_to_signals
from strategies.MR.s8_bollinger import bollinger_mr


def bollinger_mr_signals(
    prices: pd.Series,
    lookback: int,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
) -> dict:
    """
    S8 布林带均值回归策略（信号列表版）。

    参数:
        prices:   日收盘价序列
        lookback: 回望期 L
        entry_z:  入场 Z-Score 阈值
        exit_z:   出场 Z-Score 阈值

    返回:
        dict: {
            "num_units": pd.Series,   # 可直接传给 run_backtest
            "signals": list[Signal],  # SET 1.0 / CLOSE
            "z_score": pd.Series,
        }
    """
    result = bollinger_mr(prices, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    signals = num_units_to_signals(result["num_units"])
    num_units = interpret_signals(prices, signals)

    return {
        "num_units": num_units,
        "signals": signals,
        "z_score": result["z_score"],
    }


def run_validation() -> bool:
    """与原策略做 parity 校验。"""
    import numpy as np

    from signals.stats import generate_ou_paths

    print("=" * 60)
    print("S8 布林带 MR（信号列表版）— 验证协议")
    print("=" * 60)

    all_pass = True
    n = 200
    idx = pd.date_range("2024-01-01", periods=n)
    ou_raw = generate_ou_paths(1, n, theta=0.05, mu=0.0, sigma=1.0, dt=1.0, seed=42)
    prices = pd.Series(np.exp(ou_raw[0]), index=idx)

    result = bollinger_mr_signals(prices, lookback=20, entry_z=1.0, exit_z=0.0)
    orig = bollinger_mr(prices, lookback=20, entry_z=1.0, exit_z=0.0)

    parity = result["num_units"].equals(orig["num_units"])
    if not parity:
        all_pass = False
    print(f"  与原策略 num_units 完全一致: {'PASS' if parity else 'FAIL'}")

    only_set_close = all(
        s.action in {"SET", "CLOSE"} for s in result["signals"]
    )
    if not only_set_close:
        all_pass = False
    print(f"  仅含 SET/CLOSE 信号: {'PASS' if only_set_close else 'FAIL'}")

    bounds_ok = result["num_units"].isin({0.0, 1.0}).all()
    if not bounds_ok:
        all_pass = False
    print(f"  num_units ∈ {{0, 1}}: {'PASS' if bounds_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] S8 信号版验证")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
