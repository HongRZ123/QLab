"""
s4_linear_signals.py — S4 线性均值回归策略（信号列表版）
=========================================================

把原 ``strategies.MR.s4_linear.linear_mr`` 改写成信号列表形式：

- 策略层只决定目标仓位 ``target = max(0, -Z(t))``
- 当目标仓位变化时，通过 ``SET`` 信号直接设为目标仓位
- ``backtest.interpreter`` 把信号列表转成 ``num_units`` 序列

这是新接口与旧策略保持行为一致的最小包装：信号版与原 ``linear_mr``
产生的 ``num_units`` 完全相同。
"""

from __future__ import annotations

import pandas as pd

from backtest import interpret_signals, num_units_to_signals
from strategies.MR.s4_linear import linear_mr


def linear_mr_signals(
    prices: pd.Series,
    lookback: int | None = None,
    half_life: float | None = None,
) -> dict:
    """
    S4 线性均值回归策略（信号列表版）。

    参数:
        prices:    日收盘价序列
        lookback:  回望期 L，None 时自动确定
        half_life: 半衰期，lookback=None 时用于推导 L

    返回:
        dict: {
            "num_units": pd.Series,   # 可直接传给 run_backtest
            "signals": list[Signal],  # 人类可读的目标仓位信号
            "z_score": pd.Series,
            "lookback_used": int,
        }
    """
    result = linear_mr(prices, lookback=lookback, half_life=half_life)
    num_units = result["num_units"]
    signals = num_units_to_signals(num_units)

    # 通过 interpreter 再转一次，保证输出与 signal 语义严格一致
    num_units_interp = interpret_signals(prices, signals)

    return {
        "num_units": num_units_interp,
        "signals": signals,
        "z_score": result["z_score"],
        "lookback_used": result["lookback_used"],
    }


def run_validation() -> bool:
    """与原策略做 parity 校验的简化验证。"""
    import numpy as np

    from signals.stats import generate_ou_paths

    print("=" * 60)
    print("S4 线性 MR（信号列表版）— 验证协议")
    print("=" * 60)

    all_pass = True
    n = 200
    idx = pd.date_range("2024-01-01", periods=n)
    ou_raw = generate_ou_paths(1, n, theta=0.05, mu=0.0, sigma=1.0, dt=1.0, seed=42)
    prices = pd.Series(np.exp(ou_raw[0]), index=idx)

    result = linear_mr_signals(prices, lookback=20)
    orig = linear_mr(prices, lookback=20)

    parity = result["num_units"].equals(orig["num_units"])
    if not parity:
        all_pass = False
    print(f"  与原策略 num_units 完全一致: {'PASS' if parity else 'FAIL'}")

    has_signals = len(result["signals"]) > 0
    if not has_signals:
        all_pass = False
    print(f"  产生非空信号列表: {'PASS' if has_signals else 'FAIL'}")

    bounds_ok = (result["num_units"] >= 0).all()
    if not bounds_ok:
        all_pass = False
    print(f"  num_units >= 0: {'PASS' if bounds_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] S4 信号版验证")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
