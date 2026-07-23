"""
signal_demo.py - 信号列表策略示例
==================================

演示如何用人类可读的信号列表表达策略，再由 backtest.interpreter 转换成
num_units 序列交给回测引擎执行。

这是新接口的最小示例：策略只负责决定“什么时候买/卖/止损”，
回测引擎负责 T+1、手数、成本、涨跌停等执行细节。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import Signal, interpret_signals


def ma_crossover_signals(
    prices: pd.Series,
    short_window: int = 5,
    long_window: int = 20,
    stop_loss: float | None = None,
) -> dict:
    """
    均线金叉死叉策略（信号列表版）。

    参数:
        prices:       日收盘价序列
        short_window: 短期均线窗口
        long_window:  长期均线窗口
        stop_loss:    可选，止损比例（如 0.95 表示下跌 5% 止损）

    返回:
        dict: {
            "num_units": pd.Series,   # 可直接传给 run_backtest
            "signals": list[Signal],  # 人类可读的信号列表
            "short_ma": pd.Series,
            "long_ma": pd.Series,
        }
    """
    if short_window >= long_window:
        raise ValueError("short_window 必须小于 long_window")

    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    signals: list[Signal] = []
    in_position = False

    for i in range(1, len(prices)):
        prev_short = short_ma.iloc[i - 1]
        prev_long = long_ma.iloc[i - 1]
        curr_short = short_ma.iloc[i]
        curr_long = long_ma.iloc[i]

        if pd.isna(prev_short) or pd.isna(prev_long):
            continue

        if not in_position and prev_short <= prev_long and curr_short > curr_long:
            signals.append(
                Signal(date=prices.index[i], action="BUY", qty=1.0, stop_loss=stop_loss)
            )
            in_position = True
        elif in_position and prev_short >= prev_long and curr_short < curr_long:
            signals.append(Signal(date=prices.index[i], action="CLOSE"))
            in_position = False

    num_units = interpret_signals(prices, signals)
    return {
        "num_units": num_units,
        "signals": signals,
        "short_ma": short_ma,
        "long_ma": long_ma,
    }


def _sharpe(ret: pd.Series) -> float:
    """辅助：年化夏普。"""
    ret = ret.dropna()
    if ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(252))


def run_validation() -> bool:
    """验证协议：正控/负控/不变式。"""
    print("=" * 60)
    print("信号列表策略示例 (MA Crossover) - 验证协议")
    print("=" * 60)

    all_pass = True
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    # 正控：构造阶梯趋势，确保出现金叉/死叉
    print("\n【正控】趋势行情产生交易信号")
    print("-" * 60)
    step_up = np.full(n // 3, 10.0)
    step_high = np.full(n // 3, 20.0)
    step_down = np.full(n - 2 * (n // 3), 10.0)
    trend_prices = pd.Series(
        np.concatenate([step_up, step_high, step_down]), index=idx
    )
    res_trend = ma_crossover_signals(trend_prices, short_window=5, long_window=20)
    has_signals = len(res_trend["signals"]) > 0
    print(f"  信号数量: {len(res_trend['signals'])}  [{'PASS' if has_signals else 'FAIL'}]")
    if not has_signals:
        all_pass = False

    # 负控：随机游走不应产生稳定收益
    print("\n【负控】随机游走夏普 < 0.5")
    print("-" * 60)
    rng = np.random.default_rng(42)
    random_prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    res_random = ma_crossover_signals(random_prices, short_window=5, long_window=20)
    num_units = res_random["num_units"]
    positions = num_units.shift(1).fillna(0.0)
    ret = positions * random_prices.pct_change()
    sharpe = _sharpe(ret)
    neg_ok = sharpe < 0.5
    print(f"  Sharpe = {sharpe:.4f}  [{'PASS' if neg_ok else 'FAIL'}]")
    if not neg_ok:
        all_pass = False

    # 不变式
    print("\n【不变式】num_units ∈ [0, 1]")
    print("-" * 60)
    bounds_ok = (res_random["num_units"] >= 0).all() and (res_random["num_units"] <= 1).all()
    print(f"  边界检查  [{'PASS' if bounds_ok else 'FAIL'}]")
    if not bounds_ok:
        all_pass = False

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] 信号策略示例验证")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
