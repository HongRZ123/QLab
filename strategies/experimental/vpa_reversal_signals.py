"""
vpa_reversal_signals.py -- VPA 止损量反转策略（信号列表版）
=============================================================

把原 `strategies/Tech/vpa_reversal.py` 改写成信号列表形式。
策略只输出人类可读的交易信号，由 backtest.interpreter 转成 num_units。

信号逻辑（仅做多）：
    入场：stopping_volume 信号出现 -> BUY 1.0（附带止损位）
    出场：buying_climax 信号 或 价格跌破入场 K 线最低价 -> CLOSE
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import Signal, interpret_signals
from signals.vpa import buying_climax, stopping_volume


def vpa_reversal_signals(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> dict:
    """
    VPA 止损量反转策略（信号列表版）。

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 信号滚动窗口

    返回:
        dict: {
            "num_units": pd.Series,        # 可直接传给 run_backtest
            "signals": list[Signal],       # 人类可读的交易信号
            "stopping_volume": pd.Series,
            "buying_climax": pd.Series,
        }
    """
    close = ohlcv["close"]
    low = ohlcv["low"]

    sv = stopping_volume(ohlcv, lookback=lookback)
    bc = buying_climax(ohlcv, lookback=lookback)

    signals: list[Signal] = []
    in_position = False
    entry_low = np.nan

    for i in range(len(close)):
        if in_position:
            # 退出条件：买入高潮 或 价格跌破入场 K 线最低价
            exit_signal = bool(bc.iloc[i]) or (
                not np.isnan(entry_low) and close.iloc[i] < entry_low
            )
            if exit_signal:
                signals.append(Signal(date=close.index[i], action="CLOSE"))
                in_position = False
                entry_low = np.nan
        else:
            if sv.iloc[i]:
                stop_frac = low.iloc[i] / close.iloc[i] if close.iloc[i] > 0 else None
                signals.append(
                    Signal(
                        date=close.index[i],
                        action="BUY",
                        qty=1.0,
                        stop_loss=stop_frac,
                    )
                )
                in_position = True
                entry_low = float(low.iloc[i])

    num_units = interpret_signals(close, signals)
    return {
        "num_units": num_units,
        "signals": signals,
        "stopping_volume": sv,
        "buying_climax": bc,
    }


def run_validation() -> bool:
    """VPA 反转信号版策略验证协议。"""
    print("=" * 60)
    print("VPA 反转策略（信号列表版）验证协议")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 200

    # 正控：下跌趋势底部出现止损量 -> 应产生买入信号
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
    low.iloc[-1] = 9.0

    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[-1] = 5000.0

    ohlcv = pd.DataFrame(
        {
            "open": open_s,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    result = vpa_reversal_signals(ohlcv, lookback=20)
    has_buy = any(s.action == "BUY" for s in result["signals"])
    print(f"  存在 BUY 信号: {'PASS' if has_buy else 'FAIL'}")
    if not has_buy:
        all_pass = False

    if has_buy:
        last_held = result["num_units"].iloc[-1] > 0
        print(f"  最后一根仍持仓 (forward-fill): {'PASS' if last_held else 'FAIL'}")
        if not last_held:
            all_pass = False

    # 负控：上涨趋势无反转形态 -> 信号稀疏
    print("\n【负控】上涨趋势无反转形态")
    print("-" * 60)

    close_r = pd.Series(np.linspace(10, 20, n), index=idx)
    open_r = close_r.shift(1).fillna(close_r.iloc[0])
    high_r = pd.concat([open_r, close_r], axis=1).max(axis=1) + 0.05
    low_r = pd.concat([open_r, close_r], axis=1).min(axis=1) - 0.05

    ohlcv_r = pd.DataFrame(
        {
            "open": open_r,
            "high": high_r,
            "low": low_r,
            "close": close_r,
            "volume": pd.Series(np.full(n, 1000.0), index=idx),
        }
    )

    result_r = vpa_reversal_signals(ohlcv_r, lookback=20)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.10
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 10%)  [{'PASS' if neg_ok else 'FAIL'}]")
    if not neg_ok:
        all_pass = False

    # 不变式
    print("\n【不变式】num_units ∈ [0, 1]")
    print("-" * 60)
    nonneg_ok = (result["num_units"] >= 0).all()
    nonpos_ok = (result["num_units"] <= 1).all()
    print(f"  num_units >= 0: {'PASS' if nonneg_ok else 'FAIL'}")
    print(f"  num_units <= 1: {'PASS' if nonpos_ok else 'FAIL'}")
    if not nonneg_ok or not nonpos_ok:
        all_pass = False

    print("\n" + "=" * 60)
    print(f"[{'PASS' if all_pass else 'FAIL'}] VPA 反转信号版策略验证")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
