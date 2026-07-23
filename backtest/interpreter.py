"""
interpreter.py - 信号列表 → num_units 适配层
==============================================

策略输出人类可读的信号列表（BUY/SELL/止损/止盈），本模块将其转换为
backtest/core.py 所需的逐 bar num_units 序列。

这样策略作者可以按交易事件思考，而回测引擎仍然保持向量化执行。

用法:
    from backtest import Signal, interpret_signals
    from backtest import run_backtest, performance_summary

    signals = [
        Signal(date="2024-01-05", action="BUY", qty=1.0, stop_loss=0.95),
        Signal(date="2024-02-10", action="SELL", qty=1.0),
    ]
    num_units = interpret_signals(prices, signals)
    bt = run_backtest(prices, num_units)

导入方向:
    backtest/interpreter.py 不依赖 engine.py/core.py，
    只生成与 prices 同 index 的 num_units 序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signal:
    """
    单条交易信号。

    字段:
        date: 信号日期。接受 "YYYY-MM-DD"、datetime.date、datetime.datetime
              或 pd.Timestamp。若日期不在 prices.index 中，将映射到第一个
              大于等于该日期的交易日。
        action: "BUY" | "SELL" | "SET" | "CLOSE" | "HOLD"。
        qty: 仓位比例。BUY 表示增持，SELL 表示减持，SET 表示直接设为目标仓位，
             CLOSE 忽略 qty。
        stop_loss: 止损价相对于开仓价的倍数。例如 0.95 表示下跌 5% 止损。
        take_profit: 止盈价相对于开仓价的倍数。例如 1.10 表示上涨 10% 止盈。
    """

    date: str | date | datetime | pd.Timestamp
    action: Literal["BUY", "SELL", "SET", "CLOSE", "HOLD"] = "HOLD"
    qty: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None


def _parse_signal_date(
    d: str | date | datetime | pd.Timestamp, index: pd.DatetimeIndex
) -> pd.Timestamp:
    """把信号日期解析为 prices.index 中的某个交易日。"""
    if isinstance(d, str):
        parsed = pd.Timestamp(d)
    elif isinstance(d, pd.Timestamp):
        parsed = d
    elif isinstance(d, (datetime, date)):
        parsed = pd.Timestamp(d)
    else:
        raise TypeError(f"Signal.date 类型不受支持: {type(d)}")

    if parsed in index:
        return parsed

    future: pd.DatetimeIndex = index[index >= parsed]
    if len(future) == 0:
        raise ValueError(f"Signal.date {parsed.date()} 超出了 prices.index 范围")
    return future[0]


def _validate_optional_ohlc(
    series: pd.Series | None, reference: pd.Series, name: str
) -> None:
    """校验可选的 low/high 序列与 prices 对齐。"""
    if series is None:
        return
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} 必须是 pd.Series 或 None")
    if not reference.index.equals(series.index):
        raise ValueError(f"{name} 必须与 prices 使用相同的 index")


def num_units_to_signals(targets: pd.Series) -> list[Signal]:
    """
    把目标仓位序列转换成 ``SET`` 信号列表。

    只在目标仓位发生变化时生成信号，空仓目标会生成 ``CLOSE``。
    该函数不处理止损/止盈，需要止损时应额外构造 ``BUY`` 信号。
    """
    signals: list[Signal] = []
    prev = 0.0
    for dt, value in targets.items():
        target = float(value)
        if target == prev:
            continue
        if target == 0.0:
            signals.append(Signal(date=dt, action="CLOSE"))
        else:
            signals.append(Signal(date=dt, action="SET", qty=target))
        prev = target
    return signals


def interpret_signals(
    prices: pd.Series,
    signals: list[Signal],
    low: pd.Series | None = None,
    high: pd.Series | None = None,
) -> pd.Series:
    """
    把离散信号列表转换为逐 bar 的 num_units 目标仓位序列。

    语义:
        - 信号 date 表示“在该 bar 收盘后做出决策”，backtest/core.py 会通过
          shift(1) 在下一个 bar 执行。
        - BUY qty: 在当前仓位基础上增持 qty，上限 1.0。
        - SELL qty: 在当前仓位基础上减持 qty，下限 0.0。
        - SET qty: 直接把仓位设为目标值 qty（不裁剪，允许 >1 表示杠杆）。
        - CLOSE: 直接清仓（忽略 qty）。
        - stop_loss / take_profit: 以开仓价为基准的倍数；触发后从下一 bar 起空仓。

    Args:
        prices: 收盘价序列，index 为 DatetimeIndex。
        signals: 信号列表。同一天内的信号按列表顺序依次处理。
        low: 可选，最低价序列。提供时用于更真实地判断止损是否被触发。
        high: 可选，最高价序列。提供时用于更真实地判断止盈是否被触发。

    Returns:
        pd.Series: 与 prices 同 index 的 num_units 序列（非负）。
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices 必须是 pd.Series")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index 必须是 pd.DatetimeIndex")
    if not isinstance(signals, list):
        raise TypeError("signals 必须是 list[Signal]")

    _validate_optional_ohlc(low, prices, "low")
    _validate_optional_ohlc(high, prices, "high")

    index = prices.index
    num_units = pd.Series(0.0, index=index)
    if len(index) == 0:
        return num_units

    signals_by_date: dict[pd.Timestamp, list[Signal]] = {}
    for sig in signals:
        if sig.qty < 0:
            raise ValueError("Signal.qty 必须 >= 0")
        mapped_date = _parse_signal_date(sig.date, index)
        signals_by_date.setdefault(mapped_date, []).append(sig)

    position = 0.0
    entry_price = np.nan
    active_stop_loss: float | None = None
    active_take_profit: float | None = None

    for i, current_date in enumerate(index):
        close = float(prices.iloc[i])

        # 1) 按顺序处理当日信号
        for sig in signals_by_date.get(current_date, []):
            action = sig.action
            qty = float(sig.qty)

            if action == "HOLD":
                continue
            if action == "CLOSE":
                new_position = 0.0
            elif action == "BUY":
                new_position = min(1.0, position + qty)
            elif action == "SELL":
                new_position = max(0.0, position - qty)
            elif action == "SET":
                new_position = qty
            else:
                raise ValueError(f"不支持的 Signal.action: {action}")

            # 更新持仓成本（仅在仓位增加时）
            if new_position > position and close > 0:
                added = new_position - position
                if position == 0.0 or np.isnan(entry_price):
                    entry_price = close
                else:
                    entry_price = (position * entry_price + added * close) / new_position

            # 更新活跃止损/止盈
            if new_position > 0.0:
                if sig.stop_loss is not None:
                    active_stop_loss = float(sig.stop_loss)
                if sig.take_profit is not None:
                    active_take_profit = float(sig.take_profit)

            position = new_position
            if position == 0.0:
                entry_price = np.nan
                active_stop_loss = None
                active_take_profit = None

        # 2) 记录本 bar 的目标仓位
        num_units.iloc[i] = position

        # 3) 检查止损/止盈，影响下一 bar
        if position > 0.0 and not np.isnan(entry_price):
            check_low = low.iloc[i] if low is not None else close
            check_high = high.iloc[i] if high is not None else close

            stopped = active_stop_loss is not None and check_low <= entry_price * active_stop_loss
            taken = active_take_profit is not None and check_high >= entry_price * active_take_profit

            if stopped or taken:
                position = 0.0
                entry_price = np.nan
                active_stop_loss = None
                active_take_profit = None

    return num_units


def _is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    """浮点比较。"""
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) < tol


def run_validation() -> bool:
    """
    验证协议: 用已知信号序列验证 interpret_signals 输出。
    """
    print("=" * 60)
    print("信号列表 → num_units 适配层验证协议")
    print("=" * 60)

    all_pass = True

    # 测试 1: BUY + SELL
    print("\n[1] BUY 后 SELL")
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0], index=dates)
    signals = [
        Signal(date="2024-01-02", action="BUY", qty=1.0),
        Signal(date="2024-01-04", action="SELL", qty=1.0),
    ]
    units = interpret_signals(prices, signals)
    # SELL 在 01-04 收盘后发出，目标仓位从 01-04 起变为 0，backtest 在 01-05 执行清仓
    expected = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=dates)
    pass1 = units.equals(expected)
    if not pass1:
        all_pass = False
    print(f"  num_units = {list(units.values)}")
    print(f"  expected  = {list(expected.values)}  [{'PASS' if pass1 else 'FAIL'}]")

    # 测试 2: 止损
    print("\n[2] BUY + stop_loss")
    dates2 = pd.date_range("2024-01-01", periods=4)
    prices2 = pd.Series([10.0, 10.2, 9.0, 10.5], index=dates2)
    signals2 = [
        Signal(date="2024-01-01", action="BUY", qty=1.0, stop_loss=0.95),
    ]
    units2 = interpret_signals(prices2, signals2)
    # 01-03 收盘 9.0 触发止损，目标仓位从 01-04 起变为 0
    expected2 = pd.Series([1.0, 1.0, 1.0, 0.0], index=dates2)
    pass2 = units2.equals(expected2)
    if not pass2:
        all_pass = False
    print(f"  num_units = {list(units2.values)}")
    print(f"  expected  = {list(expected2.values)}  [{'PASS' if pass2 else 'FAIL'}]")

    # 测试 3: 止盈
    print("\n[3] BUY + take_profit")
    dates3 = pd.date_range("2024-01-01", periods=4)
    prices3 = pd.Series([10.0, 11.5, 11.0, 10.0], index=dates3)
    signals3 = [
        Signal(date="2024-01-01", action="BUY", qty=1.0, take_profit=1.10),
    ]
    units3 = interpret_signals(prices3, signals3)
    # 01-02 收盘 11.5 触发止盈，目标仓位从 01-03 起变为 0
    expected3 = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates3)
    pass3 = units3.equals(expected3)
    if not pass3:
        all_pass = False
    print(f"  num_units = {list(units3.values)}")
    print(f"  expected  = {list(expected3.values)}  [{'PASS' if pass3 else 'FAIL'}]")

    # 测试 4: SET 直接设置目标仓位
    print("\n[4] SET 设置目标仓位")
    dates4 = pd.date_range("2024-01-01", periods=5)
    prices4 = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0], index=dates4)
    signals4 = [
        Signal(date="2024-01-01", action="SET", qty=0.5),
        Signal(date="2024-01-03", action="SET", qty=0.8),
        Signal(date="2024-01-04", action="SET", qty=0.0),
    ]
    units4 = interpret_signals(prices4, signals4)
    expected4 = pd.Series([0.5, 0.5, 0.8, 0.0, 0.0], index=dates4)
    pass4 = units4.equals(expected4)
    if not pass4:
        all_pass = False
    print(f"  num_units = {list(units4.values)}")
    print(f"  expected  = {list(expected4.values)}  [{'PASS' if pass4 else 'FAIL'}]")

    # 测试 5: 日期映射到下一交易日
    print("\n[5] 信号日期映射到下一交易日")
    dates5 = pd.date_range("2024-01-01", periods=5)
    prices5 = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates5)
    signals5 = [
        Signal(date="2023-12-31", action="BUY", qty=1.0),
        Signal(date="2024-01-05", action="SELL", qty=1.0),
    ]
    units5 = interpret_signals(prices5, signals5)
    expected5 = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0], index=dates5)
    pass5 = units5.equals(expected5)
    if not pass5:
        all_pass = False
    print(f"  num_units = {list(units5.values)}")
    print(f"  expected  = {list(expected5.values)}  [{'PASS' if pass5 else 'FAIL'}]")

    print("\n" + "=" * 60)
    status = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"信号适配层验证: {status}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = run_validation()
    if not success:
        raise SystemExit(1)
