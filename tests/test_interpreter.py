"""
test_interpreter.py - 信号列表 → num_units 适配层测试
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.interpreter import Signal, interpret_signals, num_units_to_signals


@pytest.fixture
def prices() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=5)
    return pd.Series([10.0, 11.0, 12.0, 11.0, 10.0], index=dates)


def test_buy_sell_sequence(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-02", action="BUY", qty=1.0),
        Signal(date="2024-01-04", action="SELL", qty=1.0),
    ]
    units = interpret_signals(prices, signals)
    # SELL 在 01-04 收盘后发出，目标仓位从 01-04 起变为 0，backtest 在 01-05 执行清仓
    expected = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_partial_buy_sell(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-01", action="BUY", qty=0.3),
        Signal(date="2024-01-02", action="BUY", qty=0.5),
        Signal(date="2024-01-03", action="SELL", qty=0.4),
        Signal(date="2024-01-04", action="SELL", qty=1.0),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([0.3, 0.8, 0.4, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_close_ignores_qty(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-02", action="BUY", qty=1.0),
        Signal(date="2024-01-04", action="CLOSE", qty=0.0),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_hold_does_nothing(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-01", action="BUY", qty=1.0),
        Signal(date="2024-01-02", action="HOLD"),
        Signal(date="2024-01-04", action="SELL", qty=1.0),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_stop_loss_triggered_by_close() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 10.2, 9.0, 10.5], index=dates)
    signals = [Signal(date="2024-01-01", action="BUY", qty=1.0, stop_loss=0.95)]
    units = interpret_signals(prices, signals)
    # 01-03 收盘 9.0 触发止损，目标仓位从 01-04 起变为 0
    expected = pd.Series([1.0, 1.0, 1.0, 0.0], index=dates)
    pd.testing.assert_series_equal(units, expected)


def test_stop_loss_triggered_by_low() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 10.2, 9.6, 10.5], index=dates)
    low = pd.Series([9.9, 9.4, 9.5, 10.2], index=dates)
    signals = [Signal(date="2024-01-01", action="BUY", qty=1.0, stop_loss=0.95)]
    units = interpret_signals(prices, signals, low=low)
    # low on day 2 (index=1) is 9.4 <= 9.5 -> stop triggers, position becomes 0 from day 3
    expected = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)
    pd.testing.assert_series_equal(units, expected)


def test_take_profit_triggered_by_close() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 11.5, 11.0, 10.0], index=dates)
    signals = [Signal(date="2024-01-01", action="BUY", qty=1.0, take_profit=1.10)]
    units = interpret_signals(prices, signals)
    expected = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)
    pd.testing.assert_series_equal(units, expected)


def test_take_profit_triggered_by_high() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 10.5, 10.8, 10.0], index=dates)
    high = pd.Series([10.2, 11.1, 10.9, 10.5], index=dates)
    signals = [Signal(date="2024-01-01", action="BUY", qty=1.0, take_profit=1.10)]
    units = interpret_signals(prices, signals, high=high)
    # high on day 2 (index=1) is 11.1 >= 11.0 -> take profit triggers
    expected = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)
    pd.testing.assert_series_equal(units, expected)


def test_date_mapped_to_next_trading_day() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates)
    signals = [
        Signal(date="2023-12-31", action="BUY", qty=1.0),
        Signal(date="2024-01-05", action="SELL", qty=1.0),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0], index=dates)
    pd.testing.assert_series_equal(units, expected)


def test_empty_prices() -> None:
    prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    signals = [Signal(date="2024-01-01", action="BUY", qty=1.0)]
    units = interpret_signals(prices, signals)
    assert len(units) == 0


def test_no_signals_returns_zeros(prices: pd.Series) -> None:
    units = interpret_signals(prices, [])
    expected = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_invalid_qty_raises() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    prices = pd.Series([10.0, 11.0], index=dates)
    signals = [Signal(date="2024-01-01", action="BUY", qty=-0.5)]
    with pytest.raises(ValueError, match="qty"):
        interpret_signals(prices, signals)


def test_invalid_prices_type_raises() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        interpret_signals([10.0, 11.0], [Signal(date="2024-01-01", action="BUY")])  # type: ignore[arg-type]


def test_invalid_signals_type_raises(prices: pd.Series) -> None:
    with pytest.raises(TypeError, match="list"):
        interpret_signals(prices, Signal(date="2024-01-01", action="BUY"))  # type: ignore[arg-type]


def test_misaligned_low_raises(prices: pd.Series) -> None:
    wrong_low = pd.Series([1.0, 2.0], index=pd.date_range("2024-02-01", periods=2))
    with pytest.raises(ValueError, match="low"):
        interpret_signals(prices, [], low=wrong_low)


def test_buy_capped_at_one(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-01", action="BUY", qty=0.6),
        Signal(date="2024-01-02", action="BUY", qty=0.6),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([0.6, 1.0, 1.0, 1.0, 1.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_sell_floored_at_zero(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-01", action="BUY", qty=1.0),
        Signal(date="2024-01-02", action="SELL", qty=0.3),
        Signal(date="2024-01-03", action="SELL", qty=1.0),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([1.0, 0.7, 0.0, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_set_action_sets_target_position(prices: pd.Series) -> None:
    signals = [
        Signal(date="2024-01-01", action="SET", qty=0.5),
        Signal(date="2024-01-03", action="SET", qty=0.8),
        Signal(date="2024-01-04", action="SET", qty=0.0),
    ]
    units = interpret_signals(prices, signals)
    expected = pd.Series([0.5, 0.5, 0.8, 0.0, 0.0], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_set_action_allows_leverage(prices: pd.Series) -> None:
    signals = [Signal(date="2024-01-01", action="SET", qty=1.5)]
    units = interpret_signals(prices, signals)
    expected = pd.Series([1.5, 1.5, 1.5, 1.5, 1.5], index=prices.index)
    pd.testing.assert_series_equal(units, expected)


def test_num_units_to_signals_emits_set_and_close(prices: pd.Series) -> None:
    targets = pd.Series([0.0, 0.5, 0.5, 1.0, 0.0], index=prices.index)
    signals = num_units_to_signals(targets)
    assert len(signals) == 3
    assert signals[0].action == "SET" and signals[0].qty == 0.5
    assert signals[1].action == "SET" and signals[1].qty == 1.0
    assert signals[2].action == "CLOSE"
    units = interpret_signals(prices, signals)
    pd.testing.assert_series_equal(units, targets)


def test_average_entry_price_on_additions() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 12.0, 11.0, 10.0], index=dates)
    signals = [
        Signal(date="2024-01-01", action="BUY", qty=0.5, stop_loss=0.95),
        Signal(date="2024-01-02", action="BUY", qty=0.5, stop_loss=0.95),
    ]
    # Weighted entry price = (0.5*10 + 0.5*12) / 1.0 = 11.0
    # stop at 11.0 * 0.95 = 10.45. Day 3 close = 10.0 -> stop triggers,
    # so target position becomes 0 from day 4 (which is beyond the series).
    units = interpret_signals(prices, signals)
    expected = pd.Series([0.5, 1.0, 1.0, 1.0], index=dates)
    pd.testing.assert_series_equal(units, expected)
