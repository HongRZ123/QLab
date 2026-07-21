"""
test_backtest_core.py - run_core 单元测试
==========================================

4 个测试覆盖:
  (a) run_core 无约束时与 run_backtest(check_limits=False, check_suspension=False) 输出相同
  (b) run_core 空约束列表时成本为 0（无 cost_model）
  (c) run_core 应用 PriceLimits 约束时行为正确
  (d) run_core 应用 SuspensionCheck 约束时行为正确
"""

import pandas as pd
import pytest

from backtest.constraints import (
    AShareCost,
    PriceLimits,
    SuspensionCheck,
)
from backtest.core import run_core
from backtest.engine import run_backtest

# ============================================================
# 共享 fixture
# ============================================================


@pytest.fixture()
def dates_4():
    """4 日日期序列。"""
    return pd.date_range("2024-01-01", periods=4)


@pytest.fixture()
def basic_scenario(dates_4):
    """基本价格/信号场景。"""
    prices = pd.Series([10.0, 11.0, 9.0, 12.0], index=dates_4)
    num_units = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates_4)
    return prices, num_units


# ============================================================
# (a) run_core 无约束 ≈ run_backtest(check_limits=False, check_suspension=False)
# ============================================================


def test_run_core_no_constraints_matches_engine(basic_scenario):
    """run_core 无约束 + AShareCost 应与 engine (check_limits=False) 输出一致。"""
    prices, num_units = basic_scenario

    result_core = run_core(
        prices,
        num_units,
        initial_capital=1_000_000,
        constraints=None,
        cost_model=AShareCost(),
        dynamic_sizing=False,
        lot_size=100,
    )

    result_engine = run_backtest(
        prices,
        num_units,
        initial_capital=1_000_000,
        dynamic_sizing=False,
        check_limits=False,
        check_suspension=False,
    )

    # 逐字段比较
    pd.testing.assert_series_equal(
        result_core["positions"], result_engine["positions"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result_core["pnl"], result_engine["pnl"],
        check_names=False, atol=1e-6,
    )
    pd.testing.assert_series_equal(
        result_core["equity_curve"], result_engine["equity_curve"],
        check_names=False, atol=1e-6,
    )
    assert result_core["n_trades"] == result_engine["n_trades"]
    assert abs(result_core["total_cost"] - result_engine["total_cost"]) < 1e-6


# ============================================================
# (b) run_core 空约束 + 无 cost_model → 成本为 0
# ============================================================


def test_run_core_empty_constraints_no_cost_model(basic_scenario):
    """空约束 + cost_model=None → total_cost=0。"""
    prices, num_units = basic_scenario

    result = run_core(
        prices,
        num_units,
        initial_capital=1_000_000,
        constraints=[],
        cost_model=None,
        dynamic_sizing=False,
    )

    assert result["total_cost"] == 0.0
    assert (result["positions"].diff().abs() > 0).any(), "应有持仓变化"
    # 无成本时权益 = 初始资金 + 累计 PnL
    expected_equity = 1_000_000.0 + result["pnl"].cumsum()
    pd.testing.assert_series_equal(
        result["equity_curve"], expected_equity,
        check_names=False, atol=1e-6,
    )


# ============================================================
# (c) run_core 应用 PriceLimits 约束 → 涨停日阻断买入
# ============================================================


def test_run_core_price_limits_constraint():
    """PriceLimits 约束在涨停日阻断买入。"""
    # 构造涨停场景: 10 → 10 → 11 (涨停 +10%)
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates)
    # 信号在 t=1 发出 num_units=1, T+1 在 t=2 执行
    # 但 t=2 涨停 (10→11 = +10%), PriceLimits 应阻断买入
    num_units = pd.Series([0.0, 1.0, 1.0, 1.0], index=dates)

    result = run_core(
        prices,
        num_units,
        initial_capital=1_000_000,
        constraints=[PriceLimits(board="main")],
        cost_model=AShareCost(),
        dynamic_sizing=False,
    )

    # t=2 涨停，买入被阻断 → shares[2] 应保持 shares[1]=0
    assert result["positions"].iloc[2] == 0, (
        f"涨停日应保持 prev_shares=0，得到 {result['positions'].iloc[2]}"
    )


# ============================================================
# (d) run_core 应用 SuspensionCheck 约束 → 停牌日保持持仓
# ============================================================


def test_run_core_suspension_check_constraint():
    """SuspensionCheck 约束在停牌日保持 prev_shares。"""
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates)
    num_units = pd.Series([0.0, 1.0, 1.0, 1.0], index=dates)

    # 构造停牌: t=2 日 volume=0
    price_data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 11.0, 11.0],
            "high": [10.0, 10.0, 11.0, 11.0],
            "low": [10.0, 10.0, 11.0, 11.0],
            "close": [10.0, 10.0, 11.0, 11.0],
            "volume": [1000, 1000, 0, 1000],
        },
        index=dates,
    )

    result = run_core(
        prices,
        num_units,
        initial_capital=1_000_000,
        constraints=[SuspensionCheck(price_data=price_data)],
        cost_model=AShareCost(),
        dynamic_sizing=False,
    )

    # t=2 停牌，应沿用 t=1 的持仓
    # t=1: lagged_units[1]=num_units[0]=0 → shares[1]=0
    # t=2: 停牌 → shares[2]=shares[1]=0
    assert result["positions"].iloc[2] == result["positions"].iloc[1], (
        f"停牌日应保持 prev_shares={result['positions'].iloc[1]}，"
        f"得到 {result['positions'].iloc[2]}"
    )


# ============================================================
# (d-补充) run_core 停牌日 + 已有持仓 → 保持持仓不变
# ============================================================


def test_run_core_suspension_keeps_existing_position():
    """停牌日已有持仓时保持该持仓。"""
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.Series([10.0, 10.0, 10.0, 11.0, 11.0], index=dates)
    # t=0: signal=0 → lagged=0 → shares[0]=0
    # t=1: lagged=0 → shares[1]=0
    # t=2: lagged=1 → buy at 10.0 → shares[2]>0
    # t=3: 停牌 → 保持 shares[2]
    # t=4: lagged=0 → sell → shares[4]=0
    num_units = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=dates)

    price_data = pd.DataFrame(
        {
            "open": [10, 10, 10, 11, 11],
            "high": [10, 10, 10, 11, 11],
            "low": [10, 10, 10, 11, 11],
            "close": [10, 10, 10, 11, 11],
            "volume": [1000, 1000, 1000, 0, 1000],
        },
        index=dates,
    )

    result = run_core(
        prices,
        num_units,
        initial_capital=1_000_000,
        constraints=[SuspensionCheck(price_data=price_data)],
        cost_model=AShareCost(),
        dynamic_sizing=False,
    )

    # t=3 停牌 → shares[3]=shares[2]
    assert result["positions"].iloc[3] == result["positions"].iloc[2], (
        f"停牌日应保持持仓 shares[2]={result['positions'].iloc[2]}，"
        f"得到 shares[3]={result['positions'].iloc[3]}"
    )
    # t=3 停牌时不应有交易成本（持仓不变）
    assert result["positions"].diff().iloc[3] == 0
