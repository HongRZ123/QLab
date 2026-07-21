"""
test_constraints.py - 约束与成本模型单元测试
=============================================

8 个测试覆盖:
  (a)-(c) PriceLimits 涨跌停约束
  (d)-(e) SuspensionCheck 停牌检查
  (f)-(g) default_a_share_constraints 工厂
  (h) AShareCost.compute 与 engine._compute_cost 一致性
"""

import pandas as pd
import pytest

from backtest.constraints import (
    AShareCost,
    PriceLimits,
    SuspensionCheck,
    default_a_share_constraints,
)
from backtest.engine import _compute_cost

# ============================================================
# 共享 fixture
# ============================================================


@pytest.fixture()
def dates_4():
    """4 日日期序列。"""
    return pd.date_range("2024-01-01", periods=4)


@pytest.fixture()
def price_data_with_suspension(dates_4):
    """含停牌日的 OHLCV DataFrame（第 3 日 volume=0）。"""
    return pd.DataFrame(
        {
            "open": [10.0, 10.0, 11.0, 11.0],
            "high": [10.0, 10.0, 11.0, 11.0],
            "low": [10.0, 10.0, 11.0, 11.0],
            "close": [10.0, 10.0, 11.0, 11.0],
            "volume": [1000, 1000, 0, 1000],
        },
        index=dates_4,
    )


# ============================================================
# (a) PriceLimits 在涨停日阻断买入
# ============================================================


def test_price_limits_blocks_buy_on_limit_up(dates_4):
    """主板 +10% 涨停 → 买入被阻断，返回 prev_shares。"""
    # prices[1]=10 → prices[2]=11 → 涨幅 10% → 涨停
    prices = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates_4)
    pl = PriceLimits(board="main")

    result = pl.apply(target_shares=1000, prev_shares=0, i=2, prices=prices)
    assert result == 0, f"涨停日应阻断买入，期望 0，得到 {result}"


# ============================================================
# (b) PriceLimits 在跌停日阻断卖出
# ============================================================


def test_price_limits_blocks_sell_on_limit_down(dates_4):
    """主板 -10% 跌停 → 卖出被阻断，返回 prev_shares。"""
    # prices[1]=10 → prices[2]=9 → 跌幅 10% → 跌停
    prices = pd.Series([10.0, 10.0, 9.0, 9.0], index=dates_4)
    pl = PriceLimits(board="main")

    result = pl.apply(target_shares=0, prev_shares=500, i=2, prices=prices)
    assert result == 500, f"跌停日应阻断卖出，期望 500，得到 {result}"


# ============================================================
# (c) PriceLimits 在非涨跌停日允许交易
# ============================================================


def test_price_limits_allows_trade_on_normal_day(dates_4):
    """+5% 非涨跌停 → 交易正常通过。"""
    prices = pd.Series([10.0, 10.0, 10.5, 10.5], index=dates_4)
    pl = PriceLimits(board="main")

    # 买入场景
    result_buy = pl.apply(target_shares=1000, prev_shares=0, i=2, prices=prices)
    assert result_buy == 1000

    # 卖出场景
    result_sell = pl.apply(target_shares=0, prev_shares=500, i=2, prices=prices)
    assert result_sell == 0


# ============================================================
# (d) SuspensionCheck 在停牌日保持 prev_shares
# ============================================================


def test_suspension_check_keeps_prev_on_suspension(
    dates_4, price_data_with_suspension,
):
    """停牌日（volume=0）→ 保持 prev_shares。"""
    sc = SuspensionCheck(price_data=price_data_with_suspension)
    prices = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates_4)

    # i=2 → dates_4[2] → price_data volume=0 → 停牌
    result = sc.apply(target_shares=1000, prev_shares=500, i=2, prices=prices)
    assert result == 500, f"停牌日应保持 prev_shares=500，得到 {result}"


# ============================================================
# (e) SuspensionCheck 在正常日允许交易
# ============================================================


def test_suspension_check_allows_trade_on_normal_day(
    dates_4, price_data_with_suspension,
):
    """正常交易日 → target_shares 原样返回。"""
    sc = SuspensionCheck(price_data=price_data_with_suspension)
    prices = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates_4)

    # i=1 → dates_4[1] → price_data volume=1000 → 正常
    result = sc.apply(target_shares=1000, prev_shares=500, i=1, prices=prices)
    assert result == 1000


# ============================================================
# (f) default_a_share_constraints(False, False) → 空列表
# ============================================================


def test_default_constraints_all_disabled_returns_empty():
    """全部禁用 → 返回空列表。"""
    constraints = default_a_share_constraints(
        check_limits=False, check_suspension=False,
    )
    assert constraints == []


# ============================================================
# (g) default_a_share_constraints(True, True) → [SuspensionCheck, PriceLimits]
# ============================================================


def test_default_constraints_order(price_data_with_suspension):
    """全部启用 → [SuspensionCheck, PriceLimits]（按此顺序）。"""
    constraints = default_a_share_constraints(
        check_limits=True,
        check_suspension=True,
        price_data=price_data_with_suspension,
    )
    assert len(constraints) == 2
    assert isinstance(constraints[0], SuspensionCheck)
    assert isinstance(constraints[1], PriceLimits)


# ============================================================
# (h) AShareCost.compute 与 engine._compute_cost 一致性
# ============================================================


@pytest.mark.parametrize(
    "shares_delta, price, direction",
    [
        (1000, 10.0, "buy"),
        (-500, 20.0, "sell"),
        (100, 50.0, "buy"),     # 小额: trade_amount=5000, commission=max(1.25,5)=5
        (10000, 100.0, "sell"), # 大额: trade_amount=1M
    ],
)
def test_ashare_cost_matches_engine_compute(
    shares_delta: int, price: float, direction: str,
):
    """AShareCost.compute 与 engine._compute_cost 对已知输入匹配。"""
    cost_model = AShareCost()
    actual = cost_model.compute(shares_delta, price, direction)

    trade_amount = abs(shares_delta) * price
    expected = _compute_cost(
        trade_amount=trade_amount,
        direction=direction,
        commission_rate=cost_model.commission_rate,
        stamp_tax_rate=cost_model.stamp_tax_rate,
        slippage_rate=cost_model.slippage_rate,
        min_commission=cost_model.min_commission,
    )
    assert abs(actual - expected) < 1e-9, (
        f"AShareCost({shares_delta}, {price}, {direction}) = {actual}, "
        f"engine._compute_cost = {expected}"
    )


# ============================================================
# 额外: malformed input 防护
# ============================================================


def test_default_constraints_suspension_without_price_data_raises():
    """check_suspension=True 但未提供 price_data → ValueError。"""
    with pytest.raises(ValueError, match="price_data"):
        default_a_share_constraints(check_suspension=True)
