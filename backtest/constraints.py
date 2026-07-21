"""
constraints.py - A股约束与成本模型
===================================

将A股特有的交易约束（涨跌停、停牌）和交易成本（佣金、印花税、滑点）
封装为可组合的协议实现，供回测引擎以列表方式注入。

导入方向: constraints.py <- core.py <- engine.py (单向)

用法:
    from backtest.constraints import (
        PriceLimits, SuspensionCheck, AShareCost,
        default_a_share_constraints,
    )

验证:
    python -m backtest.constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from data.rules import check_price_limit, is_tradable

# ============================================================
# Protocol 定义
# ============================================================


class Constraint(Protocol):
    """约束协议：修改目标股数，返回实际股数。"""

    def apply(
        self,
        target_shares: int,
        prev_shares: int,
        i: int,
        prices: pd.Series,
    ) -> int:
        """
        应用约束。

        Args:
            target_shares: 目标股数（可能已被前一个约束修改）
            prev_shares:   前一日股数
            i:             当前 bar 索引
            prices:        价格序列

        Returns:
            实际股数（应用约束后）
        """
        ...


class CostModel(Protocol):
    """成本模型协议：计算交易成本。"""

    def compute(self, shares_delta: int, price: float, direction: str) -> float:
        """
        计算交易成本。

        Args:
            shares_delta: 股数变化（正=买入，负=卖出）
            price:        成交价格
            direction:    "buy" 或 "sell"

        Returns:
            总交易成本
        """
        ...


# ============================================================
# 约束实现
# ============================================================


@dataclass
class PriceLimits:
    """涨跌停约束。

    涨停日不可买入，跌停日不可卖出。
    """

    board: str = "main"

    def apply(
        self,
        target_shares: int,
        prev_shares: int,
        i: int,
        prices: pd.Series,
    ) -> int:
        if i == 0:
            return target_shares

        limit_info = check_price_limit(
            prices.iloc[i], prices.iloc[i - 1], self.board,
        )

        # 涨停日不可买入
        if target_shares > prev_shares and limit_info["limit_up"]:
            return prev_shares

        # 跌停日不可卖出
        if target_shares < prev_shares and limit_info["limit_down"]:
            return prev_shares

        return target_shares


@dataclass
class SuspensionCheck:
    """停牌检查。

    停牌日保持前一日持仓，不执行交易。
    """

    price_data: pd.DataFrame

    def apply(
        self,
        target_shares: int,
        prev_shares: int,
        i: int,
        prices: pd.Series,
    ) -> int:
        if i == 0:
            return target_shares

        current_date = prices.index[i]
        if not is_tradable(current_date, self.price_data):
            return prev_shares  # 停牌日保持前一日持仓

        return target_shares


# ============================================================
# 工厂函数
# ============================================================


def default_a_share_constraints(
    check_limits: bool = True,
    check_suspension: bool = False,
    price_data: pd.DataFrame | None = None,
    board: str = "main",
) -> list[Constraint]:
    """
    创建 A 股约束列表。

    顺序：[1] SuspensionCheck（优先），[2] PriceLimits

    Args:
        check_limits:     是否启用涨跌停检查
        check_suspension: 是否启用停牌检查
        price_data:       停牌检查所需的 OHLCV 数据
        board:            板块类型

    Returns:
        约束列表（按顺序）

    Raises:
        ValueError: check_suspension=True 但未提供 price_data
    """
    constraints: list[Constraint] = []

    # 顺序重要：停牌检查优先
    if check_suspension:
        if price_data is None:
            raise ValueError("check_suspension=True 时必须提供 price_data")
        constraints.append(SuspensionCheck(price_data=price_data))

    if check_limits:
        constraints.append(PriceLimits(board=board))

    return constraints


# ============================================================
# 成本模型实现
# ============================================================


@dataclass
class AShareCost:
    """A 股成本模型。

    成本 = 佣金 + 印花税（仅卖出）+ 滑点
    """

    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.001
    min_commission: float = 5.0

    def compute(self, shares_delta: int, price: float, direction: str) -> float:
        """
        计算 A 股交易成本。

        Args:
            shares_delta: 股数变化（正=买入，负=卖出）
            price:        成交价格
            direction:    "buy" 或 "sell"

        Returns:
            总交易成本（元）
        """
        trade_amount = abs(shares_delta) * price

        # 佣金（有最低限制）
        commission = max(trade_amount * self.commission_rate, self.min_commission)

        # 印花税（仅卖出）
        stamp = trade_amount * self.stamp_tax_rate if direction == "sell" else 0.0

        # 滑点（双边）
        slippage = trade_amount * self.slippage_rate

        return commission + stamp + slippage


# ============================================================
# 验证协议
# ============================================================


def run_validation() -> bool:
    """
    验证协议：测试每个约束单独工作，验证约束顺序。
    """
    print("=" * 60)
    print("约束与成本模型验证协议")
    print("=" * 60)

    all_pass = True

    # ----------------------------------------------------------
    # 测试 1: PriceLimits — 涨停日阻断买入
    # ----------------------------------------------------------
    print("\n[1] PriceLimits: 涨停日阻断买入")
    print("-" * 60)

    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates)
    pl = PriceLimits(board="main")

    # i=2: prices[2]=11.0, prices[1]=10.0 → +10% → 涨停
    result = pl.apply(target_shares=1000, prev_shares=0, i=2, prices=prices)
    t1_pass = result == 0
    if not t1_pass:
        all_pass = False
    print(f"  result={result}, expected=0  [{'PASS' if t1_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 2: PriceLimits — 跌停日阻断卖出
    # ----------------------------------------------------------
    print("\n[2] PriceLimits: 跌停日阻断卖出")
    print("-" * 60)

    prices2 = pd.Series([10.0, 10.0, 9.0, 9.0], index=dates)
    # i=2: prices[2]=9.0, prices[1]=10.0 → -10% → 跌停
    result2 = pl.apply(target_shares=0, prev_shares=500, i=2, prices=prices2)
    t2_pass = result2 == 500
    if not t2_pass:
        all_pass = False
    print(f"  result={result2}, expected=500  [{'PASS' if t2_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 3: PriceLimits — 非涨跌停日允许交易
    # ----------------------------------------------------------
    print("\n[3] PriceLimits: 非涨跌停日允许交易")
    print("-" * 60)

    prices3 = pd.Series([10.0, 10.0, 10.5, 10.5], index=dates)
    # i=2: prices[2]=10.5, prices[1]=10.0 → +5% → 非涨跌停
    result3 = pl.apply(target_shares=1000, prev_shares=0, i=2, prices=prices3)
    t3_pass = result3 == 1000
    if not t3_pass:
        all_pass = False
    print(f"  result={result3}, expected=1000  [{'PASS' if t3_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 4: SuspensionCheck — 停牌日保持 prev_shares
    # ----------------------------------------------------------
    print("\n[4] SuspensionCheck: 停牌日保持 prev_shares")
    print("-" * 60)

    # price_data: day 2 缺失 → 视为停牌
    trade_dates = pd.date_range("2024-01-01", periods=4)
    price_data = pd.DataFrame(
        {"open": [10, 10, 11, 11], "high": [10, 10, 11, 11],
         "low": [10, 10, 11, 11], "close": [10, 10, 11, 11],
         "volume": [1000, 1000, 0, 1000]},
        index=trade_dates,
    )
    sc = SuspensionCheck(price_data=price_data)
    # i=2 → 日期为 dates[2]，price_data 该日 volume=0 → 停牌
    result4 = sc.apply(target_shares=1000, prev_shares=500, i=2, prices=prices)
    t4_pass = result4 == 500
    if not t4_pass:
        all_pass = False
    print(f"  result={result4}, expected=500  [{'PASS' if t4_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 5: SuspensionCheck — 正常日允许交易
    # ----------------------------------------------------------
    print("\n[5] SuspensionCheck: 正常日允许交易")
    print("-" * 60)

    # i=1 → 日期为 dates[1]，price_data 该日 volume=1000 → 正常
    result5 = sc.apply(target_shares=1000, prev_shares=500, i=1, prices=prices)
    t5_pass = result5 == 1000
    if not t5_pass:
        all_pass = False
    print(f"  result={result5}, expected=1000  [{'PASS' if t5_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 6: 约束顺序 — SuspensionCheck 在 PriceLimits 之前
    # ----------------------------------------------------------
    print("\n[6] 约束顺序: SuspensionCheck 在 PriceLimits 之前")
    print("-" * 60)

    constraints = default_a_share_constraints(
        check_limits=True, check_suspension=True, price_data=price_data,
    )
    t6_pass = (
        len(constraints) == 2
        and isinstance(constraints[0], SuspensionCheck)
        and isinstance(constraints[1], PriceLimits)
    )
    if not t6_pass:
        all_pass = False
    names = [type(c).__name__ for c in constraints]
    print(f"  constraints={names}  [{'PASS' if t6_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 7: AShareCost — 买入成本
    # ----------------------------------------------------------
    print("\n[7] AShareCost: 买入成本计算")
    print("-" * 60)

    cost = AShareCost()
    buy_cost = cost.compute(shares_delta=1000, price=10.0, direction="buy")
    # trade_amount = 1000 * 10.0 = 10000
    # commission = max(10000 * 0.00025, 5.0) = max(2.5, 5.0) = 5.0
    # stamp = 0 (buy)
    # slippage = 10000 * 0.001 = 10.0
    # total = 5.0 + 0 + 10.0 = 15.0
    expected_buy = 15.0
    t7_pass = abs(buy_cost - expected_buy) < 1e-9
    if not t7_pass:
        all_pass = False
    print(f"  buy_cost={buy_cost:.4f}, expected={expected_buy:.4f}  "
          f"[{'PASS' if t7_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 8: AShareCost — 卖出成本（含印花税）
    # ----------------------------------------------------------
    print("\n[8] AShareCost: 卖出成本计算（含印花税）")
    print("-" * 60)

    sell_cost = cost.compute(shares_delta=-1000, price=10.0, direction="sell")
    # trade_amount = 1000 * 10.0 = 10000
    # commission = max(10000 * 0.00025, 5.0) = 5.0
    # stamp = 10000 * 0.0005 = 5.0
    # slippage = 10000 * 0.001 = 10.0
    # total = 5.0 + 5.0 + 10.0 = 20.0
    expected_sell = 20.0
    t8_pass = abs(sell_cost - expected_sell) < 1e-9
    if not t8_pass:
        all_pass = False
    print(f"  sell_cost={sell_cost:.4f}, expected={expected_sell:.4f}  "
          f"[{'PASS' if t8_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 测试 9: malformed input — check_suspension=True 但无 price_data
    # ----------------------------------------------------------
    print("\n[9] Malformed input: check_suspension=True 无 price_data → ValueError")
    print("-" * 60)

    try:
        default_a_share_constraints(check_suspension=True)
        t9_pass = False
    except ValueError:
        t9_pass = True
    if not t9_pass:
        all_pass = False
    print(f"  ValueError raised  [{'PASS' if t9_pass else 'FAIL'}]")

    # ----------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    status = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"约束与成本模型验证: {status}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    run_validation()
