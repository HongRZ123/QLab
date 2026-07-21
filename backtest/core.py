"""
core.py - 纯 Chan PnL 循环
============================

将 Chan PnL 公式抽象为纯数学循环，不含 A 股特定逻辑。
所有 A 股约束（涨跌停、停牌）通过 Constraint 协议列表注入。

导入方向: constraints.py <- core.py <- engine.py (单向)

用法:
    from backtest.core import run_core
    from backtest.constraints import PriceLimits, AShareCost

    result = run_core(
        prices, num_units,
        constraints=[PriceLimits(board="main")],
        cost_model=AShareCost(),
    )

验证:
    python -m backtest.core
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.constraints import Constraint, CostModel
from data.rules import round_to_lot

# ============================================================
# 纯 Chan PnL 循环
# ============================================================


def run_core(
    prices: pd.Series,
    num_units: pd.Series,
    initial_capital: float = 1_000_000.0,
    constraints: list[Constraint] | None = None,
    cost_model: CostModel | None = None,
    dynamic_sizing: bool = True,
    lot_size: int = 100,
) -> dict:
    """
    纯 Chan PnL 循环。

    将 Chan 公式回测抽象为：T+1 执行 → 整数手取整 → 约束列表 → PnL → 成本 → 权益。
    不含任何 A 股特定逻辑（涨跌停/停牌），全部通过 constraints 注入。

    Args:
        prices:         日价格序列 (pd.Series, index 为日期)
        num_units:      仓位单元序列（原始信号，未做 T+1）
        initial_capital: 初始资金 (元)
        constraints:    约束列表（可选）。None 或空列表 = 不应用任何约束
        cost_model:     成本模型（可选）。None = 成本为 0
        dynamic_sizing: 动态仓位管理。True=用当前权益, False=用初始资金
        lot_size:       每手股数

    Returns:
        dict: {
            "positions":    每日持仓股数 (pd.Series, int),
            "pnl":          每日盈亏 (pd.Series, float),
            "ret":          每日收益率 (pd.Series, float),
            "equity_curve": 权益曲线 (pd.Series, float),
            "n_trades":     交易次数 (int),
            "total_cost":   总交易成本 (float),
        }
    """
    # ---- 输入验证 ----
    if not isinstance(prices, pd.Series):
        raise TypeError("prices 必须是 pd.Series")
    if not isinstance(num_units, pd.Series):
        raise TypeError("num_units 必须是 pd.Series")
    if not prices.index.equals(num_units.index):
        raise ValueError("prices 和 num_units 必须使用相同的 index")

    N = len(prices)
    if N == 0:
        return {
            "positions": pd.Series([], dtype=int),
            "pnl": pd.Series([], dtype=float),
            "ret": pd.Series([], dtype=float),
            "equity_curve": pd.Series([], dtype=float),
            "n_trades": 0,
            "total_cost": 0.0,
        }

    # ---- 规范化约束列表 ----
    if constraints is None:
        constraints = []

    # ---- T+1 执行: 信号日 t 的 num_units 在 t+1 日执行 ----
    lagged_units = num_units.shift(1).fillna(0)

    # ---- 初始化输出序列 ----
    shares = pd.Series(0, index=prices.index, dtype=int)
    pnl = pd.Series(0.0, index=prices.index)
    trade_cost = pd.Series(0.0, index=prices.index)
    ret = pd.Series(0.0, index=prices.index)
    equity_curve = pd.Series(0.0, index=prices.index)
    equity_curve.iloc[0] = initial_capital

    # ---- 逐 bar 迭代 ----
    for i in range(N):
        # 1) 确定 sizing 资金
        capital = equity_curve.iloc[i - 1] if (dynamic_sizing and i > 0) else initial_capital

        # 2) 计算目标仓位
        u = lagged_units.iloc[i]
        target_shares = 0
        if u > 0 and prices.iloc[i] > 0 and capital > 0:
            raw = u * capital / prices.iloc[i]
            target_shares = round_to_lot(raw, lot_size)

        # 3) 应用约束列表
        actual_shares = target_shares
        for c in constraints:
            actual_shares = c.apply(actual_shares, shares.iloc[i - 1], i, prices)

        shares.iloc[i] = actual_shares

        # 4) PnL (Chan 公式): pnl(t) = shares(t-1) * (price(t) - price(t-1))
        if i >= 1 and shares.iloc[i - 1] > 0:
            pnl.iloc[i] = shares.iloc[i - 1] * (prices.iloc[i] - prices.iloc[i - 1])

        # 5) 交易成本: 持仓变化时通过 cost_model 计算
        if i >= 1:
            delta = shares.iloc[i] - shares.iloc[i - 1]
            if delta != 0 and cost_model is not None:
                direction = "buy" if delta > 0 else "sell"
                trade_cost.iloc[i] = cost_model.compute(delta, prices.iloc[i], direction)

        # 6) 权益更新: equity(t) = equity(t-1) + pnl(t) - cost(t)
        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] + pnl.iloc[i] - trade_cost.iloc[i]

        # 7) 每日收益率: ret(t) = (equity(t) - equity(t-1)) / equity(t-1)
        if i >= 1 and equity_curve.iloc[i - 1] != 0:
            ret.iloc[i] = (equity_curve.iloc[i] - equity_curve.iloc[i - 1]) / equity_curve.iloc[i - 1]

    # ---- 汇总 ----
    n_trades = int((shares.diff().abs() > 0).sum())
    total_cost = float(trade_cost.sum())

    return {
        "positions": shares,
        "pnl": pnl,
        "ret": ret,
        "equity_curve": equity_curve,
        "n_trades": n_trades,
        "total_cost": total_cost,
    }


# ============================================================
# 验证协议
# ============================================================


def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    """浮点比较，处理 NaN。"""
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) < tol


def run_validation() -> bool:
    """
    验证协议: 构造已知数据，断言 run_core 输出与手算一致。
    """
    print("=" * 60)
    print("纯 Chan PnL 循环验证协议")
    print("=" * 60)

    all_pass = True

    # ============================================================
    # 测试 1: 恒定价格 → PnL = 0（纯 PnL，无约束，无成本）
    # ============================================================
    print("\n[1] 恒定价格 → PnL = 0 (无约束, 无成本)")
    print("-" * 60)

    dates = pd.date_range("2024-01-01", periods=5)
    p_const = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=dates)
    u_const = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)

    result = run_core(p_const, u_const, dynamic_sizing=False)

    pnl_sum = float(result["pnl"].sum())
    cost_sum = result["total_cost"]
    t1_pass = _is_close(pnl_sum, 0.0) and _is_close(cost_sum, 0.0)
    if not t1_pass:
        all_pass = False
    print(f"  PnL 总和 = {pnl_sum:.6f}, 预期 = 0.0  "
          f"[{'PASS' if _is_close(pnl_sum, 0.0) else 'FAIL'}]")
    print(f"  总成本   = {cost_sum:.6f}, 预期 = 0.0  "
          f"[{'PASS' if _is_close(cost_sum, 0.0) else 'FAIL'}]")

    # ============================================================
    # 测试 2: 已知序列 → 与 engine 测试 2 手算值匹配 (check_limits=False)
    # ============================================================
    print("\n[2] 已知序列 → PnL 与手算一致 (无约束, AShareCost)")
    print("-" * 60)

    from backtest.constraints import AShareCost

    dates2 = pd.date_range("2024-01-01", periods=4)
    p_known = pd.Series([10.0, 11.0, 9.0, 12.0], index=dates2)
    u_known = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates2)

    result2 = run_core(
        p_known, u_known,
        initial_capital=1_000_000,
        cost_model=AShareCost(),
        dynamic_sizing=False,
    )

    # 手算 (与 engine.py run_validation 测试 2 一致):
    # lagged_units = [0, 0, 1, 1]
    # t=0: shares=0, pnl=0
    # t=1: shares=0, pnl=0
    # t=2: target=round_to_lot(1*1000000/9, 100)=111100, 买入成本=1249.875
    # t=3: target=round_to_lot(1*1000000/12, 100)=83300, 卖出成本=583.8
    #      pnl(3)=111100*(12-9)=333300
    expected_pnl = pd.Series([0.0, 0.0, 0.0, 333300.0], index=dates2)
    expected_equity = pd.Series([1_000_000.0, 1_000_000.0, 998_750.125, 1_331_466.325], index=dates2)
    total_cost_expected = 1249.875 + 583.8

    # PnL 验证
    for i in range(4):
        ok = _is_close(result2["pnl"].iloc[i], expected_pnl.iloc[i])
        if not ok:
            all_pass = False
        print(f"  PnL[{i}]: actual={result2['pnl'].iloc[i]:,.2f}  "
              f"expected={expected_pnl.iloc[i]:,.2f}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # 总成本验证
    pass_cost = _is_close(result2["total_cost"], total_cost_expected, tol=1.0)
    if not pass_cost:
        all_pass = False
    print(f"  总成本: actual={result2['total_cost']:,.4f}  "
          f"expected={total_cost_expected:,.4f}  "
          f"[{'PASS' if pass_cost else 'FAIL'}]")

    # 权益验证
    for i in range(4):
        ok = _is_close(result2["equity_curve"].iloc[i], expected_equity.iloc[i], tol=1.0)
        if not ok:
            all_pass = False
        print(f"  Equity[{i}]: actual={result2['equity_curve'].iloc[i]:,.2f}  "
              f"expected={expected_equity.iloc[i]:,.2f}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    status = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"纯 Chan PnL 循环验证: {status}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    run_validation()
