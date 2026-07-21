"""
engine.py — 回测引擎 + A股约束
=================================

基于 Chan 公式的回测引擎，支持 T+1 执行、整数手约束、交易成本、
涨跌停和停牌检查。

核心函数:
    run_backtest            — 通用回测引擎 (num_units 可为任意非负数)
    run_backtest_long_only  — 便捷封装 (signals ∈ {0, 1})

PnL 公式 (Chan):
    pnl(t) = mkt_val(t-1) × (y(t) - y(t-1)) / y(t-1)
           = shares(t-1) × (price(t) - price(t-1))

T+1 执行:
    信号日 t 的 num_units 在 t+1 日执行 → positions = num_units.shift(1).fillna(0)

整数手约束:
    shares = round_to_lot(num_units × capital / price, lot_size)

交易成本:
    买入: 佣金 = max(amount × commission_rate, 5.0)
    卖出: 佣金 + 印花税 = amount × stamp_tax_rate
    滑点: amount × slippage_rate (双边)

涨跌停/停牌:
    涨停日不可买入, 跌停日不可卖出, 停牌日不执行交易
    通过 board 参数区分板块: main (±10%), chinext/star (±20%), st (±5%)

验证协议:
    - 恒定价格 + 恒定持仓 → PnL = 0
    - 已知价格序列 + 已知信号 → PnL 与手算一致 (含成本)
    - T+1 验证: positions 比 signals 滞后 1 天
    - 整数手验证: 所有 positions 对应的股数为 100 的倍数
    - 涨跌停验证: 涨停日买入被阻断, 跌停日卖出被阻断

用法:
    python -m backtest.engine
"""

import numpy as np
import pandas as pd

from backtest.constraints import AShareCost, default_a_share_constraints
from backtest.core import run_core

# ============================================================
# 通用回测引擎
# ============================================================

def run_backtest(
    prices: pd.Series,
    num_units: pd.Series,
    initial_capital: float = 1_000_000.0,
    lot_size: int = 100,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.0005,
    slippage_rate: float = 0.001,
    dynamic_sizing: bool = True,
    board: str = "main",
    check_limits: bool = True,
    check_suspension: bool = False,
    price_data: pd.DataFrame | None = None,
) -> dict:
    """
    基于 Chan 公式的回测引擎，含 T+1 执行、整数手约束、交易成本、涨跌停/停牌检查。

    核心逻辑:
        1. T+1 执行: num_units(t-1) 决定 positions(t)
        2. 整数手:   实际股数 = round_to_lot(num_units × capital / price, lot)
        3. 涨跌停:   涨停日不可买入, 跌停日不可卖出
        4. 停牌:     停牌日不执行交易, 沿用前一日持仓
        5. PnL:      pnl(t) = shares(t-1) × (price(t) - price(t-1))
        6. 成本:     每次持仓变化时扣除买入/卖出佣金、印花税、滑点

    参数:
        prices:           日价格序列 (pd.Series, index 为日期, 收盘价)
        num_units:        仓位单元序列 (pd.Series, 与 prices 同 index)
                          含义: 策略输出的持仓目标 (非负, 不可做空)
                          例如: Z-Score 策略中 num_units = -Z
        initial_capital:  初始资金 (元), 默认 1,000,000
        lot_size:         每手股数, 默认 100 (A股)
        commission_rate:  佣金费率, 默认 万2.5
        stamp_tax_rate:   印花税率, 默认 万5 (仅卖出)
        slippage_rate:    滑点费率, 默认 千1 (双边)
        dynamic_sizing:   动态仓位管理, 默认 True.
                          True:  用当前权益 equity(t-1) 计算仓位 (复利)
                          False: 用初始资金 initial_capital 计算仓位 (固定)
        board:            板块类型, 用于涨跌停计算
                          "main"=主板(±10%), "chinext"=创业板(±20%),
                          "star"=科创板(±20%), "st"=ST股(±5%),
                          "bse"=北交所(±30%)
        check_limits:     是否启用涨跌停检查, 默认 True
        check_suspension: 是否启用停牌检查, 默认 False
        price_data:       含 open/high/low/close/volume 的 DataFrame,
                          check_suspension=True 时必填, 用于判断停牌/零成交日

    返回:
        dict: {
            "positions":      每日持仓股数 (pd.Series, int),
            "pnl":            每日盈亏 (pd.Series, float),
            "ret":            每日收益率 (pd.Series, float),
            "equity_curve":   权益曲线 (pd.Series, float),
            "n_trades":       交易次数 (int, shares 变化的天数),
            "total_cost":     总交易成本 (float),
        }

    示例:
        >>> prices = pd.Series([10.0, 10.5, 10.3, 10.8],
        ...                    index=pd.date_range("2024-01-01", periods=4))
        >>> num_units = pd.Series([0.0, 1.0, 1.0, 0.0], index=prices.index)
        >>> result = run_backtest(prices, num_units)
        >>> result["n_trades"]
        2
    """
    # ---- 输入验证 ----
    if not isinstance(prices, pd.Series):
        raise TypeError("prices 必须是 pd.Series")
    if not isinstance(num_units, pd.Series):
        raise TypeError("num_units 必须是 pd.Series")
    if not prices.index.equals(num_units.index):
        raise ValueError("prices 和 num_units 必须使用相同的 index")
    if len(prices) == 0:
        return {
            "positions": pd.Series([], dtype=int),
            "pnl": pd.Series([], dtype=float),
            "ret": pd.Series([], dtype=float),
            "equity_curve": pd.Series([], dtype=float),
            "n_trades": 0,
            "total_cost": 0.0,
        }

    # ---- 停牌检查参数校验 ----
    if check_suspension:
        if price_data is None:
            raise ValueError("check_suspension=True 时必须提供 price_data (含 volume 的 DataFrame)")
        if "volume" not in price_data.columns:
            raise ValueError("price_data 必须包含 'volume' 列用于停牌检查")
        if not prices.index.equals(price_data.index):
            raise ValueError("prices 和 price_data 必须使用相同的 index")

    # ---- 创建成本模型 ----
    cost_model = AShareCost(
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_rate=slippage_rate,
        min_commission=5.0,
    )

    # ---- 创建约束列表 ----
    constraints = default_a_share_constraints(
        check_limits=check_limits,
        check_suspension=check_suspension,
        price_data=price_data,
        board=board,
    )

    # ---- 委托给 run_core（传递原始 num_units，run_core 内部处理 T+1）----
    return run_core(
        prices=prices,
        num_units=num_units,
        initial_capital=initial_capital,
        constraints=constraints,
        cost_model=cost_model,
        dynamic_sizing=dynamic_sizing,
        lot_size=lot_size,
    )


# ============================================================
# 仅做多便捷封装
# ============================================================

def run_backtest_long_only(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 1_000_000.0,
    lot_size: int = 100,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.0005,
    slippage_rate: float = 0.001,
    dynamic_sizing: bool = True,
    board: str = "main",
    check_limits: bool = True,
    check_suspension: bool = False,
    price_data: pd.DataFrame | None = None,
) -> dict:
    """
    仅做多回测的便捷封装，signals ∈ {0, 1}。

    signal=1 时满仓买入 (shares = round_to_lot(capital / price))
    signal=0 时空仓 (shares = 0)

    参数:
        prices:   日价格序列 (pd.Series)
        signals:  交易信号序列 (pd.Series, 仅含 0 和 1)
        board:    板块类型, 用于涨跌停计算
        check_limits:     是否启用涨跌停检查
        check_suspension: 是否启用停牌检查
        price_data:       check_suspension=True 时必填
        *args:    其余参数同 run_backtest

    返回:
        同 run_backtest 的 dict

    示例:
        >>> prices = pd.Series([10.0, 10.5, 10.3, 10.8],
        ...                    index=pd.date_range("2024-01-01", periods=4))
        >>> signals = pd.Series([0, 1, 1, 0], index=prices.index)
        >>> result = run_backtest_long_only(prices, signals)
    """
    # signal=1 → num_units=1 → shares = capital / price (满仓)
    units = signals.astype(float)
    return run_backtest(
        prices, units,
        initial_capital=initial_capital,
        lot_size=lot_size,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_rate=slippage_rate,
        dynamic_sizing=dynamic_sizing,
        board=board,
        check_limits=check_limits,
        check_suspension=check_suspension,
        price_data=price_data,
    )


# ============================================================
# 成本计算辅助函数
# ============================================================


def _compute_cost(
    trade_amount: float,
    direction: str,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.0005,
    slippage_rate: float = 0.001,
    min_commission: float = 5.0,
) -> float:
    """
    计算 A 股交易成本（纯函数）。

    Args:
        trade_amount:     交易金额（元）
        direction:        "buy" 或 "sell"
        commission_rate:  佣金费率
        stamp_tax_rate:   印花税率（仅卖出）
        slippage_rate:    滑点费率（双边）
        min_commission:   最低佣金

    Returns:
        总交易成本（元）
    """
    commission = max(trade_amount * commission_rate, min_commission)
    stamp = trade_amount * stamp_tax_rate if direction == "sell" else 0.0
    slippage = trade_amount * slippage_rate
    return commission + stamp + slippage


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
    验证协议: 构造已知数据，断言引擎输出与手算一致。
    """
    print("=" * 60)
    print("回测引擎验证协议")
    print("=" * 60)

    all_pass = True

    # ============================================================
    # 测试 1: 恒定价格 + 恒定持仓 → PnL = 0
    # ============================================================
    print("\n【正控】恒定价格 + 恒定持仓 → PnL = 0")
    print("-" * 60)

    dates = pd.date_range("2024-01-01", periods=5)
    p_const = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=dates)
    u_const = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)

    result = run_backtest(p_const, u_const, dynamic_sizing=False)

    # PnL 应为全零 (价格不变)
    pnl_sum = float(result["pnl"].sum())
    pass_const = _is_close(pnl_sum, 0.0)
    if not pass_const:
        all_pass = False
    print(f"  PnL 总和 = {pnl_sum:.6f}, 预期 = 0.0  [{'PASS' if pass_const else 'FAIL'}]")

    # 权益曲线应 = 初始资金 - 成本 (仅首日买入成本)
    eq_final = float(result["equity_curve"].iloc[-1])
    print(f"  最终权益 = {eq_final:,.2f}")
    print(f"  交易次数 = {result['n_trades']}")
    print(f"  总成本   = {result['total_cost']:,.2f}")

    # ============================================================
    # 测试 2: 已知价格序列 + 已知信号 → PnL 与手算一致 (含成本)
    # ============================================================
    print("\n【正控】已知价格序列 + 已知信号 → PnL 与手算一致")
    print("-" * 60)

    dates2 = pd.date_range("2024-01-01", periods=4)
    # 价格: 10 → 11 (+10%) → 9 (-18.18%) → 12 (+33.33%)
    p_known = pd.Series([10.0, 11.0, 9.0, 12.0], index=dates2)
    # 信号: 0, 1, 1, 0
    # T+1 后: 持仓 = [0, 0, 1, 1] → 在 t=2 和 t=3 日持有
    # 注意: num_units 直接传入, 不做 0/1 转换
    # 为了方便手算, 这里直接用 num_units 而非 run_backtest_long_only
    # num_units: [0, 1, 1, 0] → 持仓 = [0, 0, 1, 1]
    u_known = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates2)
    result2 = run_backtest(p_known, u_known, initial_capital=1_000_000, dynamic_sizing=False, check_limits=False)

    # 手算:
    # t=0: signal=0, 持仓=0 (T+1: signal从NaN来, fillna(0))
    # t=1: signal=0 (T+1: 执行t=0的num_units=0), 持仓=0
    # t=2: signal=1 (T+1: 执行t=1的num_units=1), 持仓: 1000000/9 ≈ 111111.1股
    #      → round_to_lot: 111100 股
    #      成本: 买入 111100 * 9 = 999,900 元
    #      佣金 = max(999900 * 0.00025, 5) = 249.975
    #      印花税 = 0
    #      滑点 = 999900 * 0.001 = 999.9
    #      总成本 = 249.975 + 999.9 = 1249.875
    # t=3: signal=0 (T+1: 执行t=2的num_units=1), 持仓: 1000000/12 ≈ 83333.33股
    #      → round_to_lot: 83300 股
    #      但 t=3 的 lagged_units 来自 t=2 的 num_units=1, 所以持仓仍是满仓
    #      实际上: lagged_units = [0, 0, 1, 1], 所以 t=3 持仓 = 83300
    #      同时, t=3 卖出 t=2 买入的 111100 股 → delta = 83300 - 111100 = -27800
    #      卖出金额 = 27800 * 12 = 333,600
    #      佣金 = max(333600 * 0.00025, 5) = 83.4
    #      印花税 = 333600 * 0.0005 = 166.8
    #      滑点 = 333600 * 0.001 = 333.6
    #      总卖出成本 = 83.4 + 166.8 + 333.6 = 583.8
    #
    # 但等等, t=4 (最后一天)没有后续了, 我们卖不出去。只有 t=3 的持仓变化触发成本。
    # 另外 t=5 不存在...
    #
    # 手算 PnL:
    # t=0: pnl=0 (首日无持仓)
    # t=1: pnl=0 (持仓=0)
    # t=2: pnl=0 (t=1 持仓=0)
    # t=3: shares(2)=111100, pnl(3)=111100*(12-9)=111100*3=333,300
    #
    # 手算权益:
    # t=0: 1,000,000
    # t=1: 1,000,000
    # t=2: 1,000,000 - 1249.875 = 998,750.125
    # t=3: 998,750.125 + 333,300 - 583.8 = 1,331,466.325

    # 手算 PnL
    expected_pnl = pd.Series([0.0, 0.0, 0.0, 333300.0], index=dates2)
    expected_equity = pd.Series([1_000_000.0, 1_000_000.0, 998_750.125, 1_331_466.325], index=dates2)

    pass_pnl = all(
        _is_close(result2["pnl"].iloc[i], expected_pnl.iloc[i])
        for i in range(4)
    )
    if not pass_pnl:
        all_pass = False
    for i in range(4):
        print(f"  PnL[{i}]:  actual={result2['pnl'].iloc[i]:,.2f}  "
              f"expected={expected_pnl.iloc[i]:,.2f}  "
              f"[{'PASS' if _is_close(result2['pnl'].iloc[i], expected_pnl.iloc[i]) else 'FAIL'}]")

    # 直接比较成本不便 (分到各天不同), 比较总成本
    total_cost_expected = 1249.875 + 583.8
    pass_total = _is_close(result2["total_cost"], total_cost_expected, tol=1.0)
    if not pass_total:
        all_pass = False
    print(f"  总成本: actual={result2['total_cost']:,.4f}  "
          f"expected={total_cost_expected:,.4f}  "
          f"[{'PASS' if pass_total else 'FAIL'}]")

    # 手算权益
    for i in range(4):
        eq_ok = _is_close(result2["equity_curve"].iloc[i], expected_equity.iloc[i], tol=1.0)
        if not eq_ok:
            all_pass = False
        print(f"  Equity[{i}]: actual={result2['equity_curve'].iloc[i]:,.2f}  "
              f"expected={expected_equity.iloc[i]:,.2f}  "
              f"[{'PASS' if eq_ok else 'FAIL'}]")

    # ============================================================
    # 测试 3: T+1 验证 — positions 比 signals 滞后 1 天
    # ============================================================
    print("\n【正控】T+1 验证 — positions 比 signals 滞后 1 天")
    print("-" * 60)

    dates3 = pd.date_range("2024-01-01", periods=5)
    p_t1 = pd.Series([10.0, 10.5, 10.3, 10.8, 10.0], index=dates3)
    u_t1 = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=dates3)
    result3 = run_backtest(p_t1, u_t1, initial_capital=1_000_000, dynamic_sizing=False)

    # num_units:        [0,   1,   1,   1,   0]
    # lagged (执行):    [0,   0,   1,   1,   1]
    # 所以:
    #   t=0: 持仓=0 (num_units[0]=0 → T+1 在 t=1 执行, 所以 t=0 是 NaN→fillna(0)=0)
    #   t=1: 持仓=0 (num_units[0]=0)
    #   t=2: 持仓>0 (num_units[1]=1) ← 信号日 t=1, 执行日 t=2
    #   t=3: 持仓>0 (num_units[2]=1)
    #   t=4: 持仓>0 (num_units[3]=1)
    positions = result3["positions"]
    # t=0 应为 0, t=1 应为 0, t=2 应 >0, t=3 应 >0, t=4 应 >0
    pass_t1 = (
        positions.iloc[0] == 0
        and positions.iloc[1] == 0
        and positions.iloc[2] > 0
        and positions.iloc[3] > 0
        and positions.iloc[4] > 0
    )
    if not pass_t1:
        all_pass = False
    for i in range(5):
        status = "持仓" if positions.iloc[i] > 0 else "空仓"
        print(f"  t={i}: num_units={u_t1.iloc[i]:.0f}, positions={positions.iloc[i]:,d}股 ({status})")
    print(f"  [{'PASS' if pass_t1 else 'FAIL'}] T+1 验证")

    # ============================================================
    # 测试 4: 整数手验证 — 所有 positions 对应的股数为 100 的倍数
    # ============================================================
    print("\n【正控】整数手验证 — 所有 positions 为 100 的倍数")
    print("-" * 60)

    dates4 = pd.date_range("2024-01-01", periods=3)
    p_lot = pd.Series([10.0, 10.5, 10.3], index=dates4)
    u_lot = pd.Series([0.0, 1.5, 0.8], index=dates4)
    result4 = run_backtest(p_lot, u_lot, lot_size=100, initial_capital=500_000, dynamic_sizing=False)

    pass_lot = all(s % 100 == 0 for s in result4["positions"])
    if not pass_lot:
        all_pass = False
    for i in range(3):
        raw = u_lot.iloc[i] * 500_000 / p_lot.iloc[i] if u_lot.iloc[i] > 0 else 0
        ok = result4["positions"].iloc[i] % 100 == 0
        print(f"  t={i}: raw_shares={raw:,.1f} → positions={result4['positions'].iloc[i]:,d}股 "
              f"[{'PASS' if ok else 'FAIL'}]")
    print(f"  [{'PASS' if pass_lot else 'FAIL'}] 整数手验证")

    # ============================================================
    # 测试 5: run_backtest_long_only 便捷封装
    # ============================================================
    print("\n【集成】run_backtest_long_only 便捷封装")
    print("-" * 60)

    dates5 = pd.date_range("2024-01-01", periods=4)
    p5 = pd.Series([10.0, 11.0, 9.0, 12.0], index=dates5)
    s5 = pd.Series([0, 1, 1, 0], index=dates5)
    result5 = run_backtest_long_only(p5, s5, dynamic_sizing=False, check_limits=False)

    # 应该等价于 num_units = signals
    result5b = run_backtest(p5, s5.astype(float), dynamic_sizing=False, check_limits=False)
    pass_wrapper = (
        result5["positions"].equals(result5b["positions"])
        and _is_close(result5["total_cost"], result5b["total_cost"], tol=0.01)
    )
    if not pass_wrapper:
        all_pass = False
    print(f"  便捷封装 vs 直接调用: 持仓一致={result5['positions'].equals(result5b['positions'])}")
    print(f"  [{'PASS' if pass_wrapper else 'FAIL'}]")

    # ============================================================
    # 测试 6: 空输入
    # ============================================================
    print("\n【边界】空 Series")
    print("-" * 60)

    empty_p = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    empty_u = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    result6 = run_backtest(empty_p, empty_u)
    pass_empty = (
        len(result6["positions"]) == 0
        and result6["n_trades"] == 0
        and result6["total_cost"] == 0.0
    )
    if not pass_empty:
        all_pass = False
    print(f"  [{'PASS' if pass_empty else 'FAIL'}]")

    # ============================================================
    # 测试 7: 全零 num_units → 无交易
    # ============================================================
    print("\n【边界】全零 num_units → 无交易")
    print("-" * 60)

    dates7 = pd.date_range("2024-01-01", periods=5)
    p7 = pd.Series([10.0, 10.5, 10.3, 10.8, 10.0], index=dates7)
    u7 = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates7)
    result7 = run_backtest(p7, u7)
    pass_zero = (
        result7["n_trades"] == 0
        and result7["total_cost"] == 0.0
        and float(result7["pnl"].sum()) == 0.0
        and _is_close(float(result7["equity_curve"].iloc[-1]), 1_000_000.0)
    )
    if not pass_zero:
        all_pass = False
    print(f"  交易次数={result7['n_trades']}, 总成本={result7['total_cost']:.2f}")
    print(f"  最终权益={float(result7['equity_curve'].iloc[-1]):,.2f}")
    print(f"  [{'PASS' if pass_zero else 'FAIL'}]")

    # ============================================================
    # 测试 8: 动态仓位 (复利) vs 固定仓位
    # ============================================================
    print("\n【正控】动态仓位 (复利) vs 固定仓位")
    print("-" * 60)

    dates8 = pd.date_range("2024-01-01", periods=6)
    p8 = pd.Series([10.0, 10.0, 10.0, 20.0, 20.0, 21.0], index=dates8)
    u8 = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=dates8)

    res_dyn = run_backtest(
        p8, u8, initial_capital=1000, lot_size=1,
        commission_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0,
        dynamic_sizing=True, check_limits=False,
    )
    res_fix = run_backtest(
        p8, u8, initial_capital=1000, lot_size=1,
        commission_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0,
        dynamic_sizing=False, check_limits=False,
    )

    # 手算 (含 min_commission=5):
    # 固定: t=2 buy cost=5 eq=995, t=3 sell cost=5 pnl=1000 eq=1990
    #        t=5 sell cost=5 pnl=50 eq=2035
    # 动态: t=2 buy cost=5 eq=995, t=3 capital=995 shares=49 pnl=1000 eq=1990
    #        t=4 capital=1990 shares=99 buy cost=5 eq=1985
    #        t=5 capital=1985 shares=94 pnl=99 eq=2079
    dyn_eq = float(res_dyn["equity_curve"].iloc[-1])
    fix_eq = float(res_fix["equity_curve"].iloc[-1])
    dyn_pnl5 = float(res_dyn["pnl"].iloc[-1])
    fix_pnl5 = float(res_fix["pnl"].iloc[-1])

    pass_dyn = (
        _is_close(dyn_eq, 2079.0)
        and _is_close(fix_eq, 2035.0)
        and _is_close(dyn_pnl5, 99.0)
        and _is_close(fix_pnl5, 50.0)
        and dyn_eq > fix_eq
    )
    if not pass_dyn:
        all_pass = False

    print(f"  动态仓位: 最终权益={dyn_eq:,.0f} (期望 2079), 末日PnL={dyn_pnl5:.0f} (期望 99)")
    print(f"  固定仓位: 最终权益={fix_eq:,.0f} (期望 2035), 末日PnL={fix_pnl5:.0f} (期望 50)")
    print(f"  复利效应: 动态比固定多 {dyn_eq - fix_eq:,.0f} 元")
    print(f"  [{'PASS' if pass_dyn else 'FAIL'}]")

    # ============================================================
    # 测试 9: 涨跌停 — 涨停日不可买入
    # ============================================================
    print("\n【A股约束】涨停日不可买入")
    print("-" * 60)

    dates9 = pd.date_range("2024-01-01", periods=4)
    # 主板 ±10%: 10 -> 11 为涨停
    p_limit_up = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates9)
    # 只在 day 1 发一次买入信号, day 2 执行时涨停被阻断
    u_limit_up = pd.Series([0.0, 1.0, 0.0, 0.0], index=dates9)
    result9 = run_backtest(
        p_limit_up, u_limit_up, initial_capital=1_000_000,
        dynamic_sizing=False, board="main", check_limits=True,
    )

    # T+1: signal[1]=1 在 day 2 执行, day 2 涨停, 买入应被阻断
    # positions 应为 [0, 0, 0, 0]
    pass_limit_up = (
        (result9["positions"] == 0).all()
        and result9["n_trades"] == 0
        and result9["total_cost"] == 0.0
    )
    if not pass_limit_up:
        all_pass = False
    print(f"  positions={list(result9['positions'])} 期望 [0, 0, 0, 0]")
    print(f"  [{'PASS' if pass_limit_up else 'FAIL'}] 涨停买入被阻断")

    # ============================================================
    # 测试 10: 涨跌停 — 跌停日不可卖出
    # ============================================================
    print("\n【A股约束】跌停日不可卖出")
    print("-" * 60)

    dates10 = pd.date_range("2024-01-01", periods=5)
    # 主板 ±10%: 10 -> 9 为跌停
    p_limit_down = pd.Series([10.0, 10.0, 9.0, 10.0, 10.0], index=dates10)
    # day 0 买入, day 1 卖出, day 2 执行卖出时跌停被阻断, day 3 执行卖出成功
    u_limit_down = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=dates10)
    result10 = run_backtest(
        p_limit_down, u_limit_down, initial_capital=1_000_000,
        dynamic_sizing=False, board="main", check_limits=True,
    )

    # day 2 跌停, 卖出被阻断 → positions[2] 保持买入仓位
    # day 3 不跌停, 卖出成功 → positions[3] = 0
    bought = result10["positions"].iloc[1] > 0
    blocked_day2 = result10["positions"].iloc[2] == result10["positions"].iloc[1]
    sold_day3 = result10["positions"].iloc[3] == 0
    pass_limit_down = bought and blocked_day2 and sold_day3
    if not pass_limit_down:
        all_pass = False
    print(f"  day1 买入={bought}, day2 跌停卖出被阻断={blocked_day2}, day3 卖出={sold_day3}")
    print(f"  positions={list(result10['positions'])}")
    print(f"  [{'PASS' if pass_limit_down else 'FAIL'}] 跌停卖出被阻断")

    # ============================================================
    # 测试 11: 停牌日不执行交易
    # ============================================================
    print("\n【A股约束】停牌日不执行交易")
    print("-" * 60)

    dates11 = pd.date_range("2024-01-01", periods=5)
    p_susp = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=dates11)
    u_susp = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=dates11)
    # day 2 (index=2) 停牌: volume=0
    price_data_susp = pd.DataFrame({
        "open": [10.0, 10.0, 10.0, 10.0, 10.0],
        "high": [10.0, 10.0, 10.0, 10.0, 10.0],
        "low": [10.0, 10.0, 10.0, 10.0, 10.0],
        "close": [10.0, 10.0, 10.0, 10.0, 10.0],
        "volume": [1000, 1000, 0, 1000, 1000],
    }, index=dates11)
    result11 = run_backtest(
        p_susp, u_susp, initial_capital=1_000_000,
        dynamic_sizing=False, check_limits=False,
        check_suspension=True, price_data=price_data_susp,
    )

    # signal[1]=1 在 day 2 执行, day 2 停牌 → 买入被阻断
    # signal[2]=1 在 day 3 执行, day 3 可交易 → 买入
    # signal[3]=0 在 day 4 执行 → 卖出
    pass_susp = (
        result11["positions"].iloc[2] == 0   # 停牌日未买入
        and result11["positions"].iloc[3] > 0  # 复牌后买入
        and result11["positions"].iloc[4] == 0  # 随后卖出
    )
    if not pass_susp:
        all_pass = False
    print(f"  positions={list(result11['positions'])}")
    print(f"  [{'PASS' if pass_susp else 'FAIL'}] 停牌日交易被阻断")

    # ============================================================
    # 测试 12: 板块参数 — 创业板 ±20%
    # ============================================================
    print("\n【A股约束】创业板 ±20% 涨跌停")
    print("-" * 60)

    dates12 = pd.date_range("2024-01-01", periods=3)
    # 主板 10 -> 11 为涨停 (+10%), 创业板 10 -> 11 不是涨停 (上限 12)
    p_chinext = pd.Series([10.0, 10.0, 11.0], index=dates12)
    u_chinext = pd.Series([0.0, 1.0, 0.0], index=dates12)

    # 主板: 涨停, 不可买入
    result_main = run_backtest(
        p_chinext, u_chinext, initial_capital=1_000_000,
        dynamic_sizing=False, board="main", check_limits=True,
    )
    # 创业板: 可买入
    result_chinext = run_backtest(
        p_chinext, u_chinext, initial_capital=1_000_000,
        dynamic_sizing=False, board="chinext", check_limits=True,
    )

    pass_board = (
        result_main["positions"].iloc[2] == 0
        and result_chinext["positions"].iloc[2] > 0
    )
    if not pass_board:
        all_pass = False
    print(f"  主板买入: {result_main['positions'].iloc[2] > 0}")
    print(f"  创业板涨停买入被阻断: {result_chinext['positions'].iloc[2] == 0}")
    print(f"  [{'PASS' if pass_board else 'FAIL'}]")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 回测引擎验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    success = run_validation()
    if not success:
        raise SystemExit(1)
