"""
s12_vpa_draft.py — VPA（量价分析）实验策略
==========================================

基于 signals.Tech.vpa 的量价确认信号生成交易决策。

信号逻辑:
- volume_confirmation = +2（看涨确认）→ num_units = 1
- volume_confirmation = -2（看跌确认）→ num_units = 0
- 其他（异常或中性）→ num_units = 0（保守）
- 额外确认：如果 wick_body_ratio.signal = +1（看涨反转）且 volume_confirmation >= 0 → num_units = 1

核心函数:
    vpa_strategy(prices, volume, lookback=20, **kwargs) -> dict

验证协议:
    - 正控：线性上涨价格 + 递增成交量 → num_units 中 >=50% 为 1
    - 断言：num_units ∈ {0, 1}

用法:
    python -m strategies.experimental.s12_vpa_draft
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import run_backtest
from signals.Tech.vpa import volume_confirmation, wick_body_ratio


def vpa_strategy(
    prices: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
    **kwargs,
) -> dict:
    """
    VPA 量价分析策略

    Args:
        prices: 价格序列
        volume: 成交量序列
        lookback: 滚动窗口

    Returns:
        dict: {num_units: pd.Series}，num_units ∈ {0, 1}
    """
    # 计算 VPA 信号
    vc = volume_confirmation(prices, volume, lookback=lookback)

    # 计算 K 线形态（需要 OHLC，这里简化为只用 close）
    # 实际应用中应该传入 open, high, low, close
    # 这里为了简化，假设 open = high = low = close（无 wick）
    open_prices = prices.copy()
    high = prices.copy()
    low = prices.copy()
    close = prices.copy()

    wbr = wick_body_ratio(open_prices, high, low, close)

    # 初始化 num_units
    num_units = pd.Series(0, index=prices.index, dtype=int)

    # 主要信号：volume_confirmation = +2 → 买入
    num_units[vc == 2] = 1

    # 额外确认：wick_body_ratio.signal = +1 且 volume_confirmation >= 0 → 买入
    additional_buy = (wbr["signal"] == 1) & (vc >= 0)
    num_units[additional_buy] = 1

    # 看跌确认：volume_confirmation = -2 → 卖出（已在初始化中设为 0）

    return {
        "num_units": num_units,
    }


def run_validation() -> bool:
    """VPA 策略验证协议"""
    print("=" * 60)
    print("VPA 策略验证协议")
    print("=" * 60)

    all_pass = True

    # 正控：线性上涨价格 + 递增成交量
    print("\n【正控】线性上涨价格 + 递增成交量")
    print("-" * 60)

    T = 100
    prices = pd.Series(np.linspace(10, 20, T))
    volume = pd.Series(np.linspace(1000, 2000, T))

    result = vpa_strategy(prices, volume, lookback=20)
    num_units = result["num_units"]

    # 检查 num_units ∈ {0, 1}
    unique_vals = set(num_units.unique())
    nu_ok = unique_vals.issubset({0, 1})
    print(f"  num_units 取值: {sorted(unique_vals)}  [{'PASS' if nu_ok else 'FAIL'}]")

    if not nu_ok:
        all_pass = False

    # 检查 >=50% 为 1
    ones_ratio = (num_units == 1).sum() / len(num_units)
    pos_ok = ones_ratio >= 0.50
    print(
        f"  num_units=1 占比: {ones_ratio:.2%} (要求 >= 50%)"
        f"  [{'PASS' if pos_ok else 'FAIL'}]"
    )

    if not pos_ok:
        all_pass = False

    # 回测验证 PnL > 0
    bt = run_backtest(
        prices, num_units, dynamic_sizing=False, check_limits=False,
    )
    pnl_positive = bt["pnl"].sum() > 0
    print(f"  回测 PnL: {bt['pnl'].sum():.2f}  [{'PASS' if pnl_positive else 'FAIL'}]")

    if not pnl_positive:
        all_pass = False

    # 汇总
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] VPA 策略验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = run_validation()
    if not success:
        raise SystemExit(1)
