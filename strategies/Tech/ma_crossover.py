"""
ma_crossover.py — 均线金叉死叉策略（仅做多）
=============================================

最简均值回归/趋势策略，用于测试数据管道:
    - 短期均线上穿长期均线 (金叉) → num_units = 1 (买入)
    - 短期均线下穿长期均线 (死叉) → num_units = 0 (卖出)
    - num_units ∈ {0, 1}  ← 仅做多

核心函数:
    ma_crossover(prices, short_window=5, long_window=20) -> dict

验证协议:
    - 正控: 趋势序列 (先涨后跌) → 金叉买入, 死叉卖出, 交易次数 ≥ 1
    - 负控: 恒定价格 → 无交叉, num_units 全 0
    - num_units 全部 ∈ {0, 1}

用法:
    python -m strategies.ma_crossover
"""

import numpy as np
import pandas as pd

# ============================================================
# 核心函数
# ============================================================

def ma_crossover(
    prices: pd.Series,
    short_window: int = 5,
    long_window: int = 20,
) -> dict:
    """
    均线金叉死叉策略（仅做多）。

    金叉: SMA(short) 上穿 SMA(long) → 买入 (num_units = 1)
    死叉: SMA(short) 下穿 SMA(long) → 卖出 (num_units = 0)

    参数:
        prices:       日价格序列 (pd.Series)
        short_window: 短期均线窗口 (天), 默认 5
        long_window:  长期均线窗口 (天), 默认 20

    返回:
        dict: {
            sma_short   : pd.Series  — 短期均线
            sma_long    : pd.Series  — 长期均线
            num_units   : pd.Series  — 仓位单元, ∈ {0, 1}
            signals     : pd.Series  — 交叉信号 (1=金叉, -1=死叉, 0=无)
            pnl         : pd.Series  — 理论每日盈亏 (无交易成本、无 T+1,
                                       仅用于验证; 生产回测请用 backtest.run_backtest)
            ret         : pd.Series  — 理论每日收益率 (同上)
            n_trades    : int        — 往返交易次数
        }

    示例:
        >>> prices = pd.Series([10.0, 10.5, 11.0, 10.8, 10.2, 9.8, 10.0, 10.5])
        >>> result = ma_crossover(prices, short_window=2, long_window=4)
        >>> result["num_units"].isin([0, 1]).all()
        True
    """
    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) 必须 < long_window ({long_window})"
        )

    y = prices.astype(float)

    # ── Step 1: 均线 ──
    sma_short = y.rolling(window=short_window, min_periods=short_window).mean()
    sma_long = y.rolling(window=long_window, min_periods=long_window).mean()

    # ── Step 2: 交叉信号 ──
    # 金叉: 短均线从下方穿越长均线
    # 死叉: 短均线从上方穿越长均线
    above = (sma_short > sma_long).astype(int)
    above_prev = above.shift(1).fillna(0).astype(int)

    signals = pd.Series(0, index=y.index, name="signals")
    signals[(above == 1) & (above_prev == 0)] = 1    # 金叉
    signals[(above == 0) & (above_prev == 1)] = -1   # 死叉

    # ── Step 3: 仓位 (仅做多, forward-fill) ──
    # 金叉 → 1, 死叉 → 0, 无信号 → 沿用前一日
    num_units = pd.Series(np.nan, index=y.index, name="num_units")
    num_units[signals == 1] = 1.0
    num_units[signals == -1] = 0.0
    num_units = num_units.ffill().fillna(0.0)

    # ── Step 4: 理论 PnL (Chan 公式, 无成本) ──
    mkt_val = num_units * y
    mkt_val_lag = mkt_val.shift(1).fillna(0.0)
    price_ret = y.pct_change().fillna(0.0)
    pnl = mkt_val_lag * price_ret
    ret = pd.Series(0.0, index=y.index, name="ret")
    mask = mkt_val_lag.abs() > 1e-12
    ret[mask] = pnl[mask] / mkt_val_lag[mask].abs()

    # ── Step 5: 交易次数 ──
    pos_change = num_units.diff().fillna(0.0)
    n_trades = int((pos_change != 0).sum())

    return {
        "sma_short": sma_short,
        "sma_long": sma_long,
        "num_units": num_units,
        "signals": signals,
        "pnl": pnl,
        "ret": ret,
        "n_trades": n_trades,
    }


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """正控 + 负控 + num_units 验证协议。"""
    all_pass = True

    print("=" * 60)
    print("  均线金叉死叉策略 — 验证协议")
    print("=" * 60)

    # ── 正控: 趋势序列 (先涨后跌再涨) → 应有交叉 ──
    print("\n【正控】趋势序列 → 金叉/死叉")
    np.random.seed(42)
    n = 200
    # 构造: 涨 50 天 → 跌 50 天 → 涨 50 天 → 跌 50 天
    trend = np.concatenate([
        np.linspace(10, 15, 50),
        np.linspace(15, 10, 50),
        np.linspace(10, 15, 50),
        np.linspace(15, 10, 50),
    ])
    prices_trend = pd.Series(trend + np.random.randn(n) * 0.1)
    res_trend = ma_crossover(prices_trend, short_window=5, long_window=20)

    trades_ok = res_trend["n_trades"] >= 2
    print(f"  交易次数 = {res_trend['n_trades']}  (要求 ≥ 2)  "
          f"[{'PASS' if trades_ok else 'FAIL'}]")
    if not trades_ok:
        all_pass = False

    # 检查金叉后 num_units=1, 死叉后 num_units=0
    golden = res_trend["signals"] == 1
    death = res_trend["signals"] == -1
    if golden.any():
        first_golden_idx = golden.idxmax()
        golden_pos = res_trend["num_units"].loc[first_golden_idx]
        golden_ok = golden_pos == 1.0
        print(f"  金叉后 num_units = {golden_pos}  (要求 = 1.0)  "
              f"[{'PASS' if golden_ok else 'FAIL'}]")
        if not golden_ok:
            all_pass = False

    if death.any():
        first_death_idx = death.idxmax()
        death_pos = res_trend["num_units"].loc[first_death_idx]
        death_ok = death_pos == 0.0
        print(f"  死叉后 num_units = {death_pos}  (要求 = 0.0)  "
              f"[{'PASS' if death_ok else 'FAIL'}]")
        if not death_ok:
            all_pass = False

    # ── 负控: 恒定价格 → 无交叉 ──
    print("\n【负控】恒定价格 → 无交叉")
    prices_const = pd.Series([100.0] * 100)
    res_const = ma_crossover(prices_const, short_window=5, long_window=20)

    const_ok = res_const["n_trades"] == 0 and (res_const["num_units"] == 0).all()
    print(f"  交易次数 = {res_const['n_trades']}  (要求 = 0)  "
          f"[{'PASS' if const_ok else 'FAIL'}]")
    if not const_ok:
        all_pass = False

    # ── num_units ∈ {0, 1} 断言 ──
    print("\n【断言】num_units ∈ {0, 1}")
    units_ok = res_trend["num_units"].isin([0.0, 1.0]).all()
    print(f"  全部 ∈ {{0, 1}}  [{'PASS' if units_ok else 'FAIL'}]")
    if not units_ok:
        all_pass = False

    # ── 窗口期检查 ──
    print("\n【断言】窗口期 (前 long_window-1 天) num_units = 0")
    warmup_ok = (res_trend["num_units"].iloc[:19] == 0).all()
    print(f"  前 19 天全 0  [{'PASS' if warmup_ok else 'FAIL'}]")
    if not warmup_ok:
        all_pass = False

    # ── 参数校验 ──
    print("\n【断言】short_window >= long_window → ValueError")
    try:
        ma_crossover(prices_const, short_window=20, long_window=5)
        param_ok = False
    except ValueError:
        param_ok = True
    print(f"  抛出 ValueError  [{'PASS' if param_ok else 'FAIL'}]")
    if not param_ok:
        all_pass = False

    print("\n" + "=" * 60)
    status = "PASS" if all_pass else "FAIL"
    print(f"  [{status}] 均线金叉死叉验证{'通过' if all_pass else '未通过'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    run_validation()
