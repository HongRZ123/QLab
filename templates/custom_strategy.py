"""
custom_strategy.py - 自定义策略开发模板
========================================

演示如何从零开发一个新策略，接入注册表，并做验证。

用法:
    1. 复制此文件到 strategies/ 目录
    2. 实现你的策略函数 (返回 dict, 必须含 "num_units")
    3. 实现验证协议 (正控 + 负控 + 不变式)
    4. 在 strategies/registry.py 中注册

策略函数要求:
    - 签名: (prices: pd.Series, **kwargs) -> dict
    - 返回值必须含 "num_units" 键 (pd.Series, 非负, 仅做多)
    - 可选返回: pnl, ret, signals 等中间结果

验证协议要求:
    - 正控: OU 均值回归序列上, 策略应该赚钱 (Sharpe > 0)
    - 负控: GBM 随机游走上, 策略应该不赚钱 (Sharpe < 0.5)
    - 不变式: num_units >= 0, 前若干天 = 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# 示例策略: RSI 均值回归 (简化版)
# ============================================================

def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI (Wilder 平滑)。"""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_mean_reversion(
    prices: pd.Series,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> dict:
    """
    RSI 均值回归策略 (仅做多)。

    参数:
        prices:     日价格序列
        period:     RSI 回望期
        oversold:   超卖阈值 (低于买入)
        overbought: 超买阈值 (高于卖出)

    返回:
        dict: {rsi, num_units, signals}
    """
    if oversold >= overbought:
        raise ValueError(f"oversold ({oversold}) 必须 < overbought ({overbought})")

    y = prices.astype(float)

    # Step 1: RSI
    rsi = _compute_rsi(y, period)

    # Step 2: 信号 (仅做多)
    signals = pd.Series(np.nan, index=y.index, dtype=float)
    signals.loc[rsi < oversold] = 1.0       # 超卖 -> 买入
    signals.loc[rsi > overbought] = 0.0     # 超买 -> 卖出

    # Step 3: forward-fill
    if np.isnan(signals.iloc[0]):
        signals.iloc[0] = 0.0
    num_units = signals.ffill()

    # Step 4: 理论 PnL (无成本, 仅用于验证)
    positions_lag = num_units.shift(1).fillna(0.0)
    price_ret = y.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pnl = positions_lag * price_ret
    gross_mv = positions_lag.abs()
    ret = np.where(gross_mv > 1e-12, pnl / gross_mv, 0.0)

    return {
        "rsi": rsi,
        "num_units": num_units,
        "signals": signals,
        "pnl": pnl,
        "ret": pd.Series(ret, index=y.index),
    }


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """
    验证协议: 正控 + 负控 + 不变式。

    返回:
        bool: 全部通过返回 True
    """
    from tests.s3_half_life import generate_gbm_paths, generate_ou_paths

    print("=" * 60)
    print("  RSI 均值回归策略 - 验证协议")
    print("=" * 60)
    all_pass = True

    # ── 正控: OU 均值回归序列 ──
    print("\n【正控】OU(θ=0.05, T=2000) -> Sharpe > 0")
    print("-" * 60)

    ou_prices = pd.Series(generate_ou_paths(1, 2000, theta=0.05, mu=100.0, sigma=0.5)[0])
    res_pos = rsi_mean_reversion(ou_prices, period=14, oversold=30, overbought=70)

    ret_pos = res_pos["ret"].dropna()
    sharpe_pos = float(ret_pos.mean() / ret_pos.std() * np.sqrt(252)) if ret_pos.std() > 0 else 0.0
    cum_pnl = float(res_pos["pnl"].sum())

    pos_ok = sharpe_pos > 0
    print(f"  Sharpe = {sharpe_pos:.4f}  (要求 > 0)  [{'PASS' if pos_ok else 'FAIL'}]")
    print(f"  累计 PnL = {cum_pnl:.4f}")
    if not pos_ok:
        all_pass = False

    # ── 负控: GBM 随机游走 ──
    print("\n【负控】GBM(T=2000) -> Sharpe < 0.5")
    print("-" * 60)

    gbm_prices = pd.Series(generate_gbm_paths(1, 2000, sigma=0.02)[0])
    res_neg = rsi_mean_reversion(gbm_prices, period=14, oversold=30, overbought=70)

    ret_neg = res_neg["ret"].dropna()
    sharpe_neg = float(ret_neg.mean() / ret_neg.std() * np.sqrt(252)) if ret_neg.std() > 0 else 0.0

    neg_ok = sharpe_neg < 0.5
    print(f"  Sharpe = {sharpe_neg:.4f}  (要求 < 0.5)  [{'PASS' if neg_ok else 'FAIL'}]")
    if not neg_ok:
        all_pass = False

    # ── 不变式: num_units >= 0 ──
    print("\n【不变式】num_units >= 0")
    print("-" * 60)

    nu_ok = bool((res_pos["num_units"] >= 0).all() and (res_neg["num_units"] >= 0).all())
    print(f"  全部 >= 0: {nu_ok}  [{'PASS' if nu_ok else 'FAIL'}]")
    if not nu_ok:
        all_pass = False

    # ── 不变式: num_units ∈ {0, 1} ──
    print("\n【不变式】num_units ∈ {0, 1}")
    print("-" * 60)

    vals = sorted(res_pos["num_units"].unique())
    binary_ok = all(v in (0.0, 1.0) for v in vals)
    print(f"  取值: {vals}  [{'PASS' if binary_ok else 'FAIL'}]")
    if not binary_ok:
        all_pass = False

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    if all_pass:
        print("  [PASS] RSI 策略验证通过")
    else:
        print("  [FAIL] RSI 策略验证未通过")
    print("=" * 60)

    return all_pass


# ============================================================
# 注册到 Registry (取消注释即可)
# ============================================================

# from strategies.registry import Strategy, register
#
# register(Strategy(
#     name="rsi_mr",
#     fn=rsi_mean_reversion,
#     description="RSI 均值回归策略",
#     default_kwargs={"period": 14, "oversold": 30.0, "overbought": 70.0},
# ))


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    run_validation()
