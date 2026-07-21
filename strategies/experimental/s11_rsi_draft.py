"""
strategies/experimental/s11_rsi_draft.py — RSI 均值回归策略草案
================================================================

基于 RSI (Relative Strength Index) 的均值回归策略草稿。

状态: 半成品 — 仅用于研究和调试参数, 不保证通过 run_validation()。

核心思路:
    RSI < oversold  → 买入 (num_units = 1)
    RSI > overbought → 卖出 (num_units = 0)
    无信号日 → forward-fill 沿用前一日仓位

注意:
    - RSI 计算采用 Wilder 平滑 (指数移动平均)
    - 未实现 run_validation() 协议
    - 参数未经 Walk-Forward 优化, 仅供探索
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 RSI (Wilder 平滑)。

    参数:
        prices: 日价格序列
        period: RSI 回望期, 默认 14

    返回:
        pd.Series: RSI 值, 范围 [0, 100]
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder 平滑 (指数移动平均, alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def rsi_mean_reversion(
    prices: pd.Series,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> dict:
    """
    RSI 均值回归策略 (仅做多, 草案)。

    参数:
        prices:    日价格序列 (pd.Series)
        period:    RSI 回望期, 默认 14
        oversold:  超卖阈值, 低于此值买入, 默认 30
        overbought: 超买阈值, 高于此值卖出, 默认 70

    返回:
        dict: {
            rsi       : pd.Series  — RSI 值
            num_units : pd.Series  — 仓位单元, ∈ {0, 1}
            signals   : pd.Series  — 原始信号 (NaN=无信号日)
        }

    示例:
        >>> prices = pd.Series([10.0, 10.5, 10.3, 10.8, 10.0, 9.5, 9.8, 10.2])
        >>> result = rsi_mean_reversion(prices, period=3)
        >>> result["num_units"].isin([0, 1]).all()
        True
    """
    if oversold >= overbought:
        raise ValueError(
            f"oversold ({oversold}) 必须 < overbought ({overbought})"
        )

    y = prices.astype(float)

    # ── Step 1: RSI ──
    rsi = _compute_rsi(y, period)

    # ── Step 2: 信号 (仅做多) ──
    signals = pd.Series(np.nan, index=y.index, dtype=float)
    signals.loc[rsi < oversold] = 1.0      # 超卖 → 买入
    signals.loc[rsi > overbought] = 0.0    # 超买 → 卖出

    # ── Step 3: forward-fill ──
    if np.isnan(signals.iloc[0]):
        signals.iloc[0] = 0.0
    num_units = signals.ffill()

    return {
        "rsi": rsi,
        "num_units": num_units,
        "signals": signals,
    }


# ============================================================
# 草案: 无 run_validation()
# ============================================================

if __name__ == "__main__":
    print("[s11_rsi_draft] 草案策略 — 仅用于研究, 未实现 run_validation()")
    print("  用法: from strategies.experimental.s11_rsi_draft import rsi_mean_reversion")
