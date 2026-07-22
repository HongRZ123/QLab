"""
vpa_trend.py -- VPA-T1 量价确认趋势跟踪策略

基于《量价分析》Ch4 确认/异常方法论 + Ch8 趋势健康度。

策略逻辑：
    - 量价和谐确认（confirmed）且趋势健康 -> 满仓
    - 量价确认但趋势走弱 -> 半仓
    - 异常（anomaly）或陷阱（trap） -> 空仓
    - 仅做多：只在 close > open 时产生多头信号
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.trend import trend_health
from signals.vpa import vpa_confirmation_matrix


def vpa_trend(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
) -> dict:
    """
    VPA 量价确认趋势跟踪策略。

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 信号滚动窗口

    返回:
        dict: {
            "num_units": pd.Series,  # 仓位比例 (0~1)
            "signal": pd.Series,     # 原始信号分类
            "trend_health": pd.Series,  # 趋势健康度 (+1/-1/0)
        }
    """
    open_s = ohlcv["open"]
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    matrix = vpa_confirmation_matrix(open_s, close, volume, lookback=lookback)
    health = trend_health(close, volume, lookback=lookback)
    bullish = close > open_s

    num_units = pd.Series(0.0, index=close.index)

    # confirmed + 趋势健康 + 上涨 K 线 -> 满仓
    full_long = (matrix == "confirmed") & (health == 1) & bullish
    # confirmed + 趋势走弱 + 上涨 K 线 -> 半仓
    half_long = (matrix == "confirmed") & (health == -1) & bullish

    num_units[full_long] = 1.0
    num_units[half_long] = 0.5

    return {
        "num_units": num_units,
        "signal": matrix,
        "trend_health": health,
    }


def run_validation() -> bool:
    """VPA 趋势策略验证协议"""
    print("=" * 60)
    print("VPA 趋势策略验证协议（vpa_trend.py）")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 500

    # ── 正控：健康上涨趋势 -> 应有较多持仓 ──
    print("\n【正控】健康上涨趋势")
    print("-" * 60)

    base_body = np.linspace(0.05, 0.3, n)
    body = np.maximum(base_body + np.random.randn(n) * 0.05, 0.01)
    close = pd.Series(np.cumsum(body) + 10)
    open_s = close.shift(1).fillna(close.iloc[0])
    volume = (body * 50000 + np.random.randint(100, 500, n)).clip(min=100)

    ohlcv = pd.DataFrame({
        "open": open_s,
        "high": np.maximum(open_s, close) + 0.1,
        "low": np.minimum(open_s, close) - 0.1,
        "close": close,
        "volume": volume,
    })

    result = vpa_trend(ohlcv, lookback=20)
    pos_ratio = (result["num_units"] > 0).sum() / len(result["num_units"])
    pos_ok = pos_ratio >= 0.20
    print(f"  持仓天数占比: {pos_ratio:.2%} (要求 >= 20%)  "
          f"[{'PASS' if pos_ok else 'FAIL'}]")
    if not pos_ok:
        all_pass = False

    # ── 负控：随机游走 -> 仓位应较低 ──
    print("\n【负控】随机游走")
    print("-" * 60)

    np.random.seed(43)
    close_r = pd.Series(np.cumsum(np.random.randn(n)) + 10)
    open_r = close_r.shift(1).fillna(close_r.iloc[0])
    volume_r = pd.Series(np.random.randint(500, 1500, n))

    ohlcv_r = pd.DataFrame({
        "open": open_r,
        "high": np.maximum(open_r, close_r) + 0.1,
        "low": np.minimum(open_r, close_r) - 0.1,
        "close": close_r,
        "volume": volume_r,
    })

    result_r = vpa_trend(ohlcv_r, lookback=20)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.40
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 40%)  "
          f"[{'PASS' if neg_ok else 'FAIL'}]")
    if not neg_ok:
        all_pass = False

    # ── 不变式：num_units >= 0 ──
    print("\n【不变式】num_units >= 0")
    print("-" * 60)
    nonneg_ok = (result["num_units"] >= 0).all()
    print(f"  num_units 全部 >= 0: {'PASS' if nonneg_ok else 'FAIL'}")
    if not nonneg_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] VPA 趋势策略验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
