"""
vpa_reversal.py -- VPA-T2 反转形态策略

基于《量价分析》Ch6 K 线形态。

策略逻辑（仅做多）：
    - 锤头线 + 高成交量 + 下跌趋势 -> 满仓（买入）
    - 射击十字星 + 高成交量 + 上涨趋势 -> 空仓（卖出/退出）
    - 吊人线 + 高成交量 + 上涨趋势 -> 空仓
    - 其他 -> 空仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.vpa import volume_percentile, wick_body_ratio


def vpa_reversal(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    volume_threshold: float = 0.7,
    trend_lookback: int = 20,
) -> dict:
    """
    VPA 反转形态策略。

    参数:
        ohlcv: DataFrame[open, high, low, close, volume]
        lookback: 成交量百分位窗口
        volume_threshold: 成交量分位阈值（>此值为高量）
        trend_lookback: 趋势判断均线窗口

    返回:
        dict: {
            "num_units": pd.Series,
            "wick_signal": pd.Series,
            "volume_pct": pd.Series,
        }
    """
    open_s = ohlcv["open"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    wbr = wick_body_ratio(open_s, high, low, close)
    vol_pct = volume_percentile(volume, lookback=lookback)
    high_vol = vol_pct > volume_threshold

    # 简单趋势过滤器：close > MA = 上涨，close < MA = 下跌
    ma = close.rolling(window=trend_lookback, min_periods=1).mean()
    uptrend = close > ma
    downtrend = close < ma

    wick_signal = wbr["signal"]

    num_units = pd.Series(0.0, index=close.index)

    # 锤头线 + 高量 + 下跌趋势 -> 买入
    hammer_buy = (wick_signal == 1) & high_vol & downtrend
    # 射击十字星 / 吊人线 + 高量 + 上涨趋势 -> 退出
    reversal_sell = (wick_signal == -1) & high_vol & uptrend

    num_units[hammer_buy] = 1.0
    num_units[reversal_sell] = 0.0  # 空仓

    return {
        "num_units": num_units,
        "wick_signal": wick_signal,
        "volume_pct": vol_pct,
    }


def run_validation() -> bool:
    """VPA 反转策略验证协议"""
    print("=" * 60)
    print("VPA 反转策略验证协议（vpa_reversal.py）")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 200

    # ── 正控：下跌趋势底部出现锤头线 + 高量 -> 应产生买入信号 ──
    print("\n【正控】下跌趋势 + 锤头线 + 高量")
    print("-" * 60)

    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    # 构造下跌趋势
    close = pd.Series(np.linspace(20, 10, n), index=idx)
    open_s = pd.Series(np.linspace(20, 10, n), index=idx)

    # 最后一根锤头线：下影线长，收盘价在高位
    open_s.iloc[-1] = 10.2
    close.iloc[-1] = 10.5
    high = close + 0.1
    low = pd.Series(np.full(n, 9.0), index=idx)
    low.iloc[-1] = 9.0  # 长下影线

    volume = pd.Series(np.full(n, 1000.0), index=idx)
    volume.iloc[-1] = 10000.0  # 高量

    ohlcv = pd.DataFrame({
        "open": open_s,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    result = vpa_reversal(ohlcv, lookback=20)
    has_buy = (result["num_units"] > 0).any()
    print(f"  存在买入信号: {'PASS' if has_buy else 'FAIL'}")
    if not has_buy:
        all_pass = False

    # ── 负控：上涨趋势无反转形态 -> 信号稀疏 ──
    print("\n【负控】上涨趋势无反转形态")
    print("-" * 60)

    close_r = pd.Series(np.linspace(10, 20, n), index=idx)
    open_r = close_r.copy()
    ohlcv_r = pd.DataFrame({
        "open": open_r,
        "high": open_r + 0.1,
        "low": open_r - 0.1,
        "close": close_r,
        "volume": pd.Series(np.full(n, 1000.0), index=idx),
    })

    result_r = vpa_reversal(ohlcv_r, lookback=20)
    pos_ratio_r = (result_r["num_units"] > 0).sum() / len(result_r["num_units"])
    neg_ok = pos_ratio_r <= 0.10
    print(f"  持仓天数占比: {pos_ratio_r:.2%} (要求 <= 10%)  "
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
        print("[PASS] VPA 反转策略验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
