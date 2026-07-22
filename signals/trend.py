"""
trend.py -- 趋势健康度信号

基于《量价分析》Ch8 动态趋势 + Ch10 实战案例。

Functions:
    trend_health: 趋势健康度（VPA-S7）
    run_validation: 趋势信号验证协议
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_health(
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """趋势健康度（VPA-S7, Ch8/Ch10）

    判断量价关系是否支持当前趋势：
        +1 = 趋势健康（上涨放量 + 回调缩量，或下跌放量 + 反弹缩量）
        -1 = 趋势走弱（上涨缩量 或 回调放量）
         0 = 中性

    判定逻辑：
        1. 计算收盘价变动方向（涨/跌）
        2. 计算成交量相对均值的偏离
        3. 上涨 + 放量 = 健康；上涨 + 缩量 = 走弱
        4. 下跌 + 缩量 = 健康（正常回调）；下跌 + 放量 = 走弱（抛压加剧）

    Args:
        close: 收盘价序列
        volume: 成交量序列
        lookback: 滚动均值窗口

    Returns:
        整数 Series: +1（健康）/ -1（走弱）/ 0（中性）
    """
    price_change = close.diff()
    vol_mean = volume.rolling(window=lookback, min_periods=1).mean()
    vol_above_mean = volume > vol_mean

    up = price_change > 0
    down = price_change < 0

    result = pd.Series(0, index=close.index, dtype=int)

    # 上涨 + 放量 = 趋势健康
    result[up & vol_above_mean] = 1
    # 上涨 + 缩量 = 趋势走弱（动力不足）
    result[up & ~vol_above_mean] = -1
    # 下跌 + 缩量 = 正常回调，趋势健康
    result[down & ~vol_above_mean] = 1
    # 下跌 + 放量 = 抛压加剧，趋势走弱
    result[down & vol_above_mean] = -1

    return result


def run_validation() -> bool:
    """趋势信号验证协议"""
    print("=" * 60)
    print("趋势信号验证协议（trend.py）")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 500

    # ── 正控：上涨趋势 + 上涨放量 ──
    print("\n【正控】上涨趋势 + 上涨放量")
    print("-" * 60)

    # 构造趋势：实体大小随时间递增，成交量与实体正相关
    # 这样当前实体通常大于滚动均值，形成持续上涨放量
    base_body = np.linspace(0.05, 0.3, n)
    body = np.maximum(base_body + np.random.randn(n) * 0.05, 0.01)
    close = pd.Series(np.cumsum(body) + 10)
    volume = pd.Series(body * 50000 + np.random.randint(100, 500, n))

    th = trend_health(close, volume, lookback=20)
    healthy_ratio = (th == 1).sum() / len(th)
    healthy_ok = healthy_ratio >= 0.50
    print(f"  健康信号占比: {healthy_ratio:.2%} (要求 >= 50%)  "
          f"[{'PASS' if healthy_ok else 'FAIL'}]")
    if not healthy_ok:
        all_pass = False

    # ── 负控：随机游走 + 独立成交量 ──
    print("\n【负控】随机游走 + 独立成交量")
    print("-" * 60)

    np.random.seed(43)
    close_random = pd.Series(np.cumsum(np.random.randn(n)) + 10)
    volume_random = pd.Series(np.random.randint(500, 1500, n))

    th_random = trend_health(close_random, volume_random, lookback=20)
    healthy_random = (th_random == 1).sum() / len(th_random)
    weak_random = (th_random == -1).sum() / len(th_random)

    # 随机数据下健康和走弱应大致均衡
    balanced_ok = abs(healthy_random - weak_random) < 0.20
    print(f"  健康: {healthy_random:.2%}, 走弱: {weak_random:.2%}, "
          f"差值: {abs(healthy_random - weak_random):.2%} (要求 < 20%)  "
          f"[{'PASS' if balanced_ok else 'FAIL'}]")
    if not balanced_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 趋势信号验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
