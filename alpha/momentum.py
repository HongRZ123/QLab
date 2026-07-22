"""
alpha/momentum.py — 趋势动量选股

对一批标的做趋势跟踪适合度评分，输出按分数排序的候选列表。

评分维度:
    Hurst > 0.5 → 趋势性 (weight: 0.4)
    趋势健康度 → 近期趋势是否健康 (weight: 0.3)
    价格动量 → 近期收益方向性 (weight: 0.3)

函数:
    score_momentum(close, volume) -> dict     单标的趋势评分
    screen_momentum(universe, top_n) -> list  批量筛选

用法:
    from alpha.momentum import screen_momentum
    candidates = screen_momentum(universe, top_n=3)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from data.fetcher import read_day
from stats.univariate import hurst_exponent


def score_momentum(close: pd.Series, volume: pd.Series) -> dict:
    """对单标的做趋势跟踪适合度评分。

    三个维度加权:
        Hurst > 0.5 (趋势性)     — 40%
        趋势健康度 (up+high_vol = healthy) — 30%
        价格动量 (近期收益方向)   — 30%

    Returns:
        dict: hurst, trend_health_ratio, momentum_pct, momentum_score (0~1)
    """
    n = len(close)

    # 1) Hurst: H > 0.5 = trending, weight 0.4
    try:
        h = hurst_exponent(close, max_lag=min(100, n // 2))
        hurst_val = h["hurst"]
    except Exception:
        hurst_val = 0.5
    score_hurst = max(0.0, min(1.0, (hurst_val - 0.4) / 0.3))  # H > 0.4 starts scoring, H >= 0.7 = 1.0

    # 2) Trend health: ratio of healthy trend days in recent period
    lookback = min(60, n)
    recent_close = close.iloc[-lookback:]
    recent_vol = volume.iloc[-lookback:]

    price_change = recent_close.diff()
    up_day = price_change > 0
    vol_mean = recent_vol.rolling(window=min(20, lookback), min_periods=1).mean()
    vol_high = recent_vol > vol_mean

    ma = recent_close.rolling(window=min(20, lookback), min_periods=1).mean()
    uptrend = recent_close > ma
    downtrend = recent_close < ma

    healthy_up = (uptrend & up_day & vol_high).sum()
    healthy_down = (downtrend & ~up_day & vol_high).sum()
    total = max((uptrend & up_day).sum() + (downtrend & ~up_day).sum(), 1)
    health_ratio = (healthy_up + healthy_down) / total
    score_health = max(0.0, min(1.0, health_ratio))

    # 3) Price momentum: recent directional bias
    half = max(n // 2, 1)
    first_half = close.iloc[:half].mean()
    second_half = close.iloc[half:].mean()
    momentum_pct = (second_half / first_half - 1) if first_half > 0 else 0.0
    score_momentum_pct = max(0.0, min(1.0, abs(momentum_pct) / 0.5))

    composite = score_hurst * 0.40 + score_health * 0.30 + score_momentum_pct * 0.30

    return {
        "hurst": round(hurst_val, 4),
        "trend_health_ratio": round(health_ratio, 4),
        "momentum_pct": round(momentum_pct, 4),
        "momentum_score": round(composite, 4),
    }


def screen_momentum(
    universe: list[str],
    top_n: int = 5,
) -> list[dict]:
    """对一批标的做趋势评分，返回按 momentum_score 降序排列的候选列表。

    筛选条件: hurst > 0.5 (趋势性) 且 momentum_score > 0.3

    Args:
        universe: 标的代码列表
        top_n: 返回前 N 只

    Returns:
        list[dict]: [{"symbol": ..., "momentum_score": ..., ...}, ...]
    """
    results: list[dict] = []
    for symbol in universe:
        try:
            df = read_day(symbol)
        except FileNotFoundError:
            continue
        if len(df) < 250:
            continue

        score = score_momentum(df["close"], df["volume"])
        if score["hurst"] <= 0.5:
            continue
        if score["momentum_score"] < 0.3:
            continue

        results.append({"symbol": symbol, **score})

    results.sort(key=lambda x: x["momentum_score"], reverse=True)
    return results[:top_n]
