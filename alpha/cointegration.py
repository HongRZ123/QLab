"""
alpha/cointegration.py — 协整选股

对一批标的做协整关系评分，输出配对候选列表和组合候选列表。

函数:
    screen_pairs(universe, top_n) -> list[dict]
        对 universe 中所有两两组合做 CADF 检验，按 p-value 排序返回前 N 对

    screen_portfolio(universe, top_n) -> list[dict]
        在 universe 中取不同子集做 Johansen 检验，按协整秩排序

用法:
    from alpha.cointegration import screen_pairs
    pairs = screen_pairs(universe, top_n=3)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.fetcher import read_day
from stats.cointegration import cadf_test, johansen_test


def screen_pairs(
    universe: list[str],
    top_n: int = 5,
    max_pairs: int = 50,
) -> list[dict]:
    """对 universe 中所有两两组合做 CADF 检验，按 p-value 排序返回前 N 对。

    限制: 最多尝试 max_pairs 个组合（避免 N² 爆炸）。

    Args:
        universe: 标的代码列表
        top_n: 返回前 N 对
        max_pairs: 最多尝试的组合数

    Returns:
        list[dict]: [{"pair": (sym_y, sym_x), "p_value": ..., "hedge_ratio": ..., ...}, ...]
    """
    # 加载数据
    closes: dict[str, pd.Series] = {}
    for sym in universe:
        try:
            closes[sym] = read_day(sym)["close"]
        except FileNotFoundError:
            continue

    symbols = list(closes.keys())
    n = len(symbols)
    if n < 2:
        return []

    results: list[dict] = []
    tried = 0
    for i in range(n):
        for j in range(i + 1, n):
            if tried >= max_pairs:
                break
            tried += 1

            y_sym, x_sym = symbols[i], symbols[j]
            y, x = closes[y_sym], closes[x_sym]
            common = y.index.intersection(x.index)
            if len(common) < 250:
                continue

            y_aligned = y.loc[common]
            x_aligned = x.loc[common]

            try:
                c = cadf_test(y_aligned, x_aligned)
            except Exception:
                continue

            results.append({
                "pair": (y_sym, x_sym),
                "p_value": round(c["p_value"], 4),
                "hedge_ratio": round(c["hedge_ratio"], 4),
                "half_life_spread": round(c["half_life_spread"], 1),
                "adf_stat": round(c["adf_stat"], 4),
            })

    results.sort(key=lambda x: x["p_value"])
    return results[:top_n]


def screen_portfolio(
    universe: list[str],
    top_n: int = 3,
    group_size: int = 5,
    max_groups: int = 10,
) -> list[dict]:
    """在 universe 中取不同子集做 Johansen 检验，按协整秩排序。

    从 universe 中随机选取 group_size 个标的的组合，做 Johansen 检验，
    返回协整秩最高（rank 最大）的前 top_n 组。

    限制: 最多尝试 max_groups 个组合。

    Args:
        universe: 标的代码列表
        top_n: 返回前 N 组
        group_size: 每组包含的标个数
        max_groups: 最多尝试的组合数

    Returns:
        list[dict]: [{"symbols": [...], "rank": ..., "half_life": ..., ...}, ...]
    """
    closes: dict[str, pd.Series] = {}
    for sym in universe:
        try:
            closes[sym] = read_day(sym)["close"]
        except FileNotFoundError:
            continue

    symbols = list(closes.keys())
    n = len(symbols)
    if n < group_size:
        return []

    results: list[dict] = []
    rng = np.random.default_rng(42)
    tried = 0
    while tried < max_groups:
        tried += 1
        group = rng.choice(symbols, size=min(group_size, n), replace=False).tolist()

        common = closes[group[0]].index
        for sym in group[1:]:
            common = common.intersection(closes[sym].index)
        if len(common) < 250:
            continue

        prices_df = pd.DataFrame({sym: closes[sym].loc[common] for sym in group})

        try:
            joh = johansen_test(prices_df, lag=1)
        except Exception:
            continue

        results.append({
            "symbols": group,
            "rank": joh["rank"],
            "half_life": round(joh["half_life"], 1),
            "is_cointegrated": joh["is_cointegrated"],
        })

    results.sort(key=lambda x: (-x["rank"], x["half_life"]))
    return results[:top_n]
