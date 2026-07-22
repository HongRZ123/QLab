"""
alpha/defaults.py — ETF 分类宇宙与标的查询

提供 ETF 分类宇宙数据和按类别取标的的函数。不包含候选选取逻辑——
候选选取由各 run/ 脚本直接调用 alpha/stationarity.py / momentum.py /
cointegration.py 完成。

Functions:
    get_universe(category) -> list[str]    按类别取标的
    list_categories() -> list[str]         所有可用类别
    ETF_UNIVERSE: dict[str, list[str]]     标的分类宇宙 (38只)
"""
from __future__ import annotations

# ============================================================
# ETF 分类宇宙 (与 explore/scan_stationarity.py 保持同步)
# ============================================================

ETF_UNIVERSE: dict[str, list[str]] = {
    "control_index": [
        "sh000001", "sh000300", "sh000905", "sh000016", "sz399006",
    ],
    "broad_etf": [
        "sh510050", "sh510300", "sh510500", "sz159915", "sh588000",
    ],
    "industry": [
        "sh512880", "sh512800", "sh512010", "sh512690", "sh515030",
        "sh512480", "sh512660", "sh512200", "sh512400", "sh515220",
        "sh515210", "sh512980", "sh512720", "sh512760", "sh515790",
        "sz159996", "sh512170", "sh515050", "sh512670", "sh561660",
    ],
    "cross_border": [
        "sh513100", "sh513050", "sh513600", "sh513880",
        "sh518880", "sz159985", "sh501018", "sh513030",
    ],
}

# ============================================================
# Public API
# ============================================================

def get_universe(category: str | None = None) -> list[str]:
    """返回标的宇宙。

    Args:
        category: 类别名 ("broad_etf", "industry", "cross_border", "control_index")
                  或 None (返回全部标的，去重排序)
    """
    if category is None:
        all_symbols: set[str] = set()
        for symbols in ETF_UNIVERSE.values():
            all_symbols.update(symbols)
        return sorted(all_symbols)
    if category not in ETF_UNIVERSE:
        available = ", ".join(ETF_UNIVERSE.keys())
        raise ValueError(f"未知类别 '{category}'。可选: {available}")
    return ETF_UNIVERSE[category]


def list_categories() -> list[str]:
    """返回所有可用的标的类别名。"""
    return list(ETF_UNIVERSE.keys())
