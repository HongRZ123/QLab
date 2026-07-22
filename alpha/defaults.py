"""
alpha/defaults.py — 默认标的配置与候选选取

所有端到端模板从本模块获取标的列表，不再硬编码 SYMBOL。

设计意图:
    今天是静态列表，明天可以切到 screen_stationarity() / screen_momentum() /
    screen_cointegration()。模板调 alpha.defaults 这一个入口，
    内部是静态 dict 还是评分函数对模板透明。

Functions:
    get_mr_candidates() -> list[str]          均值回归候选标的
    get_trend_candidates() -> list[str]        趋势跟踪候选标的
    get_pair_candidates() -> list[tuple]       配对交易候选
    get_portfolio_candidates() -> list[list]   组合策略候选
    get_universe(category) -> list[str]        按类别取标的
    ETF_UNIVERSE: dict[str, list[str]]         标的分类宇宙
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

def get_mr_candidates() -> list[str]:
    """均值回归策略的默认候选标的。

    调用 alpha.stationarity.screen_stationarity() 按 ADF+Hurst+HL 评分。
    扫描 universe 中除 index 外的 ETF，返回 top 3。
    如果评分筛选结果为空，回退到硬编码列表。
    """
    from alpha.stationarity import screen_stationarity

    universe = get_universe("broad_etf") + get_universe("industry")
    results = screen_stationarity(universe, top_n=3)
    if results:
        return [r["symbol"] for r in results]

    # fallback: 如果所有 ETF 都不满足严格筛选条件，用已知的 MR 候选
    return ["sh512670", "sh510300", "sz159915"]


def get_trend_candidates() -> list[str]:
    """趋势跟踪策略的默认候选标的。

    调用 alpha.momentum.screen_momentum() 按 Hurst+趋势健康度+动量评分。
    扫描 universe 中除 index 外的 ETF，返回 top 3。
    """
    from alpha.momentum import screen_momentum

    universe = get_universe("broad_etf") + get_universe("industry")
    results = screen_momentum(universe, top_n=3)
    return [r["symbol"] for r in results]


def get_pair_candidates() -> list[tuple[str, str]]:
    """配对交易的默认候选标的对。

    调用 alpha.cointegration.screen_pairs() 做 CADF 配对筛选。
    扫描 broad_etf 中的标的，返回 top 3 对。
    """
    from alpha.cointegration import screen_pairs

    universe = get_universe("broad_etf")
    results = screen_pairs(universe, top_n=3)
    return [r["pair"] for r in results]


def get_portfolio_candidates() -> list[list[str]]:
    """组合策略的默认候选标的组。

    调用 alpha.cointegration.screen_portfolio() 做 Johansen 组合筛选。
    扫描 broad_etf 中的标的，返回 top 3 组。
    """
    from alpha.cointegration import screen_portfolio

    universe = get_universe("broad_etf")
    results = screen_portfolio(universe, top_n=3)
    return [r["symbols"] for r in results]


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
