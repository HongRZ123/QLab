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
# 默认候选 (过渡期静态列表，未来切到评分函数)
# ============================================================

# 这几个标的是前序研究中跑出来有效果的:
#   sh512670: 银行 ETF, 半衰期 44.5d, 在探索扫描中唯一通过 Holm-Bonferroni 的标的
#   sh510300: 沪深 300 ETF, 流动性好, 适合做基准
#   sz159915: 创业板 ETF, 趋势性强, VPA 策略在此类标的上表现好
_DEFAULT_MR_SYMBOLS = ["sh512670", "sh510300", "sz159915"]
_DEFAULT_TREND_SYMBOLS = ["sh512760", "sz159915", "sh510050"]
_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("sh512670", "sh512760"),   # 银行 vs 半导体
    ("sh510300", "sh510500"),   # 沪深300 vs 中证500
]
_DEFAULT_PORTFOLIO_SYMBOLS: list[list[str]] = [
    ["sh512670", "sh510300", "sh512760", "sh510050", "sz159915"],
]

# ============================================================
# Public API
# ============================================================

def get_mr_candidates() -> list[str]:
    """均值回归策略的默认候选标的。

    适合 linear_mr / bollinger_mr 等策略。
    当前为静态列表，未来切到 screen_stationarity(universe, top_n=3)。
    """
    return _DEFAULT_MR_SYMBOLS


def get_trend_candidates() -> list[str]:
    """趋势跟踪策略的默认候选标的。

    适合 vpa_trend / ma_crossover 等策略。
    当前为静态列表，未来切到 screen_momentum(universe, top_n=3)。
    """
    return _DEFAULT_TREND_SYMBOLS


def get_pair_candidates() -> list[tuple[str, str]]:
    """配对交易的默认候选标的对。

    当前为静态列表，未来切到 screen_cointegration(universe, top_n=3)。
    """
    return _DEFAULT_PAIRS


def get_portfolio_candidates() -> list[list[str]]:
    """组合策略的默认候选标的组。

    当前为静态列表，未来切到 Johansen 评选最优协整组合。
    """
    return _DEFAULT_PORTFOLIO_SYMBOLS


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
