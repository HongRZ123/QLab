"""
strategies/registry.py — 策略注册表
====================================

仅作为策略的**目录索引**。提供按名称发现策略的能力，
不再做运行时 dispatch —— 端到端运行由 run/ 脚本负责，直接 import 调用。

设计:
    - Strategy 数据类封装名称、函数指针、默认参数、签名说明
    - register() 让策略在 import 时自报家门
    - list_names() / get_strategy() 供发现和自查
    - 端到端脚本直接 `from strategies.MR.s4_linear import linear_mr` 调用，
      不经过 register。register 仅用于 `python -m strategies` 列目录。

为何不做 dispatch:
    原本的 run_strategy(name, data) 想把"加载什么数据"和"调用什么策略"统一，
    但策略输入形态多样（close Series / OHLCV DataFrame / 配对 / 组合矩阵 / 信号列表),
    二选一的 data_requirements 字段无法表达，会引起抽象泄漏。
    策略输入由其函数签名直接表达，端到端脚本组装输入并直接调用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 数据类
# ============================================================

@dataclass(frozen=True)
class Strategy:
    """
    策略注册条目（目录索引用途）。

    属性:
        name:            策略名称 (唯一标识)
        fn:              策略函数指针，签名由各策略模块自行声明
        description:     简短描述
        default_kwargs:  默认参数字典（供发现者参考，调用方应在 import 后直接覆盖）
        input_signature: 文字描述的输入签名，如 "close: pd.Series" 或 "ohlcv: pd.DataFrame"
                         用于 --list 等目录输出，不做运行时类型检查
    """

    name: str
    fn: Callable[..., dict[str, Any]]
    description: str = ""
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    input_signature: str = "close: pd.Series"


# ============================================================
# 全局注册表
# ============================================================

_REGISTRY: dict[str, Strategy] = {}


def register(entry: Strategy) -> None:
    """
    注册一个策略到全局注册表（用于目录发现）。

    参数:
        entry: Strategy 数据类实例

    异常:
        ValueError: 同名策略已存在
    """
    if entry.name in _REGISTRY:
        raise ValueError(f"策略 '{entry.name}' 已注册")
    _REGISTRY[entry.name] = entry


def get_strategy(name: str) -> Strategy:
    """
    按名称获取策略条目（仅目录查询，不调用）。

    异常:
        KeyError: 名称未注册
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"策略 '{name}' 未注册。可用策略: {available}")
    return _REGISTRY[name]


def list_names() -> list[str]:
    """返回所有已注册策略名称 (按字母排序)。"""
    return sorted(_REGISTRY.keys())


def list_strategies() -> list[Strategy]:
    """返回所有已注册策略条目 (按字母排序)。"""
    return [_REGISTRY[k] for k in list_names()]


# ============================================================
# 注册内置策略
# ============================================================

def _register_builtins() -> None:
    """注册项目内置的稳定策略。仅做目录用途，运行请直接 import。"""
    from strategies.MR.s4_linear import linear_mr
    from strategies.MR.s8_bollinger import bollinger_mr
    from strategies.Tech.ma_crossover import ma_crossover
    from strategies.Tech.vpa_breakout import vpa_breakout
    from strategies.Tech.vpa_reversal import vpa_reversal
    from strategies.Tech.vpa_trend import vpa_trend

    register(Strategy(
        name="linear_mr",
        fn=linear_mr,
        description="S4 线性均值回归 (Chan 2013 Ch2)",
        default_kwargs={},
        input_signature="prices: pd.Series",
    ))

    register(Strategy(
        name="bollinger_mr",
        fn=bollinger_mr,
        description="S8 布林带均值回归 (Chan 2013 Ch3)",
        default_kwargs={"lookback": 20, "entry_z": 1.0, "exit_z": 0.0},
        input_signature="prices: pd.Series",
    ))

    register(Strategy(
        name="ma_crossover",
        fn=ma_crossover,
        description="均线金叉死叉策略",
        default_kwargs={"short_window": 5, "long_window": 20},
        input_signature="prices: pd.Series",
    ))

    register(Strategy(
        name="vpa_trend",
        fn=vpa_trend,
        description="VPA 量价确认趋势跟踪 (Anna Coulling Ch4/Ch8)",
        default_kwargs={"lookback": 20, "confirm_low": 0.7, "confirm_high": 1.5},
        input_signature="ohlcv: pd.DataFrame",
    ))

    register(Strategy(
        name="vpa_reversal",
        fn=vpa_reversal,
        description="VPA 止损量反转策略 (Anna Coulling Ch5/Ch6)",
        default_kwargs={"lookback": 20},
        input_signature="ohlcv: pd.DataFrame",
    ))

    register(Strategy(
        name="vpa_breakout",
        fn=vpa_breakout,
        description="VPA 放量突破策略 (Anna Coulling Ch7)",
        default_kwargs={"lookback": 20, "breakout_lookback": 20,
                        "vol_threshold": 1.5, "spread_threshold": 1.5},
        input_signature="ohlcv: pd.DataFrame",
    ))


_register_builtins()