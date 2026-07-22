"""
strategies/registry.py — 策略注册表
===================================

提供轻量级的策略注册机制，允许按名称查找和运行单资产策略。

设计:
    - Strategy 数据类封装名称、函数、默认参数
    - 全局注册表支持 register() / get() / list_names() / run()
    - 所有注册的策略必须返回含 "num_units" 键的 dict

用法:
    from strategies.registry import get_strategy, run_strategy

    entry = get_strategy("linear_mr")
    result = run_strategy("linear_mr", prices)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ============================================================
# 数据类
# ============================================================

@dataclass(frozen=True)
class Strategy:
    """
    策略注册条目。

    属性:
        name:        策略名称 (唯一标识)
        fn:          策略函数, 签名 (prices, **kwargs) -> dict 或 (ohlcv, **kwargs) -> dict
        description: 简短描述
        default_kwargs: 默认参数字典
        data_requirements: "close" 或 "ohlcv"，策略需要的数据类型
    """

    name: str
    fn: Callable[..., dict[str, Any]]
    description: str = ""
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    data_requirements: str = "close"

    def __post_init__(self) -> None:
        if self.data_requirements not in ("close", "ohlcv"):
            raise ValueError(
                f"策略 '{self.name}' 的 data_requirements 必须是 'close' 或 'ohlcv'，"
                f"得到: {self.data_requirements}"
            )


# ============================================================
# 全局注册表
# ============================================================

_REGISTRY: dict[str, Strategy] = {}


def register(entry: Strategy) -> None:
    """
    注册一个策略到全局注册表。

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
    按名称获取策略注册条目。

    参数:
        name: 策略名称

    返回:
        Strategy 数据类

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


def run_strategy(
    name: str,
    data: pd.Series | pd.DataFrame,
    **override_kwargs: Any,
) -> dict[str, Any]:
    """
    按名称运行策略, 合并默认参数与覆盖参数。

    参数:
        name:            策略名称
        data:            价格数据 (close: pd.Series 或 ohlcv: pd.DataFrame)
        override_kwargs: 覆盖默认参数

    返回:
        dict: 策略函数返回值 (必须含 "num_units" 键)

    异常:
        KeyError:   策略未注册
        TypeError:  数据类型与策略 data_requirements 不匹配
        ValueError: 返回结果缺少 "num_units" 键
    """
    entry = get_strategy(name)

    kwargs = {**entry.default_kwargs, **override_kwargs}

    if entry.data_requirements == "ohlcv":
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"策略 '{name}' 需要 OHLCV DataFrame，得到 {type(data).__name__}"
            )
        result = entry.fn(data, **kwargs)
    else:
        if not isinstance(data, pd.Series):
            raise TypeError(
                f"策略 '{name}' 需要 close price Series，得到 {type(data).__name__}"
            )
        result = entry.fn(data, **kwargs)

    if "num_units" not in result:
        raise ValueError(
            f"策略 '{name}' 返回的 dict 缺少 'num_units' 键。"
            f"可用键: {list(result.keys())}"
        )

    return result


# ============================================================
# 注册内置策略
# ============================================================

def _register_builtins() -> None:
    """注册项目内置的稳定策略。"""
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
    ))

    register(Strategy(
        name="bollinger_mr",
        fn=bollinger_mr,
        description="S8 布林带均值回归 (Chan 2013 Ch3)",
        default_kwargs={"lookback": 20, "entry_z": 1.0, "exit_z": 0.0},
    ))

    register(Strategy(
        name="ma_crossover",
        fn=ma_crossover,
        description="均线金叉死叉策略",
        default_kwargs={"short_window": 5, "long_window": 20},
    ))

    register(Strategy(
        name="vpa_trend",
        fn=vpa_trend,
        description="VPA 量价确认趋势跟踪 (Anna Coulling)",
        default_kwargs={"lookback": 20},
        data_requirements="ohlcv",
    ))

    register(Strategy(
        name="vpa_reversal",
        fn=vpa_reversal,
        description="VPA 反转形态策略 (Anna Coulling Ch6)",
        default_kwargs={"lookback": 20, "volume_threshold": 0.7, "trend_lookback": 20},
        data_requirements="ohlcv",
    ))

    register(Strategy(
        name="vpa_breakout",
        fn=vpa_breakout,
        description="VPA 放量突破策略 (Anna Coulling Ch7)",
        default_kwargs={"lookback": 20, "breakout_lookback": 20, "vol_threshold": 0.6},
        data_requirements="ohlcv",
    ))


_register_builtins()
