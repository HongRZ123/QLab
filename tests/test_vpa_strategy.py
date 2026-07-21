"""VPA 策略单元测试"""
import numpy as np
import pandas as pd

from strategies.experimental.s12_vpa_draft import (
    run_validation,
    vpa_strategy,
)

# ── vpa_strategy ─────────────────────────────────────────────


def test_vpa_strategy_returns_dict_with_num_units():
    """(a) vpa_strategy 返回包含 num_units 键的字典"""
    prices = pd.Series(np.linspace(10, 15, 50))
    volume = pd.Series(np.linspace(1000, 2000, 50))

    result = vpa_strategy(prices, volume, lookback=10)

    assert isinstance(result, dict)
    assert "num_units" in result
    assert isinstance(result["num_units"], pd.Series)


def test_vpa_strategy_num_units_binary():
    """(b) num_units 值 ∈ {0, 1}"""
    prices = pd.Series(np.linspace(10, 20, 100))
    volume = pd.Series(np.linspace(1000, 2000, 100))

    result = vpa_strategy(prices, volume, lookback=20)
    num_units = result["num_units"]

    assert set(num_units.unique()).issubset({0, 1})


def test_vpa_strategy_empty_series():
    """(c) vpa_strategy 在空 Series 上返回空 num_units"""
    prices = pd.Series([], dtype=float)
    volume = pd.Series([], dtype=float)

    result = vpa_strategy(prices, volume, lookback=20)
    num_units = result["num_units"]

    assert isinstance(num_units, pd.Series)
    assert len(num_units) == 0


def test_vpa_strategy_run_validation():
    """(d) run_validation() 通过"""
    assert run_validation() is True
