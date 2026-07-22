"""
run/_common.py -- shared utilities for end-to-end runner scripts

All run/ strategy scripts use these helpers for data loading, output, and backtest.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from data.dividend import adjust_close_prices, detect_ex_dividend
from data.fetcher import read_day


def adjust_dividend(symbol: str) -> pd.DataFrame:
    """Load OHLCV and adjust close for ex-dividend events."""
    df = read_day(symbol)
    ex_div = detect_ex_dividend(df["close"], df["open"])
    n_ex = int(ex_div.sum())
    if n_ex > 0:
        print(f"  [div] {n_ex} ex-dividend events, prices adjusted")
        df["close"] = adjust_close_prices(df["close"], df["open"], ex_div)
    return df


def load_close(symbol: str) -> pd.Series:
    """Load adjusted close Series."""
    return adjust_dividend(symbol)["close"]


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load adjusted OHLCV DataFrame."""
    return adjust_dividend(symbol)


def save_report(rows: list[dict] | pd.DataFrame, path: str | Path) -> None:
    """Save performance report to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Report saved: {output_path}")
