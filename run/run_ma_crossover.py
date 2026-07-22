"""
run/run_ma_crossover.py 鈥?鍧囩嚎閲戝弶姝诲弶 绔埌绔?=================================================

data 鈫?alpha.defaults.get_trend_candidates() 鈫?strategies.Tech.ma_crossover 鈫?backtest 鈫?output

鐢ㄦ硶:
    python run/run_ma_crossover.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from strategies.Tech.ma_crossover import ma_crossover as strategy_fn

STRATEGY_NAME = "ma_crossover"
STRATEGY_KWARGS: dict = {"short_window": 5, "long_window": 20}
DYNAMIC_SIZING = True
OUTPUT_CSV: str | None = None


def main() -> None:
    from alpha.defaults import get_trend_candidates
    from backtest.engine import run_backtest
    from backtest.metrics import performance_summary
    from run._common import load_close

    candidates = get_trend_candidates()
    rows: list[dict] = []

    print(f"=== {STRATEGY_NAME} 脳 {len(candidates)} candidates: {', '.join(candidates)}")

    for sym in candidates:
        close = load_close(sym)
        result = strategy_fn(close, **STRATEGY_KWARGS)
        nu = result["num_units"]
        common = close.index.intersection(nu.index)
        bt = run_backtest(close.loc[common], nu.loc[common], dynamic_sizing=DYNAMIC_SIZING)
        perf = performance_summary(bt["ret"], (nu.loc[common] > 0).astype(float))
        print(f"  {sym:12s}  APR={perf['apr']:8.3%}  Sharpe={perf['sharpe']:7.3f}  "
              f"MaxDD={perf['maxdd']:8.3%}  Trades={perf['trade_count']:4.0f}")
        rows.append({"symbol": sym, "strategy": STRATEGY_NAME, **perf})

    if rows:
        print()
        print(pd.DataFrame(rows).to_string(index=False, formatters={
            "apr": "{:.3%}".format, "sharpe": "{:.3f}".format, "maxdd": "{:.3%}".format}))
    if OUTPUT_CSV and rows:
        from run._common import save_report
        save_report(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
