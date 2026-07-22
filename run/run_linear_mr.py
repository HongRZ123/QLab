"""
templates/run_linear_mr.py 鈥?S4 绾挎€у潎鍊煎洖褰?绔埌绔?===================================================

data 鈫?alpha.defaults.get_mr_candidates() 鈫?strategies.MR.s4_linear 鈫?backtest 鈫?output

鐢ㄦ硶:
    python templates/run_linear_mr.py
    cp templates/run_linear_mr.py my_run.py && python my_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from strategies.MR.s4_linear import linear_mr as strategy_fn

STRATEGY_NAME = "linear_mr"
STRATEGY_KWARGS: dict = {}
DYNAMIC_SIZING = True
OUTPUT_CSV: str | None = None


def main() -> None:
    from alpha.defaults import get_mr_candidates
    from backtest.engine import run_backtest
    from backtest.metrics import performance_summary
    from run._common import load_close

    candidates = get_mr_candidates()
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
