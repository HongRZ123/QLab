"""
templates/run_vpa_reversal.py 鈥?VPA 姝㈡崯閲忓弽杞?绔埌绔?====================================================

data 鈫?alpha.defaults.get_trend_candidates() 鈫?signals.vpa.stopping_volume 鈫?vpa_reversal 鈫?backtest 鈫?output

鐢ㄦ硶:
    python templates/run_vpa_reversal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from strategies.Tech.vpa_reversal import vpa_reversal as strategy_fn

STRATEGY_NAME = "vpa_reversal"
STRATEGY_KWARGS: dict = {"lookback": 20}
DYNAMIC_SIZING = True
OUTPUT_CSV: str | None = None


def main() -> None:
    from alpha.defaults import get_trend_candidates
    from backtest.engine import run_backtest
    from backtest.metrics import performance_summary
    from run._common import load_ohlcv

    candidates = get_trend_candidates()
    rows: list[dict] = []

    print(f"=== {STRATEGY_NAME} 脳 {len(candidates)} candidates: {', '.join(candidates)}")

    for sym in candidates:
        ohlcv = load_ohlcv(sym)
        result = strategy_fn(ohlcv, **STRATEGY_KWARGS)
        nu = result["num_units"]
        close = ohlcv["close"]
        common = close.index.intersection(nu.index)
        bt = run_backtest(close.loc[common], nu.loc[common], dynamic_sizing=DYNAMIC_SIZING)
        perf = performance_summary(bt["ret"], (nu.loc[common] > 0).astype(float))
        sv = result.get("stopping_volume")
        sv_str = f" sv={int(sv.sum())}" if sv is not None else ""
        print(f"  {sym:12s}  APR={perf['apr']:8.3%}  Sharpe={perf['sharpe']:7.3f}  "
              f"MaxDD={perf['maxdd']:8.3%}  Trades={perf['trade_count']:4.0f}{sv_str}")
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
