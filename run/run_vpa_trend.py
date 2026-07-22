"""
run/run_vpa_trend.py 鈥?VPA 閲忎环瓒嬪娍璺熻釜 绔埌绔?==================================================

data 鈫?alpha.defaults.get_trend_candidates() 鈫?signals.vpa.effort_vs_result 鈫?vpa_trend 鈫?backtest 鈫?output

鐢ㄦ硶:
    python run/run_vpa_trend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from strategies.Tech.vpa_trend import vpa_trend as strategy_fn

STRATEGY_NAME = "vpa_trend"
STRATEGY_KWARGS: dict = {"lookback": 20, "confirm_low": 0.7, "confirm_high": 1.5}
DYNAMIC_SIZING = True
OUTPUT_CSV: str | None = None


def main() -> None:
    from alpha.defaults import get_universe
    from alpha.momentum import screen_momentum
    from backtest.engine import run_backtest
    from backtest.metrics import performance_summary
    from run._common import load_ohlcv

    UNIVERSE = get_universe("broad_etf") + get_universe("industry")
    TOP_N = 3
    candidates = [r["symbol"] for r in screen_momentum(UNIVERSE, top_n=TOP_N)]
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
        evr = result.get("effort_vs_result")
        evr_str = f" evr={evr.median():.2f}" if evr is not None else ""
        print(f"  {sym:12s}  APR={perf['apr']:8.3%}  Sharpe={perf['sharpe']:7.3f}  "
              f"MaxDD={perf['maxdd']:8.3%}  Trades={perf['trade_count']:4.0f}{evr_str}")
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
