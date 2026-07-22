"""
run/run_linear_portfolio.py -- S7 Linear Portfolio MR end-to-end
==================================================================

data -> alpha.defaults.get_universe("broad_etf") -> Johansen -> linear_portfolio -> backtest -> output

Usage:
    python run/run_linear_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.fetcher import read_day
from signals.stats import estimate_half_life
from signals.stats_cointegration import johansen_test
from strategies.MR.s7_linear_portfolio import linear_portfolio as strategy_fn

STRATEGY_NAME = "linear_portfolio"
STRATEGY_KWARGS: dict = {}
OUTPUT_CSV: str | None = None


def main() -> None:
    from alpha.defaults import get_universe

    symbols = get_universe("broad_etf")
    print(f"=== {STRATEGY_NAME} x {len(symbols)} symbols: {', '.join(symbols)}")

    # data - load and align
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        closes[sym] = read_day(sym)["close"]
    common = closes[symbols[0]].index
    for sym in symbols[1:]:
        common = common.intersection(closes[sym].index)
    prices_df = pd.DataFrame({sym: closes[sym].loc[common] for sym in symbols})

    # signal - Johansen
    joh = johansen_test(prices_df, lag=1)
    print(f"  Johansen rank={joh['rank']}  yport_HL={joh['half_life']:.1f}d  "
          f"is_cointegrated={joh['is_cointegrated']}")

    if not joh["is_cointegrated"]:
        print("  No cointegration found - portfolio strategy not applicable")
        return

    # strategy
    v1 = joh["eigenvectors"][:, 0]
    result = strategy_fn(prices_df, v1, **STRATEGY_KWARGS)
    ret = result.get("ret", pd.Series(0.0, index=prices_df.index))
    yport = joh["yport"]

    valid = ret.iloc[1:] if len(ret) > 1 else ret
    if len(valid) > 0 and valid.std() > 0:
        sharpe = valid.mean() / valid.std() * np.sqrt(252)
        apr = (1 + valid.mean()) ** 252 - 1
    else:
        sharpe = apr = 0.0

    hl = estimate_half_life(yport, use_log=False)
    print(f"  yport_HL={hl['half_life']:.1f}d  "
          f"Sharpe(theoretical)={sharpe:.3f}  APR(theoretical)={apr:.3%}")
    print(f"  eigenvector: {np.array2string(v1, precision=4, suppress_small=True)}")

    if OUTPUT_CSV:
        from run._common import save_report
        save_report([{"strategy": STRATEGY_NAME, "n_symbols": len(symbols),
                      "rank": joh["rank"], "half_life": joh["half_life"],
                      "sharpe": sharpe, "apr": apr}], OUTPUT_CSV)


if __name__ == "__main__":
    main()
