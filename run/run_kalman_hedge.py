"""
run/run_kalman_hedge.py 鈥?S9 鍗″皵鏇煎姩鎬佸鍐?绔埌绔?========================================================

data 鈫?alpha.defaults.get_pair_candidates() 鈫?CADF 鈫?kalman_hedge 鈫?backtest 鈫?output

姣忓鏍囩殑鏄嫭绔嬬殑閰嶅浜ゆ槗: CADF 妫€楠?鈫?鍗″皵鏇兼护娉㈠姩鎬?尾 鈫?鐞嗚鏀剁泭鐜囥€?
鐢ㄦ硶:
    python run/run_kalman_hedge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from data.fetcher import read_day
from signals.stats_cointegration import cadf_test
from strategies.MR.s9_kalman_hedge import kalman_hedge as strategy_fn

STRATEGY_NAME = "kalman_hedge"
BURN_IN = 60
OUTPUT_CSV: str | None = None


def main() -> None:
    from alpha.defaults import get_pair_candidates

    pairs = get_pair_candidates()
    rows: list[dict] = []

    print(f"=== {STRATEGY_NAME} 脳 {len(pairs)} pairs")
    for sym_y, sym_x in pairs:
        print(f"\n  {sym_y} vs {sym_x}")

        # run/run_kalman_hedge.py -- S9 Kalman Hedge end-to-end
        y = read_day(sym_y)["close"]
        x = read_day(sym_x)["close"]
        common = y.index.intersection(x.index)
        y, x = y.loc[common], x.loc[common]

        # signal 鈥?CADF
        c = cadf_test(y, x)
        print(f"  CADF: h={c['hedge_ratio']:.4f}  p={c['p_value']:.4f}  HL={c['half_life_spread']:.1f}d")

        # strategy
        result = strategy_fn(y, x, burn_in=BURN_IN)
        ret = result["ret"][BURN_IN:]
        beta = result["beta_slope"]

        if len(ret) > 0 and ret.std() > 0:
            sharpe = ret.mean() / ret.std() * np.sqrt(252)
            apr = (1 + ret.mean()) ** 252 - 1
        else:
            sharpe = apr = 0.0

        print(f"  beta_mean={np.mean(beta[BURN_IN:]):.4f}  beta_sd={np.std(beta[BURN_IN:]):.4f}  "
              f"Sharpe(theoretical)={sharpe:.3f}  APR(theoretical)={apr:.3%}")
        rows.append({"pair": f"{sym_y}/{sym_x}", "sharpe": sharpe, "apr": apr,
                     "hedge_ratio": c["hedge_ratio"], "p_value": c["p_value"]})

    if OUTPUT_CSV and rows:
        import pandas as pd

        from run._common import save_report
        save_report(pd.DataFrame(rows), OUTPUT_CSV)


if __name__ == "__main__":
    main()
