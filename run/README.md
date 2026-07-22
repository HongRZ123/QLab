# run/ -- end-to-end strategy entry points

Each strategy has its own standalone runner script. The complete pipeline:
data -> alpha -> strategy -> backtest -> output.

Parameters are hardcoded at the top of each file. Copy and edit to customize.

## Strategy runners

| Script | Strategy | Type | Alpha source |
|--------|----------|------|-------------|
| run_linear_mr.py | S4 Linear MR | close | get_mr_candidates() |
| run_bollinger_mr.py | S8 Bollinger MR | close | get_mr_candidates() |
| run_ma_crossover.py | MA Crossover | close | get_trend_candidates() |
| run_vpa_trend.py | VPA Trend | OHLCV | get_trend_candidates() |
| run_vpa_reversal.py | VPA Reversal | OHLCV | get_trend_candidates() |
| run_vpa_breakout.py | VPA Breakout | OHLCV | get_trend_candidates() |
| run_kalman_hedge.py | S9 Kalman Hedge | pair | get_pair_candidates() |
| run_linear_portfolio.py | S7 Portfolio MR | matrix | get_universe("broad_etf") |
| run_walk_forward.py | Walk-Forward | multi | hardcoded sh512670/sh512760 |
| research_workflow.py | Full workflow | multi | single-symbol stats + multi-strategy |
| _common.py | Shared utils | -- | internal dependency |

## Usage

```bash
python run/run_vpa_trend.py
cp run/run_vpa_trend.py my_run.py && python my_run.py
```

## Design

Each script forms a complete data -> alpha -> strategy -> backtest -> output pipeline.
Symbols come from alpha.defaults -- currently static lists, can switch to scoring
functions without changing the runner scripts. The strategy is hardcoded in the import;
the filename IS the strategy name.

## FAQ

### How to change symbols?
Edit `alpha/defaults.py` or replace the `get_xxx_candidates()` call in the script.

### Symbol format?
`{market}{code}`: Shanghai `sh`, Shenzhen `sz`. E.g. `sh512670` (Bank ETF).

### Where does data come from?
TDX after-market data, default path `D:\new_tdx64\vipdoc`.
Edit `data/fetcher.py` to change.
