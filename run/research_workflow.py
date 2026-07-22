"""
research_workflow.py - 完整研究工作流模板
==========================================

从统计检验到策略对比的完整流程：
    1. 加载数据 + 除权除息
    2. 统计检验 (ADF / Hurst / 半衰期)
    3. 多策略对比 (所有已注册策略)
    4. Walk-Forward 滚动回测
    5. 汇总报告

用法:
    1. 复制此文件到项目根目录: cp run/research_workflow.py my_workflow.py
    2. 修改下方 CONFIG 区域
    3. 运行: python my_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import performance_summary
from data.dividend import adjust_close_prices, detect_ex_dividend
from data.fetcher import read_day
from run.run_walk_forward import walk_forward_bollinger, walk_forward_linear_mr
from stats.univariate import estimate_half_life, hurst_exponent, run_adf
from strategies.registry import list_names

# ============================================================
# CONFIG
# ============================================================

SYMBOL = "sh512670"
ADJUST_DIVIDEND = True
DO_WALK_FORWARD = True       # 是否做 Walk-Forward (较慢)
INITIAL_CAPITAL = 1_000_000.0

# ============================================================
# 主流程
# ============================================================

def main() -> None:
    print("=" * 70)
    print("  QLab 完整研究工作流")
    print("=" * 70)
    print(f"  标的: {SYMBOL}")

    # ── 1. 数据 ──
    print(f"\n{'=' * 70}")
    print("  Phase 1: 数据加载")
    print(f"{'=' * 70}")

    df = read_day(SYMBOL)
    close = df["close"]

    if ADJUST_DIVIDEND:
        ex_div = detect_ex_dividend(close, df["open"])
        n_ex = int(ex_div.sum())
        if n_ex > 0:
            close = adjust_close_prices(close, df["open"], ex_div)

    print(f"  {len(close)} 天 ({close.index[0].date()} ~ {close.index[-1].date()})")
    print(f"  除权除息日: {n_ex}")

    # ── 2. 统计检验 ──
    print(f"\n{'=' * 70}")
    print("  Phase 2: 统计检验")
    print(f"{'=' * 70}")

    adf = run_adf(close)
    h = hurst_exponent(close)
    hl = estimate_half_life(close)

    print(f"  {'检验':<20s} {'值':>10s} {'阈值':>10s} {'判定':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'ADF p-value (AIC)':<20s} {adf['p_value_aic']:10.4f} {'0.05':>10s} "
          f"{'✅ 平稳' if adf['p_value_aic'] < 0.05 else '❌ 非平稳':>8s}")
    print(f"  {'Hurst 指数':<20s} {h['hurst']:10.4f} {'0.5':>10s} "
          f"{'✅ MR' if h['hurst'] < 0.5 else '❌ 趋势':>8s}")
    print(f"  {'半衰期 (天)':<20s} {hl['half_life']:10.1f} {'<60':>10s} "
          f"{'✅' if hl['half_life'] < 60 else '⚠':>8s}")

    mr_score = sum([
        adf["p_value_aic"] < 0.05,
        h["hurst"] < 0.5,
        hl["half_life"] < 60,
    ])
    print(f"\n  均值回归评分: {mr_score}/3")
    if mr_score == 3:
        print("  ✅ 统计检验强烈支持均值回归")
    elif mr_score >= 2:
        print("  ⚠ 统计检验部分支持均值回归")
    else:
        print("  ❌ 统计检验不支持均值回归, 策略可能效果不佳")

    # ── 3. 多策略对比 ──
    print(f"\n{'=' * 70}")
    print("  Phase 3: 多策略对比")
    print(f"{'=' * 70}")

    strategy_names = list_names()
    rows: list[dict] = []

    # close-only strategies (OHLCV strategies need ohlcv input, use run scripts instead)
    from strategies.MR.s4_linear import linear_mr
    from strategies.MR.s8_bollinger import bollinger_mr
    from strategies.Tech.ma_crossover import ma_crossover
    close_strategies = {
        "linear_mr": linear_mr,
        "bollinger_mr": bollinger_mr,
        "ma_crossover": ma_crossover,
    }

    for name in strategy_names:
        if name not in close_strategies:
            continue  # skip OHLCV strategies here
        try:
            result = close_strategies[name](close)
            num_units = result["num_units"]

            common_idx = close.index.intersection(num_units.index)
            cl = close.loc[common_idx]
            nu = num_units.loc[common_idx]

            bt = run_backtest(cl, nu, initial_capital=INITIAL_CAPITAL)
            signals = (nu > 0).astype(float)
            perf = performance_summary(bt["ret"], signals)

            rows.append({
                "strategy": name,
                "apr": perf["apr"],
                "sharpe": perf["sharpe"],
                "max_dd": perf["maxdd"],
                "win_rate": perf["win_rate"],
                "n_trades": perf["trade_count"],
                "avg_holding": perf["avg_holding"],
            })
        except Exception as e:
            print(f"  [跳过] {name}: {e}")

    if rows:
        df_perf = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        print(f"\n  {'strategy':<18s} {'APR':>9s} {'Sharpe':>8s} {'MaxDD':>8s} "
              f"{'Win%':>7s} {'Trades':>7s} {'Hold':>6s}")
        print(f"  {'-' * 65}")
        for _, r in df_perf.iterrows():
            print(f"  {r['strategy']:<18s} {r['apr']:9.3%} {r['sharpe']:8.3f} "
                  f"{r['max_dd']:8.3%} {r['win_rate']:7.3%} "
                  f"{r['n_trades']:7.0f} {r['avg_holding']:6.1f}")

        best = df_perf.iloc[0]
        print(f"\n  最佳策略: {best['strategy']} (Sharpe={best['sharpe']:.3f})")

    # ── 4. Walk-Forward ──
    if DO_WALK_FORWARD:
        print(f"\n{'=' * 70}")
        print("  Phase 4: Walk-Forward 滚动回测")
        print(f"{'=' * 70}")

        wf_rows: list[dict] = []

        # S4 Walk-Forward
        try:
            wf_s4 = walk_forward_linear_mr(close, reest_interval=63, min_warmup=252)
            bt_s4 = run_backtest(close, wf_s4["num_units"], initial_capital=INITIAL_CAPITAL)
            sig_s4 = (wf_s4["num_units"] > 0).astype(float)
            perf_s4 = performance_summary(bt_s4["ret"], sig_s4)
            n_reest = len(wf_s4["lookback_log"])

            print(f"\n  S4 Walk-Forward (重估 {n_reest} 次):")
            print(f"    APR={perf_s4['apr']:.3%}  Sharpe={perf_s4['sharpe']:.3f}  "
                  f"MaxDD={perf_s4['maxdd']:.3%}  Trades={perf_s4['trade_count']:.0f}")

            wf_rows.append({
                "strategy": "S4 (WF)",
                "apr": perf_s4["apr"],
                "sharpe": perf_s4["sharpe"],
                "max_dd": perf_s4["maxdd"],
            })
        except Exception as e:
            print(f"  [S4 WF 跳过] {e}")

        # S8 Walk-Forward
        try:
            wf_s8 = walk_forward_bollinger(close, reest_interval=63, min_warmup=252)
            bt_s8 = run_backtest(close, wf_s8["num_units"], initial_capital=INITIAL_CAPITAL)
            sig_s8 = (wf_s8["num_units"] > 0).astype(float)
            perf_s8 = performance_summary(bt_s8["ret"], sig_s8)
            n_reest = len(wf_s8.get("param_log", []))

            print(f"\n  S8 Walk-Forward (重估 {n_reest} 次):")
            print(f"    APR={perf_s8['apr']:.3%}  Sharpe={perf_s8['sharpe']:.3f}  "
                  f"MaxDD={perf_s8['maxdd']:.3%}  Trades={perf_s8['trade_count']:.0f}")

            wf_rows.append({
                "strategy": "S8 (WF)",
                "apr": perf_s8["apr"],
                "sharpe": perf_s8["sharpe"],
                "max_dd": perf_s8["maxdd"],
            })
        except Exception as e:
            print(f"  [S8 WF 跳过] {e}")

        if wf_rows:
            print("\n  Walk-Forward vs 静态对比:")
            print(f"  {'type':<18s} {'APR':>9s} {'Sharpe':>8s} {'MaxDD':>8s}")
            print(f"  {'-' * 45}")
            for r in wf_rows:
                print(f"  {r['strategy']:<18s} {r['apr']:9.3%} {r['sharpe']:8.3f} "
                      f"{r['max_dd']:8.3%}")

    # ── 5. 总结 ──
    print(f"\n{'=' * 70}")
    print("  Phase 5: 研究总结")
    print(f"{'=' * 70}")

    print(f"  标的: {SYMBOL}")
    print(f"  均值回归评分: {mr_score}/3")

    if rows:
        best = max(rows, key=lambda r: r["sharpe"])
        print(f"  最佳静态策略: {best['strategy']} (Sharpe={best['sharpe']:.3f})")

    print("\n  下一步建议:")
    if mr_score >= 3 and rows and best["sharpe"] > 0:
        print("    1. 标的适合均值回归, 策略有正收益")
        print("    2. 尝试参数优化 (grid search)")
        print("    3. 做样本外验证")
    elif mr_score >= 2:
        print("    1. 均值回归特征不够强, 考虑换标的")
        print("    2. 查看扫描结果: python explore/scan_stationarity.py")
    else:
        print("    1. 该标的不适合均值回归")
        print("    2. 运行全市场扫描寻找合适标的:")
        print("       python explore/scan_stationarity.py")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
