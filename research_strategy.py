"""
research_strategy.py — 实验策略研究脚本
========================================

演示如何加载和研究半成品策略 (strategies/experimental/)。

功能:
    1. 导入实验策略 rsi_mean_reversion
    2. 加载数据并运行策略
    3. 打印中间结果 (RSI 分布、仓位统计)
    4. 通过回测引擎评估绩效
    5. 展示参数敏感性调试

用法:
    python research_strategy.py
    python research_strategy.py --period 7 --oversold 25 --overbought 75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import performance_summary
from data.fetcher import read_day
from strategies.experimental.s11_rsi_draft import rsi_mean_reversion

# ============================================================
# 配置
# ============================================================

DEFAULT_SYMBOL: str = "sh512670"


# ============================================================
# 研究函数
# ============================================================

def _load_data(symbol: str) -> pd.Series:
    """加载收盘价。"""
    df = read_day(symbol)
    return df["close"]


def _inspect_result(result: dict, close: pd.Series) -> None:
    """
    打印实验策略的中间结果, 辅助研究。

    参数:
        result: rsi_mean_reversion 返回的 dict
        close:  价格序列
    """
    rsi = result["rsi"]
    num_units = result["num_units"]

    print(f"\n{'─' * 50}")
    print("  RSI 分布")
    print(f"{'─' * 50}")
    valid_rsi = rsi.dropna()
    if len(valid_rsi) > 0:
        print(f"  均值: {valid_rsi.mean():.1f}")
        print(f"  中位: {valid_rsi.median():.1f}")
        print(f"  最小: {valid_rsi.min():.1f}")
        print(f"  最大: {valid_rsi.max():.1f}")

    print(f"\n{'─' * 50}")
    print("  仓位统计")
    print(f"{'─' * 50}")
    n_total = len(num_units)
    n_in = int((num_units > 0).sum())
    pct_in = 100.0 * n_in / n_total if n_total > 0 else 0.0
    print(f"  总天数: {n_total}")
    print(f"  持仓天数: {n_in} ({pct_in:.1f}%)")
    print(f"  空仓天数: {n_total - n_in} ({100 - pct_in:.1f}%)")

    # RSI 分位数与仓位的关系
    print(f"\n{'─' * 50}")
    print("  RSI 分位数 vs 仓位")
    print(f"{'─' * 50}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        rsi_thresh = valid_rsi.quantile(q)
        mask = rsi <= rsi_thresh
        avg_pos = num_units[mask].mean() if mask.any() else np.nan
        print(f"  RSI ≤ P{int(q*100):2d} ({rsi_thresh:5.1f}): "
              f"平均仓位 = {avg_pos:.3f}")


def _backtest_and_report(
    close: pd.Series, result: dict,
) -> dict:
    """
    通过回测引擎评估策略绩效。

    参数:
        close:  价格序列
        result: 策略返回的 dict

    返回:
        dict: 绩效汇总
    """
    num_units = result["num_units"]

    # 对齐 index
    common_idx = close.index.intersection(num_units.index)
    num_units_aligned = num_units.loc[common_idx]
    close_aligned = close.loc[common_idx]

    # 回测引擎
    bt = run_backtest(close_aligned, num_units_aligned, dynamic_sizing=True)

    # 绩效汇总
    signals = (num_units_aligned > 0).astype(float)
    perf = performance_summary(bt["ret"], signals)

    print(f"\n{'─' * 50}")
    print("  绩效指标 (引擎: T+1, 整数手, 成本, 复利)")
    print(f"{'─' * 50}")
    print(f"  年化收益: {perf['apr']:.3%}")
    print(f"  夏普比率: {perf['sharpe']:.3f}")
    print(f"  最大回撤: {perf['maxdd']:.3%}")
    print(f"  胜率:     {perf['win_rate']:.3%}")
    print(f"  交易次数: {perf['trade_count']:.0f}")
    print(f"  平均持仓: {perf['avg_holding']:.1f} 天")
    print(f"  总交易日: {perf['n_days']}")

    return perf


def _parameter_sweep(
    close: pd.Series,
    period_range: list[int],
    oversold_range: list[float],
    overbought_range: list[float],
) -> pd.DataFrame:
    """
    参数敏感性扫描, 展示不同参数组合下的绩效。

    参数:
        close:           价格序列
        period_range:    RSI period 候选列表
        oversold_range:  超卖阈值候选列表
        overbought_range: 超买阈值候选列表

    返回:
        DataFrame: 参数网格结果
    """
    rows: list[dict] = []

    for period in period_range:
        for oversold in oversold_range:
            for overbought in overbought_range:
                if oversold >= overbought:
                    continue

                result = rsi_mean_reversion(
                    close, period=period,
                    oversold=oversold, overbought=overbought,
                )
                num_units = result["num_units"]

                common_idx = close.index.intersection(num_units.index)
                nu = num_units.loc[common_idx]
                cl = close.loc[common_idx]

                bt = run_backtest(cl, nu, dynamic_sizing=True)
                signals = (nu > 0).astype(float)
                perf = performance_summary(bt["ret"], signals)

                rows.append({
                    "period": period,
                    "oversold": oversold,
                    "overbought": overbought,
                    "apr": perf["apr"],
                    "sharpe": perf["sharpe"],
                    "maxdd": perf["maxdd"],
                })

    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="实验策略研究: RSI 均值回归草案",
    )
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help=f"标的代码 (默认: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--period", type=int, default=14,
        help="RSI period (默认: 14)",
    )
    parser.add_argument(
        "--oversold", type=float, default=30.0,
        help="超卖阈值 (默认: 30)",
    )
    parser.add_argument(
        "--overbought", type=float, default=70.0,
        help="超买阈值 (默认: 70)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="运行参数敏感性扫描",
    )
    return parser.parse_args()


# ============================================================
# 入口
# ============================================================

def main() -> None:
    """主流程: 加载数据 → 运行实验策略 → 检查中间结果 → 回测 → 参数扫描。"""
    args = _parse_args()
    symbol = args.symbol

    print("=" * 60)
    print("  实验策略研究 — RSI 均值回归草案")
    print("=" * 60)
    print(f"  标的: {symbol}")
    print(f"  参数: period={args.period}, oversold={args.oversold}, "
          f"overbought={args.overbought}")

    # ── 1. 加载数据 ──
    print(f"\n{'=' * 60}")
    print("  数据加载")
    print(f"{'=' * 60}")

    try:
        close = _load_data(symbol)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        return

    print(f"  {len(close)} 天 ({close.index[0].date()} ~ {close.index[-1].date()})")

    # ── 2. 运行实验策略 ──
    print(f"\n{'=' * 60}")
    print("  运行实验策略: rsi_mean_reversion")
    print(f"{'=' * 60}")

    result = rsi_mean_reversion(
        close,
        period=args.period,
        oversold=args.oversold,
        overbought=args.overbought,
    )

    # ── 3. 检查中间结果 ──
    _inspect_result(result, close)

    # ── 4. 回测评估 ──
    print(f"\n{'=' * 60}")
    print("  回测评估")
    print(f"{'=' * 60}")

    _backtest_and_report(close, result)

    # ── 5. 参数敏感性 (可选) ──
    if args.sweep:
        print(f"\n{'=' * 60}")
        print("  参数敏感性扫描")
        print(f"{'=' * 60}")

        sweep_df = _parameter_sweep(
            close,
            period_range=[7, 14, 21],
            oversold_range=[20.0, 30.0],
            overbought_range=[70.0, 80.0],
        )

        if not sweep_df.empty:
            sweep_df = sweep_df.sort_values("sharpe", ascending=False)
            print(f"\n  {'period':>6s} {'oversold':>8s} {'overbought':>10s} "
                  f"{'APR':>9s} {'Sharpe':>8s} {'MaxDD':>8s}")
            print(f"  {'-' * 55}")
            for _, row in sweep_df.iterrows():
                print(f"  {int(row['period']):6d} {row['oversold']:8.1f} "
                      f"{row['overbought']:10.1f} {row['apr']:9.3%} "
                      f"{row['sharpe']:8.3f} {row['maxdd']:8.3%}")

    print(f"\n{'=' * 60}")
    print("  研究完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
