"""
run_strategy.py — 端到端策略编排脚本
=====================================

通过策略注册表加载数据、运行策略、执行回测并输出绩效报告。

用法:
    python run_strategy.py                          # 运行所有稳定策略
    python run_strategy.py --strategy linear_mr     # 运行指定策略
    python run_strategy.py --symbol sh512760        # 使用其他标的
    python run_strategy.py --save                   # 保存 CSV 报告
"""

from __future__ import annotations

import argparse
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
from strategies.registry import get_strategy, list_names, run_strategy

# ============================================================
# 配置
# ============================================================

DEFAULT_SYMBOL: str = "sh512670"
STABLE_STRATEGIES: list[str] = ["linear_mr", "bollinger_mr", "ma_crossover"]

REPORT_COLUMNS: list[str] = [
    "symbol", "strategy", "apr", "sharpe", "max_dd",
    "win_rate", "n_trades", "avg_holding", "n_days",
]


# ============================================================
# 辅助
# ============================================================

def _to_binary(num_units: pd.Series) -> pd.Series:
    """将连续仓位转为 0/1 信号, 用于交易次数统计。"""
    return (num_units > 0).astype(float)


def _load_and_adjust(symbol: str) -> pd.DataFrame:
    """
    加载完整 OHLCV 数据并处理除权除息。

    参数:
        symbol: 标的代码

    返回:
        pd.DataFrame: 含 open/high/low/close/volume 列，close 已复权

    异常:
        FileNotFoundError: 数据文件不存在
    """
    df = read_day(symbol)

    close = df["close"]
    open_prices = df["open"]

    ex_div = detect_ex_dividend(close, open_prices)
    n_ex = int(ex_div.sum())
    if n_ex > 0:
        print(f"  检测到 {n_ex} 个除权除息日, 已调整价格")
        df["close"] = adjust_close_prices(close, open_prices, ex_div)

    return df



def _run_single(
    symbol: str,
    ohlcv: pd.DataFrame,
    strategy_name: str,
) -> dict:
    """
    对单个策略执行完整回测流程。

    参数:
        symbol:         标的代码
        ohlcv:          含 open/high/low/close/volume 的 DataFrame
        strategy_name:  策略名称 (注册表中的键)

    返回:
        dict: 报告行 (含 symbol, strategy, apr, sharpe, ...)
    """
    entry = get_strategy(strategy_name)
    print(f"  [{strategy_name}] {entry.description}")

    # 1) 运行策略, 获取 num_units
    if entry.data_requirements == "ohlcv":
        result = run_strategy(strategy_name, ohlcv)
        close = ohlcv["close"]
    else:
        close = ohlcv["close"]
        result = run_strategy(strategy_name, close)

    num_units = result["num_units"]

    # 对齐 index
    common_idx = close.index.intersection(num_units.index)
    num_units_aligned = num_units.loc[common_idx]
    close_aligned = close.loc[common_idx]

    # 2) 回测引擎 (T+1, 整数手, 成本, 动态仓位)
    bt = run_backtest(close_aligned, num_units_aligned, dynamic_sizing=True)

    # 3) 绩效汇总
    perf = performance_summary(bt["ret"], _to_binary(num_units_aligned))

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "apr": perf["apr"],
        "sharpe": perf["sharpe"],
        "max_dd": perf["maxdd"],
        "win_rate": perf["win_rate"],
        "n_trades": perf["trade_count"],
        "avg_holding": perf["avg_holding"],
        "n_days": perf["n_days"],
    }


# ============================================================
# 打印
# ============================================================

def _print_report(rows: list[dict]) -> None:
    """打印控制台汇总表。"""
    if not rows:
        print("\n  无回测结果")
        return

    df = pd.DataFrame(rows)

    print(f"\n{'=' * 100}")
    print("  策略回测汇总")
    print(f"{'=' * 100}")
    print(f"  {'symbol':<12s} {'strategy':<18s} {'APR':>9s} {'Sharpe':>8s} "
          f"{'MaxDD':>8s} {'Win%':>7s} {'Trades':>7s} {'Hold':>6s} {'Days':>6s}")
    print(f"  {'-' * 90}")

    for _, row in df.iterrows():
        n_trades = row["n_trades"]
        avg_holding = row["avg_holding"]

        trades_str = f"{n_trades:7.0f}" if not pd.isna(n_trades) else "   N/A"
        hold_str = f"{avg_holding:6.1f}" if not pd.isna(avg_holding) else "  N/A"

        print(f"  {str(row['symbol']):<12s} {str(row['strategy']):<18s} "
              f"{row['apr']:9.3%} {row['sharpe']:8.3f} {row['max_dd']:8.3%} "
              f"{row['win_rate']:7.3%} {trades_str} {hold_str} "
              f"{int(row['n_days']):6d}")

    print(f"  {'-' * 90}")
    print(f"  共 {len(df)} 条记录")


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="端到端策略回测 (通过策略注册表)",
    )
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help=f"标的代码 (默认: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help=f"策略名称 (默认: 运行所有稳定策略). 可选: {', '.join(list_names())}",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存 CSV 报告到 output/strategy_run_report.csv",
    )
    return parser.parse_args()


# ============================================================
# 入口
# ============================================================

def main() -> None:
    """主流程: 加载数据 → 运行策略 → 回测 → 输出报告。"""
    args = _parse_args()
    symbol = args.symbol
    strategy_names = [args.strategy] if args.strategy else STABLE_STRATEGIES

    print("=" * 60)
    print("  端到端策略编排 — 注册表驱动")
    print("=" * 60)
    print(f"  标的: {symbol}")
    print(f"  策略: {', '.join(strategy_names)}")

    # ── 1. 加载数据 ──
    print(f"\n{'=' * 60}")
    print("  数据加载")
    print(f"{'=' * 60}")

    try:
        ohlcv = _load_and_adjust(symbol)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        return

    print(f"  {len(ohlcv)} 天 ({ohlcv.index[0].date()} ~ {ohlcv.index[-1].date()})")

    # ── 2. 运行策略 ──
    print(f"\n{'=' * 60}")
    print("  策略回测 (引擎: T+1, 整数手, 成本, 复利)")
    print(f"{'=' * 60}")

    rows: list[dict] = []
    for name in strategy_names:
        try:
            row = _run_single(symbol, ohlcv, name)
            rows.append(row)

            print(f"    Sharpe={row['sharpe']:7.3f}  "
                  f"APR={row['apr']:8.3%}  MaxDD={row['max_dd']:8.3%}")
        except KeyError as e:
            print(f"  [跳过] {e}")

    # ── 3. 输出报告 ──
    _print_report(rows)

    if args.save and rows:
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "strategy_run_report.csv"
        report = pd.DataFrame(rows)[REPORT_COLUMNS]
        report.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n  报告已保存: {output_path}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
