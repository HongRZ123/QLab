"""
run_strategy.py — 端到端集成运行器 (Walk-Forward 版)
====================================================

对 Phase 1 候选标的 (sh512670 银行ETF, sh512760 芯片ETF) 执行完整策略回测流程:
    1. 加载数据 (data.fetcher.read_day)
    2. 除权除息检测与价格调整 (data.dividend)
    3. Walk-Forward S4 线性 MR (滚动半衰期, 每 63 天重估)
    4. Walk-Forward S8 布林带 (滚动半衰期 + 训练期 entry_z 选择)
    5. Walk-Forward S7 组合线性 MR (滚动 Johansen, 每 63 天重估)
    6. S9 卡尔曼对冲 (含 burn_in=50 预热)
    7. 回测引擎: T+1 执行, 整数手, 交易成本, 动态仓位 (复利)

所有参数仅使用历史数据估计, 消除前视偏差。
S4/S8 通过回测引擎计算真实 PnL (含 T+1, 整数手, 成本, 复利)。
S7/S9 为多资产/价差策略, 使用策略自身收益率。

输出 `output/strategy_report.csv`: 每行一个策略×标的×参数组合
列: symbol, strategy, lookback, entry_z, apr, sharpe, max_dd, win_rate, n_trades, avg_holding

用法:
    python run/run_walk_forward.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import run_backtest
from backtest.metrics import performance_summary
from backtest.walk_forward import (
    walk_forward_bollinger,
    walk_forward_linear_mr,
    walk_forward_portfolio,
)
from data.dividend import adjust_close_prices, detect_ex_dividend
from data.fetcher import read_day
from strategies import kalman_hedge

# ============================================================
# 配置
# ============================================================

CANDIDATES: list[str] = ["sh512670", "sh512760"]
ENTRY_Z_CANDIDATES: list[float] = [1.0, 1.5, 2.0]
REEST_INTERVAL: int = 63
MIN_WARMUP: int = 252

OUTPUT_COLUMNS: list[str] = [
    "symbol", "strategy", "lookback", "entry_z",
    "apr", "sharpe", "max_dd", "win_rate", "n_trades", "avg_holding",
]


# ============================================================
# 辅助
# ============================================================

def _to_binary(num_units: pd.Series) -> pd.Series:
    """将连续仓位转为 0/1 信号, 用于交易次数统计。"""
    return (num_units > 0).astype(float)


def _engine_perf(close: pd.Series, num_units: pd.Series) -> dict:
    """通过回测引擎计算绩效 (T+1, 整数手, 成本, 动态仓位)。"""
    result = run_backtest(close, num_units, dynamic_sizing=True)
    return performance_summary(result["ret"], _to_binary(num_units))


def _make_row(
    symbol: str, strategy: str, lookback: str, entry_z: str, perf: dict,
) -> dict:
    """构造报告行。"""
    return {
        "symbol": symbol,
        "strategy": strategy,
        "lookback": lookback,
        "entry_z": entry_z,
        "apr": perf["apr"],
        "sharpe": perf["sharpe"],
        "max_dd": perf["maxdd"],
        "win_rate": perf["win_rate"],
        "n_trades": perf["trade_count"],
        "avg_holding": perf["avg_holding"],
    }


# ============================================================
# 单标的: Walk-Forward S4 + S8
# ============================================================

def _backtest_single(
    symbol: str, close: pd.Series, open_prices: pd.Series,
) -> list[dict]:
    """对单只标的执行 Walk-Forward S4 和 S8 回测。"""
    rows: list[dict] = []

    ex_div = detect_ex_dividend(close, open_prices)
    n_ex = int(ex_div.sum())
    if n_ex > 0:
        print(f"    检测到 {n_ex} 个除权除息日, 已调整价格")
        close = adjust_close_prices(close, open_prices, ex_div)

    # ── Walk-Forward S4 ──
    wf_s4 = walk_forward_linear_mr(close, REEST_INTERVAL, MIN_WARMUP)
    s4_perf = _engine_perf(close, wf_s4["num_units"])

    lbs = [x["lookback"] for x in wf_s4["lookback_log"] if x["lookback"] is not None]
    avg_lb = f"{np.mean(lbs):.0f}" if lbs else "-"

    rows.append(_make_row(symbol, "S4_wf_linear", avg_lb, "", s4_perf))

    # ── Walk-Forward S8 (自适应 entry_z) ──
    wf_s8 = walk_forward_bollinger(
        close, ENTRY_Z_CANDIDATES, REEST_INTERVAL, MIN_WARMUP,
    )
    s8_perf = _engine_perf(close, wf_s8["num_units"])

    ez_list = [x["entry_z"] for x in wf_s8["param_log"] if x["entry_z"] is not None]
    ez_str = f"{np.median(ez_list):.1f}" if ez_list else "-"

    rows.append(_make_row(symbol, "S8_wf_bollinger", avg_lb, ez_str, s8_perf))

    return rows


# ============================================================
# 组合: Walk-Forward S7
# ============================================================

def _backtest_portfolio(
    prices_df: pd.DataFrame, symbol_label: str,
) -> list[dict]:
    """Walk-Forward S7 组合线性均值回归 (多资产, 不经过单资产引擎)。"""
    rows: list[dict] = []

    wf_s7 = walk_forward_portfolio(prices_df, REEST_INTERVAL, MIN_WARMUP)

    valid = [x for x in wf_s7["param_log"] if x["eigenvector"] is not None]
    if not valid:
        print(f"  [跳过] {symbol_label}: 无有效协整窗口")
        return rows

    perf = performance_summary(wf_s7["ret"], _to_binary(wf_s7["num_units"]))

    lbs = [x["lookback"] for x in valid]
    avg_lb = f"{np.mean(lbs):.0f}" if lbs else "-"

    rows.append(_make_row(symbol_label, "S7_wf_portfolio", avg_lb, "", perf))
    return rows


# ============================================================
# 卡尔曼对冲: S9
# ============================================================

def _backtest_kalman(
    close_x: pd.Series, close_y: pd.Series,
    label_x: str, label_y: str,
) -> list[dict]:
    """卡尔曼滤波动态对冲回测 (含 burn_in 预热)。"""
    rows: list[dict] = []

    common_idx = close_x.index.intersection(close_y.index)
    if len(common_idx) < 100:
        print(f"  [跳过] 卡尔曼对冲: 共同交易日不足 ({len(common_idx)} < 100)")
        return rows

    x = close_x.loc[common_idx].values
    y = close_y.loc[common_idx].values

    s9 = kalman_hedge(x, y, delta=0.0001, ve=0.001, burn_in=50)
    s9_ret = pd.Series(s9["ret"], index=common_idx)
    s9_sig = pd.Series(s9["num_units"], index=common_idx)
    s9_perf = performance_summary(s9_ret, s9_sig)

    n_trades = int(np.ceil(np.sum(np.abs(np.diff(s9["num_units"]))) / 2))
    total_holding = float(np.sum(s9["num_units"]))
    avg_holding = total_holding / n_trades if n_trades > 0 else 0.0

    rows.append({
        "symbol": f"{label_y} vs {label_x}",
        "strategy": "S9_kalman_hedge",
        "lookback": "",
        "entry_z": "",
        "apr": s9_perf["apr"],
        "sharpe": s9_perf["sharpe"],
        "max_dd": s9_perf["maxdd"],
        "win_rate": s9_perf["win_rate"],
        "n_trades": n_trades,
        "avg_holding": avg_holding,
    })

    return rows


# ============================================================
# 主流程
# ============================================================

def run_all() -> pd.DataFrame:
    """
    端到端回测流程: 加载数据 → 除权调整 → Walk-Forward → 引擎 → 报告。

    返回:
        策略报告 DataFrame
    """
    all_rows: list[dict] = []
    close_map: dict[str, pd.Series] = {}
    open_map: dict[str, pd.Series] = {}

    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("=" * 60)
    print("  数据加载")
    print("=" * 60)

    for sym in CANDIDATES:
        print(f"\n  [{sym}] 加载数据...", end=" ")

        try:
            df = read_day(sym)
        except FileNotFoundError:
            print("数据文件不存在, 跳过")
            continue

        close_map[sym] = df["close"]
        open_map[sym] = df["open"]
        print(f"{len(df)} 天 ({df.index[0].date()} ~ {df.index[-1].date()})")

    if len(close_map) < 1:
        print("\n[错误] 无可用数据, 终止")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # ============================================================
    # 2. 单标的回测: Walk-Forward S4 + S8
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  单标回测: Walk-Forward S4 + S8 (引擎: T+1, 整数手, 成本, 复利)")
    print(f"{'=' * 60}")

    for sym in CANDIDATES:
        if sym not in close_map:
            continue

        print(f"\n  [{sym}] Walk-Forward 回测...")
        rows = _backtest_single(sym, close_map[sym], open_map[sym])
        all_rows.extend(rows)

        for r in rows:
            print(f"    {r['strategy']:20s}  entry_z={str(r['entry_z']):>4s}  "
                  f"Sharpe={r['sharpe']:7.3f}  APR={r['apr']:8.3%}  "
                  f"MaxDD={r['max_dd']:8.3%}  trades={r['n_trades']:3d}")

    # ============================================================
    # 3. 组合回测: Walk-Forward S7
    # ============================================================
    if len(close_map) >= 2:
        print(f"\n{'=' * 60}")
        print("  组合回测: Walk-Forward S7 (滚动 Johansen)")
        print(f"{'=' * 60}")

        sym_list = list(close_map.keys())
        common_idx = close_map[sym_list[0]].index
        for s in sym_list[1:]:
            common_idx = common_idx.intersection(close_map[s].index)

        if len(common_idx) >= MIN_WARMUP + REEST_INTERVAL:
            prices_df = pd.DataFrame(
                {s: close_map[s].loc[common_idx] for s in sym_list}
            )
            portfolio_label = f"组合({' + '.join(sym_list)})"
            portfolio_rows = _backtest_portfolio(prices_df, portfolio_label)
            all_rows.extend(portfolio_rows)

            for r in portfolio_rows:
                print(f"    {r['strategy']:20s}  "
                      f"Sharpe={r['sharpe']:7.3f}  APR={r['apr']:8.3%}  "
                      f"MaxDD={r['max_dd']:8.3%}  trades={r['n_trades']:3d}")
        else:
            print(f"  [跳过] 共同交易日不足 ({len(common_idx)} < {MIN_WARMUP + REEST_INTERVAL})")

    # ============================================================
    # 4. 卡尔曼对冲: S9
    # ============================================================
    if len(close_map) >= 2:
        print(f"\n{'=' * 60}")
        print("  卡尔曼对冲: S9 (burn_in=50)")
        print(f"{'=' * 60}")

        sym_a = CANDIDATES[0]
        sym_b = CANDIDATES[1]
        if sym_a in close_map and sym_b in close_map:
            kalman_rows = _backtest_kalman(
                close_x=close_map[sym_a],
                close_y=close_map[sym_b],
                label_x=sym_a,
                label_y=sym_b,
            )
            all_rows.extend(kalman_rows)

            for r in kalman_rows:
                print(f"    {r['strategy']:20s}  "
                      f"Sharpe={r['sharpe']:7.3f}  APR={r['apr']:8.3%}  "
                      f"MaxDD={r['max_dd']:8.3%}  trades={r['n_trades']:3d}")

    # ============================================================
    # 5. 组装 DataFrame
    # ============================================================
    if not all_rows:
        print("\n[警告] 无回测结果, 返回空报告")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    report = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]
    return report


# ============================================================
# 汇总输出
# ============================================================

def _print_report(df: pd.DataFrame) -> None:
    """打印控制台汇总表。"""
    if df.empty:
        print("\n  无数据可打印")
        return

    print(f"\n{'=' * 100}")
    print("  策略回测汇总 (Walk-Forward, 消除前视偏差)")
    print(f"{'=' * 100}")
    print(f"  {'symbol':<28s} {'strategy':<22s} {'lookback':>8s} {'entry_z':>7s} "
          f"{'APR':>9s} {'Sharpe':>8s} {'MaxDD':>8s} {'Win%':>7s} {'Trades':>7s} {'Hold':>6s}")
    print(f"  {'-' * 96}")

    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        strategy = str(row["strategy"])
        lookback = str(row["lookback"]) if row["lookback"] != "" and not pd.isna(row["lookback"]) else "-"
        entry_z = str(row["entry_z"]) if row["entry_z"] != "" and not pd.isna(row["entry_z"]) else "-"
        apr = row["apr"]
        sharpe = row["sharpe"]
        max_dd = row["max_dd"]
        win_rate = row["win_rate"]
        n_trades = row["n_trades"]
        avg_holding = row["avg_holding"]

        apr_str = f"{apr:9.3%}" if not pd.isna(apr) else "      N/A"
        sharpe_str = f"{sharpe:8.3f}" if not pd.isna(sharpe) else "    N/A"
        maxdd_str = f"{max_dd:8.3%}" if not pd.isna(max_dd) else "    N/A"
        win_str = f"{win_rate:7.3%}" if not pd.isna(win_rate) else "   N/A"
        trades_str = f"{n_trades:7.0f}" if not pd.isna(n_trades) else "   N/A"
        hold_str = f"{avg_holding:6.1f}" if not pd.isna(avg_holding) else "  N/A"

        print(f"  {symbol:<28s} {strategy:<22s} {lookback:>8s} {entry_z:>7s} "
              f"{apr_str} {sharpe_str} {maxdd_str} {win_str} {trades_str} {hold_str}")

    print(f"  {'-' * 96}")
    print(f"  共 {len(df)} 条记录")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  端到端集成运行器 — Walk-Forward 策略回测")
    print("=" * 60)
    print(f"  Phase 1 候选: {', '.join(CANDIDATES)}")
    print(f"  S8 entry_z 候选: {ENTRY_Z_CANDIDATES}")
    print(f"  重估间隔: {REEST_INTERVAL} 天, 最小预热: {MIN_WARMUP} 天")
    print("  引擎: T+1, 整数手, 成本, 动态仓位 (复利)")

    report = run_all()

    _print_report(report)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "strategy_report.csv"
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"  报告已保存: {output_path}")
    print(f"  共 {len(report)} 条记录")
    print(f"{'=' * 60}")
