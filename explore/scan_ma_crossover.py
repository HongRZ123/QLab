"""全市场扫描: MA 金叉死叉策略选股 (多进程版)

对全部 A 股个股跑 MA 交叉策略, 按 Sharpe 排序输出 Top N。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ── 项目根目录: 确保能 import data / strategies / backtest ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest import performance_summary, run_backtest
from data.fetcher import list_symbols, read_day
from strategies import ma_crossover

# ============================================================
# 股票过滤 + 板块分类
# ============================================================

STOCK_PATTERNS = {
    # 上海
    "sh600": "main", "sh601": "main", "sh603": "main", "sh605": "main",
    "sh688": "star", "sh689": "star",
    # 深圳
    "sz000": "main", "sz001": "main", "sz002": "main", "sz003": "main",
    "sz300": "chinext", "sz301": "chinext",
    # 北京 (北交所 ±30%)
    "bj43": "bse", "bj83": "bse", "bj87": "bse",
}


def classify_stock(symbol: str) -> str | None:
    """返回个股板块类型, 非个股返回 None。"""
    prefix = symbol[:5]
    return STOCK_PATTERNS.get(prefix)


def get_all_stocks() -> list[dict]:
    """获取全部个股代码和板块。"""
    rows = []
    for market in ["sh", "sz", "bj"]:
        for sym in list_symbols(market):
            board = classify_stock(sym)
            if board is not None:
                rows.append({"symbol": sym, "board": board})
    return rows


# ============================================================
# 单只股票回测 (worker)
# ============================================================

def evaluate_stock(args: dict) -> dict | None:
    """对单只股票跑 MA 交叉策略, 返回绩效指标。"""
    symbol = args["symbol"]
    board = args["board"]
    short_window = args.get("short_window", 5)
    long_window = args.get("long_window", 20)

    try:
        df = read_day(symbol)
        if df is None or len(df) < long_window + 10:
            return None

        close = df["close"]
        result = ma_crossover(close, short_window=short_window, long_window=long_window)

        # 持仓天数太少说明信号稀疏, 跳过
        if (result["num_units"] == 1).sum() < 30:
            return None

        # 回测 (个股启用涨跌停 + 停牌检查)
        bt = run_backtest(
            close,
            result["num_units"],
            board=board,
            check_limits=True,
            check_suspension=True,
            price_data=df,
            dynamic_sizing=False,
        )

        # 交易次数太少不具统计意义
        if bt["n_trades"] < 4:
            return None

        signals = (result["num_units"] > 0).astype(int)
        stats = performance_summary(bt["ret"], signals=signals)

        return {
            "symbol": symbol,
            "board": board,
            "n_days": len(df),
            "n_trades": int(bt["n_trades"]),
            "hold_days": int((result["num_units"] == 1).sum()),
            "apr": stats["apr"],
            "sharpe": stats["sharpe"],
            "maxdd": stats["maxdd"],
            "win_rate": stats["win_rate"],
            "avg_holding": stats["avg_holding"],
            "total_cost": float(bt["total_cost"]),
            "final_equity": float(bt["equity_curve"].iloc[-1]),
        }
    except Exception:
        return None


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 80)
    print("全市场 MA 金叉死叉策略扫描 (多进程)")
    print("=" * 80)

    stocks = get_all_stocks()
    print(f"\n个股总数: {len(stocks)}")

    board_counts = {}
    for s in stocks:
        board_counts[s["board"]] = board_counts.get(s["board"], 0) + 1
    for board, cnt in sorted(board_counts.items()):
        print(f"  {board}: {cnt}")

    # 构造任务参数
    tasks = [{"symbol": s["symbol"], "board": s["board"], "short_window": 5, "long_window": 20} for s in stocks]

    results = []
    start_time = time.time()
    completed = 0

    # 并行执行
    n_workers = min(os.cpu_count() or 4, 6)
    print(f"\n启动 {n_workers} 个进程...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_stock, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res is not None:
                results.append(res)

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                pct = completed / len(tasks) * 100
                eta = elapsed / completed * (len(tasks) - completed) if completed > 0 else 0
                print(f"  进度: {completed}/{len(tasks)} ({pct:.1f}%), 有效 {len(results)}, "
                      f"已用 {elapsed:.0f}s, 预计剩余 {eta:.0f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\n扫描完成: {len(results)}/{len(stocks)} 只股票有效, 总用时 {total_time:.1f}s")

    if not results:
        print("无有效结果")
        return

    df_results = pd.DataFrame(results)

    # 过滤异常值
    df_results = df_results[
        (df_results["apr"] > -1.0)
        & (df_results["apr"] < 5.0)
        & (df_results["maxdd"] < 0.99)
    ].copy()

    # 按 Sharpe 排序
    df_results = df_results.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("Top 20 (按 Sharpe 排序)")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df_results.head(20).to_string(index=False))

    # 保存 CSV
    output_path = "output/ma_crossover_scan.csv"
    os.makedirs("output", exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
