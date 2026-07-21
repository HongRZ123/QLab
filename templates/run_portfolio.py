"""
run_portfolio.py - 多资产协整组合研究模板
==========================================

研究 3 个或更多标的的协整组合：Johansen 检验 -> S7 线性组合均值回归 -> Walk-Forward。

用法:
    1. 复制此文件到项目根目录: cp templates/run_portfolio.py my_portfolio.py
    2. 修改下方 CONFIG 区域的标的列表
    3. 运行: python my_portfolio.py

适用场景:
    - 同行业多 ETF 组合 (如 银行 + 证券 + 保险)
    - 跨市场组合 (如 A股 vs 港股 vs 美股 ETF)
    - 产业链上下游组合
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.walk_forward import walk_forward_portfolio
from data.fetcher import read_day
from tests.s6_johansen import johansen_test

# ============================================================
# CONFIG
# ============================================================

SYMBOLS = [
    "sh512670",  # 银行ETF
    "sh512760",  # 芯片ETF
    "sh510050",  # 50ETF
]

# Walk-Forward 参数
REEST_INTERVAL = 63     # 重估间隔 (天), 默认季度
MIN_WARMUP = 252        # 最小预热期 (天), 默认 1 年

# ============================================================
# 主流程
# ============================================================

def main() -> None:
    print("=" * 60)
    print("  多资产协整组合研究 (Johansen + S7)")
    print("=" * 60)
    print(f"  标的: {', '.join(SYMBOLS)}")
    print(f"  数量: {len(SYMBOLS)}")

    if len(SYMBOLS) < 2:
        print("\n  ❌ 至少需要 2 个标的")
        return

    # ── 1. 加载并对齐数据 ──
    print(f"\n{'=' * 60}")
    print("  1. 数据加载与对齐")
    print(f"{'=' * 60}")

    dfs = {}
    for sym in SYMBOLS:
        df = read_day(sym)
        dfs[sym] = df["close"]
        print(f"  {sym}: {len(df)} 天")

    prices_df = pd.DataFrame(dfs)
    prices_df = prices_df.dropna()

    print(f"\n  对齐后: {len(prices_df)} 天 "
          f"({prices_df.index[0].date()} ~ {prices_df.index[-1].date()})")

    # ── 2. Johansen 协整检验 ──
    print(f"\n{'=' * 60}")
    print("  2. Johansen 协整检验")
    print(f"{'=' * 60}")

    # Johansen 检验在对数价格上进行
    log_prices = np.log(prices_df)
    joh = johansen_test(log_prices)

    rank = joh["rank"]
    eigenvectors = joh["eigenvectors"]
    half_life = joh["half_life"]

    print(f"  协整秩 r = {rank}")
    print(f"  特征值: {np.array2string(joh['eigenvalues'], precision=4)}")
    print(f"  迹统计量: {np.array2string(joh['trace_stats'], precision=2)}")
    print(f"  95% 临界值: {np.array2string(joh['trace_crit'], precision=2)}")

    if rank >= 1:
        print(f"\n  第一协整向量: {np.array2string(eigenvectors[:, 0], precision=4)}")
        print(f"  yport 半衰期: {half_life:.1f} 天")

        if half_life < 60:
            print("  ✅ 协整关系显著, 半衰期合理, 适合组合均值回归")
        else:
            print("  ⚠ 半衰期较长 (>60天), 均值回归速度慢")
    else:
        print("\n  ❌ 无协整关系 (r=0), 不适合组合策略")
        return

    # ── 3. Walk-Forward 组合策略 ──
    print(f"\n{'=' * 60}")
    print("  3. Walk-Forward S7 组合策略")
    print(f"{'=' * 60}")

    print(f"  重估间隔: {REEST_INTERVAL} 天")
    print(f"  预热期: {MIN_WARMUP} 天")

    wf = walk_forward_portfolio(
        prices_df,
        reest_interval=REEST_INTERVAL,
        min_warmup=MIN_WARMUP,
    )

    num_units = wf["num_units"]
    param_log = wf["param_log"]

    n_warmup = int((num_units.iloc[:MIN_WARMUP] == 0).all())
    n_trading = int((num_units.iloc[MIN_WARMUP:] > 0).sum())

    print(f"  预热期无交易: {'✅' if n_warmup == MIN_WARMUP else '❌'}")
    print(f"  重估次数: {len(param_log)}")
    print(f"  应用期持仓天数: {n_trading} / {len(num_units) - MIN_WARMUP}")

    # 打印重估历史
    if param_log:
        print("\n  重估记录 (前 5 次):")
        for i, log in enumerate(param_log[:5]):
            date = log.get("date", "N/A")
            lb = log.get("lookback", "N/A")
            print(f"    {i+1}. {date}  lookback={lb}")

    # ── 4. 绩效 (基于理论 ret) ──
    print(f"\n{'=' * 60}")
    print("  4. 绩效评估 (理论)")
    print(f"{'=' * 60}")

    ret = wf["ret"]
    valid_ret = ret.iloc[MIN_WARMUP:].dropna()

    if len(valid_ret) > 0 and valid_ret.std() > 0:
        apr = (1 + valid_ret.mean()) ** 252 - 1
        sharpe = valid_ret.mean() / valid_ret.std() * np.sqrt(252)
        cum = valid_ret.cumsum()
        maxdd = (cum - cum.cummax()).min()
    else:
        apr = 0.0
        sharpe = 0.0
        maxdd = 0.0

    print(f"  年化收益 (理论): {apr:.3%}")
    print(f"  夏普比率 (理论): {sharpe:.3f}")
    print(f"  最大回撤 (理论): {maxdd:.3%}")

    print("\n  ⚠ 以上为理论绩效。生产级回测需要多资产回测引擎。")

    # ── 5. 结论 ──
    print(f"\n{'=' * 60}")
    print("  5. 结论")
    print(f"{'=' * 60}")

    if rank >= 1 and half_life < 60 and sharpe > 0:
        print("  ✅ 组合具有协整关系, 均值回归速度合理, 策略正向收益")
        print("  建议: 进一步做生产级回测 (含双边成本、T+1)")
    elif rank >= 1:
        print("  ⚠ 有协整关系但绩效不佳, 可尝试调整重估间隔或换标的")
    else:
        print("  ❌ 无协整关系, 不适合组合策略")


if __name__ == "__main__":
    main()
