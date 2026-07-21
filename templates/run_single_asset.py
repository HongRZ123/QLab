"""
run_single_asset.py - 单资产均值回归研究模板
=============================================

最常见的用法：选一个标的，做统计检验，跑策略，回测，看绩效。

用法:
    1. 复制此文件到项目根目录: cp templates/run_single_asset.py my_research.py
    2. 修改下方 CONFIG 区域的参数
    3. 运行: python my_research.py

支持的策略 (通过注册表):
    linear_mr      - S4 线性均值回归 (连续仓位)
    bollinger_mr   - S8 布林带均值回归 (0/1 仓位)
    ma_crossover   - 均线金叉死叉 (0/1 仓位)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保能 import 项目模块
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import run_backtest
from backtest.metrics import performance_summary
from data.dividend import adjust_close_prices, detect_ex_dividend
from data.fetcher import read_day
from strategies.registry import get_strategy, list_names, run_strategy
from tests.s1_adf import run_adf
from tests.s2_hurst import hurst_exponent
from tests.s3_half_life import estimate_half_life

# ============================================================
# CONFIG - 在这里改参数
# ============================================================

SYMBOL = "sh512670"          # 标的代码
STRATEGY_NAME = "linear_mr"  # 策略名 (可选: linear_mr, bollinger_mr, ma_crossover)
ADJUST_DIVIDEND = True       # 是否做除权除息调整

# 策略参数覆盖 (留空则用策略默认值)
# 例: bollinger_mr 可设 {"lookback": 20, "entry_z": 1.0, "exit_z": 0.0}
STRATEGY_KWARGS: dict = {}

# 回测参数
INITIAL_CAPITAL = 1_000_000.0
DYNAMIC_SIZING = True        # True=复利, False=固定资金

# ============================================================
# 主流程
# ============================================================

def main() -> None:
    print("=" * 60)
    print("  单资产均值回归研究")
    print("=" * 60)
    print(f"  标的: {SYMBOL}")
    print(f"  策略: {STRATEGY_NAME}")
    print(f"  可用策略: {', '.join(list_names())}")

    # ── 1. 加载数据 ──
    print(f"\n{'=' * 60}")
    print("  1. 数据加载")
    print(f"{'=' * 60}")

    df = read_day(SYMBOL)
    close = df["close"]

    if ADJUST_DIVIDEND:
        ex_div = detect_ex_dividend(close, df["open"])
        n_ex = int(ex_div.sum())
        if n_ex > 0:
            print(f"  检测到 {n_ex} 个除权除息日, 已调整价格")
            close = adjust_close_prices(close, df["open"], ex_div)

    print(f"  {len(close)} 天 ({close.index[0].date()} ~ {close.index[-1].date()})")

    # ── 2. 统计检验 ──
    print(f"\n{'=' * 60}")
    print("  2. 统计检验")
    print(f"{'=' * 60}")

    adf = run_adf(close)
    print(f"  ADF p-value (AIC): {adf['p_value_aic']:.4f}  "
          f"({'平稳' if adf['p_value_aic'] < 0.05 else '非平稳'})")

    h = hurst_exponent(close)
    print(f"  Hurst 指数: {h['hurst']:.4f}  "
          f"({'均值回归' if h['hurst'] < 0.5 else '趋势' if h['hurst'] > 0.5 else '随机'})")

    hl = estimate_half_life(close)
    print(f"  半衰期: {hl['half_life']:.1f} 天")

    # ── 3. 运行策略 ──
    print(f"\n{'=' * 60}")
    print(f"  3. 策略信号: {STRATEGY_NAME}")
    print(f"{'=' * 60}")

    entry = get_strategy(STRATEGY_NAME)
    print(f"  描述: {entry.description}")

    result = run_strategy(STRATEGY_NAME, close, **STRATEGY_KWARGS)
    num_units = result["num_units"]

    n_in = int((num_units > 0).sum())
    print(f"  持仓天数: {n_in} / {len(num_units)} ({100 * n_in / len(num_units):.1f}%)")

    # ── 4. 回测 ──
    print(f"\n{'=' * 60}")
    print("  4. 回测 (T+1, 整数手, 成本)")
    print(f"{'=' * 60}")

    common_idx = close.index.intersection(num_units.index)
    close_aligned = close.loc[common_idx]
    nu_aligned = num_units.loc[common_idx]

    bt = run_backtest(
        close_aligned, nu_aligned,
        initial_capital=INITIAL_CAPITAL,
        dynamic_sizing=DYNAMIC_SIZING,
    )

    signals = (nu_aligned > 0).astype(float)
    perf = performance_summary(bt["ret"], signals)

    print(f"  年化收益 (APR):  {perf['apr']:.3%}")
    print(f"  夏普比率:        {perf['sharpe']:.3f}")
    print(f"  最大回撤:        {perf['maxdd']:.3%}")
    print(f"  胜率:            {perf['win_rate']:.3%}")
    print(f"  交易次数:        {perf['trade_count']:.0f}")
    print(f"  平均持仓 (天):   {perf['avg_holding']:.1f}")
    print(f"  总成本:          {bt['total_cost']:,.0f}")

    # ── 5. 结论 ──
    print(f"\n{'=' * 60}")
    print("  5. 结论")
    print(f"{'=' * 60}")

    if perf["sharpe"] > 1.0:
        verdict = "优秀"
    elif perf["sharpe"] > 0.5:
        verdict = "可用"
    elif perf["sharpe"] > 0:
        verdict = "勉强"
    else:
        verdict = "不可用"

    print(f"  策略评估: {verdict}")
    print(f"  权益终值: {bt['equity_curve'].iloc[-1]:,.0f}")
    print(f"  (初始资金: {INITIAL_CAPITAL:,.0f})")

    # 判断统计检验是否支持均值回归
    if adf["p_value_aic"] < 0.05 and h["hurst"] < 0.5:
        print("  统计检验支持均值回归 (ADF 平稳 + Hurst < 0.5)")
    else:
        print("  ⚠ 统计检验不完全支持均值回归, 策略表现可能不佳")


if __name__ == "__main__":
    main()
