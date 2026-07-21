"""
run_pair_trade.py - 配对交易研究模板
=====================================

研究两个标的的配对交易：CADF 协整检验 -> S9 卡尔曼动态对冲 -> 回测。

用法:
    1. 复制此文件到项目根目录: cp templates/run_pair_trade.py my_pair.py
    2. 修改下方 CONFIG 区域的两个标的代码
    3. 运行: python my_pair.py

适用场景:
    - 同行业 ETF 对 (如 银行ETF vs 证券ETF)
    - A股 vs 港股 ETF
    - 高度相关的两个标的
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.fetcher import read_day
from strategies.MR.s9_kalman_hedge import kalman_hedge
from tests.s5_cadf import cadf_test, cadf_test_both_orders

# ============================================================
# CONFIG
# ============================================================

SYMBOL_Y = "sh512670"   # 因变量标的 (如 银行ETF)
SYMBOL_X = "sh512760"   # 自变量标的 (如 芯片ETF)
BURN_IN = 60            # 卡尔曼滤波预热期 (天)
INITIAL_CAPITAL = 1_000_000.0

# ============================================================
# 主流程
# ============================================================

def main() -> None:
    print("=" * 60)
    print("  配对交易研究 (CADF + S9 卡尔曼对冲)")
    print("=" * 60)
    print(f"  Y: {SYMBOL_Y}")
    print(f"  X: {SYMBOL_X}")

    # ── 1. 加载数据 ──
    print(f"\n{'=' * 60}")
    print("  1. 数据加载")
    print(f"{'=' * 60}")

    df_y = read_day(SYMBOL_Y)
    df_x = read_day(SYMBOL_X)

    # 对齐日期
    common_idx = df_y.index.intersection(df_x.index)
    y = df_y.loc[common_idx, "close"]
    x = df_x.loc[common_idx, "close"]

    print(f"  Y: {len(y)} 天 ({y.index[0].date()} ~ {y.index[-1].date()})")
    print(f"  X: {len(x)} 天 ({x.index[0].date()} ~ {x.index[-1].date()})")
    print(f"  共同日期: {len(common_idx)} 天")

    # ── 2. CADF 协整检验 ──
    print(f"\n{'=' * 60}")
    print("  2. CADF 协整检验")
    print(f"{'=' * 60}")

    # 双向检验: Y on X 和 X on Y
    result_yx = cadf_test(y, x)
    result_xy = cadf_test(x, y)

    print(f"  Y on X: 对冲比率={result_yx['hedge_ratio']:.4f}, "
          f"p-value={result_yx['p_value']:.4f}")
    print(f"  X on Y: 对冲比率={result_xy['hedge_ratio']:.4f}, "
          f"p-value={result_xy['p_value']:.4f}")

    # 双阶检验 (更稳健)
    result_both = cadf_test_both_orders(y, x)
    print(f"  双阶检验: {result_both['recommendation']}")

    # 选择 p-value 更小的方向
    if result_yx["p_value"] <= result_xy["p_value"]:
        p_value = result_yx["p_value"]
        direction = "Y on X"
    else:
        p_value = result_xy["p_value"]
        direction = "X on Y"

    print(f"\n  选择方向: {direction} (p-value={p_value:.4f})")

    if p_value < 0.05:
        print("  ✅ 协整关系显著 (p < 0.05), 适合配对交易")
    else:
        print("  ⚠ 协整关系不显著 (p >= 0.05), 配对交易风险较高")

    # ── 3. S9 卡尔曼动态对冲 ──
    print(f"\n{'=' * 60}")
    print("  3. S9 卡尔曼动态对冲")
    print(f"{'=' * 60}")

    result = kalman_hedge(y, x, burn_in=BURN_IN)

    num_units = result["num_units"]
    beta_slope = result["beta_slope"]

    print(f"  预热期: {BURN_IN} 天")
    print(f"  β_slope 尾部均值: {np.mean(beta_slope[BURN_IN:]):.4f}")
    print(f"  β_slope 尾部标准差: {np.std(beta_slope[BURN_IN:]):.4f}")
    print(f"  持仓天数: {int((num_units > 0).sum())} / {len(num_units)}")

    # ── 4. 回测 ──
    print(f"\n{'=' * 60}")
    print("  4. 回测")
    print(f"{'=' * 60}")

    # 配对交易回测: 用 spread 的收益
    # 注意: 配对交易的回测逻辑与单资产不同
    # 这里用 S9 内置的理论 ret 做展示, 生产级回测需要自定义
    ret = result["ret"]
    pnl = result["pnl"]

    # 简单绩效 (基于理论 ret)
    valid_ret = ret[BURN_IN:]
    if len(valid_ret) > 0 and valid_ret.std() > 0:
        apr = (1 + valid_ret.mean()) ** 252 - 1
        sharpe = valid_ret.mean() / valid_ret.std() * np.sqrt(252)
        maxdd = (np.cumsum(valid_ret) - np.maximum.accumulate(np.cumsum(valid_ret))).min()
    else:
        apr = 0.0
        sharpe = 0.0
        maxdd = 0.0

    print(f"  年化收益 (理论): {apr:.3%}")
    print(f"  夏普比率 (理论): {sharpe:.3f}")
    print(f"  最大回撤 (理论): {maxdd:.3%}")
    print(f"  累计 PnL (理论): {pnl.sum():.4f}")

    print("\n  ⚠ 以上为理论绩效 (无成本、无 T+1)。")
    print("  生产级配对回测需要自定义引擎 (双边持仓 + 双边成本)。")

    # ── 5. 结论 ──
    print(f"\n{'=' * 60}")
    print("  5. 结论")
    print(f"{'=' * 60}")

    if p_value < 0.05 and sharpe > 0:
        print("  ✅ 标的对具有协整关系且策略正向收益, 值得进一步研究")
    elif p_value < 0.05:
        print("  ⚠ 有协整关系但策略收益不佳, 可能需要调参或换标的")
    else:
        print("  ❌ 协整关系不显著, 不建议配对交易")


if __name__ == "__main__":
    main()
