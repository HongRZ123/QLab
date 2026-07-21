"""
s5_cadf.py — CADF 协整检验 (Cointegrated Augmented Dickey-Fuller)
=================================================================

Engle-Granger 两步法协整检验:
    1. OLS 回归: y = h·x + c + ε → 提取对冲比率 h
    2. 构造价差: e = y - h·x - c
    3. 对价差 e 做 ADF 单位根检验
    4. 估计价差的均值回归半衰期

核心函数:
    cadf_test(y, x, lag=1) -> dict
    cadf_test_both_orders(y, x) -> dict

原理:
    给定两个价格序列 y(t) 和 x(t), 先通过 OLS 回归拟合线性关系
    y = h·x + c + ε, 然后对残差 ε(t) 做 ADF 单位根检验。
    若 ADF 拒绝单位根原假设, 则表明残差平稳, y 和 x 存在协整关系。

    关键注意事项 (Chan §2):
    - y~x 和 x~y 给出不同的 hedge_ratio, 且 h_yx ≠ 1/h_xy
    - 两种顺序都试, 取 ADF t 统计量更负者作为最优顺序
    - CADF 仅适用两资产协整, 多资产需用 Johansen 检验 (S6)

用法:
    from tests.s5_cadf import cadf_test

    result = cadf_test(prices_y, prices_x)
    print(f"hedge_ratio={result['hedge_ratio']:.3f}, p={result['p_value']:.4f}")

验证:
    python -m tests.s5_cadf

依赖:
    - tests.s1_adf.run_adf              — ADF 单位根检验
    - tests.s3_half_life.estimate_half_life — 半衰期估计
    - tests.s3_half_life.generate_ou_paths  — OU 路径生成 (正控)
    - tests.s3_half_life.generate_gbm_paths — GBM 路径生成 (负控)
"""

import numpy as np
import pandas as pd

from tests.s1_adf import run_adf
from tests.s3_half_life import estimate_half_life

# ============================================================
# 核心函数
# ============================================================

def cadf_test(y: pd.Series, x: pd.Series, lag: int = 1) -> dict:
    """
    Engle-Granger 两步法 CADF 协整检验。

    对 y~x 做 OLS 回归, 提取对冲比率和截距, 构造价差 spread,
    然后对 spread 做 ADF 单位根检验并估计半衰期。

    参数:
        y:   被解释变量价格序列 (pd.Series)
        x:   解释变量价格序列 (pd.Series, 需与 y 对齐)
        lag: 保留参数 (实际由 run_adf 的 AIC/BIC 自动选择最优滞后阶数)

    返回:
        dict, 包含以下键:
        ─────────────────────────────────────────────
        hedge_ratio      : float  — 对冲比率 h (OLS 斜率)
        intercept        : float  — 回归截距 c
        spread           : pd.Series — 残差序列 e = y - h·x - c
        adf_stat         : float  — ADF t 统计量 (AIC 选阶)
        p_value          : float  — ADF p-value (AIC 选阶)
        lambda_spread    : float  — spread 的均值回归系数 λ
        half_life_spread : float  — spread 的半衰期 (天)
        ─────────────────────────────────────────────

    示例:
        >>> from tests.s5_cadf import cadf_test
        >>> result = cadf_test(prices_y, prices_x)
        >>> print(f"hedge_ratio={result['hedge_ratio']:.3f}, p={result['p_value']:.4f}")
    """
    # 对齐长度 — 取两个序列的交集索引
    common_idx = y.index.intersection(x.index)
    y_aligned = y.loc[common_idx].astype(float)
    x_aligned = x.loc[common_idx].astype(float)
    n = len(y_aligned)

    # ── Step 1: OLS 回归  y = h·x + c + ε ──
    # 设计矩阵 [x, 1], 求解最小二乘
    X = np.column_stack([x_aligned.values, np.ones(n)])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X, y_aligned.values, rcond=None)
    hedge_ratio = float(coeffs[0])
    intercept = float(coeffs[1])

    # ── Step 2: 构造价差  e = y - h·x - c ──
    spread_vals = y_aligned.values - hedge_ratio * x_aligned.values - intercept
    spread = pd.Series(spread_vals, index=common_idx, name="spread")

    # ── Step 3: 对价差做 ADF 检验 ──
    adf_result = run_adf(spread)
    adf_stat = adf_result["adf_stat_aic"]
    p_value = adf_result["p_value_aic"]

    # ── Step 4: 估计价差半衰期 ──
    # spread 已在价格空间 (若 y/x 为对数价格, spread 也在对数空间),
    # 不重复取 log
    hl_result = estimate_half_life(spread, use_log=False)
    lambda_spread = hl_result["lambda"]
    half_life_spread = hl_result["half_life"]

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread": spread,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "lambda_spread": lambda_spread,
        "half_life_spread": half_life_spread,
    }


def cadf_test_both_orders(y: pd.Series, x: pd.Series) -> dict:
    """
    双顺序 CADF 协整检验: 尝试 y~x 和 x~y 两种顺序, 取 ADF 统计量更负者。

    由于 OLS 回归的残差最小化方向不对称, y~x 和 x~y 给出不同的
    hedge_ratio, 且 h_yx ≠ 1/h_xy。本函数两种顺序都试, 选择使
    ADF t 统计量更负 (更显著拒绝单位根) 的顺序作为最优结果。

    参数:
        y:  价格序列 (pd.Series)
        x:  价格序列 (pd.Series)

    返回:
        dict, 格式同 cadf_test。若 x~y 顺序更优, hedge_ratio
        已取倒数使得解释方向保持 y~x。

    示例:
        >>> result = cadf_test_both_orders(prices_y, prices_x)
        >>> print(f"最优 hedge_ratio={result['hedge_ratio']:.3f}")
    """
    result_yx = cadf_test(y, x)
    result_xy = cadf_test(x, y)

    # 取 ADF 统计量更负者 (更负 → 更显著拒绝单位根)
    if result_yx["adf_stat"] < result_xy["adf_stat"]:
        return result_yx
    else:
        # 反转 hedge_ratio: 若 x~y 给出 h_xy, 则 y~x 的 h ≈ 1/h_xy
        # 保持返回值始终以 y~x 方向解释
        h_xy = result_xy["hedge_ratio"]
        if abs(h_xy) < 1e-12:
            # 退化情况: x 对 y 的斜率接近 0, 无法取倒数
            # 此时 y~x 方向是唯一可用解释
            return result_yx
        result_xy["hedge_ratio"] = 1.0 / h_xy
        return result_xy


# ============================================================
# 验证协议
# ============================================================

def _run_validation() -> bool:
    """
    正控 + 负控 + 双顺序验证协议。

    正控:  y = 2·x + OU_noise  → hedge_ratio≈2, p<0.10
    负控:  y, x 为独立 GBM     → p>0.10
    双顺序: cadf_test_both_orders 的 adf_stat ≤ 单顺序
    """
    N = 2000
    SIGMA = 0.01        # GBM 日波动率 (对数空间)
    THETA = 0.05        # OU 均值回归速度 → HL = ln(2)/0.05 ≈ 13.9d
    SIGMA_OU = 0.02     # OU 噪声波动率
    SEED = 42

    # 延迟导入 — 仅验证环节使用路径生成器
    from tests.s3_half_life import generate_gbm_paths, generate_ou_paths

    all_pass = True

    print("=" * 60)
    print("  CADF 协整检验 — 验证协议")
    print("=" * 60)

    # ── 正控: y = 2·x + OU_noise ──
    print("\n【正控】y = 2·x + OU(θ=0.05) noise")
    print("-" * 60)

    gbm_x = generate_gbm_paths(1, N, SIGMA, dt=1.0, seed=SEED)
    x_pos = pd.Series(gbm_x[0], name="x")

    ou_noise = generate_ou_paths(1, N, THETA, mu=0.0, sigma=SIGMA_OU, dt=1.0, seed=SEED + 1)
    noise_pos = pd.Series(ou_noise[0], name="noise")

    y_pos = 2.0 * x_pos + noise_pos
    y_pos.name = "y"

    res_pos = cadf_test(y_pos, x_pos)

    hr_ok = 1.8 <= res_pos["hedge_ratio"] <= 2.2
    p_ok = res_pos["p_value"] < 0.10

    print(f"  hedge_ratio = {res_pos['hedge_ratio']:.4f}  (期望 ∈ [1.8, 2.2])")
    print(f"  intercept   = {res_pos['intercept']:.4f}")
    print(f"  adf_stat    = {res_pos['adf_stat']:.4f}")
    print(f"  p_value     = {res_pos['p_value']:.4f}  (期望 < 0.10)")
    print(f"  half_life   = {res_pos['half_life_spread']:.1f}d")

    if hr_ok and p_ok:
        print("  [PASS] 正控验证通过")
    else:
        if not hr_ok:
            print("  [FAIL] hedge_ratio 超出 [1.8, 2.2] 区间")
        if not p_ok:
            print("  [FAIL] p_value >= 0.10, 未能检测到协整关系")
        all_pass = False

    # ── 负控: y 和 x 为独立 GBM ──
    print("\n【负控】y 和 x 为独立 GBM (无协整关系)")
    print("-" * 60)

    gbm_y = generate_gbm_paths(1, N, SIGMA, dt=1.0, seed=SEED + 100)
    y_neg = pd.Series(gbm_y[0], name="y")
    x_neg = pd.Series(gbm_x[0], name="x")

    res_neg = cadf_test(y_neg, x_neg)

    p_neg_ok = res_neg["p_value"] > 0.10

    print(f"  hedge_ratio = {res_neg['hedge_ratio']:.4f}")
    print(f"  adf_stat    = {res_neg['adf_stat']:.4f}")
    print(f"  p_value     = {res_neg['p_value']:.4f}  (期望 > 0.10)")

    if p_neg_ok:
        print("  [PASS] 负控验证通过")
    else:
        print("  [FAIL] p_value <= 0.10, 误将独立序列判为协整")
        all_pass = False

    # ── 双顺序验证 ──
    print("\n【双顺序】cadf_test_both_orders vs 单顺序")
    print("-" * 60)

    res_both = cadf_test_both_orders(y_pos, x_pos)
    adf_single = res_pos["adf_stat"]
    adf_both = res_both["adf_stat"]

    both_ok = adf_both <= adf_single + 1e-10  # 容忍浮点误差

    print(f"  adf_stat (y~x)  = {adf_single:.4f}")
    print(f"  adf_stat (both) = {adf_both:.4f}")

    if both_ok:
        print("  [PASS] 双顺序验证通过 (both ≤ single)")
    else:
        print(f"  [FAIL] both ({adf_both:.4f}) > single ({adf_single:.4f})")
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] CADF 验证通过")
    else:
        print("[FAIL] CADF 验证失败")
    print("=" * 60)

    return all_pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    success = _run_validation()
    if not success:
        raise SystemExit(1)
