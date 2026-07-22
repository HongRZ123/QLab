"""
signals/stats_cointegration.py — 协整性质信号

多资产间协整关系的统计检验，供 alpha 模块做配对/组合筛选。

组成（从 tests/s5~s6 迁移而来）:
    cadf_test(y, x) -> dict               Engle-Granger 两步法 CADF
    cadf_test_both_orders(y, x) -> dict   双顺序 CADF
    johansen_test(prices_df) -> dict      Johansen 多变量协整检验
    construct_portfolio(prices_df, v) -> Series  组合净值构造
    generate_cointegrated_paths(...) -> ndarray  协整路径生成器
    generate_gbm_matrix(...) -> ndarray          GBM 矩阵生成器

消费方:
    alpha/ (未来 cointegration.py)
run/run_walk_forward.py
run/run_kalman_hedge.py, run_linear_portfolio.py
    strategies/MR/s7_linear_portfolio.py  (run_validation)
    backtest/walk_forward.py
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from signals.stats import estimate_half_life, run_adf

# ============================================================
# CADF — Engle-Granger 两步法协整检验
# ============================================================

def cadf_test(y: pd.Series, x: pd.Series, lag: int = 1) -> dict:
    """Engle-Granger 两步法 CADF 协整检验。

    1. OLS: y = h·x + c + ε
    2. 对残差 spread 做 ADF 检验
    3. 估计 spread 的半衰期

    Returns:
        dict: hedge_ratio, intercept, spread, adf_stat, p_value,
              lambda_spread, half_life_spread
    """
    common_idx = y.index.intersection(x.index)
    y_aligned = y.loc[common_idx].astype(float)
    x_aligned = x.loc[common_idx].astype(float)
    n = len(y_aligned)

    X_mat = np.column_stack([x_aligned.values, np.ones(n)])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X_mat, y_aligned.values, rcond=None)
    hedge_ratio = float(coeffs[0])
    intercept = float(coeffs[1])

    spread_vals = y_aligned.values - hedge_ratio * x_aligned.values - intercept
    spread = pd.Series(spread_vals, index=common_idx, name="spread")

    adf_result = run_adf(spread)
    hl_result = estimate_half_life(spread, use_log=False)

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread": spread,
        "adf_stat": adf_result["adf_stat_aic"],
        "p_value": adf_result["p_value_aic"],
        "lambda_spread": hl_result["lambda"],
        "half_life_spread": hl_result["half_life"],
    }


def cadf_test_both_orders(y: pd.Series, x: pd.Series) -> dict:
    """双顺序 CADF: 试 y~x 和 x~y 两种顺序，取 ADF 统计量更负者。

    y~x 和 x~y 的 hedge_ratio 不对称 (h_yx ≠ 1/h_xy)。
    若 x~y 更优，将 hedge_ratio 取倒数使解释方向保持 y~x。
    """
    result_yx = cadf_test(y, x)
    result_xy = cadf_test(x, y)

    if result_yx["adf_stat"] < result_xy["adf_stat"]:
        return result_yx
    else:
        h_xy = result_xy["hedge_ratio"]
        if abs(h_xy) < 1e-12:
            return result_yx
        result_xy["hedge_ratio"] = 1.0 / h_xy
        return result_xy


# ============================================================
# Johansen 检验 + 组合构建
# ============================================================

def construct_portfolio(prices_df: pd.DataFrame, eigenvector: np.ndarray) -> pd.Series:
    """yport = Y · v"""
    yport = prices_df.values @ eigenvector
    return pd.Series(yport, index=prices_df.index)


def johansen_test(prices_df: pd.DataFrame, lag: int = 1) -> dict:
    """Johansen 多变量协整检验。

    1. Johansen MLE
    2. 按 trace 统计量确定秩 r
    3. 取第一特征向量构造组合净值
    4. 估计 yport 半衰期

    Returns:
        dict: eigenvalues, eigenvectors, trace_stats, trace_crit,
              rank, yport, half_life, is_cointegrated
    """
    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df 必须是 pd.DataFrame")
    if prices_df.shape[1] < 2:
        raise ValueError("至少需要 2 列资产")
    if prices_df.shape[0] < 20:
        raise ValueError("行数至少为 20")
    if prices_df.isna().any().any() or np.isinf(prices_df.values).any():
        raise ValueError("包含 NaN 或 Inf")

    n_series = prices_df.shape[1]
    result = coint_johansen(prices_df.values, det_order=0, k_ar_diff=lag)

    eigenvalues = result.eig
    eigenvectors = result.evec
    trace_stats = result.trace_stat
    trace_crit_95 = result.trace_stat_crit_vals[:, 1]

    rank = 0
    for i in range(n_series):
        if trace_stats[i] > trace_crit_95[i]:
            rank += 1
        else:
            break

    v1 = eigenvectors[:, 0]
    yport = construct_portfolio(prices_df, v1)
    hl_result = estimate_half_life(yport, use_log=False)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "trace_stats": trace_stats,
        "trace_crit": trace_crit_95,
        "rank": rank,
        "yport": yport,
        "half_life": hl_result["half_life"],
        "is_cointegrated": rank >= 1,
    }


# ============================================================
# 路径生成器
# ============================================================

def generate_cointegrated_paths(
    n_steps: int, n_series: int,
    ou_theta: float = 0.1, ou_sigma: float = 0.02,
    random_seed: int = 42,
) -> np.ndarray:
    """生成协整序列: yᵢ = common_RW + OU_noiseᵢ"""
    rng = np.random.default_rng(random_seed)
    common = np.cumsum(rng.standard_normal(n_steps) * 0.01)
    decay = np.exp(-ou_theta)
    noise_scale = ou_sigma * np.sqrt(1 - np.exp(-2 * ou_theta))
    z = np.zeros((n_series, n_steps))
    for t in range(n_steps - 1):
        z[:, t + 1] = z[:, t] * decay + noise_scale * rng.standard_normal(n_series)
    y = common[np.newaxis, :] + z
    return y.T


def generate_gbm_matrix(n_steps: int, n_series: int, sigma: float = 0.01,
                        random_seed: int = 42) -> np.ndarray:
    """独立 GBM 对数价格 (T × n)，无协整关系。"""
    rng = np.random.default_rng(random_seed)
    drift = -0.5 * sigma ** 2
    increments = drift + sigma * rng.standard_normal((n_steps - 1, n_series))
    paths = np.zeros((n_steps, n_series))
    paths[1:] = np.cumsum(increments, axis=0)
    return paths
