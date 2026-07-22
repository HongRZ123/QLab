"""
signals/stats.py — 统计性质信号

单资产价格序列的统计属性提取。这些是"信号的信号"——不直接描述市场行为，
而是描述"这个价格序列本身有什么统计特征"，供 alpha 模块做截面筛选，
供 strategies 的 run_validation() 生成合成数据。

组成（从 tests/s1~s3 迁移而来）:
    run_adf(prices) -> dict          ADF 单位根检验
    hurst_exponent(prices) -> dict   Hurst 指数估计
    estimate_half_life(prices) -> dict  半衰期估计
    generate_ou_paths(...) -> ndarray    OU 过程路径生成器
    generate_gbm_paths(...) -> ndarray   GBM 路径生成器

消费方:
    alpha/stationarity.py  — 平稳性评分 + 筛选
    explore/scan_*         — 全市场扫描
    strategies/MR/*        — run_validation() 正控/负控
    run/run_*            — 端到端研究脚本
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ============================================================
# run_adf — ADF 单位根检验
# ============================================================

def run_adf(prices: pd.Series) -> dict:
    """对价格序列执行 ADF 单位根检验 (AIC / BIC 双准则)。

    H0: 存在单位根 → 序列非平稳
    H1: 不存在单位根 → 序列平稳

    Returns:
        dict: adf_stat_aic, p_value_aic, adf_stat_bic, p_value_bic,
              lambda_adf, used_lag_aic, used_lag_bic, n_obs,
              critical_1pct, critical_5pct, critical_10pct
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices 必须是 pd.Series, 实际为 {type(prices)}")
    if len(prices) < 7:
        raise ValueError(f"prices 长度至少为 7, 实际为 {len(prices)}")

    REG = "c"
    result_aic: tuple = adfuller(
        prices, maxlag=None, autolag="AIC", regression=REG, store=True, regresults=True,
    )
    stat_aic, pval_aic, crit_aic, res_aic = result_aic

    result_bic: tuple = adfuller(
        prices, maxlag=None, autolag="BIC", regression=REG, store=True, regresults=True,
    )
    stat_bic, pval_bic, crit_bic, res_bic = result_bic

    return {
        "adf_stat_aic": stat_aic,
        "p_value_aic": pval_aic,
        "adf_stat_bic": stat_bic,
        "p_value_bic": pval_bic,
        "lambda_adf": res_aic.resols.params[0],
        "used_lag_aic": res_aic.usedlag,
        "used_lag_bic": res_bic.usedlag,
        "n_obs": res_aic.nobs,
        "critical_1pct": crit_aic["1%"],
        "critical_5pct": crit_aic["5%"],
        "critical_10pct": crit_aic["10%"],
    }


# ============================================================
# hurst_exponent — Hurst 指数估计
# ============================================================

def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> dict:
    """用方差法估计 Hurst 指数 H。

    H < 0.5 → 均值回归,  H ≈ 0.5 → 随机游走,  H > 0.5 → 趋势。

    Returns:
        dict: hurst, r_squared, lags_used, variances
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices 必须是 pd.Series, 实际为 {type(prices)}")
    if len(prices) < 10:
        raise ValueError(f"prices 长度至少为 10, 实际为 {len(prices)}")

    vals = prices.to_numpy(dtype=float)
    if np.isnan(vals).any() or np.isinf(vals).any():
        raise ValueError("prices 包含 NaN 或 Inf")
    if (vals <= 0).any():
        raise ValueError("prices 必须全部为正数 (Hurst 对价格取 log)")
    if max_lag < 2:
        raise ValueError("max_lag 至少为 2")

    z = np.log(vals)
    n = len(z)
    max_lag = min(max_lag, n - 1)

    raw_lags = np.logspace(np.log10(2), np.log10(max_lag), 20).astype(int)
    lags = np.unique(raw_lags)
    variances = np.empty(len(lags))
    for i, tau in enumerate(lags):
        variances[i] = np.mean((z[tau:] - z[:-tau]) ** 2)

    log_lags = np.log(lags)
    log_vars = np.log(variances)
    slope, _ = np.polyfit(log_lags, log_vars, 1)
    corr = np.corrcoef(log_lags, log_vars)[0, 1]

    return {
        "hurst": float(slope / 2.0),
        "r_squared": float(corr ** 2),
        "lags_used": lags,
        "variances": variances,
    }


# ============================================================
# estimate_half_life — 半衰期估计
# ============================================================

def estimate_half_life(prices: pd.Series, use_log: bool = True) -> dict:
    """估计价格序列的均值回归半衰期 (离散精确公式)。

    Δy = λ · y_lag + μ + ε
    half_life = -ln(2) / ln(1 + λ)   (λ < 0 时)

    Returns:
        dict: lambda, half_life, r_squared, is_mean_reverting, n_obs
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices 必须是 pd.Series, 实际为 {type(prices)}")
    if len(prices) < 11:
        raise ValueError("prices 长度至少为 11")

    vals = prices.to_numpy(dtype=float)
    if np.isnan(vals).any() or np.isinf(vals).any():
        raise ValueError("prices 包含 NaN 或 Inf")
    if use_log and (vals <= 0).any():
        raise ValueError("use_log=True 时 prices 必须全部为正数")

    y_arr = np.log(vals) if use_log else vals.copy()
    y = pd.Series(y_arr, index=prices.index)
    delta_y = y.diff().iloc[1:]
    y_lag = y.shift(1).iloc[1:]
    n_obs = len(delta_y)

    if n_obs < 10:
        return {"lambda": np.nan, "half_life": np.inf, "r_squared": np.nan,
                "is_mean_reverting": False, "n_obs": n_obs}

    delta_arr = delta_y.to_numpy(dtype=float)
    y_lag_arr = y_lag.to_numpy(dtype=float)
    X = np.column_stack([y_lag_arr, np.ones(n_obs)])
    result, _, _, _ = np.linalg.lstsq(X, delta_arr, rcond=None)
    lam = result[0]

    y_hat = X @ result
    ss_res = np.sum((delta_arr - y_hat) ** 2)
    ss_tot = np.sum((delta_arr - delta_arr.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    is_mr = lam < 0
    half_life = -np.log(2) / np.log(1 + lam) if is_mr else np.inf

    return {"lambda": lam, "half_life": half_life, "r_squared": r_squared,
            "is_mean_reverting": is_mr, "n_obs": n_obs}


# ============================================================
# 路径生成器 (run_validation() 用)
# ============================================================

def generate_ou_paths(n_paths: int, n_steps: int, theta: float, mu: float,
                      sigma: float, dt: float = 1.0, seed: int | None = None) -> np.ndarray:
    """精确离散化 OU 过程: x[t+1] = μ + (x[t]-μ)·exp(-θ) + σ·√(1-exp(-2θ))·ε"""
    rng = np.random.default_rng(seed)
    x = np.full((n_paths, n_steps), mu, dtype=float)
    decay = np.exp(-theta * dt)
    noise_scale = sigma * np.sqrt(1 - np.exp(-2 * theta * dt))
    for t in range(n_steps - 1):
        x[:, t + 1] = mu + (x[:, t] - mu) * decay + noise_scale * rng.standard_normal(n_paths)
    return x


def generate_gbm_paths(n_paths: int, n_steps: int, sigma: float,
                       dt: float = 1.0, seed: int | None = None) -> np.ndarray:
    """几何布朗运动: 对数形式 x[t+1] = x[t] - 0.5·σ² + σ·ε"""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_paths, n_steps), dtype=float)
    drift = -0.5 * sigma ** 2 * dt
    diffusion = sigma * np.sqrt(dt)
    increments = drift + diffusion * rng.standard_normal((n_paths, n_steps - 1))
    x[:, 1:] = np.cumsum(increments, axis=1)
    return x
