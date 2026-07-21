"""
s2_hurst.py — Hurst 指数估计 (方差法)
======================================

Hurst 指数 H 刻画时间序列的长程记忆特性:
    H < 0.5  → 均值回复 (mean-reverting)
    H = 0.5  → 随机游走 (random walk)
    H > 0.5  → 趋势持续 (trending)

核心公式 (Chan, Algorithmic Trading, Example 2.2):
    Var(τ) = ⟨|z(t+τ) - z(t)|²⟩  ∝  τ^(2H)

其中 z = ln(prices)，对两边取对数:
    log Var(τ) = 2H · log(τ) + const

通过 OLS 拟合斜率 → H = slope / 2

用法:
    from tests.s2_hurst import hurst_exponent

    result = hurst_exponent(prices_series)
    print(result['hurst'], result['r_squared'])
"""

import numpy as np
import pandas as pd


def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> dict:
    """用方差法估计 Hurst 指数。

    对价格序列取对数后，计算不同滞后 τ 下的增量方差，
    通过对数线性回归拟合 Var(τ) ∝ τ^(2H) 得到 H。

    Parameters
    ----------
    prices : pd.Series
        价格序列 (必须为正数，内部自动取 log)
    max_lag : int
        最大滞后阶数，默认 100

    Returns
    -------
    dict
        hurst      : float, Hurst 指数
        r_squared  : float, 线性拟合 R² (越高越可信)
        lags_used  : np.ndarray, 实际使用的滞后值
        variances  : np.ndarray, 各滞后对应的增量方差
    """
    # --- 输入验证 ---
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices 必须是 pd.Series, 实际为 {type(prices)}")
    if len(prices) < 10:
        raise ValueError(f"prices 长度至少为 10, 实际为 {len(prices)}")
    vals = prices.to_numpy(dtype=float)
    if np.isnan(vals).any() or np.isinf(vals).any():
        raise ValueError("prices 包含 NaN 或 Inf, 请先清洗数据")
    if (vals <= 0).any():
        raise ValueError("prices 必须全部为正数 (Hurst 对价格取 log)")
    if max_lag < 2:
        raise ValueError(f"max_lag 至少为 2, 实际为 {max_lag}")

    # ── 对数转换: 必须对 log-prices 做 Hurst，不能直接用于原始价格 ──
    z = np.log(vals)
    n = len(z)

    # max_lag 不能超过数据长度减 1 (否则无增量可计算)
    max_lag = min(max_lag, n - 1)

    # ── 构造对数等间距滞后: 从 2 到 max_lag，约 20 个采样点 ──
    # 取整后去重，避免小 lag 区域重复
    raw_lags = np.logspace(np.log10(2), np.log10(max_lag), 20).astype(int)
    lags = np.unique(raw_lags)

    # ── 逐滞后计算增量方差: Var(τ) = mean(|z(t+τ) - z(t)|²) ──
    variances = np.empty(len(lags))
    for i, tau in enumerate(lags):
        increments = z[tau:] - z[:-tau]
        variances[i] = np.mean(increments ** 2)

    # ── 对数线性回归: log Var(τ) = 2H · log(τ) + const ──
    log_lags = np.log(lags)
    log_vars = np.log(variances)

    # polyfit 返回 [slope, intercept]
    slope, _ = np.polyfit(log_lags, log_vars, 1)
    hurst = slope / 2.0

    # ── 拟合优度 R² ──
    corr = np.corrcoef(log_lags, log_vars)[0, 1]
    r_squared = corr ** 2

    return {
        'hurst': float(hurst),
        'r_squared': float(r_squared),
        'lags_used': lags,
        'variances': variances,
    }


# ============================================================
#  Smoke Test: 用已知过程验证 Hurst 估计的正确性
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    # ── GBM (几何布朗运动): log-prices 为随机游走 → H ≈ 0.5 ──
    # S(t) = S0 · exp(W(t))，其中 W(t) = cumsum(ε), ε ~ N(0,1)
    log_prices_gbm = np.cumsum(np.random.randn(5000))
    prices_gbm = np.exp(log_prices_gbm)  # 转为正价格序列
    res_gbm = hurst_exponent(pd.Series(prices_gbm), max_lag=200)

    print("═══ GBM (随机游走) ═══")
    print(f"  H         = {res_gbm['hurst']:.4f}  (期望 ≈ 0.5)")
    print(f"  R^2       = {res_gbm['r_squared']:.4f}")
    print(f"  滞后数    = {len(res_gbm['lags_used'])}")

    # ── OU 过程 (Ornstein-Uhlenbeck): 均值回复 → H < 0.5 ──
    # dx = θ(μ - x)dt + σ dW,  取 θ=0.1, μ=0, σ=1
    theta, mu, sigma = 0.1, 0.0, 1.0
    n_steps = 5000
    x = np.zeros(n_steps)
    for t in range(1, n_steps):
        dw = np.random.randn()
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * dw
    prices_ou = np.exp(x)  # 对数价格 → 正价格
    res_ou = hurst_exponent(pd.Series(prices_ou), max_lag=200)

    print("\n═══ OU 过程 (均值回复) ═══")
    print(f"  H         = {res_ou['hurst']:.4f}  (期望 < 0.5)")
    print(f"  R^2       = {res_ou['r_squared']:.4f}")
    print(f"  滞后数    = {len(res_ou['lags_used'])}")

    # ── 断言验证 ──
    assert 0.45 <= res_gbm['hurst'] <= 0.55, \
        f"GBM Hurst 异常: {res_gbm['hurst']:.4f}, 期望 [0.45, 0.55]"
    assert res_ou['hurst'] < 0.5, \
        f"OU Hurst 异常: {res_ou['hurst']:.4f}, 期望 < 0.5"

    print("\n[OK] 全部断言通过")
