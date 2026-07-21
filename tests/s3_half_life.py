"""
s3_half_life.py — 均值回归半衰期估计与 OU/GBM 验证
===================================================

核心函数:
    estimate_half_life(prices, use_log=True) -> dict

原理:
    对对数价格序列做 OLS 回归:
        Δy = λ · y_lag + μ + ε
    其中 y = log(price), Δy = y.diff(), y_lag = y.shift(1)

    离散精确半衰期公式 (Metis B1):
        half_life = -ln(2) / ln(1 + λ)

    注意: 不能用连续近似 -ln(2)/λ, 在短半衰期 (2-60天) 误差可达 23%

验证协议:
    - OU 正控: 10,000 条精确离散化路径, 估计半衰期偏差 < ±20%
    - GBM 负控: 10,000 条随机游走路径, ≥95% 估计半衰期 > 500 天

用法:
    python tests/s3_half_life.py
"""

import numpy as np
import pandas as pd

# ============================================================
# 半衰期估计
# ============================================================

def estimate_half_life(prices: pd.Series, use_log: bool = True) -> dict:
    """
    估计价格序列的均值回归半衰期。

    参数:
        prices:  价格序列 (pd.Series)
        use_log: 是否取对数 (默认 True, 对数收益率下更合理)

    返回:
        dict with keys:
            lambda          — OLS 回归系数 (负值表示均值回归)
            half_life       — 半衰期 (天), 非均值回归时为 np.inf
            r_squared       — 回归拟合度
            is_mean_reverting — 是否存在均值回归 (λ < 0)
            n_obs           — 有效观测数
    """
    # --- 输入验证 ---
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices 必须是 pd.Series, 实际为 {type(prices)}")
    if len(prices) < 11:
        raise ValueError(f"prices 长度至少为 11 (保证有效观测 ≥10), 实际为 {len(prices)}")
    vals = prices.to_numpy(dtype=float)
    if np.isnan(vals).any() or np.isinf(vals).any():
        raise ValueError("prices 包含 NaN 或 Inf, 请先清洗数据")
    if use_log and (vals <= 0).any():
        raise ValueError("use_log=True 时 prices 必须全部为正数")

    # Step 1: 构造回归变量
    y_arr = np.log(vals) if use_log else vals.copy()
    y = pd.Series(y_arr, index=prices.index)

    delta_y = y.diff().iloc[1:]       # Δy_t = y_t - y_{t-1}
    y_lag = y.shift(1).iloc[1:]       # y_{t-1}

    # 对齐 (diff 和 shift 各丢一个, iloc[1:] 后长度一致)
    n_obs = len(delta_y)
    if n_obs < 10:
        return {
            "lambda": np.nan,
            "half_life": np.inf,
            "r_squared": np.nan,
            "is_mean_reverting": False,
            "n_obs": n_obs,
        }

    # Step 2: OLS — Δy = λ · y_lag + μ
    # 设计矩阵 [y_lag, 1], 求解最小二乘
    delta_arr = delta_y.to_numpy(dtype=float)
    y_lag_arr = y_lag.to_numpy(dtype=float)
    X = np.column_stack([y_lag_arr, np.ones(n_obs)])
    result, _, _, _ = np.linalg.lstsq(X, delta_arr, rcond=None)
    lam = result[0]

    # Step 3: R²
    y_hat = X @ result
    ss_res = np.sum((delta_arr - y_hat) ** 2)
    ss_tot = np.sum((delta_arr - delta_arr.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Step 4: 半衰期 — 离散精确公式
    is_mean_reverting = lam < 0
    half_life = -np.log(2) / np.log(1 + lam) if is_mean_reverting else np.inf

    return {
        "lambda": lam,
        "half_life": half_life,
        "r_squared": r_squared,
        "is_mean_reverting": is_mean_reverting,
        "n_obs": n_obs,
    }


# ============================================================
# 路径生成器
# ============================================================

def generate_ou_paths(n_paths, n_steps, theta, mu, sigma, dt=1.0, seed=None):
    """
    精确离散化 OU 过程 (避免 Euler-Maruyama 的 O(Δt) 偏差)。

    x[t+1] = μ + (x[t] - μ) · exp(-θ·dt) + σ · √(1 - exp(-2θ·dt)) · ε

    参数:
        n_paths:  路径数量
        n_steps:  每条路径的步数
        theta:    均值回归速度 (>0)
        mu:       长期均值
        sigma:    波动率
        dt:       时间步长 (默认 1 天)
        seed:     随机种子
    """
    rng = np.random.default_rng(seed)
    x = np.full((n_paths, n_steps), mu, dtype=float)

    decay = np.exp(-theta * dt)
    noise_scale = sigma * np.sqrt(1 - np.exp(-2 * theta * dt))

    for t in range(n_steps - 1):
        x[:, t + 1] = mu + (x[:, t] - mu) * decay + noise_scale * rng.standard_normal(n_paths)

    return x


def generate_gbm_paths(n_paths, n_steps, sigma, dt=1.0, seed=None):
    """
    几何布朗运动 (GBM) — 负控: 无均值回归的随机游走。

    x[t+1] = x[t] - 0.5·σ²·dt + σ·√dt·ε
    (对数形式, 等价于 dS = σS dW)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_paths, n_steps), dtype=float)

    drift = -0.5 * sigma ** 2 * dt
    diffusion = sigma * np.sqrt(dt)

    increments = drift + diffusion * rng.standard_normal((n_paths, n_steps - 1))
    x[:, 1:] = np.cumsum(increments, axis=1)

    return x


# ============================================================
# 验证协议
# ============================================================

def run_validation():
    """
    OU + GBM 验证协议。

    正控: OU 过程, 已知半衰期, 验证估计精度
    负控: GBM 过程, 无均值回归, 验证不误判
    """
    N_PATHS = 10_000
    N_STEPS = 1200
    MU = 0.0
    SIGMA = 1.0

    # OU 真值参数: θ → 理论半衰期
    ou_configs = [
        (0.0693,  10),   # θ=ln(2)/10 → HL=10
        (0.03466, 20),   # θ=ln(2)/20 → HL=20
        (0.01386, 50),   # θ=ln(2)/50 → HL=50
    ]

    print("=" * 60)
    print("半衰期估计验证协议")
    print("=" * 60)

    all_pass = True

    # --- 正控: OU 过程 ---
    print("\n【正控】OU 过程半衰期估计 (N=10,000 paths, T=1,200 steps)")
    print("-" * 60)

    for theta, true_hl in ou_configs:
        paths = generate_ou_paths(N_PATHS, N_STEPS, theta, MU, SIGMA, dt=1.0, seed=42)

        estimates = np.empty(N_PATHS)
        for i in range(N_PATHS):
            # OU 路径本身在对数空间, 直接传入 (use_log=False)
            res = estimate_half_life(pd.Series(paths[i]), use_log=False)
            estimates[i] = res["half_life"]

        # 排除 inf (不应出现, 但防御性处理)
        finite = estimates[np.isfinite(estimates)]
        mean_est = np.mean(finite)
        median_est = np.median(finite)
        pct_error = abs(mean_est - true_hl) / true_hl

        status = "PASS" if pct_error < 0.20 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  θ={theta:.4f}  真值HL={true_hl:3d}d  "
              f"mean={mean_est:6.2f}  median={median_est:6.2f}  "
              f"误差={pct_error:.1%}  [{status}]")

    # --- 负控: GBM 过程 ---
    print("\n【负控】GBM 随机游走 (N=10,000 paths, T=1,200 steps)")
    print("-" * 60)

    gbm_paths = generate_gbm_paths(N_PATHS, N_STEPS, SIGMA, dt=1.0, seed=42)

    gbm_estimates = np.empty(N_PATHS)
    for i in range(N_PATHS):
        # GBM 路径本身就在对数空间, 直接传入 (use_log=False)
        # 避免 exp() 溢出导致 float64 问题
        res = estimate_half_life(pd.Series(gbm_paths[i]), use_log=False)
        gbm_estimates[i] = res["half_life"]

    # 判定: 多少比例估计半衰期 > 500 天
    pct_long = np.mean(gbm_estimates > 500) * 100
    status_gbm = "PASS" if pct_long >= 95.0 else "FAIL"
    if status_gbm == "FAIL":
        all_pass = False

    print(f"  HL > 500d 比例: {pct_long:.1f}%  "
          f"(要求 ≥95%)  [{status_gbm}]")
    print(f"  GBM mean HL: {np.mean(gbm_estimates[np.isfinite(gbm_estimates)]):.1f}d  "
          f"median: {np.median(gbm_estimates[np.isfinite(gbm_estimates)]):.1f}d")

    # --- 汇总 ---
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 全部验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    success = run_validation()
    if not success:
        raise SystemExit(1)
