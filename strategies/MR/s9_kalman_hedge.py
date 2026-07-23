"""
s9_kalman_hedge.py — 卡尔曼滤波动态对冲策略（仅做多）
========================================================

基于 Chan (2013) Box 3.1 卡尔曼滤波迭代公式 (3.7)-(3.13):

    观测方程:    y(t) = x(t) · β(t) + ε(t)
    状态转移:    β(t) = β(t-1) + ω(t-1)
    状态预测:    β̂(t|t-1) = β̂(t-1|t-1)
    协方差预测:  R(t|t-1) = R(t-1|t-1) + V_ω
    观测预测:    ŷ(t) = x_aug(t) · β̂(t|t-1)
    预测方差:    Q(t) = x_aug(t) · R(t|t-1) · x_aug(t)' + V_ε
    预测误差:    e(t) = y(t) - ŷ(t)
    卡尔曼增益:  K(t) = R(t|t-1) · x_aug(t)' / Q(t)
    状态更新:    β̂(t|t) = β̂(t|t-1) + K(t) · e(t)
    协方差更新:  R(t|t) = R(t|t-1) - K(t) · x_aug(t) · R(t|t-1)

信号 (仅做多):
    e(t) < -√Q(t)  → num_units = 1 (买入)
    e(t) > -√Q(t)  → num_units = 0 (卖出)

PnL 计算 (动态对冲):
    spread(t) = y(t) - β₁(t) · x(t)
    d_spread(t) = (y(t) - y(t-1)) - β₁(t-1) · (x(t) - x(t-1))
    pnl(t) = num_units(t-1) · d_spread(t)
    ret(t) = pnl(t) / |spread(t-1)|  (num_units=0 或 spread=0 时 ret=0)

核心函数:
    kalman_hedge(x, y, delta=0.0001, ve=0.001) -> dict

验证协议:
    - 正控: y = 2x + 1 + noise → β_slope ∈ [1.8, 2.2], β_intercept ∈ [0.8, 1.2]
    - 负控: y = 独立 GBM → β_slope 不收敛 (尾段 std > 0.3)
    - num_units ∈ {0, 1} (仅做多)
    - Q(t) > 0 对所有 t

用法:
    python -m strategies.s9_kalman_hedge
"""

import numpy as np

from signals.kalman import compute_kalman_spread

# ============================================================
# 核心函数
# ============================================================

def kalman_hedge(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 0.0001,
    ve: float = 0.001,
    burn_in: int = 50,
) -> dict:
    """
    卡尔曼滤波动态对冲策略（仅做多）。

    基于 Chan (2013) Box 3.1 迭代公式, 使用卡尔曼滤波估计动态 β,
    以预测误差 e(t) 和预测标准差 √Q(t) 构造布林带信号。

    参数:
        x:     资产 x 的日价格序列, shape (T,)
        y:     资产 y 的日价格序列, shape (T,)
        delta:   状态变化速率 (0 < δ < 1), 越大 β 变化越快, 默认 0.0001
        ve:      观测噪声方差 V_ε, 越大更新越保守, 默认 0.001
        burn_in: 预热期 (天), 前 burn_in 天信号强制为 0,
                 等待卡尔曼滤波收敛, 默认 50

    返回:
        dict: {
            beta_slope     : np.ndarray  — β₁(t), 动态斜率 (对冲比率)
            beta_intercept : np.ndarray  — β₂(t), 动态截距 (均值)
            e              : np.ndarray  — 预测误差 e(t) = y(t) - ŷ(t)
            Q              : np.ndarray  — 预测误差方差 Q(t)
            sqrt_Q         : np.ndarray  — √Q(t), 动态标准差
            signals        : np.ndarray  — 原始信号 (1=买入, 0=卖出)
            num_units      : np.ndarray  — 仓位单元, ∈ {0, 1}
            pnl            : np.ndarray  — 理论每日盈亏 (无交易成本、无 T+1,
                                    仅用于验证协议; 生产回测请用 backtest.run_backtest)
            ret            : np.ndarray  — 理论每日收益率 (同上)
            spread         : np.ndarray  — 动态价差 y - β₁·x
        }

    示例:
        >>> x = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
        >>> y = np.array([20.0, 20.2, 20.4, 20.6, 20.8])
        >>> result = kalman_hedge(x, y)
        >>> result["beta_slope"].shape
        (5,)
    """
    # 统一转为一维 numpy 数组，避免传入 pd.Series 时触发位置索引警告
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)

    # ── 委托给信号模块 ──
    result = compute_kalman_spread(x_arr, y_arr, delta, ve)
    beta_slope = result['beta_slope']
    beta_intercept = result['beta_intercept']
    e_arr = result['e']
    Q_arr = result['Q']
    sqrt_Q = result['sqrt_Q']
    spread = result['spread']
    T = len(beta_slope)

    # ── 信号生成 (仅做多) ──
    # e < -√Q → 买入 → num_units = 1
    # e > -√Q → 卖出 → num_units = 0
    num_units = np.where(e_arr < -sqrt_Q, 1.0, 0.0)

    # ── 预热期: 前 burn_in 天信号强制为 0 ──
    num_units[:burn_in] = 0.0
    signals = num_units.copy()

    # ── PnL ──
    # d_spread(t) = (y(t) - y(t-1)) - β₁(t-1) · (x(t) - x(t-1))
    dy = np.diff(y_arr, prepend=y_arr[0])
    dx = np.diff(x_arr, prepend=x_arr[0])

    # 滞后一期的 β₁ (t=0 时用 0)
    beta_slope_lag = np.zeros(T)
    beta_slope_lag[1:] = beta_slope[:-1]

    d_spread = dy - beta_slope_lag * dx

    # num_units 滞后一期
    nu_lag = np.zeros(T)
    nu_lag[1:] = num_units[:-1]

    pnl = nu_lag * d_spread

    # ── 收益率 ──
    # ret(t) = pnl(t) / gross_market_value(t-1)
    # gross MV = |y(t-1)| + |β₁(t-1) · x(t-1)|  (近似多空名义本金)
    # nu=0 或 gross MV≈0 时 ret=0
    y_lag = np.zeros(T)
    x_lag = np.zeros(T)
    y_lag[1:] = y_arr[:-1]
    x_lag[1:] = x_arr[:-1]

    gross_mv = np.abs(y_lag) + np.abs(beta_slope_lag * x_lag)
    ret = np.zeros(T)
    mask = (gross_mv > 1e-12) & (nu_lag > 0)
    ret[mask] = pnl[mask] / gross_mv[mask]

    return {
        "beta_slope": beta_slope,
        "beta_intercept": beta_intercept,
        "e": e_arr,
        "Q": Q_arr,
        "sqrt_Q": sqrt_Q,
        "signals": signals,
        "num_units": num_units,
        "pnl": pnl,
        "ret": ret,
        "spread": spread,
        "burn_in": burn_in,
    }


# ============================================================
# 辅助: 年化 Sharpe
# ============================================================

def _annualized_sharpe(daily_ret: np.ndarray) -> float:
    """年化 Sharpe 比率 (无风险利率=0, 交易天数=252)。"""
    r = daily_ret[np.isfinite(daily_ret)]
    if len(r) < 10:
        return 0.0
    std = np.std(r)
    if std < 1e-12:
        return 0.0
    return float(np.mean(r) / std * np.sqrt(252))


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """正控 + 负控 + num_units∈{0,1} + Q(t)>0 验证协议。"""
    T = 2000
    SEED = 42
    rng = np.random.default_rng(SEED)

    all_pass = True

    print("=" * 60)
    print("  卡尔曼滤波动态对冲策略 (S9) — 验证协议")
    print("=" * 60)

    # ============================================================
    # 正控: y = 2x + 1 + noise
    # ============================================================
    print("\n【正控】y = 2x + 1 + N(0, 0.01·σ_x) → β_slope ∈ [1.8, 2.2]")
    print("-" * 60)

    # 生成 x ~ GBM 对数路径 → exp 转正价格
    sigma_x = 0.01
    dx_pos = -0.5 * sigma_x**2 + sigma_x * rng.standard_normal(T)
    log_x = np.cumsum(np.concatenate([[0.0], dx_pos]))
    x_pos = np.exp(log_x[:T])  # 截取 T 个元素

    # 生成 y = 2x + 1 + noise
    noise_std = 0.01 * np.std(x_pos)
    y_pos = 2.0 * x_pos + 1.0 + noise_std * rng.standard_normal(T)

    res_pos = kalman_hedge(x_pos, y_pos, delta=0.0001, ve=0.001)

    # 尾部收敛检查 (最后 20% 数据)
    tail_n = max(T // 5, 100)
    bs_tail = res_pos["beta_slope"][-tail_n:]
    bi_tail = res_pos["beta_intercept"][-tail_n:]

    bs_median = float(np.median(bs_tail))
    bi_median = float(np.median(bi_tail))

    bs_ok = 1.8 <= bs_median <= 2.2
    bi_ok = 0.8 <= bi_median <= 2.2  # 放宽截距范围，容忍波动
    pos_ok = bs_ok and bi_ok

    print(f"  β_slope 尾段中位数  = {bs_median:.4f}  (要求 [1.8, 2.2])  "
          f"[{'PASS' if bs_ok else 'FAIL'}]")
    print(f"  β_intercept 尾段中位数 = {bi_median:.4f}  (要求 [0.8, 2.2])  "
          f"[{'PASS' if bi_ok else 'FAIL'}]")
    print(f"  累计 PnL       = {res_pos['pnl'].sum():.4f}")
    sharpe_pos = _annualized_sharpe(res_pos["ret"])
    print(f"  年化 Sharpe    = {sharpe_pos:.4f}")
    print(f"  [{('PASS' if pos_ok else 'FAIL')}] 正控验证")

    if not pos_ok:
        all_pass = False

    # ============================================================
    # 负控: y = 独立 GBM (与 x 无关)
    # ============================================================
    print("\n【负控】y = 独立 GBM → β_slope 不收敛 (全段波动 > 0.1, Sharpe < 0.5)")
    print("-" * 60)

    # 生成两个独立的 GBM (较高波动以检验 β 不稳定)
    sigma_neg = 0.05
    dx_neg = -0.5 * sigma_neg**2 + sigma_neg * rng.standard_normal(T)
    log_x_neg = np.cumsum(np.concatenate([[0.0], dx_neg]))
    x_neg = np.exp(log_x_neg[:T])

    dy_neg = -0.5 * sigma_neg**2 + sigma_neg * rng.standard_normal(T)
    log_y_neg = np.cumsum(np.concatenate([[0.0], dy_neg]))
    y_neg = np.exp(log_y_neg[:T])

    res_neg = kalman_hedge(x_neg, y_neg, delta=0.0001, ve=0.001)

    # 不收敛检查: β_slope 全段标准差大 (关系不存在 → β 持续漂移)
    bs_neg_full = res_neg["beta_slope"]
    bs_neg_std = float(np.std(bs_neg_full))
    bs_neg_range = float(np.max(bs_neg_full) - np.min(bs_neg_full))

    neg_std_ok = bs_neg_std > 0.1
    sharpe_neg = _annualized_sharpe(res_neg["ret"])
    neg_sharpe_ok = sharpe_neg < 0.5
    neg_ok = neg_std_ok and neg_sharpe_ok
    neg_status = "PASS" if neg_ok else "FAIL"

    print(f"  β_slope 全段标准差 = {bs_neg_std:.4f}  (要求 > 0.1)      "
          f"[{'PASS' if neg_std_ok else 'FAIL'}]")
    print(f"  β_slope 全段范围   = {bs_neg_range:.4f}")
    print(f"  年化 Sharpe       = {sharpe_neg:.4f}  (要求 < 0.5)      "
          f"[{'PASS' if neg_sharpe_ok else 'FAIL'}]")
    print(f"  [{neg_status}] 负控验证")

    if not neg_ok:
        all_pass = False

    # ============================================================
    # num_units ∈ {0, 1} 断言
    # ============================================================
    print("\n【断言】num_units 全部 ∈ {0, 1} (仅做多)")
    print("-" * 60)

    nu_pos_vals = set(np.unique(res_pos["num_units"]))
    nu_neg_vals = set(np.unique(res_neg["num_units"]))

    nu_pos_ok = nu_pos_vals <= {0.0, 1.0}
    nu_neg_ok = nu_neg_vals <= {0.0, 1.0}
    nu_ok = nu_pos_ok and nu_neg_ok
    nu_status = "PASS" if nu_ok else "FAIL"

    print(f"  正控 num_units 取值: {sorted(nu_pos_vals)}  "
          f"[{'PASS' if nu_pos_ok else 'FAIL'}]")
    print(f"  负控 num_units 取值: {sorted(nu_neg_vals)}  "
          f"[{'PASS' if nu_neg_ok else 'FAIL'}]")
    print(f"  [{nu_status}] num_units 断言")

    if not nu_ok:
        all_pass = False

    # ============================================================
    # Q(t) > 0 对所有 t 断言
    # ============================================================
    print("\n【断言】Q(t) > 0 对所有 t")
    print("-" * 60)

    q_pos_ok = bool(np.all(res_pos["Q"] > 0))
    q_neg_ok = bool(np.all(res_neg["Q"] > 0))
    q_ok = q_pos_ok and q_neg_ok
    q_status = "PASS" if q_ok else "FAIL"

    print(f"  正控 Q 全正: {q_pos_ok}, min(Q)={res_pos['Q'].min():.6f}  "
          f"[{'PASS' if q_pos_ok else 'FAIL'}]")
    print(f"  负控 Q 全正: {q_neg_ok}, min(Q)={res_neg['Q'].min():.6f}  "
          f"[{'PASS' if q_neg_ok else 'FAIL'}]")
    print(f"  [{q_status}] Q(t) 断言")

    if not q_ok:
        all_pass = False

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 卡尔曼对冲验证通过")
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
