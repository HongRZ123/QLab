"""
kalman_spread.py -- Kalman 滤波 spread 信号提取
================================================

从价格序列 (x, y) 中提取卡尔曼滤波动态对冲比率与 spread,
不含任何交易决策逻辑。

基于 Chan (2013) Box 3.1 卡尔曼滤波迭代公式 (3.7)-(3.13)。
"""

import numpy as np


def compute_kalman_spread(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 0.0001,
    ve: float = 0.001,
) -> dict:
    """
    卡尔曼滤波 spread 信号。

    参数:
        x:     资产 x 的日价格序列, shape (T,)
        y:     资产 y 的日价格序列, shape (T,)
        delta: 状态变化速率 (0 < δ < 1), 越大 β 变化越快, 默认 0.0001
        ve:    观测噪声方差 V_ε, 越大更新越保守, 默认 0.001

    返回:
        dict: {
            beta_slope     : np.ndarray -- β₁(t), 动态斜率 (对冲比率)
            beta_intercept : np.ndarray -- β₂(t), 动态截距
            e              : np.ndarray -- 预测误差 e(t) = y(t) - ŷ(t)
            Q              : np.ndarray -- 预测误差方差 Q(t)
            sqrt_Q         : np.ndarray -- √Q(t), 动态标准差
            spread         : np.ndarray -- y - β₁·x
        }
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if len(x) != T:
        raise ValueError(f"x 和 y 长度必须相等: len(x)={len(x)}, len(y)={T}")

    # ── 增广 x: 加入全1列以容纳截距 ──
    x_aug = np.column_stack([x, np.ones(T)])  # shape (T, 2)

    # ── 初始化 ──
    # β̂(1|0) = [0, 0], R(1|0) = 0_{2×2}
    beta = np.zeros(2)           # 当前 β̂(t|t-1) 或 β̂(t|t)
    R = np.zeros((2, 2))         # 当前 R(t|t-1) 或 R(t|t)
    Vw = delta / (1.0 - delta) * np.eye(2)  # 状态噪声协方差
    Ve = ve                                # 观测噪声方差

    # ── 存储 ──
    beta_slope = np.zeros(T)
    beta_intercept = np.zeros(T)
    e_arr = np.zeros(T)
    Q_arr = np.zeros(T)

    # ── 每日迭代 (t = 0..T-1, 对应 Chan 的 t=1..T) ──
    for t in range(T):
        # ── 预测步 (t > 0 时执行) ──
        if t > 0:
            # (3.8) R(t|t-1) = R(t-1|t-1) + V_ω
            R = R + Vw

        x_t = x_aug[t]  # shape (2,)

        # (3.9) ŷ(t) = x(t) · β̂(t|t-1)
        yhat = np.dot(x_t, beta)

        # (3.10) Q(t) = x(t) · R(t|t-1) · x(t)' + V_ε
        Q_t = np.dot(x_t, np.dot(R, x_t)) + Ve
        Q_arr[t] = Q_t

        # e(t) = y(t) - ŷ(t)
        e_t = y[t] - yhat
        e_arr[t] = e_t

        # (3.13) K(t) = R(t|t-1) · x(t)' / Q(t)
        K = np.dot(R, x_t) / Q_t

        # (3.11) β̂(t|t) = β̂(t|t-1) + K(t) · e(t)
        beta = beta + K * e_t

        # (3.12) R(t|t) = R(t|t-1) - K(t) · x(t) · R(t|t-1)
        R = R - np.outer(K, np.dot(x_t, R))

        # 保存 β₁(t) (斜率) 和 β₂(t) (截距)
        beta_slope[t] = beta[0]
        beta_intercept[t] = beta[1]

    # ── 后处理 ──
    sqrt_Q = np.sqrt(Q_arr)
    spread = y - beta_slope * x

    return {
        "beta_slope": beta_slope,
        "beta_intercept": beta_intercept,
        "e": e_arr,
        "Q": Q_arr,
        "sqrt_Q": sqrt_Q,
        "spread": spread,
    }
