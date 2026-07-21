"""
s6_johansen.py — Johansen 协整检验 + 组合构建
===============================================

核心函数:
    johansen_test(prices_df, lag=1) -> dict
    construct_portfolio(prices_df, eigenvector) -> pd.Series

原理:
    Johansen 检验推广了单变量 ADF 到多变量情形:
        ΔY(t) = Λ·Y(t-1) + M + Σ Aᵢ·ΔY(t-i) + ε

    Λ 的秩 r = 独立的协整关系数。
    特征向量 (按特征值降序) = 对冲比率。
    第一特征向量 v₁ → 最短半衰期组合。

验证协议:
    - 正控: 3 协整序列 (共享随机游走因子 + 独立 OU 噪声) → rank ≥ 1
    - 负控: 3 独立 GBM → rank = 0
    - yport 半衰期 < 输入序列半衰期

用法:
    python -m tests.s6_johansen
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from tests.s3_half_life import estimate_half_life

# ============================================================
# 组合构建
# ============================================================

def construct_portfolio(prices_df: pd.DataFrame, eigenvector: np.ndarray) -> pd.Series:
    """
    用特征向量构造组合净值序列。

    公式: yport(t) = Σ vᵢ · yᵢ(t)

    其中 vᵢ 是特征向量的第 i 个分量, yᵢ(t) 是第 i 个价格序列 (对数空间)。

    参数:
        prices_df:   价格矩阵 (T × n), 已在 (对数) 价格空间
        eigenvector: 特征向量 (长度 n)

    返回:
        组合净值序列 (T 长度)
    """
    # 矩阵乘法: Y · v = Σ vᵢ · yᵢ
    yport = prices_df.values @ eigenvector
    return pd.Series(yport, index=prices_df.index)


# ============================================================
# Johansen 检验
# ============================================================

def johansen_test(prices_df: pd.DataFrame, lag: int = 1) -> dict:
    """
    Johansen 协整检验, 并构造平稳组合。

    步骤:
        1. 运行 Johansen MLE
        2. 按 trace 统计量确定协整秩 r
        3. 取第一特征向量 v₁ (最大特征值 → 最短半衰期)
        4. 构造 yport = Y · v₁
        5. 对 yport 估计半衰期

    参数:
        prices_df: 价格矩阵 (T × n), 列 = 资产, 行 = 时间.
                   由于 Johansen 检验针对价格水平进行, 调用方通常应传入对数价格
                   (log prices) 以保证线性组合的可解释性.
        lag:       差分滞后阶数 (默认 1)

    返回:
        dict with keys:
            eigenvalues     — 特征值 (降序)
            eigenvectors    — 特征向量矩阵 (n × n), 列对应特征值
            trace_stats     — trace 统计量 (长度 n)
            trace_crit      — trace 统计量 95% 临界值
            rank            — 协整秩 (显著协整关系数)
            yport           — 第一特征向量构造的组合净值
            half_life       — yport 的半衰期 (天)
            is_cointegrated — 是否存在至少一个协整关系
    """
    # --- 输入验证 ---
    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError(f"prices_df 必须是 pd.DataFrame, 实际为 {type(prices_df)}")
    if prices_df.shape[1] < 2:
        raise ValueError(f"prices_df 至少需要 2 列资产, 实际为 {prices_df.shape[1]}")
    if prices_df.shape[0] < 20:
        raise ValueError(f"prices_df 行数至少为 20, 实际为 {prices_df.shape[0]}")
    if prices_df.isna().any().any() or np.isinf(prices_df.values).any():
        raise ValueError("prices_df 包含 NaN 或 Inf, 请先清洗数据")

    if lag < 1:
        raise ValueError(f"lag 必须 >= 1, 实际为 {lag}")

    n_series = prices_df.shape[1]

    # Step 1: Johansen MLE
    # det_order=0: 允许常数项 (M ≠ 0), 但无时间趋势 (β=0)
    result = coint_johansen(prices_df.values, det_order=0, k_ar_diff=lag)

    eigenvalues = result.eig                        # 降序
    eigenvectors = result.evec                      # 列 = 特征向量, 按特征值降序
    trace_stats = result.trace_stat                 # trace 统计量
    # trace_stat_crit_vals: (n × 3), 列 = [90%, 95%, 99%]
    trace_crit_95 = result.trace_stat_crit_vals[:, 1]

    # Step 2: 确定协整秩 r (95% 置信水平)
    # H0: r <= i → 如果 trace_stats[i] > crit_95[i], 拒绝 H0
    # 秩 = 第一次无法拒绝之前已拒绝的 H0 数量
    rank = 0
    for i in range(n_series):
        if trace_stats[i] > trace_crit_95[i]:
            rank += 1
        else:
            break

    is_cointegrated = rank >= 1

    # Step 3: 第一特征向量 (最大特征值)
    v1 = eigenvectors[:, 0]

    # Step 4: 构造组合净值
    yport = construct_portfolio(prices_df, v1)

    # Step 5: 半衰期估计 (yport 已在 price 空间, 不需要再取 log)
    hl_result = estimate_half_life(yport, use_log=False)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "trace_stats": trace_stats,
        "trace_crit": trace_crit_95,
        "rank": rank,
        "yport": yport,
        "half_life": hl_result["half_life"],
        "is_cointegrated": is_cointegrated,
    }


# ============================================================
# 路径生成器
# ============================================================

def generate_cointegrated_paths(
    n_steps: int,
    n_series: int,
    ou_theta: float = 0.1,
    ou_sigma: float = 0.02,
    random_seed: int = 42,
) -> np.ndarray:
    """
    生成具有协整关系的多变量价格序列。

    构造方式:
        c[t] = c[t-1] + ε[t]           (公共随机游走因子)
        yᵢ[t] = c[t] + zᵢ[t]          (对数价格 = 公共因子 + 独立噪声)
        zᵢ 为 OU 过程 (均值回归, 确保 yᵢ - yⱼ 平稳)

    参数:
        n_steps:   时序长度
        n_series:  序列数量
        ou_theta:  OU 过程均值回归速度 (>0, 对应短半衰期噪声)
        ou_sigma:  OU 过程波动率
        random_seed: 随机种子

    返回:
        (T × n) 数组
    """
    rng = np.random.default_rng(random_seed)

    # 公共随机游走因子
    common = np.cumsum(rng.standard_normal(n_steps) * 0.01)

    # OU 噪声项: z[t+1] = z[t] * exp(-θ) + σ · √(1-exp(-2θ)) · ε
    decay = np.exp(-ou_theta)
    noise_scale = ou_sigma * np.sqrt(1 - np.exp(-2 * ou_theta))

    z = np.zeros((n_series, n_steps))
    for t in range(n_steps - 1):
        z[:, t + 1] = z[:, t] * decay + noise_scale * rng.standard_normal(n_series)

    # yᵢ = c + zᵢ
    y = common[np.newaxis, :] + z

    return y.T  # (T × n)


def generate_gbm_matrix(
    n_steps: int,
    n_series: int,
    sigma: float = 0.01,
    random_seed: int = 42,
) -> np.ndarray:
    """
    生成独立的 GBM 对数价格序列 (无协整关系)。

    每个序列独立地对数形式随机游走:
        x[t] = x[t-1] - 0.5·σ² + σ·ε

    参数:
        n_steps:   时序长度
        n_series:  序列数量
        sigma:     日波动率
        random_seed: 随机种子

    返回:
        (T × n) 数组
    """
    rng = np.random.default_rng(random_seed)

    drift = -0.5 * sigma ** 2
    increments = drift + sigma * rng.standard_normal((n_steps - 1, n_series))
    paths = np.zeros((n_steps, n_series))
    paths[1:] = np.cumsum(increments, axis=0)

    return paths


# ============================================================
# 验证协议
# ============================================================

def run_validation():
    """
    Johansen 检验验证协议。

    正控: 3 协整序列 (共享随机游走 + 独立 OU 噪声) → rank ≥ 1
    负控: 3 独立 GBM → rank = 0
    辅助校验: yport 半衰期 < 输入序列半衰期
    """
    N_STEPS = 1000
    LAG = 1

    print("=" * 60)
    print("Johansen 协整检验验证协议")
    print("=" * 60)

    all_pass = True

    # --- 正控: 协整序列 ---
    print(f"\n【正控】3 协整序列 (共享 RW 因子 + OU 噪声, T={N_STEPS})")
    print("-" * 60)

    coint_data = generate_cointegrated_paths(N_STEPS, 3, ou_theta=0.1, ou_sigma=0.02, random_seed=42)
    coint_df = pd.DataFrame(coint_data, columns=["A", "B", "C"])

    result = johansen_test(coint_df, lag=LAG)

    print(f"  特征值: {np.array2string(result['eigenvalues'], precision=4, suppress_small=True)}")
    print(f"  Trace 统计量: {np.array2string(result['trace_stats'], precision=2, suppress_small=True)}")
    print(f"  Trace 95% 临界值: {np.array2string(result['trace_crit'], precision=2, suppress_small=True)}")

    rank_ok = result["rank"] >= 1
    trace_ok = result["trace_stats"][0] > result["trace_crit"][0]
    status_positive = "PASS" if (rank_ok and trace_ok) else "FAIL"
    if not (rank_ok and trace_ok):
        all_pass = False

    print(f"  协整秩 r = {result['rank']} (要求 ≥1) → {'OK' if rank_ok else 'NG'}")
    print(f"  trace_stat[0]={result['trace_stats'][0]:.2f} > crit_95[0]={result['trace_crit'][0]:.2f} → {'OK' if trace_ok else 'NG'}")
    print(f"  第一特征向量: {np.array2string(result['eigenvectors'][:, 0], precision=4, suppress_small=True)}")
    print(f"  yport 半衰期: {result['half_life']:.1f} 天")
    print(f"  [{status_positive}]")

    # --- 负控: 独立 GBM ---
    print(f"\n【负控】3 独立 GBM (无协整, T={N_STEPS})")
    print("-" * 60)

    gbm_data = generate_gbm_matrix(N_STEPS, 3, sigma=0.01, random_seed=42)
    gbm_df = pd.DataFrame(gbm_data, columns=["X", "Y", "Z"])

    result_gbm = johansen_test(gbm_df, lag=LAG)

    print(f"  特征值: {np.array2string(result_gbm['eigenvalues'], precision=4, suppress_small=True)}")
    print(f"  Trace 统计量: {np.array2string(result_gbm['trace_stats'], precision=2, suppress_small=True)}")
    print(f"  Trace 95% 临界值: {np.array2string(result_gbm['trace_crit'], precision=2, suppress_small=True)}")

    rank_zero = result_gbm["rank"] == 0
    status_negative = "PASS" if rank_zero else "FAIL"
    if not rank_zero:
        all_pass = False

    print(f"  协整秩 r = {result_gbm['rank']} (要求 =0) → {'OK' if rank_zero else 'NG'}")
    print(f"  [{status_negative}]")

    # --- yport 半衰期校验 ---
    print("\n【辅助】yport 半衰期 vs 输入序列半衰期")
    print("-" * 60)

    # 协整序列下: yport 半衰期应短于原始序列
    hl_yport = result["half_life"]

    individual_hl = []
    for col in coint_df.columns:
        hl = estimate_half_life(coint_df[col], use_log=False)
        individual_hl.append(hl["half_life"])

    avg_hl = np.mean(individual_hl)
    hl_shorter = hl_yport < avg_hl
    status_hl = "PASS" if hl_shorter else "FAIL"
    if not hl_shorter:
        all_pass = False

    print(f"  原始序列半衰期: {[f'{h:.1f}' for h in individual_hl]} 天")
    print(f"  原始序列平均半衰期: {avg_hl:.1f} 天")
    print(f"  yport 半衰期: {hl_yport:.1f} 天  (要求 < {avg_hl:.1f}) → {'OK' if hl_shorter else 'NG'}")
    print(f"  yport 半衰期缩短比例: {hl_yport / avg_hl:.2%}")
    print(f"  [{status_hl}]")

    # --- 汇总 ---
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] Johansen 验证通过")
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
