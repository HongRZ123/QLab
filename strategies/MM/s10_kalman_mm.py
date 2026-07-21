"""
s10_kalman_mm.py — 卡尔曼滤波做市模型
=====================================

基于 Chan (2013) 第3章的成交量加权卡尔曼滤波:
    - 观测方程: y(t) = m(t) + ε(t)
    - 状态转移: m(t) = m(t-1) + ω(t-1),  ω ~ N(0, V_ω)
    - V_ω = δ/(1-δ)
    - V_e(t) = R(t|t-1) × (T_max / T(t) - 1)   (成交量加权观测噪声)
    - K(t) = R(t|t-1) / (R(t|t-1) + V_e(t))
    - m(t|t) = m(t|t-1) + K(t) × (y(t) - m(t|t-1))
    - R(t|t) = (1 - K(t)) × R(t|t-1)

    T=T_max 时 K=1 → 公允价格直接跳到成交价
    T<<T_max 时 K~0 -> 公允价格几乎不动
    T=T_max 时 R=0 (完美观测 → 零不确定性), 否则 R>0
    本质: VWAP 的动态升级版 (成交量加权 + 时间衰减)

核心函数:
    kalman_mm(prices, volumes, t_max=None, delta=0.0001) -> dict

验证协议:
    - 恒定价格 100 + 变化成交量 → |fair_value - 100| < 0.01 (收敛后)
    - T = T_max 日 → K = 1.0 (精确)
    - T → 0 日 → K → 0 (fair_value 不变)
    - R(t) >= 0 对所有 t (T=T_max 时 R=0 合法)

用法:
    python -m strategies.s10_kalman_mm
"""

import numpy as np
import pandas as pd

# ============================================================
# 核心函数
# ============================================================

def kalman_mm(
    prices: pd.Series,
    volumes: pd.Series,
    t_max: float | None = None,
    delta: float = 0.0001,
) -> dict:
    """
    成交量加权卡尔曼滤波做市模型。

    使用卡尔曼滤波动态估计资产的公允价格 (隐状态 m(t)),
    其中观测噪声 V_e 与成交量 T 成反比:
        - 大成交量 → V_e 小 → K 大 → 公允价格快速跟进成交价
        - 小成交量 → V_e 大 → K 小 → 公允价格几乎不动

    参数:
        prices:  成交价格序列 (pd.Series)
        volumes: 成交量序列 (pd.Series), 与 prices 等长
        t_max:   基准成交量 (最大成交量). None 时使用扩展窗口
                 max(volumes[:t+1]), 避免前视偏差. 显式传入时使用固定值
        delta:   过程噪声参数 (0 < δ < 1), V_ω = δ/(1-δ).
                 默认 0.0001 → V_ω ≈ 1e-4

    返回:
        dict: {
            fair_value  : pd.Series — 动态公允价格 m(t|t)
            kalman_gain : pd.Series — 卡尔曼增益 K(t)
            R           : pd.Series — 状态方差 R(t|t)
            deviation   : pd.Series — 偏离 y(t) - m(t|t)
        }

    示例:
        >>> prices = pd.Series([100.0, 100.5, 99.8])
        >>> volumes = pd.Series([500, 10000, 100])
        >>> result = kalman_mm(prices, volumes)
        >>> result["kalman_gain"].iloc[-1] < 0.5
        True
    """
    y = prices.astype(float)
    T = volumes.astype(float)

    n = len(y)
    if len(T) != n:
        raise ValueError(f"prices 与 volumes 长度不匹配: {len(y)} vs {len(T)}")

    # ── 过程噪声方差 ──
    V_omega = delta / (1.0 - delta)

    # ── 基准成交量 ──
    # 显式传入时使用固定值; None 时在循环内使用扩展窗口 (避免前视偏差)
    T_max_fixed = float(t_max) if t_max is not None else None
    if T_max_fixed is not None and T_max_fixed <= 0:
        raise ValueError(f"T_max 必须为正数: {T_max_fixed}")

    # ── 初始化 ──
    m_arr = np.full(n, np.nan, dtype=float)
    R_arr = np.full(n, np.nan, dtype=float)
    K_arr = np.full(n, np.nan, dtype=float)
    dev_arr = np.full(n, np.nan, dtype=float)

    # t=0: 初始状态
    m_arr[0] = y.iloc[0]
    R_arr[0] = 1.0
    K_arr[0] = 0.0
    dev_arr[0] = 0.0

    # ── 迭代 ──
    for t_idx in range(1, n):
        T_t = T.iloc[t_idx]
        y_t = y.iloc[t_idx]
        m_prev = m_arr[t_idx - 1]
        R_prev = R_arr[t_idx - 1]

        # 预测: R(t|t-1) = R(t-1|t-1) + V_ω
        R_pred = R_prev + V_omega

        # 成交量为 0 时跳过更新 (保持上一时刻状态)
        if T_t <= 0:
            m_arr[t_idx] = m_prev
            R_arr[t_idx] = R_pred
            K_arr[t_idx] = 0.0
            dev_arr[t_idx] = y_t - m_prev
            continue

        # ── 基准成交量 (扩展窗口, 避免前视偏差) ──
        T_max_t = T_max_fixed if T_max_fixed is not None else float(T.iloc[:t_idx + 1].max())

        if T_max_t <= 0:
            m_arr[t_idx] = m_prev
            R_arr[t_idx] = R_pred
            K_arr[t_idx] = 0.0
            dev_arr[t_idx] = y_t - m_prev
            continue

        # ── 观测噪声 (成交量加权) ──
        # V_e(t) = R(t|t-1) × (T_max_t / T(t) - 1)
        V_e = R_pred * (T_max_t / T_t - 1.0)

        # ── 卡尔曼增益 ──
        # K(t) = R(t|t-1) / (R(t|t-1) + V_e(t))
        K_t = R_pred / (R_pred + V_e)

        # ── 均值更新 ──
        # m(t|t) = m(t|t-1) + K(t) × (y(t) - m(t|t-1))
        # 其中 m(t|t-1) = m(t-1|t-1)
        m_t = m_prev + K_t * (y_t - m_prev)

        # ── 方差更新 ──
        # R(t|t) = (1 - K(t)) × R(t|t-1)
        R_t = (1.0 - K_t) * R_pred

        m_arr[t_idx] = m_t
        R_arr[t_idx] = R_t
        K_arr[t_idx] = K_t
        dev_arr[t_idx] = y_t - m_t

    return {
        "fair_value": pd.Series(m_arr, index=y.index, name="fair_value"),
        "kalman_gain": pd.Series(K_arr, index=y.index, name="kalman_gain"),
        "R": pd.Series(R_arr, index=y.index, name="R"),
        "deviation": pd.Series(dev_arr, index=y.index, name="deviation"),
    }


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """恒定价格收敛 + K 极端值 + R 正定性 验证协议。"""
    all_pass = True

    print("=" * 60)
    print("  卡尔曼滤波做市模型 (S10) — 验证协议")
    print("=" * 60)

    # ── 测试 1: 恒定价格 100 + 变化成交量 → fair_value 收敛至 100 ──
    print("\n【测试1】恒定价格 100 + 变化成交量 → |fair_value - 100| < 0.01 (收敛后)")
    print("-" * 60)

    np.random.seed(42)
    n_days = 200
    const_prices = pd.Series(np.full(n_days, 100.0), name="price")
    # 成交量波动: 大部分为中等成交量, 偶尔大单
    base_vol = np.random.uniform(500, 1500, n_days)
    big_order_idx = [50, 100, 150]
    base_vol[big_order_idx] = 5000.0
    const_volumes = pd.Series(base_vol, name="volume")

    res1 = kalman_mm(const_prices, const_volumes, delta=0.001)

    # 取后半段 (收敛后) 检查
    tail = res1["fair_value"].iloc[-50:]
    max_dev = float((tail - 100.0).abs().max())

    test1_ok = max_dev < 0.01
    t1_status = "PASS" if test1_ok else "FAIL"
    print(f"  fair_value 后半段最大偏离 = {max_dev:.6f}")
    print(f"  fair_value 终值           = {res1['fair_value'].iloc[-1]:.6f}")
    print(f"  [{t1_status}] 恒定价格收敛验证")

    if not test1_ok:
        all_pass = False

    # ── 测试 2: T = T_max 日 → K = 1.0 (精确) ──
    print("\n【测试2】T = T_max 日 → K = 1.0, fair_value 跳到成交价")
    print("-" * 60)

    t2_prices = pd.Series([100.0, 105.0, 110.0], name="price")
    t2_volumes = pd.Series([500.0, 10000.0, 100.0], name="volume")
    # T_max = 10000, day 1 has T = 10000 = T_max

    res2 = kalman_mm(t2_prices, t2_volumes)

    # day 1 (index 1): T = T_max → K should be 1.0
    k_day1 = float(res2["kalman_gain"].iloc[1])
    fv_day1 = float(res2["fair_value"].iloc[1])
    expected_fv = 105.0

    k_ok = abs(k_day1 - 1.0) < 1e-10
    fv_ok = abs(fv_day1 - expected_fv) < 1e-10
    test2_ok = k_ok and fv_ok

    t2_status = "PASS" if test2_ok else "FAIL"
    print(f"  Day 1 (T=T_max={t2_volumes.max():.0f}):")
    print(f"    K           = {k_day1:.10f}  (期望=1.0)    [{'PASS' if k_ok else 'FAIL'}]")
    print(f"    fair_value  = {fv_day1:.6f}     (期望=105.0)  [{'PASS' if fv_ok else 'FAIL'}]")
    print(f"  [{t2_status}] T=T_max K=1 验证")

    if not test2_ok:
        all_pass = False

    # ── 测试 3: T → 0 日 → K → 0 (fair_value 几乎不变) ──
    print("\n【测试3】T << T_max -> K ~ 0, fair_value 几乎不变")
    print("-" * 60)

    t3_prices = pd.Series([100.0, 200.0, 50.0], name="price")
    t3_volumes = pd.Series([500.0, 0.1, 10000.0], name="volume")
    # T_max = 10000, day 1 has T = 0.1 << T_max -> K ~ 0
    # day 2 has T = 10000 = T_max → K = 1

    res3 = kalman_mm(t3_prices, t3_volumes)

    k_day1_t3 = float(res3["kalman_gain"].iloc[1])
    fv_day1_t3 = float(res3["fair_value"].iloc[1])
    # After t=0: m=100. Day 1: T=0.1, huge V_e, K ≈ 0, fair_value ≈ 100
    fv_stable = abs(fv_day1_t3 - 100.0) < 0.1

    # Day 2: T=T_max, K=1, fair_value jumps to 50
    k_day2_t3 = float(res3["kalman_gain"].iloc[2])
    fv_day2_t3 = float(res3["fair_value"].iloc[2])

    k_small_ok = k_day1_t3 < 0.01
    k_large_ok = abs(k_day2_t3 - 1.0) < 1e-10
    fv_jump_ok = abs(fv_day2_t3 - 50.0) < 1e-10
    test3_ok = k_small_ok and fv_stable and k_large_ok and fv_jump_ok

    t3_status = "PASS" if test3_ok else "FAIL"
    print("  Day 1 (T=0.1 << T_max=10000):")
    print(f"    K           = {k_day1_t3:.10f}  (期望≈0)      [{'PASS' if k_small_ok else 'FAIL'}]")
    print(f"    fair_value  = {fv_day1_t3:.6f}     (期望≈100)    [{'PASS' if fv_stable else 'FAIL'}]")
    print("  Day 2 (T=T_max=10000):")
    print(f"    K           = {k_day2_t3:.10f}  (期望=1.0)    [{'PASS' if k_large_ok else 'FAIL'}]")
    print(f"    fair_value  = {fv_day2_t3:.6f}     (期望=50.0)   [{'PASS' if fv_jump_ok else 'FAIL'}]")
    print(f"  [{t3_status}] 成交量极值验证")

    if not test3_ok:
        all_pass = False

    # ── 测试 4: R(t) >= 0 对所有 t ──
    # 注: T=T_max 时 V_e=0, K=1, R(t|t)=0, 这是正确的 (完美观测 → 零不确定性)
    # R_pred = R(t-1|t-1) + V_ω > 0 始终成立, 但 R(t|t) 可为零
    print("\n【测试4】R(t) >= 0 对所有 t (T=T_max 时 R=0 是合法的)")
    print("-" * 60)

    # 使用测试1的数据 (较长序列)
    R_all_nonneg = bool((res1["R"].iloc[1:].dropna() >= 0).all())
    R_min = float(res1["R"].iloc[1:].min())
    # T_max 命中点: R 应恰好为 0 (不是负数)

    # 额外: 用另一组数据验证
    np.random.seed(123)
    extra_prices = pd.Series(np.random.randn(100).cumsum() + 100.0)
    extra_vols = pd.Series(np.abs(np.random.randn(100)) * 1000 + 100)
    res_extra = kalman_mm(extra_prices, extra_vols)
    R_extra_nonneg = bool((res_extra["R"].iloc[1:].dropna() >= 0).all())

    test4_ok = R_all_nonneg and R_extra_nonneg
    t4_status = "PASS" if test4_ok else "FAIL"

    print(f"  恒定价格序列: R_min = {R_min:.6f}, all>=0 = {R_all_nonneg}")
    print(f"  随机游走序列: all R >= 0 = {R_extra_nonneg}")
    print(f"  [{t4_status}] R 非负性验证")

    if not test4_ok:
        all_pass = False

    # ── 测试 5: 成交量全为 T_max → 完美跟踪 ──
    print("\n【测试5】成交量全部 = T_max → K≡1, fair_value ≡ price")
    print("-" * 60)

    np.random.seed(99)
    n5 = 50
    t5_prices = pd.Series(np.random.randn(n5).cumsum() + 100.0)
    # 所有成交量等于 T_max
    t5_vols = pd.Series(np.full(n5, 1000.0))

    res5 = kalman_mm(t5_prices, t5_vols, t_max=1000.0)

    # Day 0 的 K 初始化为 0, 从 Day 1 开始 K≡1
    k_from_t1 = res5["kalman_gain"].iloc[1:]
    all_k_one = bool((abs(k_from_t1 - 1.0) < 1e-10).all())

    # fair_value 从 Day 1 开始等于 prices
    fv_match = float((res5["fair_value"].iloc[1:] - t5_prices.iloc[1:]).abs().max())
    test5_ok = all_k_one and fv_match < 1e-10

    t5_status = "PASS" if test5_ok else "FAIL"
    print(f"  K≡1 从 Day 1 起: {all_k_one}")
    print(f"  fair_value 与 price 最大偏差 = {fv_match:.2e}")
    print(f"  [{t5_status}] 完美跟踪验证")

    if not test5_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 卡尔曼做市验证通过")
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
