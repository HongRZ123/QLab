"""
s7_linear_portfolio.py — 线性均值回归策略（组合版，仅做多）
===========================================================

基于 Chan (2013) 第2章组合线性均值回归策略:
    - yport(t) = Σ vᵢ · yᵢ(t) (组合净值)
    - Z(t) = (yport(t) - MA(yport, L)) / Std(yport, L)
    - num_units(t) = max(0, -Z(t))  ← 仅做多
    - positions_i(t) = num_units(t) × vᵢ × yᵢ(t) (各资产市值)
    - pnl(t) = Σ positions_i(t-1) × (yᵢ(t) - yᵢ(t-1)) / yᵢ(t-1)
    - ret(t) = pnl(t) / Σ |positions_i(t-1)|

回望期 L 设定:
    - lookback 显式传入 → 直接使用
    - lookback=None → 从 yport 半衰期自动估计, L = round(halflife)

核心函数:
    linear_portfolio(prices_df, eigenvector, lookback=None) -> dict

验证协议:
    - 正控: 3 协整序列 + Johansen v₁ → 累计 PnL > 0
    - num_units 全部 ≥ 0 (仅做多断言)
    - positions 形状 = (T, n_assets)

用法:
    python -m strategies.s7_linear_portfolio
"""

import numpy as np
import pandas as pd

from stats.cointegration import generate_cointegrated_paths, johansen_test
from stats.univariate import estimate_half_life

# ============================================================
# 核心函数
# ============================================================

def _determine_lookback(yport: pd.Series, lookback: int | None) -> int:
    """
    确定回望期 L。

    若 lookback 显式传入 → 直接使用。
    若 None → 从 yport 半衰期自动估计, L = round(halflife)。
    """
    if lookback is not None:
        return int(lookback)

    result = estimate_half_life(yport, use_log=False)
    hl = result["half_life"]
    if np.isinf(hl):
        return 20  # 非均值回归时的默认回望期
    return round(hl)


def linear_portfolio(
    prices_df: pd.DataFrame,
    eigenvector: np.ndarray,
    lookback: int | None = None,
) -> dict:
    """
    组合线性均值回归策略（仅做多）。

    基于 Chan (2013) 第2章:
        - 用特征向量构造组合净值 yport = Σ vᵢ · yᵢ
        - Z(t) = (yport(t) - MA(yport, L)) / Std(yport, L)
        - num_units(t) = max(0, -Z(t))
        - positions_i(t) = num_units(t) × vᵢ × yᵢ(t)
        - pnl(t) = Σ positions_i(t-1) × (yᵢ(t) - yᵢ(t-1)) / yᵢ(t-1)

    参数:
        prices_df:   价格矩阵 (T × n), 列 = 资产, 行 = 时间
        eigenvector: 特征向量 (长度 n), 通常来自 Johansen 检验的第一特征向量
        lookback:    回望期 L (天). None 时自动从 yport 半衰期估计

    返回:
        dict: {
            yport        : pd.Series  — 组合净值, yport(t) = Σ vᵢ · yᵢ(t)
            z_score      : pd.Series  — Z(t), 前 L-1 天为 NaN
            num_units    : pd.Series  — 仓位倍数 (≥ 0, 仅做多)
            positions    : pd.DataFrame — 各资产市值 (T × n)
            pnl          : pd.Series  — 理论每日盈亏 (无交易成本、无 T+1、无手数取整,
                                         仅用于验证协议; 生产回测请用 backtest.run_backtest)
            ret          : pd.Series  — 理论每日收益率 (同上)
            lookback_used: int        — 实际使用的回望期 L
        }

    示例:
        >>> prices = pd.DataFrame({"A": [10, 11, 10], "B": [20, 19, 21]})
        >>> v = np.array([0.6, 0.4])
        >>> result = linear_portfolio(prices, v, lookback=2)
        >>> result["positions"].shape
        (3, 2)
    """
    y = prices_df.astype(float)
    T, n = y.shape

    # ── Step 1: 构造组合净值 ──
    # yport(t) = Σ vᵢ · yᵢ(t)
    yport = pd.Series(y.values @ eigenvector, index=y.index, name="yport")

    # ── Step 2: 确定回望期 ──
    L = _determine_lookback(yport, lookback)
    if L < 1:
        L = 1

    # ── Step 3: Z-Score ──
    # Z(t) = (yport(t) - MA(yport, L)) / Std(yport, L)
    ma = yport.rolling(window=L, min_periods=L).mean()
    std = yport.rolling(window=L, min_periods=L).std()

    # 避免除零: std=0 时 Z=0
    safe_std = std.replace(0.0, np.nan)
    z_score = (yport - ma) / safe_std

    # ── Step 4: num_units (仅做多) ──
    # num_units(t) = max(0, -Z(t))
    num_units_raw = np.maximum(0.0, -z_score.to_numpy(dtype=float))
    num_units = pd.Series(num_units_raw, index=yport.index, name="num_units").fillna(0.0)

    # ── Step 5: 各资产市值 ──
    # positions_i(t) = num_units(t) × vᵢ × yᵢ(t)
    positions_arr = np.zeros((T, n))
    for t_idx in range(T):
        positions_arr[t_idx, :] = num_units.iloc[t_idx] * eigenvector * y.iloc[t_idx].values
    positions = pd.DataFrame(positions_arr, index=y.index, columns=y.columns)

    # ── Step 6: PnL ──
    # pnl(t) = Σ positions_i(t-1) × (yᵢ(t) - yᵢ(t-1)) / yᵢ(t-1)
    positions_lag = positions.shift(1).fillna(0.0)
    price_ret = (y.pct_change()
                  .replace([np.inf, -np.inf], 0.0)
                  .fillna(0.0))
    pnl = pd.Series(
        (positions_lag.values * price_ret.values).sum(axis=1),
        index=y.index,
        name="pnl",
    )

    # ── Step 7: 收益率 ──
    # ret(t) = pnl(t) / Σ |positions_i(t-1)|, 总敞口=0 时 ret=0
    gross_exposure = positions_lag.abs().sum(axis=1)
    ret = pd.Series(0.0, index=y.index, name="ret")
    mask = gross_exposure > 0
    ret.loc[mask] = pnl.loc[mask] / gross_exposure.loc[mask]

    return {
        "yport": yport,
        "z_score": z_score,
        "num_units": num_units,
        "positions": positions,
        "pnl": pnl,
        "ret": ret,
        "lookback_used": L,
    }


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """
    组合线性均值回归策略验证协议。

    正控: 3 协整序列 + Johansen v₁ → 累计 PnL > 0
    断言: num_units 全部 ≥ 0, positions 形状 = (T, n_assets)
    """
    T = 1000
    N_ASSETS = 3
    SEED = 42

    all_pass = True

    print("=" * 60)
    print("  组合线性均值回归策略 (S7) — 验证协议")
    print("=" * 60)

    # ── 生成协整序列 ──
    # generate_cointegrated_paths 返回对数价格 (T × n)
    # exp() 得到实际价格用于策略回测
    print(f"\n【准备】生成 {N_ASSETS} 协整序列 (T={T}, OU theta=0.1)")
    print("-" * 60)

    log_prices = generate_cointegrated_paths(T, N_ASSETS, ou_theta=0.1, ou_sigma=0.02, random_seed=SEED)
    log_df = pd.DataFrame(log_prices, columns=["A", "B", "C"])
    prices_df = pd.DataFrame(np.exp(log_prices), columns=["A", "B", "C"])

    # ── Johansen 检验: 获取第一特征向量 ──
    johansen_result = johansen_test(log_df, lag=1)
    v1 = johansen_result["eigenvectors"][:, 0]

    rank = johansen_result["rank"]
    hl_yport = johansen_result["half_life"]
    print(f"  Johansen 秩 r = {rank} (要求 ≥1)")
    print(f"  第一特征向量 v1 = {np.array2string(v1, precision=4, suppress_small=True)}")
    print(f"  yport (对数) 半衰期 = {hl_yport:.1f} 天")

    if rank < 1:
        print("  [FAIL] 协整秩不足, 无法验证")
        return False

    # ── 正控: 策略回测 ──
    print(f"\n【正控】{N_ASSETS} 协整序列 + Johansen v1 -> 累计 PnL > 0")
    print("-" * 60)

    result = linear_portfolio(prices_df, v1, lookback=round(hl_yport))
    cum_pnl = float(result["pnl"].sum())
    pnl_ok = cum_pnl > 0

    print(f"  累计 PnL = {cum_pnl:,.4f}  (要求 > 0)  [{'PASS' if pnl_ok else 'FAIL'}]")
    print(f"  lookback  = {result['lookback_used']}d")
    print(f"  yport(首5) = {[float(f'{x:.4f}') for x in result['yport'].iloc[:5]]}")
    print(f"  z_score(首5后) = {[f'{x:.4f}' for x in result['z_score'].dropna().iloc[:5]]}")

    if not pnl_ok:
        all_pass = False

    # ── 断言: num_units 全部 ≥ 0 ──
    print("\n【断言】num_units 全部 ≥ 0 (仅做多)")
    print("-" * 60)

    nu_min = float(result["num_units"].min())
    nu_mean = float(result["num_units"].mean())
    nu_ok = bool((result["num_units"] >= 0).all())
    nu_status = "PASS" if nu_ok else "FAIL"

    print(f"  num_units: min={nu_min:.6f}, mean={nu_mean:.4f}, all≥0={nu_ok}")
    print(f"  [{nu_status}] 仅做多断言")

    if not nu_ok:
        all_pass = False

    # ── 断言: positions 形状 ──
    print("\n【断言】positions 形状 = (T, n_assets)")
    print("-" * 60)

    pos_shape = result["positions"].shape
    expected_shape = (T, N_ASSETS)
    shape_ok = pos_shape == expected_shape
    shape_status = "PASS" if shape_ok else "FAIL"

    print(f"  positions.shape = {pos_shape}  (期望 {expected_shape})  [{'PASS' if shape_ok else 'FAIL'}]")

    # 额外: 检查 positions 正负号 — 正 vᵢ 对应做多, 负 vᵢ 对应做空
    pos_signs = []
    for i in range(N_ASSETS):
        pos_mean = float(result["positions"].iloc[:, i].mean())
        pos_signs.append(pos_mean)
    print(f"  各资产平均市值: {[f'{x:.4f}' for x in pos_signs]}")

    print(f"  [{shape_status}] 形状验证")

    if not shape_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 组合线性MR验证通过")
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
