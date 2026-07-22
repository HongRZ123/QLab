"""
s4_linear.py — 线性均值回归策略（单资产，仅做多）
===================================================

基于 Chan (2013) 第2章的线性均值回归策略:
    - Z(t) = (y(t) - MA(y, L)) / Std(y, L)
    - num_units(t) = max(0, -Z(t))  ← 仅做多
    - mkt_val(t) = num_units(t) × y(t)
    - pnl(t) = mkt_val(t-1) × (y(t) - y(t-1)) / y(t-1)
    - ret(t) = pnl(t) / |mkt_val(t-1)|  (mkt_val=0 时 ret=0)

回望期 L 设定:
    - lookback 显式传入 → 直接使用
    - lookback=None 且 half_life 给定 → lookback = round(half_life)
    - 两者均 None → 自动估计半衰期

无自由参数 → 无数据窥探偏差。

核心函数:
    linear_mr(prices, lookback=None, half_life=None) -> dict

验证协议:
    - 正控: OU(θ=0.05, T=2000) → 累计 PnL > 0, Sharpe > 0
    - 负控: GBM(T=2000) → Sharpe < 0.5
    - num_units 全部 ≥ 0 (仅做多断言)
    - lookback = round(half_life) 当 half_life 给定时

用法:
    python -m strategies.s4_linear
"""

import numpy as np
import pandas as pd

from stats.univariate import estimate_half_life, generate_gbm_paths, generate_ou_paths

# ============================================================
# 核心函数
# ============================================================

def _determine_lookback(prices: pd.Series, lookback: int | None, half_life: float | None) -> int:
    """
    确定回望期 L。

    优先级: lookback 显式传入 > half_life 给定 > 自动估计。
    """
    if lookback is not None:
        return int(lookback)

    if half_life is not None:
        return round(half_life)

    result = estimate_half_life(prices)
    hl = result["half_life"]
    if np.isinf(hl):
        return 20  # 非均值回归时的默认回望期, 可接受的最小值
    return round(hl)


def linear_mr(
    prices: pd.Series,
    lookback: int | None = None,
    half_life: float | None = None,
) -> dict:
    """
    单资产线性均值回归策略（仅做多）。

    基于 Chan (2013) 第2章策略:
        num_units(t) = max(0, -Z(t))
        其中 Z(t) = (y(t) - MA(y, L)) / Std(y, L)

    参数:
        prices:    日价格序列 (pd.Series)
        lookback:  回望期 L (天). None 时自动确定
        half_life: 半衰期 (天). lookback=None 且 half_life 给定时,
                   L = round(half_life). 两者均 None 时自动估计

    返回:
        dict: {
            z_score     : pd.Series  — Z(t), 前 L-1 天为 NaN
            num_units   : pd.Series  — 仓位单元 (≥0, 仅做多)
            mkt_val     : pd.Series  — 市值 = num_units × price
            pnl         : pd.Series  — 理论每日盈亏 (无交易成本、无 T+1、无手数取整,
                                       仅用于验证协议; 生产回测请用 backtest.run_backtest)
            ret         : pd.Series  — 理论每日收益率 (同上)
            lookback_used: int       — 实际使用的回望期 L
        }

    示例:
        >>> prices = pd.Series([10.0, 10.5, 10.3, 10.8, 10.0])
        >>> result = linear_mr(prices, half_life=3.0)
        >>> result["lookback_used"]
        3
    """
    y = prices.astype(float)

    # ── Step 1: 确定回望期 ──
    L = _determine_lookback(y, lookback, half_life)
    if L < 1:
        L = 1

    # ── Step 2: Z-Score ──
    # Z(t) = (y(t) - MA(y, L)) / Std(y, L)
    ma = y.rolling(window=L, min_periods=L).mean()
    std = y.rolling(window=L, min_periods=L).std()

    # 避免除零: std=0 时 (恒定价格) Z=0
    safe_std = std.replace(0.0, np.nan)
    z_score = (y - ma) / safe_std

    # ── Step 3: num_units (仅做多) ──
    # num_units(t) = max(0, -Z(t))
    num_units_raw = np.maximum(0.0, -z_score.to_numpy(dtype=float))
    num_units = pd.Series(num_units_raw, index=y.index, name="num_units").fillna(0.0)

    # ── Step 4: 市值 ──
    # mkt_val(t) = num_units(t) × y(t)
    mkt_val = num_units * y

    # ── Step 5: PnL ──
    # pnl(t) = mkt_val(t-1) × (y(t) - y(t-1)) / y(t-1)
    mkt_val_lag = mkt_val.shift(1).fillna(0.0)
    price_ret = ((y.diff() / y.shift(1))
                 .replace([np.inf, -np.inf], 0.0)
                 .fillna(0.0))
    pnl = mkt_val_lag * price_ret

    # ── Step 6: 收益率 ──
    # ret(t) = pnl(t) / |mkt_val(t-1)|, mkt_val=0 时 ret=0
    ret = pd.Series(0.0, index=y.index, name="ret")
    abs_mkt_val_lag = mkt_val_lag.abs()
    mask = abs_mkt_val_lag > 0
    ret.loc[mask] = pnl.loc[mask] / abs_mkt_val_lag.loc[mask]

    return {
        "z_score": z_score,
        "num_units": num_units,
        "mkt_val": mkt_val,
        "pnl": pnl,
        "ret": ret,
        "lookback_used": L,
    }


# ============================================================
# 辅助: 年化 Sharpe
# ============================================================

def _annualized_sharpe(daily_ret: pd.Series) -> float:
    """年化 Sharpe 比率 (无风险利率=0, 交易天数=252)。"""
    r = daily_ret.dropna()
    if len(r) < 10 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """正控 + 负控 + 仅做多 + lookback 验证协议。"""
    T = 2000
    SEED = 42

    all_pass = True

    print("=" * 60)
    print("  线性均值回归策略 (S4) — 验证协议")
    print("=" * 60)

    # ── 正控: OU 过程 ──
    # OU(θ=0.05) → 半衰期 ≈ ln(2)/0.05 ≈ 13.9d
    # 使用 exp 将 OU 对数路径转为正价格序列
    print("\n【正控】OU(θ=0.05, T=2000) → 累计 PnL > 0, Sharpe > 0")
    print("-" * 60)

    ou_raw = generate_ou_paths(1, T, theta=0.05, mu=0.0, sigma=1.0, dt=1.0, seed=SEED)
    ou_prices = pd.Series(np.exp(ou_raw[0]), name="ou_price")

    res_pos = linear_mr(ou_prices)
    cum_pnl = float(res_pos["pnl"].sum())
    sharpe_pos = _annualized_sharpe(res_pos["ret"])

    pnl_ok = cum_pnl > 0
    sharpe_ok = sharpe_pos > 0
    pos_status = "PASS" if (pnl_ok and sharpe_ok) else "FAIL"

    print(f"  累计 PnL   = {cum_pnl:,.4f}  (要求 > 0)  [{'PASS' if pnl_ok else 'FAIL'}]")
    print(f"  年化 Sharpe = {sharpe_pos:.4f}    (要求 > 0)  [{'PASS' if sharpe_ok else 'FAIL'}]")
    print(f"  lookback    = {res_pos['lookback_used']}d")
    print(f"  [{pos_status}] 正控验证")

    if not (pnl_ok and sharpe_ok):
        all_pass = False

    # ── 负控: GBM 过程 ──
    print("\n【负控】GBM(T=2000) → Sharpe < 0.5")
    print("-" * 60)

    gbm_raw = generate_gbm_paths(1, T, sigma=0.01, dt=1.0, seed=SEED)
    gbm_prices = pd.Series(np.exp(gbm_raw[0]), name="gbm_price")

    res_neg = linear_mr(gbm_prices)
    sharpe_neg = _annualized_sharpe(res_neg["ret"])

    neg_ok = sharpe_neg < 0.5
    neg_status = "PASS" if neg_ok else "FAIL"

    print(f"  年化 Sharpe = {sharpe_neg:.4f}    (要求 < 0.5)      [{'PASS' if neg_ok else 'FAIL'}]")
    print(f"  lookback    = {res_neg['lookback_used']}d")
    print(f"  [{neg_status}] 负控验证")

    if not neg_ok:
        all_pass = False

    # ── 仅做多断言 ──
    print("\n【断言】num_units 全部 ≥ 0 (仅做多)")
    print("-" * 60)

    nu_pos_ok = bool((res_pos["num_units"] >= 0).all())
    nu_neg_ok = bool((res_neg["num_units"] >= 0).all())
    nu_ok = nu_pos_ok and nu_neg_ok
    nu_status = "PASS" if nu_ok else "FAIL"

    print(f"  OU  num_units: min={res_pos['num_units'].min():.4f}, "
          f"all≥0={nu_pos_ok}")
    print(f"  GBM num_units: min={res_neg['num_units'].min():.4f}, "
          f"all≥0={nu_neg_ok}")
    print(f"  [{nu_status}] 仅做多断言")

    if not nu_ok:
        all_pass = False

    # ── lookback = round(half_life) 验证 ──
    print("\n【断言】lookback = round(half_life) 当 half_life 给定时")
    print("-" * 60)

    test_hls = [3.2, 14.7, 20.0, 50.99]
    lb_ok = True
    for hl in test_hls:
        expected = round(hl)
        res = linear_mr(ou_prices, half_life=hl)
        actual = res["lookback_used"]
        match = actual == expected
        if not match:
            lb_ok = False
        print(f"  half_life={hl:6.2f}  →  lookback={actual}  "
              f"(期望={expected})  [{'PASS' if match else 'FAIL'}]")

    lb_status = "PASS" if lb_ok else "FAIL"
    print(f"  [{lb_status}] lookback 验证")

    if not lb_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 线性MR验证通过")
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
