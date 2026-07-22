"""
s8_bollinger.py — 布林带均值回归策略（仅做多）
===================================================

基于 Chan (2013) 第3章 Bollinger Band 策略:
    - Z(t) = (y(t) - MA(y, L)) / Std(y, L)
    - Z < -entry_z  → num_units = 1 (买入)
    - Z >= -exit_z  → num_units = 0 (卖出)
    - 无信号日 → 沿用前一日仓位 (forward-fill)
    - num_units ∈ {0, 1}  ← 仅做多

PnL 计算 (同 S4 Chan 公式):
    - mkt_val(t) = num_units(t) × y(t)
    - pnl(t) = mkt_val(t-1) × (y(t) - y(t-1)) / y(t-1)
    - ret(t) = pnl(t) / |mkt_val(t-1)|  (mkt_val=0 时 ret=0)

核心函数:
    bollinger_mr(prices, lookback, entry_z, exit_z) -> dict
    bollinger_portfolio(yport, lookback, entry_z, exit_z) -> dict

验证协议:
    - 正控: OU(θ=0.05, T=2000, entry=1, exit=0) → n_trades > 5, 累计 PnL > 0
    - 负控: GBM → n_trades 可能 > 0 但 Sharpe 不显著 (< 0.5)
    - num_units 全部 ∈ {0, 1}
    - forward-fill: 无信号日 num_units 不变

用法:
    python -m strategies.s8_bollinger
"""

import numpy as np
import pandas as pd

from signals.stats import generate_gbm_paths, generate_ou_paths

# ============================================================
# 核心函数
# ============================================================

def bollinger_mr(
    prices: pd.Series,
    lookback: int,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
) -> dict:
    """
    布林带均值回归策略（仅做多）。

    基于 Chan (2013) 第3章 Bollinger Band 策略:
        Z < -entry_z  → 买入 (num_units = 1)
        Z >= -exit_z  → 卖出 (num_units = 0)
        无信号日 → forward-fill 沿用前一日仓位

    参数:
        prices:   日价格序列 (pd.Series)
        lookback: 回望期 L (天), 用于计算 MA 和 Std
        entry_z:  入场 Z-Score 阈值, 默认 1.0
        exit_z:   出场 Z-Score 阈值, 默认 0.0

    返回:
        dict: {
            z_score     : pd.Series  — Z(t), 前 L-1 天为 NaN
            num_units   : pd.Series  — 持仓单元, ∈ {0, 1}
            signals     : pd.Series  — 原始信号 (NaN=无信号日)
            pnl         : pd.Series  — 理论每日盈亏 (无交易成本、无 T+1、无手数取整,
                                       仅用于验证协议; 生产回测请用 backtest.run_backtest)
            ret         : pd.Series  — 理论每日收益率 (同上)
            n_trades    : int        — 往返交易次数
            avg_holding : float      — 平均持仓天数
        }

    示例:
        >>> prices = pd.Series([10.0, 10.5, 10.3, 10.8, 10.0])
        >>> result = bollinger_mr(prices, lookback=3)
        >>> result["n_trades"]
        0
    """
    y = prices.astype(float)

    # ── Step 1: Z-Score ──
    # Z(t) = (y(t) - MA(y, L)) / Std(y, L)
    ma = y.rolling(window=lookback, min_periods=lookback).mean()
    std = y.rolling(window=lookback, min_periods=lookback).std()

    # 避免除零: std=0 时 (恒定价格) Z=0
    safe_std = std.replace(0.0, np.nan)
    z_score = (y - ma) / safe_std

    # ── Step 2: 生成信号 (仅做多) ──
    # 无信号日 → NaN → forward-fill
    signals = pd.Series(np.nan, index=y.index, dtype=float)

    # 长仓入场: Z < -entry_z → num_units = 1
    signals.loc[z_score < -entry_z] = 1.0

    # 长仓出场: Z >= -exit_z → num_units = 0
    signals.loc[z_score >= -exit_z] = 0.0

    # ── Step 3: forward-fill ──
    # 初始无信号时 → 空仓 (0)
    if np.isnan(signals.iloc[0]):
        signals.iloc[0] = 0.0
    num_units = signals.ffill()

    # ── Step 4: 市值 ──
    # mkt_val(t) = num_units(t) × y(t)
    mkt_val = num_units * y

    # ── Step 5: PnL (同 S4 Chan 公式) ──
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

    # ── Step 7: 统计量 ──
    n_trades = int(num_units.diff().abs().sum() / 2)
    total_holding = float(num_units.sum())
    avg_holding = total_holding / n_trades if n_trades > 0 else 0.0

    return {
        "z_score": z_score,
        "num_units": num_units,
        "signals": signals,
        "pnl": pnl,
        "ret": ret,
        "n_trades": n_trades,
        "avg_holding": avg_holding,
    }


# ============================================================
# 组合净值便捷封装
# ============================================================

def bollinger_portfolio(
    yport: pd.Series,
    lookback: int,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
) -> dict:
    """
    对组合净值 yport 的便捷封装，直接调用 bollinger_mr。

    典型用途: 对协整组合净值或价差序列应用布林带策略。

    参数:
        yport:   组合净值序列 (pd.Series)
        lookback: 回望期 L (天)
        entry_z:  入场 Z-Score 阈值, 默认 1.0
        exit_z:   出场 Z-Score 阈值, 默认 0.0

    返回:
        同 bollinger_mr 返回的 dict
    """
    return bollinger_mr(yport, lookback, entry_z=entry_z, exit_z=exit_z)


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
    """正控 + 负控 + num_units + forward-fill 验证协议。"""
    T = 2000
    SEED = 42

    all_pass = True

    print("=" * 60)
    print("  布林带均值回归策略 (S8) — 验证协议")
    print("=" * 60)

    # ── 正控: OU 过程 ──
    # OU(θ=0.05) → 半衰期 ≈ ln(2)/0.05 ≈ 13.9d
    # lookback ≈ 14, entry=1.0, exit=0.0
    # 使用 exp 将 OU 对数路径转为正价格序列
    print("\n【正控】OU(θ=0.05, T=2000, entry=1, exit=0) → n_trades > 5, 累计 PnL > 0")
    print("-" * 60)

    ou_raw = generate_ou_paths(1, T, theta=0.05, mu=0.0, sigma=1.0, dt=1.0, seed=SEED)
    ou_prices = pd.Series(np.exp(ou_raw[0]), name="ou_price")

    LOOKBACK = 14  # ≈ round(half_life)
    res_pos = bollinger_mr(ou_prices, lookback=LOOKBACK, entry_z=1.0, exit_z=0.0)

    cum_pnl = float(res_pos["pnl"].sum())
    n_trades_pos = res_pos["n_trades"]

    pnl_ok = cum_pnl > 0
    trades_ok = n_trades_pos > 5
    pos_status = "PASS" if (pnl_ok and trades_ok) else "FAIL"

    print(f"  累计 PnL   = {cum_pnl:,.4f}    (要求 > 0)      "
          f"[{'PASS' if pnl_ok else 'FAIL'}]")
    print(f"  交易次数   = {n_trades_pos}         "
          f"(要求 > 5)       [{'PASS' if trades_ok else 'FAIL'}]")
    print(f"  lookback    = {LOOKBACK}d")
    print(f"  avg_holding = {res_pos['avg_holding']:.1f}d")
    print(f"  [{pos_status}] 正控验证")

    if not (pnl_ok and trades_ok):
        all_pass = False

    # ── 负控: GBM 过程 ──
    print("\n【负控】GBM(T=2000) → n_trades 可能 > 0 但 Sharpe < 0.5")
    print("-" * 60)

    gbm_raw = generate_gbm_paths(1, T, sigma=0.01, dt=1.0, seed=SEED)
    gbm_prices = pd.Series(np.exp(gbm_raw[0]), name="gbm_price")

    res_neg = bollinger_mr(gbm_prices, lookback=LOOKBACK, entry_z=1.0, exit_z=0.0)
    sharpe_neg = _annualized_sharpe(res_neg["ret"])
    n_trades_neg = res_neg["n_trades"]

    neg_ok = sharpe_neg < 0.5
    neg_status = "PASS" if neg_ok else "FAIL"

    print(f"  年化 Sharpe = {sharpe_neg:.4f}    (要求 < 0.5)      "
          f"[{'PASS' if neg_ok else 'FAIL'}]")
    print(f"  交易次数   = {n_trades_neg}")
    print(f"  [{neg_status}] 负控验证")

    if not neg_ok:
        all_pass = False

    # ── num_units ∈ {0, 1} 断言 ──
    print("\n【断言】num_units 全部 ∈ {0, 1} (仅做多)")
    print("-" * 60)

    nu_pos_ok = bool(res_pos["num_units"].isin({0.0, 1.0}).all())
    nu_neg_ok = bool(res_neg["num_units"].isin({0.0, 1.0}).all())
    nu_ok = nu_pos_ok and nu_neg_ok
    nu_status = "PASS" if nu_ok else "FAIL"

    pos_vals = sorted(res_pos["num_units"].unique())
    neg_vals = sorted(res_neg["num_units"].unique())
    print(f"  OU  num_units 取值: {pos_vals}")
    print(f"  GBM num_units 取值: {neg_vals}")
    print(f"  [{nu_status}] num_units 断言")

    if not nu_ok:
        all_pass = False

    # ── forward-fill 断言 ──
    print("\n【断言】forward-fill: 无信号日 num_units 不变")
    print("-" * 60)

    # 验证逻辑: 对于所有 signals 为 NaN 的日 (跳过第一天),
    # num_units 应与前一日相同
    signals_pos = res_pos["signals"]
    nu_pos = res_pos["num_units"]

    no_signal_mask = signals_pos.isna()
    # 跳过 index=0 (第一天无法比较)
    no_signal_mask.iloc[0] = False

    if no_signal_mask.any():
        # 对所有无信号日, 检查 num_units[t] == num_units[t-1]
        ff_ok = True
        for i in range(1, T):
            if pd.isna(signals_pos.iloc[i]) and nu_pos.iloc[i] != nu_pos.iloc[i - 1]:
                    ff_ok = False
                    break
    else:
        # 没有无信号日 (所有日都有信号), 也算通过
        ff_ok = True

    ff_status = "PASS" if ff_ok else "FAIL"
    n_no_signal = int(no_signal_mask.sum())
    print(f"  无信号日数 = {n_no_signal} / {T}")
    print(f"  forward-fill 一致性: {ff_ok}")
    print(f"  [{ff_status}] forward-fill 验证")

    if not ff_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 布林带验证通过")
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
