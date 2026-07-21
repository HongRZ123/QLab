"""
kalman_mu_test.py — 情绪锚均值回归策略验证 (独立实验)
=====================================================

核心假设:
    A股价格围绕情绪驱动的时变均值 μ_t 波动。
    μ_t 本身会跳跃（情绪突变），跳跃期间不应交易。

方法:
    1. 卡尔曼滤波估计 μ_t（趋势组件 = 情绪锚）
    2. Innovation 检测 μ 跳跃（regime 切换）
    3. 仅在 μ 稳定期交易围绕 μ_t 的均值回归

对比:
    A) 滚动均值 S4（固定窗口，无 regime 过滤）  ← 基线
    B) 卡尔曼 μ（更好的均值估计，无 regime 过滤）
    C) 卡尔曼 μ + regime 过滤（完整方法）

数据:
    通达信 .day 二进制文件，不依赖项目内任何模块。

用法:
    python experiments/kalman_mu_test.py
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 配置
# ============================================================

TDX_ROOT = Path(r"D:\new_tdx64\vipdoc")
RECORD_SIZE = 32
RECORD_FORMAT = "<IIIIIfII"

# A股交易成本
COMMISSION_RATE = 0.00025   # 佣金 万2.5
STAMP_RATE = 0.0005         # 印花税 万5 (仅卖出)
SLIPPAGE_RATE = 0.001       # 滑点 千1
MIN_COMMISSION = 5.0        # 最低佣金
LOT_SIZE = 100              # 整数手

# 实验标的
SYMBOLS = ["sh512670", "sh512760"]

# 参数网格
Q_RATIOS = [0.005, 0.01, 0.05]       # Q/R 比值 (μ 漂移速度)
JUMP_THRESHOLDS = [2.0, 2.5, 3.0]    # innovation z-score 阈值
ENTRY_ZS = [1.0, 1.5, 2.0]           # 入场 z-score
RESIDUAL_WINDOW = 20                  # 残差标准差滚动窗口 (~半衰期)
COOLDOWN_DAYS = 10                    # 跳跃后冷却期
WARMUP = 60                           # 预热期 (天)


# ============================================================
# 数据读取 (独立实现)
# ============================================================

def read_day(symbol: str) -> pd.DataFrame:
    """读取通达信 .day 日线数据。"""
    market, code = symbol[:2], symbol[2:]
    filepath = TDX_ROOT / market / "lday" / f"{market}{code}.day"

    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    raw = filepath.read_bytes()
    n = len(raw) // RECORD_SIZE

    records = []
    for i in range(n):
        chunk = raw[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]
        records.append(struct.unpack(RECORD_FORMAT, chunk))

    df = pd.DataFrame(
        records,
        columns=["date_raw", "open_raw", "high_raw", "low_raw",
                 "close_raw", "amount", "volume", "_"],
    )
    df["date"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y%m%d")
    df["close"] = df["close_raw"] / 100.0
    df = df[["date", "close"]].set_index("date").sort_index()
    return df


# ============================================================
# 卡尔曼滤波 — 标量水平估计
# ============================================================

def kalman_level(log_prices: np.ndarray, q_ratio: float) -> dict:
    """
    标量卡尔曼滤波，估计价格水平 μ_t。

    模型:
        状态方程:  μ_t = μ_{t-1} + w_t,   w_t ~ N(0, Q)
        观测方程:  y_t = μ_t + v_t,       v_t ~ N(0, R)

    参数:
        log_prices: 对数价格序列
        q_ratio:    Q/R 比值，控制 μ 的漂移速度
                    小 → μ 平滑 (慢漂移)
                    大 → μ 灵活 (可跳跃)

    返回:
        dict: {
            mu          : np.ndarray — 滤波后的 μ_t 估计
            P           : np.ndarray — 估计方差 P_t
            innovation  : np.ndarray — 预测误差 y_t - μ_pred
            innov_std   : np.ndarray — 预测误差标准差 sqrt(S_t)
            kalman_gain : np.ndarray — 卡尔曼增益 K_t
        }
    """
    T = len(log_prices)

    # 估计观测噪声 R: 用对数收益率方差的 1/2
    # (若 y = μ + v, 则 Δy ≈ Δμ + Δv, var(Δy) ≈ 2R 当 μ 缓慢变化时)
    returns = np.diff(log_prices)
    R = max(np.var(returns) / 2.0, 1e-10)
    Q = q_ratio * R

    # 初始化
    mu = np.zeros(T)
    P = np.zeros(T)
    innovation = np.zeros(T)
    innov_std = np.zeros(T)
    K_arr = np.zeros(T)

    mu[0] = log_prices[0]
    P[0] = R  # 初始不确定性 = 观测噪声

    for t in range(1, T):
        # ── 预测 ──
        mu_pred = mu[t - 1]
        P_pred = P[t - 1] + Q

        # ── Innovation ──
        y_t = log_prices[t]
        innov = y_t - mu_pred
        S = P_pred + R

        # ── 卡尔曼增益 ──
        K = P_pred / S

        # ── 更新 ──
        mu[t] = mu_pred + K * innov
        P[t] = (1.0 - K) * P_pred

        innovation[t] = innov
        innov_std[t] = np.sqrt(S)
        K_arr[t] = K

    return {
        "mu": mu,
        "P": P,
        "innovation": innovation,
        "innov_std": innov_std,
        "kalman_gain": K_arr,
    }


# ============================================================
# Regime 检测 — Innovation 跳跃
# ============================================================

def detect_jumps(innovation: np.ndarray, innov_std: np.ndarray,
                 threshold: float, cooldown: int) -> np.ndarray:
    """
    基于 innovation z-score 检测 μ 跳跃。

    逻辑:
        |innovation_t| > threshold × innov_std_t  →  检测到跳跃
        跳跃后 cooldown 天内标记为"不稳定"，不交易。

    参数:
        innovation: 预测误差序列
        innov_std:  预测误差标准差
        threshold:  z-score 阈值 (如 2.5)
        cooldown:   跳跃后冷却天数

    返回:
        stable: bool 数组, True = 稳定期 (可交易)
    """
    T = len(innovation)
    stable = np.ones(T, dtype=bool)

    # 避免除零
    safe_std = np.where(innov_std > 1e-12, innov_std, 1e-12)
    z_innov = np.abs(innovation) / safe_std

    # 检测跳跃并施加冷却期
    cooldown_remaining = 0
    for t in range(T):
        if cooldown_remaining > 0:
            stable[t] = False
            cooldown_remaining -= 1
        elif z_innov[t] > threshold:
            stable[t] = False
            cooldown_remaining = cooldown

    return stable


# ============================================================
# 信号生成
# ============================================================

def generate_signals(log_prices: np.ndarray, mu: np.ndarray,
                     stable: np.ndarray | None,
                     entry_z: float,
                     residual_window: int = RESIDUAL_WINDOW,
                     warmup: int = WARMUP) -> np.ndarray:
    """
    生成均值回归交易信号。

    逻辑:
        residual_t = log_price_t - μ_t
        σ_t = rolling_std(residual, residual_window)
        z_t = residual_t / σ_t

        入场: z_t < -entry_z 且 regime 稳定 → 做多
        出场: z_t > 0 (回归完成) 或 regime 变不稳定

    参数:
        log_prices:      对数价格
        mu:              卡尔曼估计的 μ_t
        stable:          regime 稳定标记 (None = 不过滤)
        entry_z:         入场 z-score 阈值
        residual_window: 残差标准差窗口
        warmup:          预热期

    返回:
        position: 0/1 数组 (0=空仓, 1=满仓)
    """
    T = len(log_prices)
    residual = log_prices - mu

    # 滚动标准差
    sigma = np.full(T, np.nan)
    for t in range(residual_window, T):
        sigma[t] = np.std(residual[t - residual_window + 1 : t + 1], ddof=1)

    # z-score
    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    # 生成仓位 (状态机)
    position = np.zeros(T, dtype=int)
    in_position = False

    for t in range(warmup, T):
        regime_ok = stable[t] if stable is not None else True

        if not in_position:
            # 入场: z 足够低 且 regime 稳定
            if z[t] < -entry_z and regime_ok:
                in_position = True
                position[t] = 1
        else:
            # 出场: 回归完成 或 regime 变不稳定
            if z[t] > 0 or not regime_ok:
                in_position = False
                position[t] = 0
            else:
                position[t] = 1

    return position


# ============================================================
# 回测引擎 (简化版, A股约束)
# ============================================================

INITIAL_CAPITAL = 1_000_000.0  # 初始资金 100万


def backtest(prices: np.ndarray, position: np.ndarray,
             dates: pd.DatetimeIndex) -> dict:
    """
    简化回测: T+1, 整数手, 交易成本。

    使用实际资金跟踪: cash + shares × price = equity。
    信号日 t 产生仓位变化, 在 t+1 日收盘价执行 (T+1)。

    参数:
        prices:   收盘价序列
        position: 0/1 仓位信号 (信号日)
        dates:    日期索引

    返回:
        dict: {
            equity   : 权益曲线 (实际金额)
            sharpe   : 年化 Sharpe
            apr      : 年化收益率 (几何)
            max_dd   : 最大回撤
            win_rate : 胜率
            n_trades : 交易次数
            avg_hold : 平均持仓天数
        }
    """
    T = len(prices)
    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0.0

    equity = np.full(T, INITIAL_CAPITAL)
    daily_ret = np.zeros(T)
    trade_returns = []
    hold_days_list = []
    current_hold_start = -1

    for t in range(1, T):
        # ── 持仓市值变化 ──
        if shares > 0:
            price_ret = (prices[t] - prices[t - 1]) / prices[t - 1]
            daily_ret[t] = price_ret

        # ── T+1 执行: 信号在 t-1, 执行在 t ──
        want_long = position[t - 1] == 1  # 昨日信号

        if want_long and shares == 0:
            # 开仓: 用 95% 资金买入
            buy_price = prices[t] * (1 + SLIPPAGE_RATE)
            available = cash * 0.95
            raw_shares = available / buy_price
            shares = int(raw_shares // LOT_SIZE) * LOT_SIZE
            if shares > 0:
                cost = shares * buy_price
                commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                cash -= (cost + commission)
                entry_price = buy_price
                current_hold_start = t

        elif not want_long and shares > 0:
            # 平仓
            sell_price = prices[t] * (1 - SLIPPAGE_RATE)
            proceeds = shares * sell_price
            commission = max(proceeds * COMMISSION_RATE, MIN_COMMISSION)
            stamp = proceeds * STAMP_RATE
            cash += (proceeds - commission - stamp)

            trade_ret = (sell_price - entry_price) / entry_price
            trade_returns.append(trade_ret)
            if current_hold_start >= 0:
                hold_days_list.append(t - current_hold_start)
            shares = 0
            current_hold_start = -1

        # ── 权益 = 现金 + 持仓市值 ──
        equity[t] = cash + shares * prices[t]

    # 强制平仓
    if shares > 0:
        sell_price = prices[-1] * (1 - SLIPPAGE_RATE)
        proceeds = shares * sell_price
        commission = max(proceeds * COMMISSION_RATE, MIN_COMMISSION)
        stamp = proceeds * STAMP_RATE
        cash += (proceeds - commission - stamp)
        trade_ret = (sell_price - entry_price) / entry_price
        trade_returns.append(trade_ret)
        if current_hold_start >= 0:
            hold_days_list.append(T - 1 - current_hold_start)
        shares = 0
        equity[-1] = cash

    # ── 指标 ──
    valid_ret = daily_ret[WARMUP:]
    sharpe = 0.0
    if len(valid_ret) > 10 and np.std(valid_ret) > 1e-12:
        sharpe = float(np.mean(valid_ret) / np.std(valid_ret) * np.sqrt(252))

    total_return = equity[-1] / equity[WARMUP] - 1
    n_days = T - WARMUP
    if total_return > 0:
        apr = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)
    elif total_return == 0:
        apr = 0.0
    else:
        apr = -1.0 + (1 + total_return) ** (252 / max(n_days, 1))

    # 最大回撤
    eq = equity[WARMUP:]
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    max_dd = float(np.min(drawdown))

    # 胜率
    win_rate = 0.0
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

    avg_hold = float(np.mean(hold_days_list)) if hold_days_list else 0.0

    return {
        "equity": equity,
        "sharpe": sharpe,
        "apr": apr,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": len(trade_returns),
        "avg_hold": avg_hold,
    }


# ============================================================
# 基线: 滚动均值 S4
# ============================================================

def rolling_mean_strategy(prices: np.ndarray, lookback: int,
                          entry_z: float, warmup: int = WARMUP) -> np.ndarray:
    """
    滚动均值 z-score 策略 (S4 简化版)。

    z_t = (price_t - MA(price, lookback)) / Std(price, lookback)
    入场: z < -entry_z → 做多
    出场: z > 0 → 平仓
    """
    T = len(prices)
    log_p = np.log(prices)

    # 滚动均值和标准差
    ma = np.full(T, np.nan)
    std = np.full(T, np.nan)
    for t in range(lookback - 1, T):
        window = log_p[t - lookback + 1 : t + 1]
        ma[t] = np.mean(window)
        std[t] = np.std(window, ddof=1)

    z = np.zeros(T)
    valid = ~np.isnan(ma) & ~np.isnan(std) & (std > 1e-12)
    z[valid] = (log_p[valid] - ma[valid]) / std[valid]

    position = np.zeros(T, dtype=int)
    in_pos = False
    for t in range(max(warmup, lookback), T):
        if not in_pos:
            if z[t] < -entry_z:
                in_pos = True
                position[t] = 1
        else:
            if z[t] > 0:
                in_pos = False
                position[t] = 0
            else:
                position[t] = 1

    return position


# ============================================================
# 单标的实验
# ============================================================

def run_experiment(symbol: str, lookback_baseline: int) -> None:
    """对单只标的运行完整对比实验。"""
    print(f"\n{'=' * 70}")
    print(f"  {symbol}")
    print(f"{'=' * 70}")

    # 读取数据
    df = read_day(symbol)
    prices = df["close"].values.astype(float)
    log_prices = np.log(prices)
    dates = df.index
    T = len(prices)

    print(f"  数据: {dates[0].date()} ~ {dates[-1].date()}, {T} 天")

    # ── 基线: 滚动均值 S4 ──
    print(f"\n  {'─' * 66}")
    print(f"  [A] 基线: 滚动均值 S4 (lookback={lookback_baseline})")
    print(f"  {'─' * 66}")

    best_baseline = None
    for ez in ENTRY_ZS:
        pos = rolling_mean_strategy(prices, lookback_baseline, ez)
        result = backtest(prices, pos, dates)
        label = f"entry_z={ez:.1f}"
        _print_result(label, result)
        if best_baseline is None or result["sharpe"] > best_baseline["sharpe"]:
            best_baseline = result

    # ── 卡尔曼 μ 实验 ──
    results_b = []  # 无 regime 过滤
    results_c = []  # 有 regime 过滤

    for qr in Q_RATIOS:
        kf = kalman_level(log_prices, qr)
        mu = kf["mu"]

        for ez in ENTRY_ZS:
            # [B] 卡尔曼 μ, 无 regime 过滤
            pos_b = generate_signals(log_prices, mu, None, ez)
            res_b = backtest(prices, pos_b, dates)
            res_b["q_ratio"] = qr
            res_b["entry_z"] = ez
            results_b.append(res_b)

            # [C] 卡尔曼 μ + regime 过滤
            for jt in JUMP_THRESHOLDS:
                stable = detect_jumps(kf["innovation"], kf["innov_std"],
                                      jt, COOLDOWN_DAYS)
                pos_c = generate_signals(log_prices, mu, stable, ez)
                res_c = backtest(prices, pos_c, dates)
                res_c["q_ratio"] = qr
                res_c["entry_z"] = ez
                res_c["jump_threshold"] = jt

                # 统计 regime 信息
                n_stable = np.sum(stable[WARMUP:])
                n_total = T - WARMUP
                res_c["stable_pct"] = n_stable / n_total * 100
                n_jumps = np.sum(np.diff(stable.astype(int)) == -1)
                res_c["n_jumps"] = n_jumps

                results_c.append(res_c)

    # 打印 [B] 最佳结果
    print(f"\n  {'─' * 66}")
    print(f"  [B] 卡尔曼 μ (无 regime 过滤) — 各参数组合")
    print(f"  {'─' * 66}")

    results_b.sort(key=lambda r: r["sharpe"], reverse=True)
    for r in results_b[:5]:
        label = f"Q/R={r['q_ratio']:.3f}, ez={r['entry_z']:.1f}"
        _print_result(label, r)

    # 打印 [C] 最佳结果
    print(f"\n  {'─' * 66}")
    print(f"  [C] 卡尔曼 μ + regime 过滤 — 各参数组合 (Top 10)")
    print(f"  {'─' * 66}")

    results_c.sort(key=lambda r: r["sharpe"], reverse=True)
    for r in results_c[:10]:
        label = (f"Q/R={r['q_ratio']:.3f}, ez={r['entry_z']:.1f}, "
                 f"k={r['jump_threshold']:.1f}, "
                 f"stable={r['stable_pct']:.0f}%, jumps={r['n_jumps']}")
        _print_result(label, r)

    # ── 汇总对比 ──
    print(f"\n  {'─' * 66}")
    print(f"  汇总对比")
    print(f"  {'─' * 66}")

    best_b = max(results_b, key=lambda r: r["sharpe"])
    best_c = max(results_c, key=lambda r: r["sharpe"])

    print(f"  {'方法':<30s} {'Sharpe':>8s} {'APR':>8s} {'MaxDD':>8s} "
          f"{'胜率':>6s} {'交易':>4s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*4}")

    _print_summary_row("A: 滚动均值 S4", best_baseline)
    _print_summary_row("B: 卡尔曼 μ", best_b)
    _print_summary_row("C: 卡尔曼 μ + regime", best_c)


def _print_result(label: str, r: dict) -> None:
    """打印单条结果。"""
    print(f"    {label:<45s}  "
          f"Sharpe={r['sharpe']:+.3f}  "
          f"APR={r['apr']:+.1%}  "
          f"MaxDD={r['max_dd']:.1%}  "
          f"胜率={r['win_rate']:.0%}  "
          f"交易={r['n_trades']}")


def _print_summary_row(label: str, r: dict) -> None:
    """打印汇总行。"""
    print(f"  {label:<30s} {r['sharpe']:>+8.3f} {r['apr']:>+7.1%} "
          f"{r['max_dd']:>7.1%} {r['win_rate']:>5.0%} {r['n_trades']:>4d}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  情绪锚均值回归策略验证")
    print("  假设: μ_t 是情绪驱动的时变均值, 用卡尔曼滤波追踪,")
    print("        仅在 μ 稳定期交易围绕 μ 的均值回归偏离。")
    print("=" * 70)

    # sh512670: 半衰期 44.5d, walk-forward lookback=110
    # sh512760: 半衰期 73d, walk-forward lookback=95
    run_experiment("sh512670", lookback_baseline=110)
    run_experiment("sh512760", lookback_baseline=95)

    print(f"\n{'=' * 70}")
    print("  实验完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
