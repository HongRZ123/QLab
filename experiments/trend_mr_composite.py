"""
trend_mr_composite.py — 趋势 + 短期均值回归 复合策略
====================================================

架构:
  外层 (慢): Kalman Linear Trend → μ(t), β(t)  趋势锚
  内层 (快): Z = (price - μ) / σ(residual, N)   短期均值回归下单信号

  趋势策略确认核心回归值, 均值回归策略决定出入场。

测试:
  短期窗口 N ∈ {10, 15, 20}
  入场阈值 entry_z ∈ {1.0, 1.5, 2.0}
  5 品种 × 全样本 + Walk-Forward (固定参数)

用法: python experiments/trend_mr_composite.py
"""

import struct
from pathlib import Path
import numpy as np

# ============================================================
# 配置
# ============================================================

TDX_ROOT = Path(r"D:\new_tdx64\vipdoc\ds\lday")

FUTURES = [
    ("29#ML9",  "豆粕 M"),
    ("30#CUL9", "铜 CU"),
    ("28#SRL9", "白糖 SR"),
    ("30#RBL9", "螺纹钢 RB"),
    ("30#AUL9", "黄金 AU"),
]

# ── 趋势层: Kalman Linear Trend (固定) ──
Q_RATIO = 0.01          # 水平过程噪声
Q_SLOPE_RATIO = 0.001   # 斜率过程噪声
R_INIT_WINDOW = 60      # R 初始估计 (无未来函数)
WARMUP = 120

# ── 均值回归层: 短期信号 (测试多组) ──
SHORT_WINDOWS = [10, 15, 20]
ENTRY_ZS = [1.0, 1.5, 2.0]
EXIT_Z = 0.0            # 回归趋势线即出场

# ── 交易成本 (期货) ──
COMMISSION_RATE = 0.00025
SLIPPAGE_RATE = 0.0005
INITIAL_CAPITAL = 1_000_000.0

# ── Walk-Forward ──
TRAIN_DAYS = 1260
TEST_DAYS = 252
STEP_DAYS = 252


# ============================================================
# 数据读取
# ============================================================

def read_futures(filename):
    filepath = TDX_ROOT / f"{filename}.day"
    raw = filepath.read_bytes()
    n = len(raw) // 32
    fmt = "<IfffffII"
    dates, closes = [], []
    for i in range(n):
        d, o, h, l, c, amt, vol, _ = struct.unpack(fmt, raw[i*32:(i+1)*32])
        dates.append(d)
        closes.append(c)
    return np.array(dates), np.array(closes, dtype=float)


# ============================================================
# 趋势层: Kalman Linear Trend
# ============================================================

def kalman_linear_trend(log_prices):
    """
    状态 x = [mu, beta]^T, 跟踪水平 + 斜率。
    R 仅用前 R_INIT_WINDOW 天估计 (因果, 无前视)。
    返回: (mu, beta)
    """
    T = len(log_prices)
    init_n = min(R_INIT_WINDOW, T // 2)
    R = max(np.var(np.diff(log_prices[:init_n + 1])) / 2.0, 1e-10)

    Q = np.array([[Q_RATIO * R, 0.0],
                  [0.0, Q_SLOPE_RATIO * R]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    I2 = np.eye(2)

    mu = np.zeros(T)
    beta = np.zeros(T)
    x = np.array([log_prices[0], 0.0])
    P = np.array([[R, 0.0], [0.0, R]])
    mu[0], beta[0] = x[0], x[1]

    for t in range(1, T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innov = log_prices[t] - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T)[0, 0] + R
        K = (P_pred @ H.T) / S
        x = x_pred + K.flatten() * innov
        P = (I2 - K @ H) @ P_pred
        mu[t], beta[t] = x[0], x[1]

    return mu, beta


# ============================================================
# 均值回归层: 短期 Z-Score + 离散信号
# ============================================================

def generate_signals(log_prices, mu, short_window, entry_z, exit_z=EXIT_Z):
    """
    短期均值回归信号:
        residual = log_price - mu (偏离趋势锚)
        sigma = rolling_std(residual, short_window)
        Z = residual / sigma
        Z < -entry_z → 买入
        Z > exit_z   → 卖出
    """
    T = len(log_prices)
    residual = log_prices - mu

    sigma = np.full(T, np.nan)
    for t in range(short_window, T):
        sigma[t] = np.std(residual[t - short_window + 1:t + 1], ddof=1)

    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    position = np.zeros(T, dtype=int)
    in_pos = False
    start = max(WARMUP, short_window)

    for t in range(start, T):
        if not in_pos:
            if z[t] < -entry_z:
                in_pos = True
                position[t] = 1
        else:
            if z[t] > exit_z:
                in_pos = False
            else:
                position[t] = 1

    return position


# ============================================================
# 回测
# ============================================================

def backtest(prices, position, warmup=0):
    T = len(prices)
    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = 0.0
    equity = np.full(T, INITIAL_CAPITAL)
    daily_ret = np.zeros(T)
    trade_rets = []

    for t in range(1, T):
        if shares > 0:
            daily_ret[t] = (prices[t] - prices[t-1]) / prices[t-1]
        want = position[t-1] == 1
        if want and shares == 0:
            bp = prices[t] * (1 + SLIPPAGE_RATE)
            shares = cash * 0.95 / bp
            if shares > 0:
                cash -= shares * bp * (1 + COMMISSION_RATE)
                entry_price = bp
        elif not want and shares > 0:
            sp = prices[t] * (1 - SLIPPAGE_RATE)
            cash += shares * sp * (1 - COMMISSION_RATE)
            trade_rets.append((sp - entry_price) / entry_price)
            shares = 0
        equity[t] = cash + shares * prices[t]

    if shares > 0:
        sp = prices[-1] * (1 - SLIPPAGE_RATE)
        cash += shares * sp * (1 - COMMISSION_RATE)
        trade_rets.append((sp - entry_price) / entry_price)
        equity[-1] = cash

    vr = daily_ret[warmup:]
    sharpe = 0.0
    if len(vr) > 10 and np.std(vr) > 1e-12:
        sharpe = float(np.mean(vr) / np.std(vr) * np.sqrt(252))
    total_ret = equity[-1] / equity[max(warmup, 1)] - 1
    eq = equity[warmup:]
    rm = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - rm) / rm))
    win_rate = sum(1 for r in trade_rets if r > 0) / max(len(trade_rets), 1)
    avg_hold = len(vr) / max(len(trade_rets), 1)

    return {"sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
            "n_trades": len(trade_rets), "total_ret": total_ret,
            "avg_hold": avg_hold}


# ============================================================
# 全样本: 参数扫描
# ============================================================

def run_full_sample(code, name):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    # 趋势层 (只算一次)
    mu, beta = kalman_linear_trend(log_prices)

    print(f"\n  {name}  ({T} days, {T/252:.0f}y)")
    print(f"  {'─' * 80}")
    print(f"  {'Window':>6s} {'EntryZ':>6s}  {'Sharpe':>8s} {'MaxDD':>8s} "
          f"{'胜率':>6s} {'交易':>5s} {'均持':>6s} {'总收益':>9s}")
    print(f"  {'─' * 80}")

    results = []
    for sw in SHORT_WINDOWS:
        for ez in ENTRY_ZS:
            pos = generate_signals(log_prices, mu, sw, ez)
            r = backtest(prices, pos, warmup=max(WARMUP, sw))
            results.append({"sw": sw, "ez": ez, **r})
            print(f"  {sw:>6d} {ez:>6.1f}  {r['sharpe']:>+8.3f} {r['max_dd']:>7.1%} "
                  f"{r['win_rate']:>5.0%} {r['n_trades']:>5d} "
                  f"{r['avg_hold']:>5.1f}d {r['total_ret']:>+8.1%}")

    # 找最稳健组合 (Sharpe 中位数 across windows 最高)
    best = max(results, key=lambda r: r["sharpe"])
    print(f"  {'─' * 80}")
    print(f"  最佳: window={best['sw']} entry_z={best['ez']:.1f} "
          f"Sharpe={best['sharpe']:+.3f} MaxDD={best['max_dd']:.1%} "
          f"交易={best['n_trades']}")

    return {"name": name, "results": results, "best": best,
            "mu": mu, "beta": beta, "prices": prices,
            "log_prices": log_prices, "dates": dates}


# ============================================================
# Walk-Forward (固定参数)
# ============================================================

def run_walkforward(code, name, fixed_sw, fixed_ez):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    mu, beta = kalman_linear_trend(log_prices)

    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    print(f"\n{'=' * 82}")
    print(f"  {name}  Walk-Forward  ({len(windows)} windows)")
    print(f"  固定: window={fixed_sw} entry_z={fixed_ez} "
          f"Kalman Q={Q_RATIO} Q_b={Q_SLOPE_RATIO}")
    print(f"{'=' * 82}")

    hdr = (f"  {'Win':>3s}  {'Period':>23s}  "
           f"{'IS Shp':>7s} {'OOS Shp':>8s} {'OOS DD':>7s}  "
           f"{'OOS#Tr':>6s} {'OOS Win%':>8s}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*23}  {'─'*7} {'─'*8} {'─'*7}  {'─'*6} {'─'*8}")

    is_list, oos_list = [], []

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        # IS
        lp_is = log_prices[tr_s:tr_e]
        p_is = prices[tr_s:tr_e]
        mu_is = mu[tr_s:tr_e]
        pos_is = generate_signals(lp_is, mu_is, fixed_sw, fixed_ez)
        r_is = backtest(p_is, pos_is, warmup=max(WARMUP, fixed_sw))

        # OOS
        ctx_s = max(0, tr_e - WARMUP)
        lp_oos = log_prices[ctx_s:te_e]
        p_oos = prices[ctx_s:te_e]
        mu_oos = mu[ctx_s:te_e]
        pos_oos = generate_signals(lp_oos, mu_oos, fixed_sw, fixed_ez)
        offset = tr_e - ctx_s
        pos_test = np.zeros(len(p_oos), dtype=int)
        pos_test[offset:] = pos_oos[offset:]
        r_oos = backtest(p_oos, pos_test, warmup=offset)

        is_list.append(r_is["sharpe"])
        oos_list.append(r_oos["sharpe"])

        period = f"{dates[tr_s]}~{dates[tr_e-1]}|{dates[tr_e]}~{dates[te_e-1]}"
        print(f"  {wi+1:>3d}  {period:>23s}  "
              f"{r_is['sharpe']:>+7.3f} {r_oos['sharpe']:>+8.3f} "
              f"{r_oos['max_dd']:>6.1%}  "
              f"{r_oos['n_trades']:>6d} {r_oos['win_rate']:>7.0%}")

    print(f"\n  {'─' * 74}")
    is_avg = np.mean(is_list)
    oos_avg = np.mean(oos_list)
    oos_med = np.median(oos_list)
    oos_pos = sum(1 for s in oos_list if s > 0)
    print(f"  IS  平均: {is_avg:+.3f}")
    print(f"  OOS 平均: {oos_avg:+.3f}  中位: {oos_med:+.3f}  "
          f"正Sharpe窗口: {oos_pos}/{len(oos_list)}")
    print(f"  {'─' * 74}")

    return {"name": name, "is": is_list, "oos": oos_list}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 82)
    print("  趋势 + 短期均值回归 复合策略")
    print("  外层: Kalman Linear Trend (趋势锚)")
    print("  内层: 短期 Z-Score (均值回归下单)")
    print(f"  测试: window={SHORT_WINDOWS} entry_z={ENTRY_ZS}")
    print("=" * 82)

    # ── Part 1: 全样本扫描 ──
    print(f"\n{'=' * 82}")
    print("  Part 1: 全样本")
    print(f"{'=' * 82}")

    all_results = []
    for code, name in FUTURES:
        all_results.append(run_full_sample(code, name))

    # 跨品种汇总: 每个 (window, entry_z) 组合的平均 Sharpe
    print(f"\n{'=' * 82}")
    print("  跨品种汇总 (平均 Sharpe)")
    print(f"{'=' * 82}")
    print(f"  {'Window':>6s} {'EntryZ':>6s}  ", end="")
    for r in all_results:
        print(f"{r['name']:>10s}", end="")
    print(f"  {'平均':>8s}")
    print(f"  {'─'*6} {'─'*6}  " + "─"*10 * len(all_results) + "  ────────")

    best_global = None
    best_global_avg = -999

    for sw in SHORT_WINDOWS:
        for ez in ENTRY_ZS:
            sharpes = []
            print(f"  {sw:>6d} {ez:>6.1f}  ", end="")
            for r in all_results:
                match = [x for x in r["results"]
                         if x["sw"] == sw and x["ez"] == ez][0]
                sharpes.append(match["sharpe"])
                print(f"{match['sharpe']:>+10.3f}", end="")
            avg = np.mean(sharpes)
            print(f"  {avg:>+8.3f}")
            if avg > best_global_avg:
                best_global_avg = avg
                best_global = (sw, ez)

    print(f"\n  全局最佳: window={best_global[0]} entry_z={best_global[1]:.1f} "
          f"平均Sharpe={best_global_avg:+.3f}")

    # ── Part 2: Walk-Forward (用全局最佳固定参数) ──
    fixed_sw, fixed_ez = best_global

    print(f"\n{'=' * 82}")
    print(f"  Part 2: Walk-Forward (固定 window={fixed_sw} entry_z={fixed_ez})")
    print(f"{'=' * 82}")

    wf_all = []
    for code, name in FUTURES:
        wf_all.append(run_walkforward(code, name, fixed_sw, fixed_ez))

    # 最终汇总
    print(f"\n{'=' * 82}")
    print("  最终汇总")
    print(f"{'=' * 82}")
    print(f"  {'品种':<12s} {'IS均值':>8s} {'OOS均值':>8s} {'OOS中位':>8s} {'正窗口':>6s}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    all_oos = []
    for r in wf_all:
        oos_pos = sum(1 for s in r["oos"] if s > 0)
        all_oos.extend(r["oos"])
        print(f"  {r['name']:<12s} {np.mean(r['is']):>+8.3f} "
              f"{np.mean(r['oos']):>+8.3f} {np.median(r['oos']):>+8.3f} "
              f"{oos_pos:>2d}/{len(r['oos']):<2d}")

    if all_oos:
        oos_pos_total = sum(1 for s in all_oos if s > 0)
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
        print(f"  {'全品种':<12s} {'':>8s} {np.mean(all_oos):>+8.3f} "
              f"{np.median(all_oos):>+8.3f} "
              f"{oos_pos_total:>2d}/{len(all_oos):<2d}")


if __name__ == "__main__":
    main()
