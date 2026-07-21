"""
kalman_mu_v2.py — 修正版: Local Linear Trend + 无 Regime 过滤
================================================================

修正三个问题:
  1. 卡尔曼模型: local level → local linear trend (跟踪水平+斜率)
  2. Regime 过滤: 移除 (冲击后恰恰是均值回归最强的时候)
  3. 参数: 固定参数, 无网格搜索, 无前视偏差

对照: 保留 local level + regime 过滤 (旧方法) 作为 baseline。

用法: python experiments/kalman_mu_v2.py
"""

import struct
from pathlib import Path
import numpy as np

# ============================================================
# 配置 — 固定参数, 无网格搜索
# ============================================================

TDX_ROOT = Path(r"D:\new_tdx64\vipdoc\ds\lday")

FUTURES = [
    ("29#ML9",  "豆粕 M"),
    ("30#CUL9", "铜 CU"),
    ("28#SRL9", "白糖 SR"),
    ("30#RBL9", "螺纹钢 RB"),
    ("30#AUL9", "黄金 AU"),
]

# ── 新方法: Local Linear Trend ──
Q_RATIO = 0.01          # 水平过程噪声 / 观测噪声
Q_SLOPE_RATIO = 0.001   # 斜率过程噪声 (比水平慢 10x)
ENTRY_Z = 1.5           # 入场: Z < -1.5
EXIT_Z = 0.0            # 出场: Z > 0 (回归趋势线)
RESIDUAL_WINDOW = 60    # 残差 σ 估计窗口
WARMUP = 120            # 预热期
R_INIT_WINDOW = 60      # R 初始估计窗口 (仅用历史, 无未来函数)

# ── 旧方法: Local Level + Regime (对照) ──
JUMP_THRESHOLD = 2.5
COOLDOWN_DAYS = 10

# ── 交易成本 ──
COMMISSION_RATE = 0.00025   # 万2.5/边
SLIPPAGE_RATE = 0.0005      # 万5
INITIAL_CAPITAL = 1_000_000.0

# ── Walk-Forward ──
TRAIN_DAYS = 1260   # 5 年
TEST_DAYS = 252     # 1 年
STEP_DAYS = 252     # 步进 1 年


# ============================================================
# 数据读取
# ============================================================

def read_futures(filename):
    """读取通达信期货 L9 连续合约 .day 文件, 返回 (dates, closes)."""
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
# 卡尔曼: Local Linear Trend (新方法)
# ============================================================

def kalman_linear_trend(log_prices, q_ratio=Q_RATIO, q_slope_ratio=Q_SLOPE_RATIO):
    """
    Local Linear Trend 卡尔曼滤波.

    状态向量 x = [μ, β]^T:
        μ = 当前水平 (趋势线截距)
        β = 当前斜率 (趋势线斜率, 即日均对数收益率)

    状态转移:
        μ(t) = μ(t-1) + β(t-1) + ε₁,  ε₁ ~ N(0, Q_μ)
        β(t) = β(t-1) + ε₂,            ε₂ ~ N(0, Q_β)

    观测方程:
        y(t) = μ(t) + η,  η ~ N(0, R)

    R 仅用前 R_INIT_WINDOW 天估计, 无未来函数。

    返回: (mu, beta, innovation, innov_std)
    """
    T = len(log_prices)

    # R: 仅用前 N 天估计 (因果, 无前视)
    init_n = min(R_INIT_WINDOW, T // 2)
    R = max(np.var(np.diff(log_prices[:init_n + 1])) / 2.0, 1e-10)

    # 过程噪声矩阵
    Q = np.array([[q_ratio * R, 0.0],
                  [0.0, q_slope_ratio * R]])

    # 状态转移 & 观测矩阵
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    I2 = np.eye(2)

    mu = np.zeros(T)
    beta = np.zeros(T)
    innovation = np.zeros(T)
    innov_std = np.zeros(T)

    # 初始化
    x = np.array([log_prices[0], 0.0])
    P = np.array([[R, 0.0],
                  [0.0, R]])
    mu[0] = x[0]
    beta[0] = x[1]

    for t in range(1, T):
        # ── 预测 ──
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # ── 更新 ──
        innov = log_prices[t] - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T)[0, 0] + R
        K = (P_pred @ H.T) / S          # 2×1

        x = x_pred + K.flatten() * innov
        P = (I2 - K @ H) @ P_pred

        mu[t] = x[0]
        beta[t] = x[1]
        innovation[t] = innov
        innov_std[t] = np.sqrt(S)

    return mu, beta, innovation, innov_std


# ============================================================
# 卡尔曼: Local Level (旧方法, 对照)
# ============================================================

def kalman_level(log_prices, q_ratio=Q_RATIO):
    """
    Local Level 卡尔曼滤波 (旧方法).
    状态: μ(t) = μ(t-1) + ε,  ε ~ N(0, Q)
    R 仅用前 R_INIT_WINDOW 天估计。
    """
    T = len(log_prices)
    init_n = min(R_INIT_WINDOW, T // 2)
    R = max(np.var(np.diff(log_prices[:init_n + 1])) / 2.0, 1e-10)
    Q = q_ratio * R

    mu = np.zeros(T)
    P = np.zeros(T)
    innovation = np.zeros(T)
    innov_std = np.zeros(T)

    mu[0] = log_prices[0]
    P[0] = R

    for t in range(1, T):
        P_pred = P[t-1] + Q
        innov = log_prices[t] - mu[t-1]
        S = P_pred + R
        K = P_pred / S
        mu[t] = mu[t-1] + K * innov
        P[t] = (1.0 - K) * P_pred
        innovation[t] = innov
        innov_std[t] = np.sqrt(S)

    return mu, innovation, innov_std


# ============================================================
# Regime 检测 (旧方法, 对照)
# ============================================================

def detect_jumps(innovation, innov_std, threshold=JUMP_THRESHOLD, cooldown=COOLDOWN_DAYS):
    """旧方法: |innovation| > threshold × σ → 不稳定, 冷却期内不交易."""
    T = len(innovation)
    stable = np.ones(T, dtype=bool)
    safe = np.where(innov_std > 1e-12, innov_std, 1e-12)
    z = np.abs(innovation) / safe
    cd = 0
    for t in range(T):
        if cd > 0:
            stable[t] = False
            cd -= 1
        elif z[t] > threshold:
            stable[t] = False
            cd = cooldown
    return stable


# ============================================================
# 信号生成
# ============================================================

def generate_signals(log_prices, mu, entry_z=ENTRY_Z, exit_z=EXIT_Z,
                     stable=None, warmup=WARMUP):
    """
    均值回归信号:
        Z = (log_price - μ) / σ(residual)
        Z < -entry_z → 买入 (价格低于趋势)
        Z > exit_z   → 卖出 (价格回归趋势)

    stable: 可选 regime 掩码 (None = 不过滤, 旧方法传入)
    """
    T = len(log_prices)
    residual = log_prices - mu

    # 滚动 σ (因果: 只用 [t-window+1, t])
    sigma = np.full(T, np.nan)
    for t in range(RESIDUAL_WINDOW, T):
        sigma[t] = np.std(residual[t - RESIDUAL_WINDOW + 1:t + 1], ddof=1)

    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    position = np.zeros(T, dtype=int)
    in_pos = False
    start = max(warmup, RESIDUAL_WINDOW)

    for t in range(start, T):
        regime_ok = stable[t] if stable is not None else True

        if not in_pos:
            if z[t] < -entry_z and regime_ok:
                in_pos = True
                position[t] = 1
        else:
            if z[t] > exit_z or not regime_ok:
                in_pos = False
            else:
                position[t] = 1

    return position


# ============================================================
# 回测
# ============================================================

def backtest(prices, position, warmup=0):
    """简单回测: 95% 仓位, 佣金+滑点, 返回绩效字典."""
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
            proceeds = shares * sp
            cash += proceeds * (1 - COMMISSION_RATE)
            trade_rets.append((sp - entry_price) / entry_price)
            shares = 0

        equity[t] = cash + shares * prices[t]

    # 强制平仓
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

    return {"sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
            "n_trades": len(trade_rets), "total_ret": total_ret}


def buy_and_hold_sharpe(prices, warmup=WARMUP):
    """买入持有基准 Sharpe."""
    rets = np.diff(prices[warmup:]) / prices[warmup:-1]
    if len(rets) < 10 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * np.sqrt(252))


# ============================================================
# 全样本对比
# ============================================================

def run_full_sample(code, name):
    """全样本: 新方法 vs 旧方法 vs 买入持有."""
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    # ── 新方法: Local Linear Trend, 无 Regime ──
    mu_new, beta_new, _, _ = kalman_linear_trend(log_prices)
    pos_new = generate_signals(log_prices, mu_new)
    r_new = backtest(prices, pos_new, warmup=WARMUP)

    # ── 旧方法: Local Level + Regime ──
    mu_old, innov_old, istd_old = kalman_level(log_prices)
    stable_old = detect_jumps(innov_old, istd_old)
    pos_old = generate_signals(log_prices, mu_old, stable=stable_old)
    r_old = backtest(prices, pos_old, warmup=WARMUP)

    # ── 基准 ──
    bh = buy_and_hold_sharpe(prices)

    # ── 趋势诊断 ──
    avg_beta = np.mean(beta_new[WARMUP:]) * 252  # 年化斜率
    pct_up = np.mean(beta_new[WARMUP:] > 0) * 100

    stable_pct = np.mean(stable_old[WARMUP:]) * 100

    print(f"\n  {name}  ({T} days, {T/252:.0f}y)")
    print(f"  {'─' * 72}")
    print(f"  {'方法':<28s} {'Sharpe':>8s} {'MaxDD':>8s} {'胜率':>6s} {'交易':>5s} {'总收益':>8s}")
    print(f"  {'─' * 72}")
    print(f"  {'新: Linear Trend (无Regime)':<28s} "
          f"{r_new['sharpe']:>+8.3f} {r_new['max_dd']:>7.1%} "
          f"{r_new['win_rate']:>5.0%} {r_new['n_trades']:>5d} {r_new['total_ret']:>+7.1%}")
    print(f"  {'旧: Level + Regime':<28s} "
          f"{r_old['sharpe']:>+8.3f} {r_old['max_dd']:>7.1%} "
          f"{r_old['win_rate']:>5.0%} {r_old['n_trades']:>5d} {r_old['total_ret']:>+7.1%}")
    print(f"  {'买入持有':<28s} {bh:>+8.3f}")
    print(f"  {'─' * 72}")
    print(f"  趋势诊断: 年化β={avg_beta:+.1%}, β>0占比={pct_up:.0f}%, "
          f"旧方法稳定期={stable_pct:.0f}%")

    return {"name": name, "new": r_new, "old": r_old, "bh": bh,
            "avg_beta": avg_beta, "pct_up": pct_up}


# ============================================================
# Walk-Forward (固定参数, 无网格搜索)
# ============================================================

def run_walkforward(code, name):
    """
    Walk-Forward: 固定参数, 无网格搜索.
    5年训练 → 1年测试 → 滚动前进.
    对比 IS vs OOS, 新方法 vs 旧方法.
    """
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    # 预计算卡尔曼 (因果: μ(t) 只依赖 [0, t])
    # R 用前 60 天估计, 无未来函数
    mu_new, beta_new, _, _ = kalman_linear_trend(log_prices)
    mu_old, innov_old, istd_old = kalman_level(log_prices)
    stable_old = detect_jumps(innov_old, istd_old)

    # 滚动窗口
    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    print(f"\n{'=' * 82}")
    print(f"  {name}  Walk-Forward  ({T} days, {len(windows)} windows)")
    print(f"  Train={TRAIN_DAYS}d ({TRAIN_DAYS//252}y)  Test={TEST_DAYS}d  "
          f"固定参数: Q={Q_RATIO} entry_z={ENTRY_Z} exit_z={EXIT_Z}")
    print(f"{'=' * 82}")

    hdr = (f"  {'Win':>3s}  {'Period':>23s}  "
           f"{'新IS':>7s} {'新OOS':>7s} {'旧IS':>7s} {'旧OOS':>7s}  "
           f"{'新#Tr':>5s} {'旧#Tr':>5s} {'新DD':>7s}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*23}  {'─'*7} {'─'*7} {'─'*7} {'─'*7}  {'─'*5} {'─'*5} {'─'*7}")

    stats = {"new_is": [], "new_oos": [], "old_is": [], "old_oos": []}

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        # ── 新方法 IS ──
        lp_is = log_prices[tr_s:tr_e]
        p_is = prices[tr_s:tr_e]
        mu_is = mu_new[tr_s:tr_e]
        pos_is = generate_signals(lp_is, mu_is, warmup=WARMUP)
        r_new_is = backtest(p_is, pos_is, warmup=WARMUP)

        # ── 新方法 OOS ──
        ctx_s = max(0, tr_e - WARMUP)
        lp_oos = log_prices[ctx_s:te_e]
        p_oos = prices[ctx_s:te_e]
        mu_oos = mu_new[ctx_s:te_e]
        pos_oos = generate_signals(lp_oos, mu_oos, warmup=WARMUP)
        offset = tr_e - ctx_s
        pos_test = np.zeros(len(p_oos), dtype=int)
        pos_test[offset:] = pos_oos[offset:]
        r_new_oos = backtest(p_oos, pos_test, warmup=offset)

        # ── 旧方法 IS ──
        mu_old_is = mu_old[tr_s:tr_e]
        stable_is = stable_old[tr_s:tr_e]
        pos_old_is = generate_signals(lp_is, mu_old_is, stable=stable_is, warmup=WARMUP)
        r_old_is = backtest(p_is, pos_old_is, warmup=WARMUP)

        # ── 旧方法 OOS ──
        mu_old_oos = mu_old[ctx_s:te_e]
        stable_oos = stable_old[ctx_s:te_e]
        pos_old_oos = generate_signals(lp_oos, mu_old_oos, stable=stable_oos, warmup=WARMUP)
        pos_old_test = np.zeros(len(p_oos), dtype=int)
        pos_old_test[offset:] = pos_old_oos[offset:]
        r_old_oos = backtest(p_oos, pos_old_test, warmup=offset)

        stats["new_is"].append(r_new_is["sharpe"])
        stats["new_oos"].append(r_new_oos["sharpe"])
        stats["old_is"].append(r_old_is["sharpe"])
        stats["old_oos"].append(r_old_oos["sharpe"])

        period = f"{dates[tr_s]}~{dates[tr_e-1]}|{dates[tr_e]}~{dates[te_e-1]}"
        print(f"  {wi+1:>3d}  {period:>23s}  "
              f"{r_new_is['sharpe']:>+7.3f} {r_new_oos['sharpe']:>+7.3f} "
              f"{r_old_is['sharpe']:>+7.3f} {r_old_oos['sharpe']:>+7.3f}  "
              f"{r_new_oos['n_trades']:>5d} {r_old_oos['n_trades']:>5d} "
              f"{r_new_oos['max_dd']:>6.1%}")

    # ── 汇总 ──
    print(f"\n  {'─' * 78}")
    for label, key_is, key_oos in [
        ("新 (Linear Trend)", "new_is", "new_oos"),
        ("旧 (Level+Regime)", "old_is", "old_oos"),
    ]:
        is_avg = np.mean(stats[key_is])
        oos_avg = np.mean(stats[key_oos])
        decay = (oos_avg - is_avg) / abs(is_avg) * 100 if abs(is_avg) > 0.01 else 0
        oos_std = np.std(stats[key_oos])
        print(f"  {label:<22s}  IS={is_avg:+.3f}  OOS={oos_avg:+.3f}  "
              f"衰减={decay:+.0f}%  OOS_std={oos_std:.3f}")
    print(f"  {'─' * 78}")

    return stats


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 82)
    print("  修正版实验: Local Linear Trend + 无 Regime 过滤")
    print(f"  固定参数: Q={Q_RATIO} Q_β={Q_SLOPE_RATIO} "
          f"entry_z={ENTRY_Z} exit_z={EXIT_Z} window={RESIDUAL_WINDOW}")
    print("=" * 82)

    # ── Part 1: 全样本对比 ──
    print(f"\n{'=' * 82}")
    print("  Part 1: 全样本对比")
    print(f"{'=' * 82}")

    full_results = []
    for code, name in FUTURES:
        r = run_full_sample(code, name)
        full_results.append(r)

    # 全样本汇总表
    print(f"\n{'=' * 82}")
    print("  全样本汇总")
    print(f"{'=' * 82}")
    print(f"  {'品种':<12s} {'新Sharpe':>9s} {'旧Sharpe':>9s} {'B&H':>7s} "
          f"{'新MaxDD':>8s} {'旧MaxDD':>8s} {'新#Tr':>6s} {'旧#Tr':>6s}")
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*7} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    new_sharpes, old_sharpes = [], []
    for r in full_results:
        new_sharpes.append(r["new"]["sharpe"])
        old_sharpes.append(r["old"]["sharpe"])
        print(f"  {r['name']:<12s} "
              f"{r['new']['sharpe']:>+9.3f} {r['old']['sharpe']:>+9.3f} {r['bh']:>+7.3f} "
              f"{r['new']['max_dd']:>7.1%} {r['old']['max_dd']:>7.1%} "
              f"{r['new']['n_trades']:>6d} {r['old']['n_trades']:>6d}")

    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*7} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    print(f"  {'平均':<12s} "
          f"{np.mean(new_sharpes):>+9.3f} {np.mean(old_sharpes):>+9.3f}")

    # ── Part 2: Walk-Forward ──
    print(f"\n{'=' * 82}")
    print("  Part 2: Walk-Forward (固定参数, 无网格搜索)")
    print(f"{'=' * 82}")

    wf_results = {}
    for code, name in FUTURES:
        wf_results[name] = run_walkforward(code, name)

    # Walk-Forward 最终汇总
    print(f"\n{'=' * 82}")
    print("  Walk-Forward 最终汇总")
    print(f"{'=' * 82}")
    print(f"  {'品种':<12s} {'新IS':>7s} {'新OOS':>7s} {'旧IS':>7s} {'旧OOS':>7s}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    all_new_oos, all_old_oos = [], []
    for name, s in wf_results.items():
        new_is = np.mean(s["new_is"])
        new_oos = np.mean(s["new_oos"])
        old_is = np.mean(s["old_is"])
        old_oos = np.mean(s["old_oos"])
        all_new_oos.extend(s["new_oos"])
        all_old_oos.extend(s["old_oos"])
        print(f"  {name:<12s} {new_is:>+7.3f} {new_oos:>+7.3f} "
              f"{old_is:>+7.3f} {old_oos:>+7.3f}")

    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'全品种OOS均值':<12s} {'':>7s} {np.mean(all_new_oos):>+7.3f} "
          f"{'':>7s} {np.mean(all_old_oos):>+7.3f}")
    print(f"  {'全品种OOS中位':<12s} {'':>7s} {np.median(all_new_oos):>+7.3f} "
          f"{'':>7s} {np.median(all_old_oos):>+7.3f}")


if __name__ == "__main__":
    main()
