"""
kalman_mu_walkforward.py — Walk-Forward 验证 (独立实验)
======================================================

对铜(CU)和白糖(SR)做严格的 walk-forward 验证:
    5年训练 → 1年测试 → 滚动前进

验证: 全样本优化的 Sharpe 是否在样本外保持。

用法: python experiments/kalman_mu_walkforward.py
"""

import struct
from pathlib import Path
import numpy as np

# ============================================================
# 配置
# ============================================================

TDX_ROOT = Path(r"D:\new_tdx64\vipdoc\ds\lday")

FUTURES = [
    ("30#CUL9", "铜 CU"),
    ("28#SRL9", "白糖 SR"),
]

COMMISSION_RATE = 0.00025
SLIPPAGE_RATE = 0.0005
INITIAL_CAPITAL = 1_000_000.0

# Walk-forward 参数
TRAIN_DAYS = 1260    # 5 年
TEST_DAYS = 252      # 1 年
STEP_DAYS = 252      # 步进 1 年
WARMUP = 120

# 参数网格
Q_RATIOS = [0.005, 0.01, 0.05]
ENTRY_ZS = [1.0, 1.5, 2.0]
JUMP_THRESHOLDS = [2.0, 2.5, 3.0]
RESIDUAL_WINDOW = 20
COOLDOWN_DAYS = 10


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
# 核心算法
# ============================================================

def kalman_level(log_prices, q_ratio):
    T = len(log_prices)
    R = max(np.var(np.diff(log_prices)) / 2.0, 1e-10)
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


def detect_jumps(innovation, innov_std, threshold, cooldown):
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


def generate_signals(log_prices, mu, stable, entry_z, warmup=0):
    T = len(log_prices)
    residual = log_prices - mu
    sigma = np.full(T, np.nan)
    for t in range(RESIDUAL_WINDOW, T):
        sigma[t] = np.std(residual[t-RESIDUAL_WINDOW+1:t+1], ddof=1)
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
            if z[t] > 0 or not regime_ok:
                in_pos = False
            else:
                position[t] = 1
    return position


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
            proceeds = shares * sp
            cash += proceeds * (1 - COMMISSION_RATE)
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
    return {"sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
            "n_trades": len(trade_rets), "total_ret": total_ret}


# ============================================================
# Walk-Forward 引擎
# ============================================================

def walk_forward(code, name):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    # 预计算所有 Q/R 的卡尔曼滤波 (因果的, 不泄露未来)
    kf_cache = {}
    for qr in Q_RATIOS:
        mu, innov, istd = kalman_level(log_prices, qr)
        kf_cache[qr] = (mu, innov, istd)

    # 滚动窗口
    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    print(f"\n{'=' * 78}")
    print(f"  {name}  Walk-Forward  ({T} days, {len(windows)} windows)")
    print(f"  Train={TRAIN_DAYS}d ({TRAIN_DAYS//252}y)  Test={TEST_DAYS}d  Step={STEP_DAYS}d")
    print(f"{'=' * 78}")

    hdr = (f"  {'Window':>6s}  {'Period':>23s}  "
           f"{'IS Sharpe':>9s}  {'OOS Sharpe':>10s}  {'OOS MaxDD':>9s}  "
           f"{'OOS Win%':>8s}  {'OOS #Tr':>7s}  {'Best Params'}")
    print(hdr)
    print(f"  {'-'*6}  {'-'*23}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*25}")

    is_sharpes = []
    oos_sharpes = []
    oos_dds = []
    oos_wins = []
    oos_trades = []

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        # ── 训练: 网格搜索 ──
        best_is = None
        best_params = None

        for qr in Q_RATIOS:
            mu_full, innov_full, istd_full = kf_cache[qr]
            # 截取训练窗口
            mu_tr = mu_full[tr_s:tr_e]
            innov_tr = innov_full[tr_s:tr_e]
            istd_tr = istd_full[tr_s:tr_e]
            lp_tr = log_prices[tr_s:tr_e]
            p_tr = prices[tr_s:tr_e]

            for ez in ENTRY_ZS:
                for jt in JUMP_THRESHOLDS:
                    stable_tr = detect_jumps(innov_tr, istd_tr, jt, COOLDOWN_DAYS)
                    pos_tr = generate_signals(lp_tr, mu_tr, stable_tr, ez, warmup=WARMUP)
                    r_tr = backtest(p_tr, pos_tr, warmup=WARMUP)
                    if best_is is None or r_tr["sharpe"] > best_is["sharpe"]:
                        best_is = r_tr
                        best_params = (qr, ez, jt)

        # ── 测试: 用训练最优参数 ──
        qr, ez, jt = best_params
        mu_full, innov_full, istd_full = kf_cache[qr]

        # 测试窗口需要前面的数据做 warmup
        # 用 [tr_e - WARMUP : te_e] 但信号只在 [tr_e : te_e] 评估
        ctx_start = max(0, tr_e - WARMUP)
        mu_te = mu_full[ctx_start:te_e]
        innov_te = innov_full[ctx_start:te_e]
        istd_te = istd_full[ctx_start:te_e]
        lp_te = log_prices[ctx_start:te_e]
        p_te = prices[ctx_start:te_e]

        stable_te = detect_jumps(innov_te, istd_te, jt, COOLDOWN_DAYS)
        pos_te = generate_signals(lp_te, mu_te, stable_te, ez, warmup=WARMUP)

        # 只评估测试部分
        test_offset = tr_e - ctx_start
        pos_test_only = np.zeros(len(p_te), dtype=int)
        pos_test_only[test_offset:] = pos_te[test_offset:]
        r_te = backtest(p_te, pos_test_only, warmup=test_offset)

        is_sharpes.append(best_is["sharpe"])
        oos_sharpes.append(r_te["sharpe"])
        oos_dds.append(r_te["max_dd"])
        oos_wins.append(r_te["win_rate"])
        oos_trades.append(r_te["n_trades"])

        period = f"{dates[tr_s]}~{dates[tr_e-1]}|{dates[tr_e]}~{dates[te_e-1]}"
        params_str = f"Q/R={qr:.3f} ez={ez:.1f} k={jt:.1f}"
        print(f"  {wi+1:>6d}  {period:>23s}  "
              f"{best_is['sharpe']:>+9.3f}  {r_te['sharpe']:>+10.3f}  "
              f"{r_te['max_dd']:>8.1%}  {r_te['win_rate']:>7.0%}  "
              f"{r_te['n_trades']:>7d}  {params_str}")

    # ── 汇总 ──
    print(f"\n  {'─' * 74}")
    is_avg = np.mean(is_sharpes)
    oos_avg = np.mean(oos_sharpes)
    oos_dd_avg = np.mean(oos_dds)
    oos_win_avg = np.mean(oos_wins)
    oos_tr_avg = np.mean(oos_trades)
    decay = (oos_avg - is_avg) / abs(is_avg) * 100 if abs(is_avg) > 0.01 else 0

    print(f"  样本内平均 Sharpe:  {is_avg:+.3f}")
    print(f"  样本外平均 Sharpe:  {oos_avg:+.3f}")
    print(f"  Sharpe 衰减:        {decay:+.0f}%")
    print(f"  样本外平均 MaxDD:   {oos_dd_avg:.1%}")
    print(f"  样本外平均胜率:     {oos_win_avg:.0%}")
    print(f"  样本外平均交易数:   {oos_tr_avg:.1f}")
    print(f"  {'─' * 74}")

    return {"name": name, "is_avg": is_avg, "oos_avg": oos_avg,
            "decay": decay, "oos_dd": oos_dd_avg}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 78)
    print("  Walk-Forward 验证: 卡尔曼 μ + Regime 过滤")
    print("  训练 5 年 → 测试 1 年 → 滚动前进")
    print("=" * 78)

    results = []
    for code, name in FUTURES:
        r = walk_forward(code, name)
        results.append(r)

    print(f"\n{'=' * 78}")
    print("  最终汇总")
    print(f"{'=' * 78}")
    for r in results:
        print(f"  {r['name']:<12s}  IS={r['is_avg']:+.3f}  OOS={r['oos_avg']:+.3f}  "
              f"decay={r['decay']:+.0f}%  OOS_MaxDD={r['oos_dd']:.1%}")


if __name__ == "__main__":
    main()
