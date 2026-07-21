"""
kalman_mu_longshort.py — 多空双向验证 (独立实验)
================================================

期货可以双向交易。加入做空信号:
    z < -entry_z → 做多 (价格低于情绪锚)
    z > +entry_z → 做空 (价格高于情绪锚)

对比: 仅多 vs 多空, 5 个商品期货品种。

用法: python experiments/kalman_mu_longshort.py
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

COMMISSION_RATE = 0.00025
SLIPPAGE_RATE = 0.0005
INITIAL_CAPITAL = 1_000_000.0

Q_RATIOS = [0.005, 0.01, 0.05]
ENTRY_ZS = [1.0, 1.5, 2.0]
JUMP_THRESHOLDS = [2.0, 2.5, 3.0]
RESIDUAL_WINDOW = 20
COOLDOWN_DAYS = 10
WARMUP = 120


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


# ============================================================
# 信号: 仅多 vs 多空
# ============================================================

def signals_long_only(log_prices, mu, stable, entry_z):
    """仅做多: z < -entry_z → long, z > 0 → exit."""
    T = len(log_prices)
    residual = log_prices - mu
    sigma = np.full(T, np.nan)
    for t in range(RESIDUAL_WINDOW, T):
        sigma[t] = np.std(residual[t-RESIDUAL_WINDOW+1:t+1], ddof=1)
    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    pos = np.zeros(T, dtype=int)  # 0=flat, 1=long
    in_pos = False
    for t in range(WARMUP, T):
        ok = stable[t] if stable is not None else True
        if not in_pos:
            if z[t] < -entry_z and ok:
                in_pos = True
                pos[t] = 1
        else:
            if z[t] > 0 or not ok:
                in_pos = False
            else:
                pos[t] = 1
    return pos


def signals_long_short(log_prices, mu, stable, entry_z):
    """多空双向: z < -entry_z → long, z > +entry_z → short."""
    T = len(log_prices)
    residual = log_prices - mu
    sigma = np.full(T, np.nan)
    for t in range(RESIDUAL_WINDOW, T):
        sigma[t] = np.std(residual[t-RESIDUAL_WINDOW+1:t+1], ddof=1)
    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    pos = np.zeros(T, dtype=int)  # 0=flat, 1=long, -1=short
    state = 0  # 0=flat, 1=long, -1=short
    for t in range(WARMUP, T):
        ok = stable[t] if stable is not None else True

        if state == 0:
            if z[t] < -entry_z and ok:
                state = 1
                pos[t] = 1
            elif z[t] > entry_z and ok:
                state = -1
                pos[t] = -1
        elif state == 1:
            if z[t] > 0 or not ok:
                state = 0
                pos[t] = 0
            else:
                pos[t] = 1
        elif state == -1:
            if z[t] < 0 or not ok:
                state = 0
                pos[t] = 0
            else:
                pos[t] = -1
    return pos


# ============================================================
# 回测 (支持多空)
# ============================================================

def backtest(prices, position):
    """
    支持 position ∈ {-1, 0, 1}。
    +1 = 做多, -1 = 做空, 0 = 空仓。

    空头机制:
        开空: 借入卖出, cash += shares × price (收到现金)
        持空: equity = cash - shares × current_price (现金 - 负债)
        平空: 买回归还, cash -= shares × price (付出现金)
    """
    T = len(prices)
    cash = INITIAL_CAPITAL
    shares = 0.0
    entry_price = 0.0
    direction = 0      # +1 long, -1 short

    equity = np.full(T, INITIAL_CAPITAL)
    daily_ret = np.zeros(T)
    trade_rets = []

    for t in range(1, T):
        price_ret = (prices[t] - prices[t-1]) / prices[t-1]

        # 持仓日收益
        if direction == 1:
            daily_ret[t] = price_ret
        elif direction == -1:
            daily_ret[t] = -price_ret

        want = position[t-1]  # T+1 执行

        if want != direction:
            # ── 平旧仓 ──
            if direction == 1 and shares > 0:
                sp = prices[t] * (1 - SLIPPAGE_RATE)
                cash += shares * sp * (1 - COMMISSION_RATE)
                trade_rets.append((sp - entry_price) / entry_price)
                shares = 0
                direction = 0
            elif direction == -1 and shares > 0:
                # 平空: 买回归还
                bp = prices[t] * (1 + SLIPPAGE_RATE)
                cash -= shares * bp * (1 + COMMISSION_RATE)
                trade_rets.append((entry_price - bp) / entry_price)
                shares = 0
                direction = 0

            # ── 开新仓 ──
            if want == 1:
                bp = prices[t] * (1 + SLIPPAGE_RATE)
                shares = cash * 0.95 / bp
                if shares > 0:
                    cash -= shares * bp * (1 + COMMISSION_RATE)
                    entry_price = bp
                    direction = 1
            elif want == -1:
                sp = prices[t] * (1 - SLIPPAGE_RATE)
                # 做空: 用 95% 资金作为保证金, 借入卖出
                notional = cash * 0.95
                shares = notional / sp
                if shares > 0:
                    cash += shares * sp * (1 - COMMISSION_RATE)
                    entry_price = sp
                    direction = -1

        # 权益 = 现金 + 多头市值 - 空头负债
        equity[t] = cash
        if direction == 1:
            equity[t] += shares * prices[t]
        elif direction == -1:
            equity[t] -= shares * prices[t]

    # 强制平仓
    if direction == 1 and shares > 0:
        sp = prices[-1] * (1 - SLIPPAGE_RATE)
        cash += shares * sp * (1 - COMMISSION_RATE)
        trade_rets.append((sp - entry_price) / entry_price)
        equity[-1] = cash
    elif direction == -1 and shares > 0:
        bp = prices[-1] * (1 + SLIPPAGE_RATE)
        cash -= shares * bp * (1 + COMMISSION_RATE)
        trade_rets.append((entry_price - bp) / entry_price)
        equity[-1] = cash

    vr = daily_ret[WARMUP:]
    sharpe = 0.0
    if len(vr) > 10 and np.std(vr) > 1e-12:
        sharpe = float(np.mean(vr) / np.std(vr) * np.sqrt(252))
    total_ret = equity[-1] / equity[WARMUP] - 1
    eq = equity[WARMUP:]
    rm = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - rm) / rm))
    win_rate = sum(1 for r in trade_rets if r > 0) / max(len(trade_rets), 1)
    return {"sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
            "n_trades": len(trade_rets), "total_ret": total_ret}


# ============================================================
# 单品种实验
# ============================================================

def run_one(code, name):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    print(f"\n{'=' * 78}")
    print(f"  {name}  ({T} days = {T/252:.1f}y)")
    print(f"{'=' * 78}")

    best_lo = None  # long-only
    best_ls = None  # long-short

    for qr in Q_RATIOS:
        mu, innov, istd = kalman_level(log_prices, qr)
        for ez in ENTRY_ZS:
            for jt in JUMP_THRESHOLDS:
                stable = detect_jumps(innov, istd, jt, COOLDOWN_DAYS)

                # Long-only
                pos_lo = signals_long_only(log_prices, mu, stable, ez)
                r_lo = backtest(prices, pos_lo)
                r_lo["params"] = f"Q/R={qr:.3f} ez={ez:.1f} k={jt:.1f}"
                if best_lo is None or r_lo["sharpe"] > best_lo["sharpe"]:
                    best_lo = r_lo

                # Long-short
                pos_ls = signals_long_short(log_prices, mu, stable, ez)
                r_ls = backtest(prices, pos_ls)
                r_ls["params"] = f"Q/R={qr:.3f} ez={ez:.1f} k={jt:.1f}"
                if best_ls is None or r_ls["sharpe"] > best_ls["sharpe"]:
                    best_ls = r_ls

    hdr = f"  {'Mode':<16s} {'Sharpe':>7s} {'TotalRet':>9s} {'MaxDD':>7s} {'Win%':>5s} {'#Tr':>4s} {'Params'}"
    print(hdr)
    print(f"  {'-'*16} {'-'*7} {'-'*9} {'-'*7} {'-'*5} {'-'*4} {'-'*25}")

    def row(label, r):
        print(f"  {label:<16s} {r['sharpe']:>+7.3f} {r['total_ret']:>+8.1%} "
              f"{r['max_dd']:>6.1%} {r['win_rate']:>4.0%} {r['n_trades']:>4d} {r['params']}")

    row("Long-only", best_lo)
    row("Long-Short", best_ls)

    delta = best_ls["sharpe"] - best_lo["sharpe"]
    print(f"\n  Long-Short vs Long-only Sharpe delta: {delta:+.3f}")

    return {"name": name, "lo": best_lo, "ls": best_ls, "delta": delta}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 78)
    print("  多空双向验证: 卡尔曼 μ + Regime + 做空")
    print("  z < -entry_z → 做多 | z > +entry_z → 做空")
    print("=" * 78)

    results = []
    for code, name in FUTURES:
        r = run_one(code, name)
        results.append(r)

    print(f"\n{'=' * 78}")
    print("  汇总")
    print(f"{'=' * 78}")
    print(f"  {'品种':<14s} {'LO Sharpe':>10s} {'LS Sharpe':>10s} {'Delta':>8s} {'LO MaxDD':>9s} {'LS MaxDD':>9s}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*9} {'-'*9}")
    for r in results:
        print(f"  {r['name']:<14s} {r['lo']['sharpe']:>+10.3f} {r['ls']['sharpe']:>+10.3f} "
              f"{r['delta']:>+8.3f} {r['lo']['max_dd']:>8.1%} {r['ls']['max_dd']:>8.1%}")

    lo_avg = np.mean([r["lo"]["sharpe"] for r in results])
    ls_avg = np.mean([r["ls"]["sharpe"] for r in results])
    print(f"\n  平均: LO={lo_avg:+.3f}  LS={ls_avg:+.3f}  Delta={ls_avg-lo_avg:+.3f}")


if __name__ == "__main__":
    main()
