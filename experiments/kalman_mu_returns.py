"""
kalman_mu_returns.py — 收益率清洗 + 拼接跳空处理 (独立实验)
==========================================================

L9 连续合约在主力换月时存在价格跳空 (拼接伪影)。
本实验用收益率 winsorize 消除跳空, 重建干净价格序列,
对比调整前后的策略表现。

方法:
    1. 计算对数收益率 r_t = log(P_t / P_{t-1})
    2. 检测拼接跳空: |r_t| > 5σ (滚动60日)
    3. Winsorize: 将极端收益率截断到 ±5σ
    4. 从清洗后收益率重建 log_price_clean
    5. 在 clean 价格上跑卡尔曼 μ + regime 策略

用法: python experiments/kalman_mu_returns.py
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

# 拼接检测
SPLICE_WINDOW = 60     # 滚动标准差窗口
SPLICE_SIGMA = 5.0     # 超过 5σ 视为拼接跳空


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
# 收益率清洗
# ============================================================

def clean_splices(prices):
    """
    检测并清洗 L9 拼接跳空。

    返回:
        clean_log_prices: 清洗后的对数价格
        n_splices: 检测到的拼接跳空数量
        splice_indices: 跳空位置
    """
    log_p = np.log(prices)
    returns = np.diff(log_p)  # T-1 个收益率
    T = len(returns)

    # 滚动标准差
    roll_std = np.full(T, np.nan)
    for t in range(SPLICE_WINDOW, T):
        roll_std[t] = np.std(returns[t-SPLICE_WINDOW+1:t+1], ddof=1)

    # 检测跳空
    threshold = SPLICE_SIGMA * roll_std
    splice_mask = np.abs(returns) > threshold
    # 前 SPLICE_WINDOW 天无法检测, 默认不标记
    splice_mask[:SPLICE_WINDOW] = False

    splice_indices = np.where(splice_mask)[0]
    n_splices = len(splice_indices)

    # Winsorize: 截断到 ±5σ
    clean_returns = returns.copy()
    for idx in splice_indices:
        limit = SPLICE_SIGMA * roll_std[idx]
        clean_returns[idx] = np.clip(returns[idx], -limit, limit)

    # 重建干净对数价格
    clean_log_p = np.zeros(len(prices))
    clean_log_p[0] = log_p[0]
    for t in range(1, len(prices)):
        clean_log_p[t] = clean_log_p[t-1] + clean_returns[t-1]

    return clean_log_p, n_splices, splice_indices


# ============================================================
# 核心算法 (同前)
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


def generate_signals(log_prices, mu, stable, entry_z):
    T = len(log_prices)
    residual = log_prices - mu
    sigma = np.full(T, np.nan)
    for t in range(RESIDUAL_WINDOW, T):
        sigma[t] = np.std(residual[t-RESIDUAL_WINDOW+1:t+1], ddof=1)
    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    pos = np.zeros(T, dtype=int)
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


def backtest(prices, position):
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

def find_best(prices_for_bt, log_prices_for_signal, label_prefix=""):
    """网格搜索最佳参数。"""
    best = None
    for qr in Q_RATIOS:
        mu, innov, istd = kalman_level(log_prices_for_signal, qr)
        for ez in ENTRY_ZS:
            for jt in JUMP_THRESHOLDS:
                stable = detect_jumps(innov, istd, jt, COOLDOWN_DAYS)
                pos = generate_signals(log_prices_for_signal, mu, stable, ez)
                r = backtest(prices_for_bt, pos)
                r["params"] = f"Q/R={qr:.3f} ez={ez:.1f} k={jt:.1f}"
                if best is None or r["sharpe"] > best["sharpe"]:
                    best = r
    return best


def run_one(code, name):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    # 清洗拼接跳空
    clean_log_p, n_splices, splice_idx = clean_splices(prices)
    clean_prices = np.exp(clean_log_p)

    print(f"\n{'=' * 78}")
    print(f"  {name}  ({T} days = {T/252:.1f}y)  拼接跳空: {n_splices} 次")
    if n_splices > 0:
        # 显示前5个跳空
        show = splice_idx[:5]
        for idx in show:
            r_orig = log_prices[idx+1] - log_prices[idx]
            r_clean = clean_log_p[idx+1] - clean_log_p[idx]
            print(f"    day {idx+1} ({dates[idx+1]}): "
                  f"raw_ret={r_orig:+.4f} → clean_ret={r_clean:+.4f}")
        if n_splices > 5:
            print(f"    ... 共 {n_splices} 次")
    print(f"{'=' * 78}")

    # 原始数据
    r_orig = find_best(prices, log_prices)
    # 清洗后数据
    r_clean = find_best(clean_prices, clean_log_p)

    hdr = f"  {'Data':<20s} {'Sharpe':>7s} {'TotalRet':>9s} {'MaxDD':>7s} {'Win%':>5s} {'#Tr':>4s} {'Params'}"
    print(hdr)
    print(f"  {'-'*20} {'-'*7} {'-'*9} {'-'*7} {'-'*5} {'-'*4} {'-'*25}")

    def row(label, r):
        print(f"  {label:<20s} {r['sharpe']:>+7.3f} {r['total_ret']:>+8.1%} "
              f"{r['max_dd']:>6.1%} {r['win_rate']:>4.0%} {r['n_trades']:>4d} {r['params']}")

    row("Original L9", r_orig)
    row("Splice-cleaned", r_clean)

    delta = r_clean["sharpe"] - r_orig["sharpe"]
    print(f"\n  Sharpe delta (clean - original): {delta:+.3f}")

    return {"name": name, "orig": r_orig, "clean": r_clean,
            "delta": delta, "n_splices": n_splices}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 78)
    print("  收益率清洗: 处理 L9 连续合约拼接跳空")
    print(f"  检测: |return| > {SPLICE_SIGMA}σ (滚动{SPLICE_WINDOW}日)")
    print(f"  处理: Winsorize 截断到 ±{SPLICE_SIGMA}σ, 重建干净价格")
    print("=" * 78)

    results = []
    for code, name in FUTURES:
        r = run_one(code, name)
        results.append(r)

    print(f"\n{'=' * 78}")
    print("  汇总")
    print(f"{'=' * 78}")
    print(f"  {'品种':<14s} {'跳空数':>5s} {'Orig Sharpe':>11s} {'Clean Sharpe':>12s} {'Delta':>8s}")
    print(f"  {'-'*14} {'-'*5} {'-'*11} {'-'*12} {'-'*8}")
    for r in results:
        print(f"  {r['name']:<14s} {r['n_splices']:>5d} "
              f"{r['orig']['sharpe']:>+11.3f} {r['clean']['sharpe']:>+12.3f} "
              f"{r['delta']:>+8.3f}")

    orig_avg = np.mean([r["orig"]["sharpe"] for r in results])
    clean_avg = np.mean([r["clean"]["sharpe"] for r in results])
    print(f"\n  平均: Original={orig_avg:+.3f}  Clean={clean_avg:+.3f}  "
          f"Delta={clean_avg-orig_avg:+.3f}")


if __name__ == "__main__":
    main()
