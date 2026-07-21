"""
kalman_mu_futures.py — 商品期货均值回归验证 (独立实验)
=====================================================

验证: 卡尔曼 μ + regime 过滤在商品期货上是否比 A股 ETF 更有效。
商品期货均值回归性更强 (供需驱动, 季节性, 无涨跌停, T+0)。

品种:
    豆粕(M) 25y, 铜(CU) 25y, 白糖(SR) 20y, 螺纹钢(RB) 17y, 黄金(AU) 18y

对比:
    A) 滚动均值 (固定窗口)
    B) 卡尔曼 μ (无 regime 过滤)
    C) 卡尔曼 μ + regime 过滤

数据: 通达信期货 L9 (最长连续合约), float32 价格格式。

用法: python experiments/kalman_mu_futures.py
"""

import struct
from pathlib import Path

import numpy as np


# ============================================================
# 配置
# ============================================================

TDX_ROOT = Path(r"D:\new_tdx64\vipdoc\ds\lday")

# 期货品种 (代码, 名称, L9文件名)
FUTURES = [
    ("29#ML9",  "豆粕 M"),
    ("30#CUL9", "铜 CU"),
    ("28#SRL9", "白糖 SR"),
    ("30#RBL9", "螺纹钢 RB"),
    ("30#AUL9", "黄金 AU"),
]

# 期货交易成本 (简化)
COMMISSION_RATE = 0.00025   # 佣金 万2.5/边
SLIPPAGE_RATE = 0.0005      # 滑点 万5 (期货流动性好, 比ETF小)
# 无印花税, 无整数手限制

# 参数网格
Q_RATIOS = [0.005, 0.01, 0.05]
JUMP_THRESHOLDS = [2.0, 2.5, 3.0]
ENTRY_ZS = [1.0, 1.5, 2.0]
RESIDUAL_WINDOW = 20
COOLDOWN_DAYS = 10
WARMUP = 120                # 期货数据长, 预热期可以大一些
BASELINE_LOOKBACK = 60      # 基线滚动窗口


# ============================================================
# 数据读取 (期货 float32 格式)
# ============================================================

def read_futures(filename: str) -> tuple:
    """
    读取通达信期货 .day 文件。

    期货格式: date(uint32) + OHLC+amount(float32×5) + volume(uint32) + reserved(uint32)
    """
    filepath = TDX_ROOT / f"{filename}.day"
    if not filepath.exists():
        raise FileNotFoundError(f"不存在: {filepath}")

    raw = filepath.read_bytes()
    n = len(raw) // 32
    fmt = "<IfffffII"

    dates = []
    closes = []
    for i in range(n):
        chunk = raw[i * 32 : (i + 1) * 32]
        d, o, h, l, c, amt, vol, _ = struct.unpack(fmt, chunk)
        dates.append(d)
        closes.append(c)

    return np.array(dates), np.array(closes, dtype=float)


# ============================================================
# 卡尔曼滤波
# ============================================================

def kalman_level(log_prices: np.ndarray, q_ratio: float) -> dict:
    """标量卡尔曼滤波估计 μ_t。"""
    T = len(log_prices)
    returns = np.diff(log_prices)
    R = max(np.var(returns) / 2.0, 1e-10)
    Q = q_ratio * R

    mu = np.zeros(T)
    P = np.zeros(T)
    innovation = np.zeros(T)
    innov_std = np.zeros(T)

    mu[0] = log_prices[0]
    P[0] = R

    for t in range(1, T):
        mu_pred = mu[t - 1]
        P_pred = P[t - 1] + Q
        innov = log_prices[t] - mu_pred
        S = P_pred + R
        K = P_pred / S
        mu[t] = mu_pred + K * innov
        P[t] = (1.0 - K) * P_pred
        innovation[t] = innov
        innov_std[t] = np.sqrt(S)

    return {"mu": mu, "innovation": innovation, "innov_std": innov_std}


# ============================================================
# Regime 检测
# ============================================================

def detect_jumps(innovation, innov_std, threshold, cooldown):
    """Innovation z-score 跳跃检测 + 冷却期。"""
    T = len(innovation)
    stable = np.ones(T, dtype=bool)
    safe_std = np.where(innov_std > 1e-12, innov_std, 1e-12)
    z = np.abs(innovation) / safe_std

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

def generate_signals(log_prices, mu, stable, entry_z,
                     residual_window=RESIDUAL_WINDOW, warmup=WARMUP):
    """z-score 均值回归信号, 可选 regime 过滤。"""
    T = len(log_prices)
    residual = log_prices - mu

    sigma = np.full(T, np.nan)
    for t in range(residual_window, T):
        sigma[t] = np.std(residual[t - residual_window + 1 : t + 1], ddof=1)

    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    position = np.zeros(T, dtype=int)
    in_pos = False
    for t in range(warmup, T):
        regime_ok = stable[t] if stable is not None else True
        if not in_pos:
            if z[t] < -entry_z and regime_ok:
                in_pos = True
                position[t] = 1
        else:
            if z[t] > 0 or not regime_ok:
                in_pos = False
                position[t] = 0
            else:
                position[t] = 1
    return position


# ============================================================
# 回测 (期货: 无印花税, 无整数手, T+1执行)
# ============================================================

INITIAL_CAPITAL = 1_000_000.0


def backtest(prices, position):
    """简化期货回测。"""
    T = len(prices)
    cash = INITIAL_CAPITAL
    shares = 0.0  # 期货用浮点数 (无整数手)
    entry_price = 0.0

    equity = np.full(T, INITIAL_CAPITAL)
    daily_ret = np.zeros(T)
    trade_returns = []
    hold_days = []
    hold_start = -1

    for t in range(1, T):
        if shares > 0:
            daily_ret[t] = (prices[t] - prices[t - 1]) / prices[t - 1]

        want_long = position[t - 1] == 1

        if want_long and shares == 0:
            buy_price = prices[t] * (1 + SLIPPAGE_RATE)
            available = cash * 0.95
            shares = available / buy_price
            if shares > 0:
                commission = shares * buy_price * COMMISSION_RATE
                cash -= (shares * buy_price + commission)
                entry_price = buy_price
                hold_start = t

        elif not want_long and shares > 0:
            sell_price = prices[t] * (1 - SLIPPAGE_RATE)
            proceeds = shares * sell_price
            commission = proceeds * COMMISSION_RATE
            cash += (proceeds - commission)
            trade_ret = (sell_price - entry_price) / entry_price
            trade_returns.append(trade_ret)
            if hold_start >= 0:
                hold_days.append(t - hold_start)
            shares = 0
            hold_start = -1

        equity[t] = cash + shares * prices[t]

    if shares > 0:
        sell_price = prices[-1] * (1 - SLIPPAGE_RATE)
        proceeds = shares * sell_price
        commission = proceeds * COMMISSION_RATE
        cash += (proceeds - commission)
        trade_ret = (sell_price - entry_price) / entry_price
        trade_returns.append(trade_ret)
        if hold_start >= 0:
            hold_days.append(T - 1 - hold_start)
        equity[-1] = cash

    # 指标
    vr = daily_ret[WARMUP:]
    sharpe = 0.0
    if len(vr) > 10 and np.std(vr) > 1e-12:
        sharpe = float(np.mean(vr) / np.std(vr) * np.sqrt(252))

    total_ret = equity[-1] / equity[WARMUP] - 1
    n_days = T - WARMUP
    if total_ret > 0:
        apr = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
    elif total_ret == 0:
        apr = 0.0
    else:
        apr = -1.0 + (1 + total_ret) ** (252 / max(n_days, 1))

    eq = equity[WARMUP:]
    rm = np.maximum.accumulate(eq)
    dd = (eq - rm) / rm
    max_dd = float(np.min(dd))

    win_rate = 0.0
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

    avg_hold = float(np.mean(hold_days)) if hold_days else 0.0

    return {
        "sharpe": sharpe, "apr": apr, "max_dd": max_dd,
        "win_rate": win_rate, "n_trades": len(trade_returns),
        "avg_hold": avg_hold, "total_ret": total_ret,
    }


# ============================================================
# 基线: 滚动均值
# ============================================================

def rolling_mean_strategy(prices, lookback, entry_z, warmup=WARMUP):
    """滚动均值 z-score 策略。"""
    T = len(prices)
    log_p = np.log(prices)
    ma = np.full(T, np.nan)
    std = np.full(T, np.nan)
    for t in range(lookback - 1, T):
        w = log_p[t - lookback + 1 : t + 1]
        ma[t] = np.mean(w)
        std[t] = np.std(w, ddof=1)

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
# 单品种实验
# ============================================================

def run_one(code, name):
    """单品种完整对比。"""
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)
    years = T / 252

    print(f"\n{'=' * 72}")
    print(f"  {name}  ({code})  {T} days = {years:.1f}y  "
          f"[{dates[0]} ~ {dates[-1]}]")
    print(f"{'=' * 72}")

    # [A] 基线
    best_a = None
    for ez in ENTRY_ZS:
        pos = rolling_mean_strategy(prices, BASELINE_LOOKBACK, ez)
        r = backtest(prices, pos)
        if best_a is None or r["sharpe"] > best_a["sharpe"]:
            best_a = r
            best_a["entry_z"] = ez

    # [B] 卡尔曼 μ
    best_b = None
    for qr in Q_RATIOS:
        kf = kalman_level(log_prices, qr)
        for ez in ENTRY_ZS:
            pos = generate_signals(log_prices, kf["mu"], None, ez)
            r = backtest(prices, pos)
            r["q_ratio"] = qr
            r["entry_z"] = ez
            if best_b is None or r["sharpe"] > best_b["sharpe"]:
                best_b = r

    # [C] 卡尔曼 μ + regime
    best_c = None
    for qr in Q_RATIOS:
        kf = kalman_level(log_prices, qr)
        for ez in ENTRY_ZS:
            for jt in JUMP_THRESHOLDS:
                stable = detect_jumps(kf["innovation"], kf["innov_std"],
                                      jt, COOLDOWN_DAYS)
                pos = generate_signals(log_prices, kf["mu"], stable, ez)
                r = backtest(prices, pos)
                r["q_ratio"] = qr
                r["entry_z"] = ez
                r["jump_k"] = jt
                n_stable = np.sum(stable[WARMUP:])
                r["stable_pct"] = n_stable / (T - WARMUP) * 100
                if best_c is None or r["sharpe"] > best_c["sharpe"]:
                    best_c = r

    # 打印
    hdr = f"  {'Method':<32s} {'Sharpe':>7s} {'APR':>7s} {'MaxDD':>7s} {'Win%':>5s} {'#Tr':>4s} {'AvgH':>5s} {'Params'}"
    print(hdr)
    print(f"  {'-' * 32} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*4} {'-'*5} {'-'*20}")

    def row(label, r, params=""):
        print(f"  {label:<32s} {r['sharpe']:>+7.3f} {r['apr']:>+6.1%} "
              f"{r['max_dd']:>6.1%} {r['win_rate']:>4.0%} {r['n_trades']:>4d} "
              f"{r['avg_hold']:>5.1f} {params}")

    row("A: Rolling Mean", best_a,
        f"lb={BASELINE_LOOKBACK} ez={best_a.get('entry_z','')}")
    row("B: Kalman mu", best_b,
        f"Q/R={best_b['q_ratio']:.3f} ez={best_b['entry_z']:.1f}")
    row("C: Kalman mu + regime", best_c,
        f"Q/R={best_c['q_ratio']:.3f} ez={best_c['entry_z']:.1f} "
        f"k={best_c['jump_k']:.1f} stb={best_c['stable_pct']:.0f}%")

    return {"name": name, "A": best_a, "B": best_b, "C": best_c, "years": years}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 72)
    print("  商品期货 — 卡尔曼情绪锚均值回归验证")
    print("  数据: 通达信 L9 最长连续合约 (15-25年)")
    print("=" * 72)

    results = []
    for code, name in FUTURES:
        try:
            r = run_one(code, name)
            results.append(r)
        except Exception as e:
            print(f"\n  [SKIP] {name}: {e}")

    # 汇总
    print(f"\n{'=' * 72}")
    print("  汇总: 各品种最佳 Sharpe")
    print(f"{'=' * 72}")
    print(f"  {'品种':<14s} {'年数':>4s}  {'A:RollMean':>10s}  {'B:Kalman':>10s}  {'C:K+Regime':>10s}")
    print(f"  {'-'*14} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")

    for r in results:
        a = r["A"]["sharpe"]
        b = r["B"]["sharpe"]
        c = r["C"]["sharpe"]
        print(f"  {r['name']:<14s} {r['years']:>4.0f}  {a:>+10.3f}  {b:>+10.3f}  {c:>+10.3f}")

    # 结论
    print(f"\n  {'─' * 68}")
    a_avg = np.mean([r["A"]["sharpe"] for r in results])
    b_avg = np.mean([r["B"]["sharpe"] for r in results])
    c_avg = np.mean([r["C"]["sharpe"] for r in results])
    print(f"  平均 Sharpe:  A={a_avg:+.3f}  B={b_avg:+.3f}  C={c_avg:+.3f}")

    c_dd = np.mean([r["C"]["max_dd"] for r in results])
    a_dd = np.mean([r["A"]["max_dd"] for r in results])
    print(f"  平均 MaxDD:   A={a_dd:.1%}  C={c_dd:.1%}")
    print(f"  {'─' * 68}")


if __name__ == "__main__":
    main()
