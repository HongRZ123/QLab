"""
trend_mr_portfolio.py — 趋势+均值回归 复合策略 (组合版)
========================================================

三项改进:
  1. 多品种等权组合 (分散风险, 提高统计功效)
  2. 趋势稳定性过滤 (β 波动过大时不交易, 排除趋势转向期)
  3. 仓位分级 (偏离越深仓位越大, 入场时确定, 持仓期不变)

固定参数: window=20, entry_z=2.0, Kalman Q=0.01, beta_ratio=1.5

用法: python experiments/trend_mr_portfolio.py
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

# 趋势层 (Kalman Linear Trend)
Q_RATIO = 0.01
Q_SLOPE_RATIO = 0.001
R_INIT_WINDOW = 60
WARMUP = 120

# 均值回归层
SHORT_WINDOW = 20
ENTRY_Z = 2.0
EXIT_Z = 0.0
MAX_WEIGHT = 0.95

# 趋势过滤
BETA_STD_WINDOW = 20
BETA_STD_RATIO = 1.5

# 成本
COMMISSION_RATE = 0.00025
SLIPPAGE_RATE = 0.0005
COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE

# Walk-Forward
TRAIN_DAYS = 1260
TEST_DAYS = 252
STEP_DAYS = 252


# ============================================================
# 数据
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


def load_all():
    """加载全部品种, 预计算 Kalman, 返回字典."""
    data = {}
    for code, name in FUTURES:
        dates, prices = read_futures(code)
        log_prices = np.log(prices)
        mu, beta = kalman_linear_trend(log_prices)
        data[name] = {"dates": dates, "prices": prices,
                      "log_prices": log_prices, "mu": mu, "beta": beta}
    return data


def overlap_range(data):
    """找到所有品种的重叠日期范围, 返回各品种的 [start_idx, end_idx)."""
    latest_start = max(d["dates"][0] for d in data.values())
    earliest_end = min(d["dates"][-1] for d in data.values())
    ranges = {}
    for name, d in data.items():
        s = int(np.searchsorted(d["dates"], latest_start))
        e = int(np.searchsorted(d["dates"], earliest_end)) + 1
        ranges[name] = (s, e)
    return ranges, latest_start, earliest_end


# ============================================================
# 趋势层: Kalman Linear Trend
# ============================================================

def kalman_linear_trend(log_prices):
    T = len(log_prices)
    init_n = min(R_INIT_WINDOW, T // 2)
    R = max(np.var(np.diff(log_prices[:init_n + 1])) / 2.0, 1e-10)

    Q = np.array([[Q_RATIO * R, 0.0], [0.0, Q_SLOPE_RATIO * R]])
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
# 趋势稳定性过滤
# ============================================================

def trend_stability(beta, window=BETA_STD_WINDOW, ratio=BETA_STD_RATIO):
    """
    beta 近期标准差 < ratio * 扩展中位数 → 趋势稳定。
    扩展中位数保证因果性。
    """
    T = len(beta)
    beta_std = np.full(T, np.nan)
    for t in range(window, T):
        beta_std[t] = np.std(beta[t - window + 1:t + 1], ddof=1)

    stable = np.ones(T, dtype=bool)
    min_hist = window * 2

    for t in range(min_hist, T):
        hist = beta_std[window:t]
        hist = hist[~np.isnan(hist)]
        if len(hist) > 10:
            med = np.median(hist)
            if med > 1e-12:
                stable[t] = beta_std[t] < ratio * med

    return stable


# ============================================================
# 信号 + 仓位分级
# ============================================================

def generate_weights(log_prices, mu, beta, use_filter=True):
    """
    入场: Z < -ENTRY_Z 且趋势稳定
    仓位: min(1, |Z| / (2*ENTRY_Z)) * MAX_WEIGHT  (入场时锁定)
    出场: Z > EXIT_Z 或趋势失稳
    """
    T = len(log_prices)
    residual = log_prices - mu

    sigma = np.full(T, np.nan)
    for t in range(SHORT_WINDOW, T):
        sigma[t] = np.std(residual[t - SHORT_WINDOW + 1:t + 1], ddof=1)

    z = np.zeros(T)
    valid = ~np.isnan(sigma) & (sigma > 1e-12)
    z[valid] = residual[valid] / sigma[valid]

    stable = trend_stability(beta) if use_filter else np.ones(T, dtype=bool)

    weights = np.zeros(T)
    in_pos = False
    entry_w = 0.0
    start = max(WARMUP, SHORT_WINDOW)

    for t in range(start, T):
        if not in_pos:
            if z[t] < -ENTRY_Z and stable[t]:
                in_pos = True
                entry_w = min(1.0, abs(z[t]) / (2.0 * ENTRY_Z)) * MAX_WEIGHT
                weights[t] = entry_w
        else:
            if z[t] > EXIT_Z or not stable[t]:
                in_pos = False
            else:
                weights[t] = entry_w

    return weights


# ============================================================
# 回测 (收益率法, 支持连续仓位)
# ============================================================

def strategy_returns(prices, weights, warmup=0):
    """ret(t) = w(t-1)*asset_ret(t) - cost*|dw|. 返回逐日策略收益数组."""
    T = len(prices)
    asset_ret = np.zeros(T)
    asset_ret[1:] = np.diff(prices) / prices[:-1]

    sr = np.zeros(T)
    for t in range(1, T):
        sr[t] = weights[t - 1] * asset_ret[t]
        if t >= 2:
            sr[t] -= COST_RATE * abs(weights[t - 1] - weights[t - 2])
    return sr


def metrics(sr, warmup=0):
    vr = sr[warmup:]
    sharpe = 0.0
    if len(vr) > 10 and np.std(vr) > 1e-12:
        sharpe = float(np.mean(vr) / np.std(vr) * np.sqrt(252))
    eq = np.cumprod(1.0 + vr)
    rm = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - rm) / rm)) if len(eq) > 0 else 0.0
    total_ret = float(eq[-1] - 1.0) if len(eq) > 0 else 0.0
    return {"sharpe": sharpe, "max_dd": max_dd, "total_ret": total_ret}


def count_trades(weights, warmup=0):
    """入场次数 (权重从 0 跳到 >0)."""
    w = weights[warmup:]
    n = 0
    for t in range(1, len(w)):
        if w[t] > 0.01 and w[t - 1] <= 0.01:
            n += 1
    return n


# ============================================================
# 全样本
# ============================================================

def run_full_sample(data):
    print(f"\n{'=' * 86}")
    print("  Part 1: 全样本")
    print(f"{'=' * 86}")

    ind_results = {}

    for name, d in data.items():
        lp, p, mu, beta = d["log_prices"], d["prices"], d["mu"], d["beta"]
        T = len(p)

        # 改进版
        w_new = generate_weights(lp, mu, beta, use_filter=True)
        sr_new = strategy_returns(p, w_new)
        m_new = metrics(sr_new, WARMUP)
        n_new = count_trades(w_new, WARMUP)

        # 基线 (无过滤, 二值仓位)
        w_base = generate_weights(lp, mu, beta, use_filter=False)
        w_bin = (w_base > 0.01).astype(float) * MAX_WEIGHT
        sr_base = strategy_returns(p, w_bin)
        m_base = metrics(sr_base, WARMUP)
        n_base = count_trades(w_bin, WARMUP)

        stable_pct = np.mean(trend_stability(beta)[WARMUP:]) * 100
        avg_w = np.mean(w_new[WARMUP:][w_new[WARMUP:] > 0.01]) if n_new > 0 else 0

        ind_results[name] = {"sr": sr_new, "m": m_new}

        print(f"\n  {name}  ({T}d, {T/252:.0f}y)")
        print(f"  {'─' * 76}")
        print(f"  {'版本':<22s} {'Sharpe':>8s} {'MaxDD':>8s} "
              f"{'总收益':>9s} {'交易':>5s} {'均仓':>6s}")
        print(f"  {'─' * 76}")
        print(f"  {'改进(过滤+分级)':<22s} {m_new['sharpe']:>+8.3f} "
              f"{m_new['max_dd']:>7.1%} {m_new['total_ret']:>+8.1%} "
              f"{n_new:>5d} {avg_w:>5.0%}")
        print(f"  {'基线(无过滤,二值)':<22s} {m_base['sharpe']:>+8.3f} "
              f"{m_base['max_dd']:>7.1%} {m_base['total_ret']:>+8.1%} "
              f"{n_base:>5d} {MAX_WEIGHT:>5.0%}")
        print(f"  趋势稳定期: {stable_pct:.0f}%")

    # ── 组合 (重叠期) ──
    ranges, d0, d1 = overlap_range(data)
    print(f"\n{'=' * 86}")
    print(f"  等权组合 (重叠期 {d0}~{d1})")
    print(f"{'=' * 86}")

    port_rets = []
    for name, d in data.items():
        s, e = ranges[name]
        lp = d["log_prices"][s:e]
        p = d["prices"][s:e]
        mu = d["mu"][s:e]
        beta = d["beta"][s:e]
        w = generate_weights(lp, mu, beta, use_filter=True)
        sr = strategy_returns(p, w)
        port_rets.append(sr)

    min_len = min(len(r) for r in port_rets)
    stacked = np.column_stack([r[:min_len] for r in port_rets])
    port_sr = np.mean(stacked, axis=1)
    m_port = metrics(port_sr, WARMUP)

    ind_avg = np.mean([ind_results[n]["m"]["sharpe"] for n in ind_results])
    ind_dds = [ind_results[n]["m"]["max_dd"] for n in ind_results]

    print(f"  单品种平均 Sharpe:  {ind_avg:+.3f}")
    print(f"  单品种最大 MaxDD:   {min(ind_dds):.1%}")
    print(f"  组合 Sharpe:        {m_port['sharpe']:+.3f}")
    print(f"  组合 MaxDD:         {m_port['max_dd']:.1%}")
    print(f"  组合总收益:         {m_port['total_ret']:+.1%}")

    return port_sr


# ============================================================
# Walk-Forward
# ============================================================

def run_walkforward(data):
    ranges, d0, d1 = overlap_range(data)
    min_T = min(e - s for s, e in ranges.values())

    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= min_T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    print(f"\n{'=' * 86}")
    print(f"  Part 2: Walk-Forward  ({len(windows)} windows)")
    print(f"  重叠期 {min_T}d, Train={TRAIN_DAYS}d Test={TEST_DAYS}d")
    print(f"{'=' * 86}")

    hdr = (f"  {'Win':>3s}  {'Period':>23s}  "
           f"{'组合IS':>7s} {'组合OOS':>8s} {'OOS DD':>7s}  {'#Tr':>4s}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*23}  {'─'*7} {'─'*8} {'─'*7}  {'─'*4}")

    is_list, oos_list = [], []
    ref_dates = data[FUTURES[0][1]]["dates"]
    ref_s = ranges[FUTURES[0][1]][0]

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        is_rets, oos_rets = [], []
        oos_trades = 0

        for name, d in data.items():
            s, e = ranges[name]
            lp = d["log_prices"][s:e]
            p = d["prices"][s:e]
            mu = d["mu"][s:e]
            beta = d["beta"][s:e]

            # IS
            w_is = generate_weights(lp[tr_s:tr_e], mu[tr_s:tr_e],
                                    beta[tr_s:tr_e])
            sr_is = strategy_returns(p[tr_s:tr_e], w_is)
            is_rets.append(sr_is)

            # OOS
            ctx = max(0, tr_e - WARMUP)
            w_oos = generate_weights(lp[ctx:te_e], mu[ctx:te_e],
                                     beta[ctx:te_e])
            off = tr_e - ctx
            w_test = np.zeros(te_e - ctx)
            w_test[off:] = w_oos[off:]
            sr_oos = strategy_returns(p[ctx:te_e], w_test)
            oos_rets.append(sr_oos)
            oos_trades += count_trades(w_test, off)

        min_is = min(len(r) for r in is_rets)
        min_oos = min(len(r) for r in oos_rets)
        port_is = np.mean(np.column_stack([r[:min_is] for r in is_rets]), axis=1)
        port_oos = np.mean(np.column_stack([r[:min_oos] for r in oos_rets]), axis=1)

        m_is = metrics(port_is, WARMUP)
        m_oos = metrics(port_oos, off)
        is_list.append(m_is["sharpe"])
        oos_list.append(m_oos["sharpe"])

        gi_s = ref_s + tr_s
        gi_e = ref_s + tr_e
        gi_te = ref_s + te_e
        period = f"{ref_dates[gi_s]}~{ref_dates[gi_e-1]}|{ref_dates[gi_e]}~{ref_dates[gi_te-1]}"
        print(f"  {wi+1:>3d}  {period:>23s}  "
              f"{m_is['sharpe']:>+7.3f} {m_oos['sharpe']:>+8.3f} "
              f"{m_oos['max_dd']:>6.1%}  {oos_trades:>4d}")

    print(f"\n  {'─' * 74}")
    is_avg = np.mean(is_list)
    oos_avg = np.mean(oos_list)
    oos_med = np.median(oos_list)
    oos_pos = sum(1 for s in oos_list if s > 0)
    print(f"  组合 IS  平均:  {is_avg:+.3f}")
    print(f"  组合 OOS 平均:  {oos_avg:+.3f}")
    print(f"  组合 OOS 中位:  {oos_med:+.3f}")
    print(f"  正 Sharpe 窗口: {oos_pos}/{len(oos_list)}")
    print(f"  {'─' * 74}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 86)
    print("  趋势+均值回归 复合策略 (组合版)")
    print("  1. 多品种等权组合  2. 趋势稳定性过滤  3. 仓位分级")
    print(f"  固定: window={SHORT_WINDOW} entry_z={ENTRY_Z} "
          f"Q={Q_RATIO} beta_ratio={BETA_STD_RATIO}")
    print("=" * 86)

    data = load_all()
    run_full_sample(data)
    run_walkforward(data)


if __name__ == "__main__":
    main()
