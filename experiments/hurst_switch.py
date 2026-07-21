"""
hurst_switch.py — Hurst 指数切换/混合: 均值回归 <-> 动量
====================================================

架构:
  滚动 Hurst 指数 (方差-时间图法) 判断市场状态:
    H < 0.5 → 均值回归模式 (Kalman + Z-Score, 仅做多)
    H ≥ 0.5 → 动量模式 (N日收益符号, 多空)

  三种组合方式:
    A) 二值切换: H_ma < threshold → MR, 否则 → 动量
    B) Sigmoid 加权混合: w_mr = σ(k×(thr - H_ma)), 连续权重
    C) 纯动量 / 纯MR 基线

  均值回归层: 已验证基线 (window=20, entry_z=2.0)
  动量层:     Chapter 6 时间序列动量 (sign of N-day return)

测试:
  动量回望 ∈ {120, 250}
  Hurst MA窗口 ∈ {10, 20, 40, 60}
  阈值 ∈ {0.45, 0.50, 0.55}
  Sigmoid 陡峭度 ∈ {10, 20}
  5 品种 × 全样本 + Walk-Forward

用法: python experiments/hurst_switch.py
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
Q_RATIO = 0.01
Q_SLOPE_RATIO = 0.001
R_INIT_WINDOW = 60
WARMUP = 120

# ── 均值回归层 (已验证基线) ──
MR_WINDOW = 20
MR_ENTRY_Z = 2.0
MR_EXIT_Z = 0.0

# ── 动量层 (测试多组) ──
MOM_LOOKBACKS = [120, 250]

# ── Hurst MA 切换/混合 (网格搜索) ──
HURST_WINDOWS = [252]                    # Hurst 估计窗口
HURST_MA_WINDOWS = [10, 20, 40, 60]     # Hurst MA 平滑窗口
HURST_THRESHOLDS = [0.45, 0.50, 0.55]   # 切换阈值
BLEND_STEEPNESS = [10, 20]              # Sigmoid 陡峭度 k

# ── 固定权重组合 (无 Hurst) ──
FIXED_WEIGHTS = [0.2, 0.3, 0.5]  # MR 权重, 动量权重 = 1 - w_mr

# ── 交易成本 (期货) ──
COMMISSION_RATE = 0.00025
SLIPPAGE_RATE = 0.0005
COST_PER_TURN = COMMISSION_RATE + SLIPPAGE_RATE

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
    """状态 x = [mu, beta]^T, 跟踪水平 + 斜率。"""
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
# Hurst 指数: 方差-时间图法
# ============================================================

def rolling_hurst(log_prices, window=252):
    """
    方差-时间图法估计滚动 Hurst 指数。
    Var(z(t+τ) - z(t)) ~ τ^(2H)
    对 log(Var) vs log(τ) 线性回归, 斜率/2 = H。
    τ 取 2 的幂次: 2, 4, 8, ..., window//4。
    """
    T = len(log_prices)
    max_tau = window // 4
    taus = []
    tau = 2
    while tau <= max_tau:
        taus.append(tau)
        tau *= 2
    log_taus = np.log(np.array(taus, dtype=float))

    H = np.full(T, np.nan)
    for t in range(window, T):
        z = log_prices[t - window:t]
        log_vars = np.empty(len(taus))
        for i, tau in enumerate(taus):
            diffs = z[tau:] - z[:-tau]
            v = np.var(diffs)
            log_vars[i] = np.log(v) if v > 1e-15 else np.nan

        if np.any(np.isnan(log_vars)):
            continue
        slope = np.polyfit(log_taus, log_vars, 1)[0]
        H[t] = slope / 2.0

    return H


def smooth_hurst(H, ma_window=20):
    """Hurst 移动均线平滑, 减少边界抖动。"""
    T = len(H)
    H_ma = np.full(T, np.nan)
    for t in range(T):
        # 向前取 ma_window 个有效值
        valid = H[max(0, t - ma_window + 1):t + 1]
        valid = valid[~np.isnan(valid)]
        if len(valid) >= ma_window // 2:
            H_ma[t] = np.mean(valid)
    return H_ma


# ============================================================
# 均值回归层: 短期 Z-Score (仅做多)
# ============================================================

def mr_signals(log_prices, mu, short_window=MR_WINDOW,
               entry_z=MR_ENTRY_Z, exit_z=MR_EXIT_Z):
    """
    Z = (log_price - mu) / rolling_std(residual, N)
    Z < -entry_z → 买入 (价格低于趋势, 等回归)
    Z > exit_z   → 卖出 (回归趋势线)
    仅做多。
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
# 动量层: N日收益符号 (多空)
# ============================================================

def momentum_signals(log_prices, lookback):
    """
    时间序列动量 (Chapter 6):
      N日对数收益 > 0 → 做多 (+1)
      N日对数收益 < 0 → 做空 (-1)
    """
    T = len(log_prices)
    position = np.zeros(T, dtype=int)
    for t in range(lookback, T):
        ret = log_prices[t] - log_prices[t - lookback]
        if ret > 0:
            position[t] = 1
        elif ret < 0:
            position[t] = -1
    return position


# ============================================================
# Hurst 切换
# ============================================================

def hurst_switch(log_prices, mu, H, mom_lookback,
                 ma_window=20, threshold=0.5):
    """
    Hurst MA 切换:
      H_ma < threshold → MR 模式 (仅做多)
      H_ma ≥ threshold → 动量模式 (多空)
    """
    T = len(log_prices)
    H_ma = smooth_hurst(H, ma_window)

    mr_pos = mr_signals(log_prices, mu)
    mom_pos = momentum_signals(log_prices, mom_lookback)

    position = np.zeros(T, dtype=int)
    for t in range(T):
        if np.isnan(H_ma[t]):
            continue
        if H_ma[t] < threshold:
            position[t] = mr_pos[t]
        else:
            position[t] = mom_pos[t]

    return position, H_ma


# ============================================================
# Hurst Sigmoid 加权混合
# ============================================================

def hurst_blend(log_prices, mu, H, mom_lookback,
                ma_window=20, threshold=0.5, steepness=10.0):
    """
    Sigmoid 加权混合 (连续权重, 非二值切换):
      w_mr = 1 / (1 + exp(k × (H_ma - threshold)))
      position = w_mr × MR仓位 + (1 - w_mr) × 动量仓位
    H_ma 远低于 threshold → 近似纯 MR
    H_ma 远高于 threshold → 近似纯动量
    H_ma ≈ threshold → 各半混合, 平滑过渡
    """
    T = len(log_prices)
    H_ma = smooth_hurst(H, ma_window)

    mr_pos = mr_signals(log_prices, mu)
    mom_pos = momentum_signals(log_prices, mom_lookback)

    position = np.zeros(T)
    for t in range(T):
        if np.isnan(H_ma[t]):
            continue
        w_mr = 1.0 / (1.0 + np.exp(steepness * (H_ma[t] - threshold)))
        position[t] = w_mr * mr_pos[t] + (1.0 - w_mr) * mom_pos[t]

    return position, H_ma


# ============================================================
# 回测 (收益率法, 支持多空)
# ============================================================

def backtest(prices, position, warmup=0):
    """
    收益率回测:
      daily_ret = position[t-1] * (price[t]/price[t-1] - 1)
      换手成本 = |Δposition| * (commission + slippage)
    """
    T = len(prices)
    price_ret = np.zeros(T)
    price_ret[1:] = prices[1:] / prices[:-1] - 1.0

    strat_ret = np.zeros(T)
    strat_ret[1:] = position[:-1] * price_ret[1:]

    # 换手成本
    turnover = np.abs(np.diff(position, prepend=0)).astype(float)
    strat_ret -= turnover * COST_PER_TURN

    # 指标
    vr = strat_ret[warmup:]
    sharpe = 0.0
    if len(vr) > 10 and np.std(vr) > 1e-12:
        sharpe = float(np.mean(vr) / np.std(vr) * np.sqrt(252))

    cum = np.cumprod(1.0 + vr)
    total_ret = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0
    rm = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - rm) / rm)) if len(cum) > 0 else 0.0

    # 交易统计 (仓位变化 = 交易)
    trades = np.where(turnover[warmup:] > 0)[0]
    n_trades = len(trades)

    # 胜率: 每段持仓的收益
    trade_rets = []
    pos = position[warmup:]
    rets = strat_ret[warmup:]
    in_trade = False
    cum_r = 0.0
    for i in range(len(pos)):
        if pos[i] != 0 and not in_trade:
            in_trade = True
            cum_r = 0.0
        if in_trade:
            cum_r += rets[i]
        if in_trade and (i == len(pos) - 1 or pos[i + 1] == 0
                         or np.sign(pos[i + 1]) != np.sign(pos[i])):
            trade_rets.append(cum_r)
            in_trade = False

    win_rate = sum(1 for r in trade_rets if r > 0) / max(len(trade_rets), 1)

    return {"sharpe": sharpe, "max_dd": max_dd, "total_ret": total_ret,
            "n_trades": n_trades, "win_rate": win_rate}


# ============================================================
# 全样本: Hurst MA 网格搜索
# ============================================================

def run_full_sample(code, name):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    mu, beta = kalman_linear_trend(log_prices)

    # 预计算 Hurst (每个估计窗口只算一次)
    H_cache = {}
    for hw in HURST_WINDOWS:
        H_cache[hw] = rolling_hurst(log_prices, hw)

    print(f"\n  {name}  ({T}d, {T/252:.0f}y)")
    print(f"  {'─' * 95}")
    print(f"  {'策略':<36s} {'Sharpe':>8s} {'MaxDD':>8s} "
          f"{'总收益':>9s} {'交易':>5s} {'MR%':>5s}")
    print(f"  {'─' * 95}")

    results = {}

    # ── 基线: 纯 MR ──
    mr_pos = mr_signals(log_prices, mu)
    r = backtest(prices, mr_pos, warmup=WARMUP)
    results["mr_only"] = r
    print(f"  {'纯MR (基线)':<36s} {r['sharpe']:>+8.3f} {r['max_dd']:>7.1%} "
          f"{r['total_ret']:>+8.1%} {r['n_trades']:>5d}")

    # ── 纯动量 ──
    for lb in MOM_LOOKBACKS:
        mom_pos = momentum_signals(log_prices, lb)
        r = backtest(prices, mom_pos, warmup=lb)
        results[f"mom_{lb}"] = r
        print(f"  {'纯动量 lb=' + str(lb):<36s} {r['sharpe']:>+8.3f} "
              f"{r['max_dd']:>7.1%} {r['total_ret']:>+8.1%} "
              f"{r['n_trades']:>5d}")

    # ── 固定权重组合 (无 Hurst) ──
    for lb in MOM_LOOKBACKS:
        mom_pos = momentum_signals(log_prices, lb)
        for w_mr in FIXED_WEIGHTS:
            pos = w_mr * mr_pos + (1.0 - w_mr) * mom_pos
            warmup = max(WARMUP, lb)
            r = backtest(prices, pos, warmup=warmup)
            tag = f"Fixed w{w_mr:.1f} Mom{lb}"
            results[tag] = {**r, "mode": "fixed", "w_mr": w_mr, "lb": lb}
            print(f"  {tag:<36s} {r['sharpe']:>+8.3f} "
                  f"{r['max_dd']:>7.1%} {r['total_ret']:>+8.1%} "
                  f"{r['n_trades']:>5d}")

    # ── Hurst MA 切换网格 ──
    for hw in HURST_WINDOWS:
        H = H_cache[hw]
        for lb in MOM_LOOKBACKS:
            for mw in HURST_MA_WINDOWS:
                for thr in HURST_THRESHOLDS:
                    pos, H_ma = hurst_switch(log_prices, mu, H, lb,
                                             ma_window=mw, threshold=thr)
                    warmup = max(WARMUP, hw, lb)
                    r = backtest(prices, pos, warmup=warmup)

                    valid_H = H_ma[~np.isnan(H_ma)]
                    pct_mr = np.mean(valid_H < thr) * 100

                    tag = f"MA{mw} thr{thr:.2f} Mom{lb}"
                    results[tag] = {**r, "pct_mr": pct_mr, "hw": hw,
                                    "mw": mw, "thr": thr, "lb": lb}
                    print(f"  {tag:<36s} {r['sharpe']:>+8.3f} "
                          f"{r['max_dd']:>7.1%} {r['total_ret']:>+8.1%} "
                          f"{r['n_trades']:>5d} {pct_mr:>4.0f}%")

    # ── Hurst Sigmoid 加权混合网格 ──
    for hw in HURST_WINDOWS:
        H = H_cache[hw]
        for lb in MOM_LOOKBACKS:
            for mw in HURST_MA_WINDOWS:
                for thr in HURST_THRESHOLDS:
                    for k in BLEND_STEEPNESS:
                        pos, H_ma = hurst_blend(log_prices, mu, H, lb,
                                                ma_window=mw, threshold=thr,
                                                steepness=k)
                        warmup = max(WARMUP, hw, lb)
                        r = backtest(prices, pos, warmup=warmup)

                        valid_H = H_ma[~np.isnan(H_ma)]
                        pct_mr = np.mean(valid_H < thr) * 100

                        tag = f"Blend MA{mw} t{thr:.2f} k{k} M{lb}"
                        results[tag] = {**r, "pct_mr": pct_mr, "hw": hw,
                                        "mw": mw, "thr": thr, "lb": lb,
                                        "k": k, "mode": "blend"}
                        print(f"  {tag:<36s} {r['sharpe']:>+8.3f} "
                              f"{r['max_dd']:>7.1%} {r['total_ret']:>+8.1%} "
                              f"{r['n_trades']:>5d} {pct_mr:>4.0f}%")

    print(f"  {'─' * 95}")
    return {"name": name, "results": results, "prices": prices,
            "log_prices": log_prices, "mu": mu, "dates": dates,
            "H_cache": H_cache}


# ============================================================
# Walk-Forward: Hurst MA 切换
# ============================================================

def run_walkforward(code, name, hw, mw, thr, lb,
                    mode="switch", steepness=10.0):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    mu, beta = kalman_linear_trend(log_prices)
    H = rolling_hurst(log_prices, hw)

    warmup = max(WARMUP, hw, lb)
    pos_fn = hurst_blend if mode == "blend" else hurst_switch
    pos_kw = dict(ma_window=mw, threshold=thr)
    if mode == "blend":
        pos_kw["steepness"] = steepness

    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    mode_str = "Sigmoid混合" if mode == "blend" else "二值切换"
    tag = (f"Blend MA{mw} t{thr:.2f} k{steepness:.0f} M{lb}"
           if mode == "blend" else f"MA{mw} thr{thr:.2f} Mom{lb}")
    print(f"\n{'=' * 82}")
    print(f"  {name}  Walk-Forward  ({len(windows)} windows)")
    print(f"  {tag}  H_win={hw}  MR: w={MR_WINDOW} z={MR_ENTRY_Z}  [{mode_str}]")
    print(f"{'=' * 82}")

    hdr = (f"  {'Win':>3s}  {'Period':>23s}  "
           f"{'IS Shp':>7s} {'OOS Shp':>8s} {'OOS DD':>7s}  "
           f"{'#Tr':>4s} {'MR%':>5s}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*23}  {'─'*7} {'─'*8} {'─'*7}  {'─'*4} {'─'*5}")

    is_list, oos_list = [], []

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        lp_is = log_prices[tr_s:tr_e]
        p_is = prices[tr_s:tr_e]
        mu_is = mu[tr_s:tr_e]
        H_is = H[tr_s:tr_e]
        pos_is, _ = pos_fn(lp_is, mu_is, H_is, lb, **pos_kw)
        r_is = backtest(p_is, pos_is, warmup=warmup)

        ctx_s = max(0, tr_e - warmup)
        lp_oos = log_prices[ctx_s:te_e]
        p_oos = prices[ctx_s:te_e]
        mu_oos = mu[ctx_s:te_e]
        H_oos = H[ctx_s:te_e]
        pos_oos, H_ma = pos_fn(lp_oos, mu_oos, H_oos, lb, **pos_kw)
        offset = tr_e - ctx_s
        pos_test = np.zeros(len(p_oos), dtype=int)
        pos_test[offset:] = pos_oos[offset:]
        r_oos = backtest(p_oos, pos_test, warmup=offset)

        valid_H = H_ma[offset:]
        valid_H = valid_H[~np.isnan(valid_H)]
        pct_mr = np.mean(valid_H < thr) * 100 if len(valid_H) > 0 else 0

        is_list.append(r_is["sharpe"])
        oos_list.append(r_oos["sharpe"])

        period = f"{dates[tr_s]}~{dates[tr_e-1]}|{dates[tr_e]}~{dates[te_e-1]}"
        print(f"  {wi+1:>3d}  {period:>23s}  "
              f"{r_is['sharpe']:>+7.3f} {r_oos['sharpe']:>+8.3f} "
              f"{r_oos['max_dd']:>6.1%}  "
              f"{r_oos['n_trades']:>4d} {pct_mr:>4.0f}%")

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
# Walk-Forward: 纯动量
# ============================================================

def run_walkforward_momentum(code, name, lookback):
    dates, prices = read_futures(code)
    log_prices = np.log(prices)
    T = len(prices)

    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    print(f"\n{'=' * 82}")
    print(f"  {name}  Walk-Forward  ({len(windows)} windows)")
    print(f"  纯动量 lb={lookback}")
    print(f"{'=' * 82}")

    hdr = (f"  {'Win':>3s}  {'Period':>23s}  "
           f"{'IS Shp':>7s} {'OOS Shp':>8s} {'OOS DD':>7s}  {'#Tr':>4s}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*23}  {'─'*7} {'─'*8} {'─'*7}  {'─'*4}")

    is_list, oos_list = [], []

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        lp_is = log_prices[tr_s:tr_e]
        p_is = prices[tr_s:tr_e]
        pos_is = momentum_signals(lp_is, lookback)
        r_is = backtest(p_is, pos_is, warmup=lookback)

        ctx_s = max(0, tr_e - lookback)
        lp_oos = log_prices[ctx_s:te_e]
        p_oos = prices[ctx_s:te_e]
        pos_oos = momentum_signals(lp_oos, lookback)
        offset = tr_e - ctx_s
        pos_test = np.zeros(len(p_oos), dtype=int)
        pos_test[offset:] = pos_oos[offset:]
        r_oos = backtest(p_oos, pos_test, warmup=offset)

        is_list.append(r_is["sharpe"])
        oos_list.append(r_oos["sharpe"])

        period = f"{dates[tr_s]}~{dates[tr_e-1]}|{dates[tr_e]}~{dates[te_e-1]}"
        print(f"  {wi+1:>3d}  {period:>23s}  "
              f"{r_is['sharpe']:>+7.3f} {r_oos['sharpe']:>+8.3f} "
              f"{r_oos['max_dd']:>6.1%}  {r_oos['n_trades']:>4d}")

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
# Walk-Forward 汇总
# ============================================================

def print_wf_summary(wf_all):
    print(f"\n{'=' * 82}")
    print("  最终汇总")
    print(f"{'=' * 82}")
    print(f"  {'品种':<12s} {'IS均值':>8s} {'OOS均值':>8s} "
          f"{'OOS中位':>8s} {'正窗口':>6s}")
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


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 82)
    print("  Hurst MA 切换/混合: 均值回归 <-> 动量")
    print("  A) 二值切换: H_ma < thr → MR, 否则 → 动量")
    print("  B) Sigmoid混合: w_mr = σ(k×(thr - H_ma)), 连续权重")
    print(f"  MR: window={MR_WINDOW} entry_z={MR_ENTRY_Z}")
    print(f"  动量回望: {MOM_LOOKBACKS}")
    print(f"  Hurst MA窗口: {HURST_MA_WINDOWS}  阈值: {HURST_THRESHOLDS}")
    print(f"  Sigmoid陡峭度: {BLEND_STEEPNESS}")
    print("=" * 82)

    # ── Part 1: 全样本网格搜索 ──
    print(f"\n{'=' * 82}")
    print("  Part 1: 全样本")
    print(f"{'=' * 82}")

    all_results = []
    for code, name in FUTURES:
        all_results.append(run_full_sample(code, name))

    # ── 跨品种汇总 ──
    print(f"\n{'=' * 82}")
    print("  跨品种汇总 (平均 Sharpe)")
    print(f"{'=' * 82}")

    tags = set()
    for r in all_results:
        tags.update(r["results"].keys())

    def tag_sort_key(t):
        if t == "mr_only":
            return (0, 0, 0, 0, 0)
        if t.startswith("mom_"):
            return (1, int(t.split("_")[1]), 0, 0, 0)
        if t.startswith("Fixed "):
            # Fixed w0.3 Mom250
            parts = t.split()
            w_mr = float(parts[1].replace("w", ""))
            lb = int(parts[2].replace("Mom", ""))
            return (2, w_mr, lb, 0, 0)
        if t.startswith("Blend "):
            # Blend MA20 t0.50 k10 M250
            parts = t.split()
            mw = int(parts[1].replace("MA", ""))
            thr = float(parts[2].replace("t", ""))
            k = int(parts[3].replace("k", ""))
            lb = int(parts[4].replace("M", ""))
            return (4, mw, thr, k, lb)
        # MA20 thr0.50 Mom250
        parts = t.split()
        mw = int(parts[0].replace("MA", ""))
        thr = float(parts[1].replace("thr", ""))
        lb = int(parts[2].replace("Mom", ""))
        return (3, mw, thr, lb, 0)

    tags = sorted(tags, key=tag_sort_key)

    print(f"  {'策略':<36s}", end="")
    for r in all_results:
        print(f" {r['name']:>10s}", end="")
    print(f"  {'平均':>8s}")
    print(f"  {'─'*36} " + "─"*10 * len(all_results) + "  ────────")

    best_tag = None
    best_avg = -999

    for tag in tags:
        sharpes = []
        print(f"  {tag:<36s}", end="")
        for r in all_results:
            if tag in r["results"]:
                s = r["results"][tag]["sharpe"]
                sharpes.append(s)
                print(f" {s:>+10.3f}", end="")
            else:
                print(f" {'—':>10s}", end="")
        avg = np.mean(sharpes) if sharpes else 0
        print(f"  {avg:>+8.3f}")
        if avg > best_avg:
            best_avg = avg
            best_tag = tag

    print(f"\n  全局最佳: {best_tag}  平均Sharpe={best_avg:+.3f}")

    # ── Part 2: Walk-Forward ──
    if best_tag == "mr_only":
        print("\n  最佳为纯MR, 无需Walk-Forward")
        return

    if best_tag.startswith("mom_"):
        lb = int(best_tag.split("_")[1])
        print(f"\n{'=' * 82}")
        print(f"  Part 2: Walk-Forward (纯动量 lb={lb})")
        print(f"{'=' * 82}")
        wf_all = []
        for code, name in FUTURES:
            wf_all.append(run_walkforward_momentum(code, name, lb))
        print_wf_summary(wf_all)
        return

    # 解析标签 → Walk-Forward
    hw = HURST_WINDOWS[0]
    if best_tag.startswith("Blend "):
        parts = best_tag.split()
        mw = int(parts[1].replace("MA", ""))
        thr = float(parts[2].replace("t", ""))
        k = float(parts[3].replace("k", ""))
        lb = int(parts[4].replace("M", ""))
        mode, steepness = "blend", k
    else:
        parts = best_tag.split()
        mw = int(parts[0].replace("MA", ""))
        thr = float(parts[1].replace("thr", ""))
        lb = int(parts[2].replace("Mom", ""))
        mode, steepness = "switch", 10.0

    print(f"\n{'=' * 82}")
    print(f"  Part 2: Walk-Forward ({best_tag})")
    print(f"{'=' * 82}")

    wf_all = []
    for code, name in FUTURES:
        wf_all.append(run_walkforward(code, name, hw, mw, thr, lb,
                                      mode=mode, steepness=steepness))
    print_wf_summary(wf_all)


if __name__ == "__main__":
    main()
