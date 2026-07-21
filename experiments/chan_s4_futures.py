"""
chan_s4_futures.py — Chan S4 原版策略: 商品期货测试
====================================================

严格还原 Ernest P. Chan《Algorithmic Trading》Ch.2 的 S4 策略:
  1. 半衰期回归: Δy = λ·y_lag + c + ε → HL = -ln2 / ln(1+λ)
  2. lookback = round(HL), 无自由参数
  3. Z = (y - MA(y, L)) / Std(y, L), 价格本身的滚动统计量
  4. numUnits = -Z, 连续仓位, 多空双向
  5. 无入场/出场阈值, 无 regime 过滤, 无卡尔曼

对照: 仅多版本 numUnits = max(0, -Z)

用法: python experiments/chan_s4_futures.py
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

# 交易成本 (期货: 无印花税, T+0)
COMMISSION_RATE = 0.00025   # 万2.5/边
SLIPPAGE_RATE = 0.0005      # 万5
COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE  # 单边总成本 0.075%

# 半衰期有效范围
HL_MIN = 5      # 太短 → 交易成本吞噬利润
HL_MAX = 250    # 太长 → 均值回归太慢, 无实际意义

# 仓位上限 (Chan 原书无上限; ±3 为实用约束, 防止极端 Z 导致过度杠杆)
MAX_UNITS = 3.0

# Walk-Forward
TRAIN_DAYS = 1260   # 5 年
TEST_DAYS = 252     # 1 年
STEP_DAYS = 252
WARMUP = 120


# ============================================================
# 数据读取
# ============================================================

def read_futures(filename):
    """读取通达信期货 L9 连续合约 .day 文件."""
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
# 半衰期估计 (Chan Ch.2 原法)
# ============================================================

def estimate_half_life(prices):
    """
    OLS 回归: Δy(t) = λ·y(t-1) + c + ε

    均衡点 y* = -c/λ, 偏离半衰期 HL = -ln2 / ln(1+λ).
    λ ≥ 0 表示非均值回归, 返回 HL = inf.

    返回: (half_life, lambda, r_squared)
    """
    y = np.asarray(prices, dtype=float)
    dy = np.diff(y)
    y_lag = y[:-1]

    X = np.column_stack([y_lag, np.ones(len(y_lag))])
    beta = np.linalg.lstsq(X, dy, rcond=None)[0]
    lam = beta[0]

    ss_res = np.sum((dy - X @ beta) ** 2)
    ss_tot = np.sum((dy - np.mean(dy)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if lam >= 0:
        return np.inf, lam, r2

    half_life = -np.log(2) / np.log(1.0 + lam)
    return half_life, lam, r2


# ============================================================
# Z-Score (Chan 原法: 价格的滚动均值和标准差)
# ============================================================

def compute_zscore(prices, lookback):
    """
    Z(t) = (y(t) - MA(y, L)) / Std(y, L)

    注意: Std 是价格本身的滚动标准差, 包含趋势分量。
    这是 Chan 原法, 与"残差标准差"不同。
    """
    T = len(prices)
    z = np.zeros(T)
    for t in range(lookback, T):
        window = prices[t - lookback + 1:t + 1]
        ma = np.mean(window)
        std = np.std(window, ddof=1)
        if std > 1e-12:
            z[t] = (prices[t] - ma) / std
    return z


# ============================================================
# 回测 (连续仓位 + 成本)
# ============================================================

def backtest_continuous(prices, num_units, warmup=0):
    """
    Chan 原法回测:
        ret(t) = numUnits(t-1) × asset_ret(t) - cost × |ΔnumUnits|

    numUnits 为连续值: 正=多, 负=空, |值|=杠杆倍数。
    成本在仓位变化时扣除 (每日再平衡)。
    """
    T = len(prices)
    asset_ret = np.zeros(T)
    asset_ret[1:] = np.diff(prices) / prices[:-1]

    strat_ret = np.zeros(T)
    for t in range(1, T):
        strat_ret[t] = num_units[t-1] * asset_ret[t]
        if t >= 2:
            strat_ret[t] -= COST_RATE * abs(num_units[t-1] - num_units[t-2])

    # ── 绩效 ──
    vr = strat_ret[warmup:]
    sharpe = 0.0
    if len(vr) > 10 and np.std(vr) > 1e-12:
        sharpe = float(np.mean(vr) / np.std(vr) * np.sqrt(252))

    equity = np.cumprod(1.0 + vr)
    rm = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - rm) / rm)) if len(equity) > 0 else 0.0
    total_ret = float(equity[-1] - 1.0) if len(equity) > 0 else 0.0

    # ── 仓位统计 ──
    active = num_units[warmup:]
    avg_abs_pos = float(np.mean(np.abs(active)))
    pct_long = float(np.mean(active > 0.01))
    pct_short = float(np.mean(active < -0.01))
    turnover = float(np.sum(np.abs(np.diff(active))))

    return {
        "sharpe": sharpe, "max_dd": max_dd, "total_ret": total_ret,
        "avg_abs_pos": avg_abs_pos, "pct_long": pct_long,
        "pct_short": pct_short, "turnover": turnover,
    }


# ============================================================
# 全样本测试
# ============================================================

def run_full_sample(code, name):
    """全样本: 估计半衰期 → Z-Score → 连续仓位回测."""
    dates, prices = read_futures(code)
    T = len(prices)

    hl, lam, r2 = estimate_half_life(prices)
    hl_valid = HL_MIN <= hl <= HL_MAX

    print(f"\n  {name}  ({T} days, {T/252:.0f}y)")
    print(f"  {'─' * 74}")
    print(f"  半衰期: {hl:.1f}d  lam={lam:.6f}  R2={r2:.4f}  "
          f"{'OK' if hl_valid else 'X 超出 [' + str(HL_MIN) + ',' + str(HL_MAX) + ']'}")

    if not hl_valid:
        print(f"  → 跳过 (半衰期不在有效范围)")
        return {"name": name, "hl": hl, "valid": False}

    lookback = int(round(hl))
    z = compute_zscore(prices, lookback)

    # 多空 (Chan 原版)
    units_ls = np.clip(-z, -MAX_UNITS, MAX_UNITS)
    r_ls = backtest_continuous(prices, units_ls, warmup=lookback)

    # 仅多
    units_lo = np.clip(np.maximum(0.0, -z), 0, MAX_UNITS)
    r_lo = backtest_continuous(prices, units_lo, warmup=lookback)

    # 买入持有
    bh_ret = np.diff(prices[WARMUP:]) / prices[WARMUP:-1]
    bh_sharpe = (float(np.mean(bh_ret) / np.std(bh_ret) * np.sqrt(252))
                 if np.std(bh_ret) > 1e-12 else 0.0)

    print(f"  lookback = {lookback}d")
    print(f"  {'─' * 74}")
    print(f"  {'模式':<14s} {'Sharpe':>8s} {'MaxDD':>8s} {'总收益':>9s} "
          f"{'平均|仓|':>8s} {'多%':>5s} {'空%':>5s} {'年换手':>8s}")
    print(f"  {'─' * 74}")
    yrs = (T - lookback) / 252
    print(f"  {'多空(原版)':<14s} {r_ls['sharpe']:>+8.3f} {r_ls['max_dd']:>7.1%} "
          f"{r_ls['total_ret']:>+8.1%} {r_ls['avg_abs_pos']:>8.2f} "
          f"{r_ls['pct_long']:>4.0%} {r_ls['pct_short']:>4.0%} "
          f"{r_ls['turnover']/yrs:>8.1f}")
    print(f"  {'仅多':<14s} {r_lo['sharpe']:>+8.3f} {r_lo['max_dd']:>7.1%} "
          f"{r_lo['total_ret']:>+8.1%} {r_lo['avg_abs_pos']:>8.2f} "
          f"{r_lo['pct_long']:>4.0%} {r_lo['pct_short']:>4.0%} "
          f"{r_lo['turnover']/yrs:>8.1f}")
    print(f"  {'买入持有':<14s} {bh_sharpe:>+8.3f}")

    return {"name": name, "hl": hl, "lookback": lookback, "valid": True,
            "ls": r_ls, "lo": r_lo, "bh": bh_sharpe}


# ============================================================
# Walk-Forward (半衰期在训练窗口估计, 无自由参数)
# ============================================================

def run_walkforward(code, name):
    """
    Walk-Forward: 训练窗口估计半衰期 → 测试窗口评估。
    无网格搜索, 无自由参数。半衰期完全由数据决定。
    """
    dates, prices = read_futures(code)
    T = len(prices)

    windows = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= T:
        windows.append((start - TRAIN_DAYS, start, start + TEST_DAYS))
        start += STEP_DAYS

    print(f"\n{'=' * 86}")
    print(f"  {name}  Walk-Forward  ({T} days, {len(windows)} windows)")
    print(f"  Train={TRAIN_DAYS}d ({TRAIN_DAYS//252}y)  Test={TEST_DAYS}d  "
          f"半衰期由训练窗口决定, 无自由参数")
    print(f"{'=' * 86}")

    hdr = (f"  {'Win':>3s}  {'Period':>23s}  "
           f"{'HL':>6s} {'LB':>4s}  "
           f"{'IS Shp':>7s} {'OOS Shp':>8s} {'OOS DD':>7s}  "
           f"{'OOS|仓|':>7s} {'OOS换手':>7s}")
    print(hdr)
    print(f"  {'─'*3}  {'─'*23}  {'─'*6} {'─'*4}  "
          f"{'─'*7} {'─'*8} {'─'*7}  {'─'*7} {'─'*7}")

    is_sharpes, oos_sharpes, oos_dds = [], [], []
    skipped = 0

    for wi, (tr_s, tr_e, te_e) in enumerate(windows):
        # 训练窗口估计半衰期
        hl, lam, r2 = estimate_half_life(prices[tr_s:tr_e])

        period = f"{dates[tr_s]}~{dates[tr_e-1]}|{dates[tr_e]}~{dates[te_e-1]}"

        if not (HL_MIN <= hl <= HL_MAX):
            skipped += 1
            print(f"  {wi+1:>3d}  {period:>23s}  {hl:>6.0f} {'--':>4s}  "
                  f"{'skip':>7s} {'skip':>8s} {'skip':>7s}  "
                  f"{'skip':>7s} {'skip':>7s}")
            continue

        lookback = int(round(hl))

        # IS
        z_is = compute_zscore(prices[tr_s:tr_e], lookback)
        units_is = np.clip(-z_is, -MAX_UNITS, MAX_UNITS)
        r_is = backtest_continuous(prices[tr_s:tr_e], units_is, warmup=lookback)

        # OOS (需要 lookback 天上下文)
        ctx_s = max(0, tr_e - lookback)
        z_oos = compute_zscore(prices[ctx_s:te_e], lookback)
        units_oos = np.clip(-z_oos, -MAX_UNITS, MAX_UNITS)

        offset = tr_e - ctx_s
        units_test = np.zeros(te_e - ctx_s)
        units_test[offset:] = units_oos[offset:]
        r_oos = backtest_continuous(prices[ctx_s:te_e], units_test, warmup=offset)

        is_sharpes.append(r_is["sharpe"])
        oos_sharpes.append(r_oos["sharpe"])
        oos_dds.append(r_oos["max_dd"])

        print(f"  {wi+1:>3d}  {period:>23s}  {hl:>6.1f} {lookback:>4d}  "
              f"{r_is['sharpe']:>+7.3f} {r_oos['sharpe']:>+8.3f} "
              f"{r_oos['max_dd']:>6.1%}  "
              f"{r_oos['avg_abs_pos']:>7.2f} {r_oos['turnover']:>7.1f}")

    # 汇总
    print(f"\n  {'─' * 80}")
    if oos_sharpes:
        is_avg = np.mean(is_sharpes)
        oos_avg = np.mean(oos_sharpes)
        oos_med = np.median(oos_sharpes)
        oos_dd_avg = np.mean(oos_dds)
        decay = (oos_avg - is_avg) / abs(is_avg) * 100 if abs(is_avg) > 0.01 else 0
        print(f"  有效窗口: {len(oos_sharpes)}/{len(windows)}  (跳过 {skipped})")
        print(f"  IS  平均 Sharpe:  {is_avg:+.3f}")
        print(f"  OOS 平均 Sharpe:  {oos_avg:+.3f}")
        print(f"  OOS 中位 Sharpe:  {oos_med:+.3f}")
        print(f"  Sharpe 衰减:      {decay:+.0f}%")
        print(f"  OOS 平均 MaxDD:   {oos_dd_avg:.1%}")
    else:
        print(f"  所有窗口半衰期超出范围, 无有效结果")
    print(f"  {'─' * 80}")

    return {"name": name, "is": is_sharpes, "oos": oos_sharpes,
            "oos_dd": oos_dds, "skipped": skipped}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 86)
    print("  Chan S4 原版策略 — 商品期货测试")
    print("  半衰期 lookback · 价格 Std · 连续仓位 numUnits = -Z")
    print("  无入场阈值 · 无 regime 过滤 · 无卡尔曼")
    print(f"  成本: 佣金{COMMISSION_RATE*10000:.1f}万/边 + "
          f"滑点{SLIPPAGE_RATE*10000:.0f}万  仓位上限 ±{MAX_UNITS:.0f}")
    print("=" * 86)

    # ── Part 1: 全样本 ──
    print(f"\n{'=' * 86}")
    print("  Part 1: 全样本")
    print(f"{'=' * 86}")

    full_results = []
    for code, name in FUTURES:
        full_results.append(run_full_sample(code, name))

    print(f"\n{'=' * 86}")
    print("  全样本汇总")
    print(f"{'=' * 86}")
    print(f"  {'品种':<12s} {'HL':>6s} {'LB':>4s} "
          f"{'多空Shp':>8s} {'仅多Shp':>8s} {'B&H':>7s} "
          f"{'多空DD':>8s} {'仅多DD':>8s}")
    print(f"  {'─'*12} {'─'*6} {'─'*4} "
          f"{'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

    for r in full_results:
        if not r.get("valid"):
            print(f"  {r['name']:<12s} {r['hl']:>6.0f} {'--':>4s} "
                  f"{'N/A':>8s} {'N/A':>8s} {'N/A':>7s} "
                  f"{'N/A':>8s} {'N/A':>8s}")
        else:
            print(f"  {r['name']:<12s} {r['hl']:>6.1f} {r['lookback']:>4d} "
                  f"{r['ls']['sharpe']:>+8.3f} {r['lo']['sharpe']:>+8.3f} "
                  f"{r['bh']:>+7.3f} "
                  f"{r['ls']['max_dd']:>7.1%} {r['lo']['max_dd']:>7.1%}")

    # ── Part 2: Walk-Forward ──
    print(f"\n{'=' * 86}")
    print("  Part 2: Walk-Forward (半衰期由训练窗口决定)")
    print(f"{'=' * 86}")

    wf_results = []
    for code, name in FUTURES:
        wf_results.append(run_walkforward(code, name))

    print(f"\n{'=' * 86}")
    print("  Walk-Forward 最终汇总")
    print(f"{'=' * 86}")
    print(f"  {'品种':<12s} {'窗口':>6s} {'IS均值':>8s} {'OOS均值':>8s} {'OOS中位':>8s}")
    print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    all_oos = []
    for r in wf_results:
        n_valid = len(r["oos"])
        n_total = n_valid + r["skipped"]
        if n_valid > 0:
            all_oos.extend(r["oos"])
            print(f"  {r['name']:<12s} {n_valid:>2d}/{n_total:<2d}  "
                  f"{np.mean(r['is']):>+8.3f} {np.mean(r['oos']):>+8.3f} "
                  f"{np.median(r['oos']):>+8.3f}")
        else:
            print(f"  {r['name']:<12s} {0:>2d}/{n_total:<2d}  "
                  f"{'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")

    if all_oos:
        print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
        print(f"  {'全品种':<12s} {len(all_oos):>6d}  "
              f"{'':>8s} {np.mean(all_oos):>+8.3f} {np.median(all_oos):>+8.3f}")


if __name__ == "__main__":
    main()
