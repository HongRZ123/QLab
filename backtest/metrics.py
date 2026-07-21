"""
metrics.py — 绩效指标计算模块
================================

提供回测绩效评估的纯函数集合，全部接收 pd.Series 日收益率序列。

核心函数:
    annualized_return   — 年化收益率 (APR)
    sharpe_ratio        — 夏普比率
    max_drawdown        — 最大回撤
    win_rate            — 胜率
    trade_count         — 交易次数
    avg_holding_days    — 平均持仓天数
    performance_summary  — 汇总全部指标

验证协议:
    - 已知收益率序列 → APR/Sharpe/MaxDD 与手算一致 (误差 < 1e-6)
    - 全零收益率 → APR=0, Sharpe=0, MaxDD=0
    - 全正收益率 → MaxDD=0, win_rate=1.0
    - 空 Series → 返回 NaN (不报错)

用法:
    python backtest/metrics.py
"""

import numpy as np
import pandas as pd

# ============================================================
# 年化收益率
# ============================================================

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算几何年化收益率 (APR)。

    使用累计净值的几何平均, 不假设收益率独立同分布:
        total_return = Π(1 + r_i)
        APR = total_return ^ (periods_per_year / n) - 1

    参数:
        returns:          日收益率序列 (pd.Series, 小数形式如 0.01 = 1%)
        periods_per_year: 年化周期数, 默认 252 (交易日)

    返回:
        几何年化收益率 (float)

    示例:
        >>> annualized_return(pd.Series([0.01, -0.005, 0.003]))
        0.xxx
    """
    if len(returns) == 0:
        return np.nan
    n = len(returns)
    total_return = float(np.prod((1 + returns).to_numpy(dtype=float)))
    if total_return <= 0:
        return -1.0
    return float(total_return ** (periods_per_year / n) - 1)


# ============================================================
# 夏普比率
# ============================================================

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252,
                 rf: float = 0.0) -> float:
    """
    计算年化夏普比率。

    参数:
        returns:          日收益率序列
        periods_per_year: 年化周期数, 默认 252
        rf:               年化无风险利率, 默认 0

    返回:
        年化夏普比率 (float), std=0 时返回 0

    示例:
        >>> sharpe_ratio(pd.Series([0.01, -0.005, 0.008]))
        x.xxx
    """
    if len(returns) == 0:
        return np.nan
    std = returns.std()
    if std == 0:
        return 0.0
    excess = returns.mean() - rf / periods_per_year
    return float(excess / std * np.sqrt(periods_per_year))


# ============================================================
# 最大回撤
# ============================================================

def max_drawdown(returns: pd.Series) -> float:
    """
    从累计净值曲线计算最大回撤。

    参数:
        returns: 日收益率序列

    返回:
        最大回撤幅度 (float, ≥0), 无回撤时返回 0

    示例:
        >>> max_drawdown(pd.Series([0.01, -0.005, 0.008]))
        0.0049505...
    """
    if len(returns) == 0:
        return np.nan
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(-dd.min())


# ============================================================
# 胜率
# ============================================================

def win_rate(returns: pd.Series) -> float:
    """
    计算正收益天数占比 (胜率)。

    参数:
        returns: 日收益率序列

    返回:
        胜率 (float, 0~1), 正收益天数 / 总天数

    示例:
        >>> win_rate(pd.Series([0.01, -0.005, 0.008, -0.002]))
        0.5
    """
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).mean())


# ============================================================
# 交易次数
# ============================================================

def trade_count(signals: pd.Series) -> int:
    """
    计算信号变化次数，每次 0→1 或 1→0 算半次交易 (一次完整往返 = 1 次交易)。

    参数:
        signals: 交易信号序列 (0/1, pd.Series)

    返回:
        交易次数 (int), 无变化时返回 0

    示例:
        >>> trade_count(pd.Series([0, 1, 0, 1, 0]))
        2
        >>> trade_count(pd.Series([0, 0, 0, 0]))
        0
    """
    if len(signals) == 0:
        return 0
    return int(np.ceil(signals.diff().abs().sum() / 2))


# ============================================================
# 平均持仓天数
# ============================================================

def avg_holding_days(signals: pd.Series) -> float:
    """
    计算平均持仓天数。

    参数:
        signals: 交易信号序列 (0/1, pd.Series)

    返回:
        平均持仓天数 (float), 持仓总天数 / 交易次数, 无交易时返回 0

    示例:
        >>> sig = pd.Series([0, 1, 1, 1, 0])
        >>> avg_holding_days(sig)
        3.0
    """
    if len(signals) == 0:
        return np.nan
    total_holding = signals.sum()
    trades = trade_count(signals)
    if trades == 0:
        return 0.0
    return float(total_holding / trades)


# ============================================================
# 绩效汇总
# ============================================================

def performance_summary(returns: pd.Series, signals: pd.Series | None = None) -> dict:
    """
    汇总计算所有绩效指标，返回完整字典。

    参数:
        returns: 日收益率序列
        signals: 交易信号序列 (可选, 0/1), 不传则 trade_count/avg_holding 为 NaN

    返回:
        dict: {
            "apr":         年化收益率,
            "sharpe":      夏普比率,
            "maxdd":       最大回撤,
            "win_rate":    胜率,
            "trade_count": 交易次数,
            "avg_holding": 平均持仓天数,
            "n_days":      总交易日数,
        }

    示例:
        >>> r = pd.Series([0.01, -0.005, 0.008, 0.003, -0.002])
        >>> s = pd.Series([0, 1, 1, 0, 0])
        >>> performance_summary(r, s)
        {'apr': ..., 'sharpe': ..., ...}
    """
    summary = {
        "apr": annualized_return(returns),
        "sharpe": sharpe_ratio(returns),
        "maxdd": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "trade_count": trade_count(signals) if signals is not None else np.nan,
        "avg_holding": avg_holding_days(signals) if signals is not None else np.nan,
        "n_days": len(returns),
    }
    return summary


# ============================================================
# 验证协议
# ============================================================

def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    """误差比较 (处理 NaN 情况)。"""
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) < tol


def run_validation() -> bool:
    """
    验证协议: 构造已知收益率序列, 断言指标与手算一致。
    """
    print("=" * 60)
    print("绩效指标验证协议")
    print("=" * 60)

    all_pass = True

    # --------------------------------------------------------
    # 测试 1: 已知收益率序列
    # --------------------------------------------------------
    print("\n【正控】已知收益率序列 [0.01, -0.005, 0.008, 0.003, -0.002]")
    print("-" * 60)

    r = pd.Series([0.01, -0.005, 0.008, 0.003, -0.002])
    s = pd.Series([0, 1, 1, 0, 0])  # 买一次, 持 2 天

    mean_r = r.mean()
    std_r = r.std(ddof=1)

    # 手算 APR (几何年化)
    total_r = float(np.prod((1 + r).to_numpy(dtype=float)))
    expected_apr = total_r ** (252 / len(r)) - 1
    apr = annualized_return(r)
    pass_apr = _is_close(apr, expected_apr)
    if not pass_apr:
        all_pass = False
    print(f"  APR:    actual={apr:.6f}  expected={expected_apr:.6f}  "
          f"[{'PASS' if pass_apr else 'FAIL'}]")

    # 手算 Sharpe
    expected_sharpe = (mean_r / std_r) * np.sqrt(252) if std_r != 0 else 0.0
    sr = sharpe_ratio(r)
    pass_sr = _is_close(sr, expected_sharpe)
    if not pass_sr:
        all_pass = False
    print(f"  Sharpe: actual={sr:.6f}  expected={expected_sharpe:.6f}  "
          f"[{'PASS' if pass_sr else 'FAIL'}]")

    # 手算 MaxDD
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd_series = (cum - peak) / peak
    expected_mdd = float(-dd_series.min())
    mdd = max_drawdown(r)
    pass_mdd = _is_close(mdd, expected_mdd)
    if not pass_mdd:
        all_pass = False
    print(f"  MaxDD:  actual={mdd:.6f}  expected={expected_mdd:.6f}  "
          f"[{'PASS' if pass_mdd else 'FAIL'}]")

    # 手算 Win Rate
    expected_win = (r > 0).mean()
    wr = win_rate(r)
    pass_wr = _is_close(wr, expected_win)
    if not pass_wr:
        all_pass = False
    print(f"  WinRate: actual={wr:.4f}  expected={expected_win:.4f}  "
          f"[{'PASS' if pass_wr else 'FAIL'}]")

    # 手算 Trade Count — signals [0,1,1,0,0]
    # diff: [NaN, 1, 0, -1, 0], abs sum = 2, /2 = 1
    expected_tc = int(s.diff().abs().sum() / 2)
    tc = trade_count(s)
    pass_tc = tc == expected_tc
    if not pass_tc:
        all_pass = False
    print(f"  Trades: actual={tc}  expected={expected_tc}  "
          f"[{'PASS' if pass_tc else 'FAIL'}]")

    # 手算 Avg Holding — 持仓 2 天 / 1 次交易 = 2
    expected_ah = float(s.sum() / max(expected_tc, 1))
    ah = avg_holding_days(s)
    pass_ah = _is_close(ah, expected_ah)
    if not pass_ah:
        all_pass = False
    print(f"  AvgHolding: actual={ah:.2f}  expected={expected_ah:.2f}  "
          f"[{'PASS' if pass_ah else 'FAIL'}]")

    # --------------------------------------------------------
    # 测试 2: 全零收益率
    # --------------------------------------------------------
    print("\n【边界】全零收益率 [0, 0, 0, 0, 0]")
    print("-" * 60)

    rz = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

    apr_z = annualized_return(rz)
    sr_z = sharpe_ratio(rz)
    mdd_z = max_drawdown(rz)
    wr_z = win_rate(rz)

    pass_zero = (
        _is_close(apr_z, 0.0) and _is_close(sr_z, 0.0) and _is_close(mdd_z, 0.0)
    )
    if not pass_zero:
        all_pass = False
    print(f"  APR=0:     {apr_z:.6f}  [{'PASS' if _is_close(apr_z, 0.0) else 'FAIL'}]")
    print(f"  Sharpe=0:  {sr_z:.6f}  [{'PASS' if _is_close(sr_z, 0.0) else 'FAIL'}]")
    print(f"  MaxDD=0:   {mdd_z:.6f}  [{'PASS' if _is_close(mdd_z, 0.0) else 'FAIL'}]")
    print(f"  WinRate:   {wr_z:.4f}")

    # --------------------------------------------------------
    # 测试 3: 全正收益率
    # --------------------------------------------------------
    print("\n【边界】全正收益率 [0.01, 0.02, 0.015, 0.025, 0.005]")
    print("-" * 60)

    rp = pd.Series([0.01, 0.02, 0.015, 0.025, 0.005])
    mdd_p = max_drawdown(rp)
    wr_p = win_rate(rp)

    pass_positive = _is_close(mdd_p, 0.0) and _is_close(wr_p, 1.0)
    if not pass_positive:
        all_pass = False
    print(f"  MaxDD=0:   {mdd_p:.6f}  [{'PASS' if _is_close(mdd_p, 0.0) else 'FAIL'}]")
    print(f"  WinRate=1: {wr_p:.4f}  [{'PASS' if _is_close(wr_p, 1.0) else 'FAIL'}]")

    # --------------------------------------------------------
    # 测试 4: 空 Series
    # --------------------------------------------------------
    print("\n【边界】空 Series")
    print("-" * 60)

    re = pd.Series([], dtype=float)

    pass_empty = (
        np.isnan(annualized_return(re))
        and np.isnan(sharpe_ratio(re))
        and np.isnan(max_drawdown(re))
        and np.isnan(win_rate(re))
    )
    if not pass_empty:
        all_pass = False
    print(f"  空Series → NaN  [{'PASS' if pass_empty else 'FAIL'}]")

    # --------------------------------------------------------
    # 测试 5: performance_summary 端到端
    # --------------------------------------------------------
    print("\n【集成】performance_summary 端到端")
    print("-" * 60)

    summary = performance_summary(r, s)
    expected_keys = {"apr", "sharpe", "maxdd", "win_rate", "trade_count", "avg_holding", "n_days"}
    pass_summary = set(summary.keys()) == expected_keys
    if not pass_summary:
        all_pass = False
    print(f"  Keys: {sorted(summary.keys())}")
    print(f"  [{'PASS' if pass_summary else 'FAIL'}]")

    # 无 signals 时 trade_count/avg_holding 为 NaN
    summary_no_sig = performance_summary(r)
    pass_nosig = np.isnan(summary_no_sig["trade_count"]) and np.isnan(summary_no_sig["avg_holding"])
    if not pass_nosig:
        all_pass = False
    print(f"  No signals → trade_count/avg_holding=NaN  "
          f"[{'PASS' if pass_nosig else 'FAIL'}]")

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 全部验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    success = run_validation()
    if not success:
        raise SystemExit(1)
