"""
walk_forward.py — Walk-Forward 滚动参数估计框架
================================================

消除参数级前视偏差: 所有策略参数 (半衰期, 特征向量, entry_z)
仅使用历史数据估计, 每 reest_interval 天重估一次。

核心函数:
    walk_forward_linear_mr   — S4 滚动半衰期 → 滚动 lookback
    walk_forward_bollinger   — S8 滚动半衰期 + 训练期 entry_z 选择
    walk_forward_portfolio   — S7 滚动 Johansen → 滚动特征向量 + lookback

设计:
    - min_warmup:     最小预热期 (天), 之前不交易
    - reest_interval: 重估间隔 (天), 默认 63 (季度)
    - 每个窗口: 用 [0, t) 估计参数, 在 [t, t+interval) 应用
    - Z-Score 使用扩展段 [t-lb+1, end) 避免边界 NaN

验证协议:
    - OU 过程: lookback 随时间变化 (非常数)
    - min_warmup 之前 num_units 全为 0
    - 截断一致性: 无未来数据泄露

用法:
    python -m backtest.walk_forward
"""

import numpy as np
import pandas as pd

from signals.stats import estimate_half_life
from signals.stats_cointegration import johansen_test
from strategies.MR.s4_linear import linear_mr
from strategies.MR.s7_linear_portfolio import linear_portfolio
from strategies.MR.s8_bollinger import bollinger_mr

# ============================================================
# 内部辅助
# ============================================================

def _sharpe(ret: pd.Series) -> float:
    """年化 Sharpe (rf=0, 252 交易日)。"""
    r = ret.dropna()
    if len(r) < 10 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _estimate_lookback(close: pd.Series, use_log: bool = True) -> int | None:
    """
    从历史数据估计 lookback。

    返回 None 表示不满足均值回归条件 (不交易)。
    """
    hl_res = estimate_half_life(close, use_log=use_log)
    hl = hl_res["half_life"]
    lam = hl_res["lambda"]
    if lam < 0 and 2 <= hl <= 252:
        return round(hl)
    return None


# ============================================================
# S4: 滚动半衰期 → 滚动 lookback
# ============================================================

def walk_forward_linear_mr(
    close: pd.Series,
    reest_interval: int = 63,
    min_warmup: int = 252,
) -> dict:
    """
    Walk-Forward S4 线性均值回归。

    每 reest_interval 天用历史数据重估半衰期 → lookback,
    在下一个窗口应用 S4 策略。

    参数:
        close:          收盘价序列
        reest_interval: 重估间隔 (天), 默认 63
        min_warmup:     最小预热期 (天), 默认 252

    返回:
        dict: {
            num_units:    pd.Series — 全序列仓位 (预热期为 0)
            lookback_log: list[dict] — 每次重估的记录
        }

    示例:
        >>> prices = pd.Series(np.exp(np.cumsum(np.random.randn(500) * 0.01)))
        >>> wf = walk_forward_linear_mr(prices, reest_interval=63, min_warmup=252)
        >>> wf["num_units"].iloc[:252].sum()
        0.0
    """
    n = len(close)
    num_units = pd.Series(0.0, index=close.index, name="num_units")
    lookback_log: list[dict] = []

    for t in range(min_warmup, n, reest_interval):
        lb = _estimate_lookback(close.iloc[:t], use_log=True)
        end = min(t + reest_interval, n)

        if lb is None:
            lookback_log.append({
                "date": close.index[t], "lookback": None,
                "half_life": estimate_half_life(close.iloc[:t])["half_life"],
            })
            continue

        start = max(0, t - lb + 1)
        segment = close.iloc[start:end]
        s4 = linear_mr(segment, lookback=lb)

        offset = t - start
        num_units.iloc[t:end] = s4["num_units"].values[offset:]

        lookback_log.append({
            "date": close.index[t], "lookback": lb,
            "half_life": estimate_half_life(close.iloc[:t])["half_life"],
        })

    return {"num_units": num_units, "lookback_log": lookback_log}


# ============================================================
# S8: 滚动半衰期 + 训练期 entry_z 选择
# ============================================================

def walk_forward_bollinger(
    close: pd.Series,
    entry_z_candidates: list[float] | None = None,
    reest_interval: int = 63,
    min_warmup: int = 252,
) -> dict:
    """
    Walk-Forward S8 布林带均值回归。

    每 reest_interval 天:
        1. 用 [0, t) 估计半衰期 → lookback
        2. 用 [0, t) 评估各 entry_z 的 Sharpe → 选最佳
        3. 在 [t, t+interval) 应用

    参数:
        close:              收盘价序列
        entry_z_candidates: entry_z 候选列表, 默认 [1.0, 1.5, 2.0]
        reest_interval:     重估间隔 (天)
        min_warmup:         最小预热期 (天)

    返回:
        dict: {
            num_units: pd.Series — 全序列仓位
            param_log: list[dict] — 每次重估的参数记录
        }

    示例:
        >>> prices = pd.Series(np.exp(np.cumsum(np.random.randn(500) * 0.01)))
        >>> wf = walk_forward_bollinger(prices, [1.0, 1.5, 2.0])
        >>> wf["num_units"].iloc[:252].sum()
        0.0
    """
    if entry_z_candidates is None:
        entry_z_candidates = [1.0, 1.5, 2.0]

    n = len(close)
    num_units = pd.Series(0.0, index=close.index, name="num_units")
    param_log: list[dict] = []

    for t in range(min_warmup, n, reest_interval):
        train = close.iloc[:t]
        lb = _estimate_lookback(train, use_log=True)
        end = min(t + reest_interval, n)

        if lb is None:
            param_log.append({
                "date": close.index[t], "lookback": None,
                "entry_z": None, "train_sharpe": None,
            })
            continue

        best_ez = entry_z_candidates[0]
        best_sr = -np.inf
        for ez in entry_z_candidates:
            s8_train = bollinger_mr(train, lookback=lb, entry_z=ez, exit_z=0.0)
            sr = _sharpe(s8_train["ret"])
            if sr > best_sr:
                best_sr = sr
                best_ez = ez

        start = max(0, t - lb + 1)
        segment = close.iloc[start:end]
        s8 = bollinger_mr(segment, lookback=lb, entry_z=best_ez, exit_z=0.0)

        offset = t - start
        num_units.iloc[t:end] = s8["num_units"].values[offset:]

        param_log.append({
            "date": close.index[t], "lookback": lb,
            "entry_z": best_ez, "train_sharpe": best_sr,
        })

    return {"num_units": num_units, "param_log": param_log}


# ============================================================
# S7: 滚动 Johansen → 滚动特征向量 + lookback
# ============================================================

def walk_forward_portfolio(
    prices_df: pd.DataFrame,
    reest_interval: int = 63,
    min_warmup: int = 252,
    lag: int = 1,
) -> dict:
    """
    Walk-Forward S7 组合线性均值回归。

    每 reest_interval 天:
        1. 用 [0, t) 的对数价格做 Johansen 检验
        2. 取第一特征向量 v₁ 和组合半衰期 → lookback
        3. 在 [t, t+interval) 应用 S7

    参数:
        prices_df:      价格矩阵 (T × n)
        reest_interval: 重估间隔 (天)
        min_warmup:     最小预热期 (天)
        lag:            Johansen 差分滞后阶数

    返回:
        dict: {
            num_units: pd.Series — 全序列仓位
            ret:       pd.Series — 策略收益率 (多资产, 不经过单资产引擎)
            param_log: list[dict] — 每次重估的参数记录
        }

    示例:
        >>> prices = pd.DataFrame(np.exp(np.cumsum(np.random.randn(500, 2) * 0.01, axis=0)))
        >>> wf = walk_forward_portfolio(prices)
        >>> wf["num_units"].iloc[:252].sum()
        0.0
    """
    n = len(prices_df)
    num_units = pd.Series(0.0, index=prices_df.index, name="num_units")
    ret = pd.Series(0.0, index=prices_df.index, name="ret")
    param_log: list[dict] = []

    for t in range(min_warmup, n, reest_interval):
        train = prices_df.iloc[:t]
        log_train = np.log(train)

        try:
            jres = johansen_test(log_train, lag=lag)
        except Exception:
            param_log.append({
                "date": prices_df.index[t], "lookback": None,
                "half_life": None, "eigenvector": None, "rank": None,
            })
            continue

        if not jres["is_cointegrated"] or jres["rank"] < 1:
            param_log.append({
                "date": prices_df.index[t], "lookback": None,
                "half_life": jres["half_life"], "eigenvector": None,
                "rank": jres["rank"],
            })
            continue

        v1 = jres["eigenvectors"][:, 0]
        hl = jres["half_life"]
        lb = round(hl) if np.isfinite(hl) and 2 <= hl <= 252 else 20

        end = min(t + reest_interval, n)
        start = max(0, t - lb + 1)
        segment = prices_df.iloc[start:end]

        s7 = linear_portfolio(segment, v1, lookback=lb)

        offset = t - start
        num_units.iloc[t:end] = s7["num_units"].values[offset:]
        ret.iloc[t:end] = s7["ret"].values[offset:]

        param_log.append({
            "date": prices_df.index[t], "lookback": lb,
            "half_life": hl, "eigenvector": v1.copy(),
            "rank": jres["rank"],
        })

    return {"num_units": num_units, "ret": ret, "param_log": param_log}


# ============================================================
# 验证协议
# ============================================================

def run_validation() -> bool:
    """Walk-Forward 框架验证协议。"""
    from signals.stats import generate_ou_paths

    print("=" * 60)
    print("Walk-Forward 框架验证协议")
    print("=" * 60)

    all_pass = True
    T = 600
    SEED = 42

    # ── 测试 1: OU 过程 → 预热期无交易, 应用期有交易 ──
    print("\n【正控】OU(θ=0.05, T=600) → 预热期无交易, 应用期有交易")
    print("-" * 60)

    ou_raw = generate_ou_paths(1, T, theta=0.05, mu=0.0, sigma=1.0, dt=1.0, seed=SEED)
    ou_prices = pd.Series(np.exp(ou_raw[0]), name="ou_price")

    wf = walk_forward_linear_mr(ou_prices, reest_interval=63, min_warmup=252)

    warmup_ok = bool((wf["num_units"].iloc[:252] == 0).all())
    if not warmup_ok:
        all_pass = False
    print(f"  预热期 (0-251) num_units 全为 0: {warmup_ok}  [{'PASS' if warmup_ok else 'FAIL'}]")

    n_reest = len(wf["lookback_log"])
    lookbacks = [x["lookback"] for x in wf["lookback_log"] if x["lookback"] is not None]
    if n_reest == 0:
        all_pass = False
    print(f"  重估次数: {n_reest}")
    print(f"  lookback 值: {lookbacks}")
    print(f"  [{'PASS' if n_reest > 0 else 'FAIL'}] 重估执行")

    has_trades = bool((wf["num_units"].iloc[252:] > 0).any())
    if not has_trades:
        all_pass = False
    print(f"  应用期有交易信号: {has_trades}  [{'PASS' if has_trades else 'FAIL'}]")

    # ── 测试 2: 截断一致性 (无未来数据泄露) ──
    print("\n【断言】截断一致性: 前 400 天结果与完整数据一致")
    print("-" * 60)

    wf_trunc = walk_forward_linear_mr(
        ou_prices.iloc[:400], reest_interval=63, min_warmup=252,
    )
    match = wf["num_units"].iloc[:400].equals(wf_trunc["num_units"])
    if not match:
        all_pass = False
    print(f"  截断一致性: {match}  [{'PASS' if match else 'FAIL'}]")

    # ── 测试 3: S8 walk-forward ──
    print("\n【集成】S8 walk-forward (entry_z 自适应选择)")
    print("-" * 60)

    wf_s8 = walk_forward_bollinger(
        ou_prices, [1.0, 1.5, 2.0], reest_interval=63, min_warmup=252,
    )
    s8_warmup_ok = bool((wf_s8["num_units"].iloc[:252] == 0).all())
    s8_has_trades = bool((wf_s8["num_units"].iloc[252:] > 0).any())
    s8_ok = s8_warmup_ok and s8_has_trades
    if not s8_ok:
        all_pass = False

    selected_ez = [x["entry_z"] for x in wf_s8["param_log"] if x["entry_z"] is not None]
    print(f"  预热期无交易: {s8_warmup_ok}")
    print(f"  应用期有交易: {s8_has_trades}")
    print(f"  选择的 entry_z: {selected_ez}")
    print(f"  [{'PASS' if s8_ok else 'FAIL'}]")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] Walk-Forward 验证通过")
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
