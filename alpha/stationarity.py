"""
alpha/stationarity.py — 平稳性选股

对一批标的做均值回归适合度评分，输出候选标的列表。

position 设计:
    alpha 是 cross-section 筛选（"哪只标的适合我的交易逻辑？"），
    不输出 num_units，不计算 PnL，只输出标的列表或排序。

    strategy 拿到这个列表后再做 time-series 决策（"这只标的现在该多少仓位？"），
    两者的边界是: alpha 输出 symbol list → strategy 输出 num_units。

    signals 是两者的公共语言——alpha 用 signals 做截面评分，
    strategy 用 signals 做时序决策。

Functions:
    score_stationarity(close) -> dict
        对单只标的的 close 序列做 ADF / Hurst / 半衰期评分。
        返回原始统计量的 dict + 合成分数 stationarity_score (0~1, 越高越适合 MR)。

    screen_stationarity(universe, top_n, min_hl, max_hl) -> list[str]
        对 universe 列表做批量评分，按 stationarity_score 排序，
        返回 top_n 只作为候选。
        支持快速模式 (skip_rolling=True，跳过滚动窗口分析)。

用法:
    from alpha.stationarity import screen_stationarity

    # 在端到端脚本中:
    candidates = screen_stationarity(["sh512670", "sh510300", "sh512760"], top_n=2)
    for sym in candidates:
        close = load_close(sym)
        result = linear_mr(close)
        bt = run_backtest(close, result["num_units"])
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保能 import 项目模块（无论从哪个目录运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.fetcher import read_day
from stats.univariate import estimate_half_life, hurst_exponent, run_adf

# ============================================================
# 配置
# ============================================================

# 滚动窗口参数（仅在 do_rolling=True 时使用）
ROLLING_WINDOW: int = 250
ROLLING_STEP: int = 60
MIN_FULL_SAMPLE: int = 500


# ============================================================
# 单标的评分
# ============================================================

def score_stationarity(
    close: pd.Series,
    *,
    do_rolling: bool = False,
) -> dict:
    """对单只标的的 close 序列做平稳性评分。

    Parameters:
        close: 价格序列（任意频率，建议日频）
        do_rolling: 是否做滚动窗口稳健性分析（较慢）

    Returns:
        dict with keys:
            n_obs: int              样本数
            adf_stat: float         ADF 统计量 (AIC)
            adf_pvalue: float       ADF p-value (越小越平稳)
            hurst: float            Hurst 指数 (<0.5 均值回归, >0.5 趋势)
            hurst_r2: float         Hurst 回归拟合度
            half_life: float        半衰期 (天)
            lambda_: float          均值回归速度 (负值=回归)
            is_mean_reverting: bool 是否均值回归 (lambda < 0)
            stationarity_score: float    合成分数 0~1 (越高越适合 MR)

            # 仅在 do_rolling=True 时:
            rolling_pass_rate: float     滚动窗口通过率
            rolling_hl_median: float      通过窗口的半衰期中位数
            rolling_hl_std: float         通过窗口的半衰期标准差
    """
    n_obs = len(close)

    # 统计检验
    adf = run_adf(close)
    hurst = hurst_exponent(close)
    hl = estimate_half_life(close, use_log=True)

    lam = hl["lambda"]
    h = hurst["hurst"]
    p_val = adf["p_value_aic"]
    half_life_val = hl["half_life"]

    # 合成评分: 三个维度加权平均
    #   ADF: p < 0.01 → 1.0,  p > 0.10 → 0.0,  中间线性
    #   Hurst: H < 0.3 → 1.0,  H > 0.6 → 0.0,  中间线性
    #   半衰期: 2 < HL < 60 → 1.0,  HL > 200 → 0.0,  中间线性
    score_adf = max(0.0, min(1.0, (0.10 - p_val) / (0.10 - 0.01)))
    score_hurst = max(0.0, min(1.0, (0.6 - h) / (0.6 - 0.3)))
    score_hl = max(0.0, min(1.0, (200.0 - half_life_val) / (200.0 - 2.0))) if lam < 0 else 0.0

    composite = (score_adf * 0.35 + score_hurst * 0.25 + score_hl * 0.40)

    result: dict = {
        "n_obs": n_obs,
        "adf_stat": adf["adf_stat_aic"],
        "adf_pvalue": p_val,
        "hurst": h,
        "hurst_r2": hurst["r_squared"],
        "half_life": half_life_val,
        "lambda_": lam,
        "is_mean_reverting": lam < 0,
        "stationarity_score": round(composite, 4),
    }

    if do_rolling:
        rolling = _rolling_analysis(close)
        result.update(rolling)

    return result


def _rolling_analysis(close: pd.Series) -> dict:
    """滚动窗口分析: ADF 通过率 + 半衰期稳定性。

    使用位置索引 (iloc), 250 天窗口, 60 天步进。
    """
    n = len(close)
    passing_hl: list[float] = []
    total_windows = 0

    start = 0
    while start + ROLLING_WINDOW <= n:
        window = close.iloc[start:start + ROLLING_WINDOW]
        if len(window) < 200:
            start += ROLLING_STEP
            continue
        total_windows += 1
        try:
            adf_r = run_adf(window)
            hl_r = estimate_half_life(window, use_log=True)
        except Exception:
            start += ROLLING_STEP
            continue

        if (adf_r["p_value_aic"] < 0.10) and (hl_r["lambda"] < 0) and (2 <= hl_r["half_life"] <= 60):
            passing_hl.append(hl_r["half_life"])
        start += ROLLING_STEP

    if total_windows == 0 or len(passing_hl) == 0:
        return {
            "rolling_pass_rate": 0.0,
            "rolling_hl_median": np.nan,
            "rolling_hl_std": np.nan,
        }

    hl_arr = np.array(passing_hl)
    return {
        "rolling_pass_rate": round(len(passing_hl) / total_windows, 4),
        "rolling_hl_median": round(float(np.median(hl_arr)), 1),
        "rolling_hl_std": round(float(np.std(hl_arr, ddof=1)), 1) if len(hl_arr) > 1 else 0.0,
    }


# ============================================================
# 批量筛选
# ============================================================

def screen_stationarity(
    universe: list[str],
    *,
    top_n: int = 5,
    min_hl: float = 2.0,
    max_hl: float = 60.0,
    skip_rolling: bool = True,
) -> list[dict]:
    """对一批标的做平稳性评分，返回按分数排序的候选列表。

    筛选条件:
        1. lamda < 0 (均值回归)
        2. min_hl <= half_life <= max_hl
        3. adf_pvalue < 0.10

    Parameters:
        universe: 标的代码列表 (如 ["sh512670", "sh510300", ...])
        top_n: 返回前 N 只候选
        min_hl / max_hl: 半衰期范围（天）
        skip_rolling: True 时跳滚动窗口（快模式，适用于大范围粗筛）

    Returns:
        list[dict], 每个 dict 包含 symbol + score_stationarity 的全部字段，
        按 stationarity_score 降序排列，最多 top_n 条。
    """
    results = []
    for symbol in universe:
        try:
            df = read_day(symbol)
        except FileNotFoundError:
            continue

        if len(df) < MIN_FULL_SAMPLE:
            continue

        score = score_stationarity(df["close"], do_rolling=not skip_rolling)

        # 候选判断
        if not score["is_mean_reverting"]:
            continue
        if not (min_hl <= score["half_life"] <= max_hl):
            continue
        if score["adf_pvalue"] >= 0.10:
            continue

        results.append({"symbol": symbol, **score})

    results.sort(key=lambda x: x["stationarity_score"], reverse=True)
    return results[:top_n]
