"""
s1_adf.py — ADF 单位根检验 (Augmented Dickey-Fuller)
=====================================================

Step 1 of the mean-reversion screening pipeline:
    检验价格序列是否具有单位根 (非平稳)。

统计假设:
    H0: 存在单位根 → 序列非平稳 (价格走势持续, 不均值回归)
    H1: 不存在单位根 → 序列平稳 (围绕均值波动)

    若 p-value < 0.05, 拒绝 H0 → 平稳
    若 p-value > 0.05, 不能拒绝 H0 → 非平稳 (对照组预期结果)

两种信息准则自动选择最优滞后阶数:
    AIC (Akaike)  — 倾向更多滞后, 降低遗漏偏误
    BIC (Schwarz) — 倾向更少滞后, 降低过度拟合

输出:
    ADF 统计量、p-value、最优滞后阶数、临界值, 以及 ADF 回归中
    y(t-1) 的系数 λ_ADF (用于后续 half-life 计算参考)。

用法:
    from tests.s1_adf import run_adf

    result = run_adf(prices)
    print(result["p_value_aic"])

依赖:
    statsmodels >= 0.14.6
"""

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ============================================================
# 核心函数
# ============================================================

def run_adf(prices: pd.Series) -> dict:
    """
    对价格序列执行 ADF 单位根检验, 分别以 AIC / BIC 选择最优滞后阶数。

    参数:
        prices: 收盘价序列 (pd.Series, index=日期, values=价格)
                建议取 log 或直接用, adfuller 对线性变换不变

    返回:
        dict, 包含以下键:
        ─────────────────────────────────────────────
        adf_stat_aic      : ADF 统计量 (AIC 选阶)
        p_value_aic       : ADF p-value (AIC 选阶)
        adf_stat_bic      : ADF 统计量 (BIC 选阶)
        p_value_bic       : ADF p-value (BIC 选阶)
        lambda_adf        : ADF 回归中 y(t-1) 的系数 γ
                            (Δy_t = γ·y(t-1) + ... 中的 γ)
        used_lag_aic      : AIC 选择的最优滞后阶数
        used_lag_bic      : BIC 选择的最优滞后阶数
        n_obs             : 实际使用的样本数
        critical_1pct     : 1% 临界值
        critical_5pct     : 5% 临界值
        critical_10pct    : 10% 临界值
        ─────────────────────────────────────────────
    """
    # --- 输入验证 ---
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices 必须是 pd.Series, 实际为 {type(prices)}")
    if len(prices) < 7:
        raise ValueError(f"prices 长度至少为 7, 实际为 {len(prices)}")
    if prices.isna().any() or np.isinf(prices.values).any():
        raise ValueError("prices 包含 NaN 或 Inf, 请先清洗数据")

    # --- 参数统一 ---
    REG = "c"          # 仅截距, 不加趋势 (Chan: 日频 βt ≈ 0)
    STORE = True       # 返回完整回归结果, 以便提取 λ
    REGRESULTS = True  # 存储完整回归对象

    # --- AIC 选阶 ---
    # statsmodels 0.14+ 返回 4 元组: (adfstat, pvalue, critical, resstore)
    # usedlag 和 nobs 存储在 resstore 对象中
    result_aic: Any = adfuller(
        prices,
        maxlag=None,          # statsmodels 自动确定 maxlag 上界
        autolag="AIC",
        regression=REG,
        store=STORE,
        regresults=REGRESULTS,
    )
    stat_aic, pval_aic, crit_aic, res_aic = result_aic

    # --- BIC 选阶 ---
    result_bic: Any = adfuller(
        prices,
        maxlag=None,
        autolag="BIC",
        regression=REG,
        store=STORE,
        regresults=REGRESULTS,
    )
    stat_bic, pval_bic, crit_bic, res_bic = result_bic

    # --- 提取关键指标 ---
    # usedlag: 最优滞后阶数 (由信息准则自动选择)
    lag_aic = res_aic.usedlag
    lag_bic = res_bic.usedlag

    # nobs: 实际使用的样本数 (去除滞后和差分后的有效观测)
    nobs = res_aic.nobs

    # λ_ADF: ADF 回归中 y(t-1) 的系数 γ
    # adfuller 的 OLS 参数顺序: [γ(y_{t-1}), 截距, (趋势), Δy_{t-1}, ..., Δy_{t-p}]
    # params[0] 即为 γ, 也就是 λ_ADF
    lambda_adf = res_aic.resols.params[0]

    return {
        "adf_stat_aic": stat_aic,
        "p_value_aic": pval_aic,
        "adf_stat_bic": stat_bic,
        "p_value_bic": pval_bic,
        "lambda_adf": lambda_adf,
        "used_lag_aic": lag_aic,
        "used_lag_bic": lag_bic,
        "n_obs": nobs,
        "critical_1pct": crit_aic["1%"],
        "critical_5pct": crit_aic["5%"],
        "critical_10pct": crit_aic["10%"],
    }


# ============================================================
# 格式化输出
# ============================================================

def _format_table(result: dict, symbol: str = "") -> str:
    """将 result 格式化为可读的文本表格。"""
    lines = [
        f"{'=' * 56}",
        f"  ADF 单位根检验结果  {symbol}",
        f"{'=' * 56}",
        "",
        f"  样本数 (n_obs):          {result['n_obs']}",
        "",
        "  ┌──────────────┬──────────┬──────────┬──────────┐",
        "  │    指标       │   AIC    │   BIC    │  备注    │",
        "  ├──────────────┼──────────┼──────────┼──────────┤",
        f"  │  ADF 统计量   │ {result['adf_stat_aic']:>8.4f} │ {result['adf_stat_bic']:>8.4f} │          │",
        f"  │  p-value     │ {result['p_value_aic']:>8.4f} │ {result['p_value_bic']:>8.4f} │          │",
        f"  │  最优滞后阶数 │ {result['used_lag_aic']:>8d} │ {result['used_lag_bic']:>8d} │          │",
        "  └──────────────┴──────────┴──────────┴──────────┘",
        "",
        f"  λ_ADF (y(t-1) 系数):     {result['lambda_adf']:.6f}",
        "",
        "  临界值:",
        f"    1%  :  {result['critical_1pct']:.4f}",
        f"    5%  :  {result['critical_5pct']:.4f}",
        f"    10% :  {result['critical_10pct']:.4f}",
        "",
        "  判定 (AIC):",
    ]

    p = result["p_value_aic"]
    if p < 0.01:
        verdict = "p < 0.01 -> 强烈拒绝 H0, 序列平稳 [PASS]"
    elif p < 0.05:
        verdict = "p < 0.05 -> 拒绝 H0, 序列平稳 [PASS]"
    elif p < 0.10:
        verdict = "p < 0.10 -> 边际拒绝, 弱平稳"
    else:
        verdict = "p >= 0.05 -> 不能拒绝 H0, 序列非平稳 [FAIL]"

    lines.append(f"    {verdict}")
    lines.append(f"{'=' * 56}")
    return "\n".join(lines)


# ============================================================
# 直接运行: 冒烟测试
# ============================================================

if __name__ == "__main__":
    from data.fetcher import read_day

    print("加载上证指数日线数据 ...")
    df = read_day("sh000001")
    prices = df["close"]
    print(f"  样本: {len(prices)} 个交易日")
    print(f"  区间: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print()

    # 执行 ADF 检验
    result = run_adf(prices)

    # 打印结果表
    print(_format_table(result, symbol="sh000001 (上证指数)"))
    print()

    # --- 对照组断言: 大盘指数应为非平稳 ---
    assert result["p_value_aic"] > 0.05, (
        f"对照组失败: sh000001 的 p_value_aic = {result['p_value_aic']:.4f}, "
        f"预期 > 0.05 (非平稳), 实际 <= 0.05 (平稳)"
    )
    print("[PASS] 对照组验证通过: sh000001 p_value_aic > 0.05 (非平稳)")
