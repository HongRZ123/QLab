"""
scan_stationarity.py — 分层平稳性扫描: 全样本 + 滚动窗口
=========================================================

对 ~38 只 ETF / 指数执行三层检验:
    Step 1  ADF 单位根检验 (AIC / BIC 双准则)
    Step 2  Hurst 指数 (方差法)
    Step 3  半衰期估计 (离散精确公式)

并叠加滚动窗口稳健性检验:
    250 天窗口, 60 天步进, 统计 ADF 通过率 & 半衰期分布

最终应用 Holm-Bonferroni 多重检验校正, 输出 CSV 报告。

用法:
    python explore/scan_stationarity.py

输出:
    output/stationarity_report.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 项目根目录: 确保能 import data / tests ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ETF 宇宙统一来源: alpha.defaults
from alpha.defaults import ETF_UNIVERSE
from data.fetcher import read_day
from stats.univariate import estimate_half_life, hurst_exponent, run_adf

# ============================================================
# 滚动窗口参数
# ============================================================

ROLLING_WINDOW = 250   # 窗口长度 (交易日)
ROLLING_STEP = 60      # 步进 (交易日)
MIN_OBS = 200          # 窗口内最少有效观测数

# 最少全样本天数: 不足则跳过
MIN_FULL_SAMPLE = 500


# ============================================================
# CSV 输出列 (严格顺序)
# ============================================================

OUTPUT_COLUMNS = [
    "symbol", "category", "n_obs",
    "adf_stat_aic", "p_value_aic", "adf_stat_bic", "p_value_bic",
    "lambda_half", "half_life", "half_life_formula_used",
    "hurst", "hurst_r_squared",
    "p_value_holm",
    "rolling_pass_rate", "rolling_hl_median", "rolling_hl_std",
    "price_limit_pct",
    "is_candidate", "is_preferred",
]


# ============================================================
# 辅助函数
# ============================================================

def _get_price_limit_pct(symbol: str) -> float:
    """
    根据代码推断涨跌停幅度。

    规则:
        科创板 (688/588) / 创业板 (300/301/159915/159996) → 20%
        其余 → 10%
    """
    code = symbol[2:]  # 去掉 sh/sz 前缀

    # 科创板 ETF (588xxx)
    if code.startswith("588"):
        return 0.20
    # 创业板 ETF (159xxx 中部分)
    if symbol.startswith("sz") and code.startswith("3"):
        return 0.20
    # 创业板指数 (399006)
    if code.startswith("399"):
        return 0.20
    # 其余默认 10%
    return 0.10


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni 逐步校正法。

    对一组 p-value 做多重检验校正, 控制族错误率 (FWER)。
    比 Bonferroni 更有力 (less conservative)。

    参数:
        p_values: 原始 p-value 数组

    返回:
        校正后的 p-value 数组 (与输入等长)
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    sorted_indices = np.argsort(p)
    adjusted = np.empty(n)

    # 逐步校正: 第 i 小的 p-value 乘以 (n - i)
    for i, idx in enumerate(sorted_indices):
        adjusted[idx] = min(p[idx] * (n - i), 1.0)

    # 保证单调性: 校正后的值不得小于前一个 (按排序顺序)
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])

    return adjusted


def _rolling_analysis(close: pd.Series) -> dict:
    """
    滚动窗口分析: 在多个子窗口上重复 ADF + 半衰期检验。

    使用位置索引 (iloc), 不使用日期索引 (.loc)。

    返回:
        dict with keys:
            rolling_pass_rate  — 通过窗口占比
            rolling_hl_median  — 通过窗口的半衰期中位数
            rolling_hl_std     — 通过窗口的半衰期标准差
    """
    n = len(close)
    passing_half_lives = []
    total_windows = 0

    start = 0
    while start + ROLLING_WINDOW <= n:
        window = close.iloc[start:start + ROLLING_WINDOW]

        # 跳过样本不足的窗口
        if len(window) < MIN_OBS:
            start += ROLLING_STEP
            continue

        total_windows += 1

        # ADF 检验
        try:
            adf_res = run_adf(window)
        except Exception:
            start += ROLLING_STEP
            continue

        # 半衰期
        try:
            hl_res = estimate_half_life(window, use_log=True)
        except Exception:
            start += ROLLING_STEP
            continue

        # 判定: 窗口是否 "通过" (平稳 + 均值回归 + 合理半衰期)
        lam = hl_res["lambda"]
        hl = hl_res["half_life"]
        p_val = adf_res["p_value_aic"]

        if (p_val < 0.10) and (lam < 0) and (2 <= hl <= 60):
            passing_half_lives.append(hl)

        start += ROLLING_STEP

    # 汇总
    if total_windows == 0 or len(passing_half_lives) == 0:
        return {
            "rolling_pass_rate": 0.0,
            "rolling_hl_median": np.nan,
            "rolling_hl_std": np.nan,
        }

    hl_arr = np.array(passing_half_lives)
    return {
        "rolling_pass_rate": len(passing_half_lives) / total_windows,
        "rolling_hl_median": float(np.median(hl_arr)),
        "rolling_hl_std": float(np.std(hl_arr, ddof=1)) if len(hl_arr) > 1 else 0.0,
    }


# ============================================================
# 核心扫描: 单只标的
# ============================================================

def _scan_single(symbol: str, category: str) -> dict | None:
    """
    扫描单只标的: 全样本三层检验 + 滚动窗口稳健性。

    返回:
        dict (一行 CSV 数据), 或 None (数据不足时跳过)
    """
    # ── 加载数据 ──
    try:
        df = read_day(symbol)
    except FileNotFoundError:
        print(f"  [跳过] {symbol}: 数据文件不存在")
        return None

    if len(df) < MIN_FULL_SAMPLE:
        print(f"  [跳过] {symbol}: 数据不足 ({len(df)} < {MIN_FULL_SAMPLE} 天)")
        return None

    close = df["close"]
    n_obs = len(close)

    # ── Step 1: 全样本 ADF ──
    adf_res = run_adf(close)

    # ── Step 2: 全样本 Hurst ──
    hurst_res = hurst_exponent(close)

    # ── Step 3: 全样本半衰期 ──
    hl_res = estimate_half_life(close, use_log=True)

    # ── Step 4: 滚动窗口 ──
    rolling = _rolling_analysis(close)

    # ── 涨跌停幅度 ──
    price_limit_pct = _get_price_limit_pct(symbol)

    # ── 候选判定 ──
    lam = hl_res["lambda"]
    hl = hl_res["half_life"]
    p_val = adf_res["p_value_aic"]

    is_candidate = (lam < 0) and (2 <= hl <= 60) and (p_val < 0.10)

    # 半衰期公式标记
    formula_used = "discrete_exact" if (lam < 0 and np.isfinite(hl)) else "N/A"

    return {
        "symbol": symbol,
        "category": category,
        "n_obs": n_obs,
        "adf_stat_aic": adf_res["adf_stat_aic"],
        "p_value_aic": adf_res["p_value_aic"],
        "adf_stat_bic": adf_res["adf_stat_bic"],
        "p_value_bic": adf_res["p_value_bic"],
        "lambda_half": lam,
        "half_life": hl,
        "half_life_formula_used": formula_used,
        "hurst": hurst_res["hurst"],
        "hurst_r_squared": hurst_res["r_squared"],
        "p_value_holm": np.nan,  # 占位, 后续 Holm-Bonferroni 填充
        "rolling_pass_rate": rolling["rolling_pass_rate"],
        "rolling_hl_median": rolling["rolling_hl_median"],
        "rolling_hl_std": rolling["rolling_hl_std"],
        "price_limit_pct": price_limit_pct,
        "is_candidate": is_candidate,
        "is_preferred": False,  # 占位, 后续填充
    }


# ============================================================
# 主扫描流程
# ============================================================

def scan_all() -> pd.DataFrame:
    """
    扫描 ETF_UNIVERSE 中所有标的, 返回结果 DataFrame。

    流程:
        1. 逐类别、逐标的执行全样本 + 滚动窗口分析
        2. 对所有 p_value_aic 应用 Holm-Bonferroni 校正
        3. 根据校正后 p-value 判定 is_preferred
    """
    rows = []

    for category, symbols in ETF_UNIVERSE.items():
        print(f"\n{'=' * 56}")
        print(f"  类别: {category}  ({len(symbols)} 只)")
        print(f"{'=' * 56}")

        for symbol in symbols:
            print(f"  扫描 {symbol} ...", end=" ")
            row = _scan_single(symbol, category)
            if row is not None:
                rows.append(row)
                # 简要状态
                p = row["p_value_aic"]
                hl = row["half_life"]
                tag = "CANDIDATE" if row["is_candidate"] else "-"
                print(f"p={p:.4f}  HL={hl:.1f}  [{tag}]")
            else:
                print("(skipped)")

    if not rows:
        print("\n[警告] 没有有效数据, 无法生成报告")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)

    # ── Holm-Bonferroni 多重检验校正 ──
    pvals = df["p_value_aic"].to_numpy(dtype=float)
    df["p_value_holm"] = holm_bonferroni(pvals)

    # ── is_preferred 判定 ──
    # 需要: is_candidate AND rolling_pass_rate > 0.30
    #       AND rolling_hl_std < rolling_hl_median * 0.5
    mask_candidate = df["is_candidate"]
    mask_rolling = df["rolling_pass_rate"] > 0.30
    mask_stable = (
        df["rolling_hl_std"] < df["rolling_hl_median"] * 0.5
    ) & df["rolling_hl_median"].notna()
    df["is_preferred"] = mask_candidate & mask_rolling & mask_stable

    # 确保列顺序
    df = df[OUTPUT_COLUMNS]

    return df


# ============================================================
# 汇总输出
# ============================================================

def _print_summary(df: pd.DataFrame) -> None:
    """按类别打印扫描汇总。"""
    print(f"\n{'=' * 56}")
    print("  扫描汇总")
    print(f"{'=' * 56}")

    for category in ETF_UNIVERSE:
        sub = df[df["category"] == category]
        if sub.empty:
            print(f"\n  [{category}] 无数据")
            continue

        n_total = len(sub)
        n_candidate = sub["is_candidate"].sum()
        n_preferred = sub["is_preferred"].sum()

        print(f"\n  [{category}]  {n_total} 只已扫描")
        print(f"    候选 (is_candidate):  {n_candidate}")
        print(f"    优选 (is_preferred):  {n_preferred}")

        # 列出优选标的
        preferred = sub[sub["is_preferred"]]
        if not preferred.empty:
            for _, row in preferred.iterrows():
                print(f"      * {row['symbol']}  "
                      f"HL={row['half_life']:.1f}  "
                      f"p_holm={row['p_value_holm']:.4f}  "
                      f"pass_rate={row['rolling_pass_rate']:.0%}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 56)
    print("  分层平稳性扫描 — 全样本 + 滚动窗口")
    print("=" * 56)
    print(f"  滚动窗口: {ROLLING_WINDOW} 天, 步进 {ROLLING_STEP} 天")
    print(f"  ETF 总数: {sum(len(v) for v in ETF_UNIVERSE.values())}")

    # 执行扫描
    report = scan_all()

    # 打印汇总
    _print_summary(report)

    # 保存 CSV
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "stationarity_report.csv"
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 56}")
    print(f"  报告已保存: {output_path}")
    print(f"  共 {len(report)} 条记录")
    print(f"{'=' * 56}")
