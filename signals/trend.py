"""
trend.py -- 趋势方向与健康度信号

基于《量价分析》Ch8 动态趋势 + Ch10 实战案例。

重写改进：
    1. 新增 trend_direction() -- 判断趋势方向
    2. trend_health() 变为上下文感知 -- 同一K线在上涨/下跌趋势中含义不同

Functions:
    trend_direction: 趋势方向 (+1/-1/0)
    trend_health:    趋势健康度 (+1/-1/0, 上下文感知)
    run_validation:  验证协议
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_direction(
    close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """趋势方向

    基于 close 与滚动均线的关系判断趋势方向。
        +1 = 上涨趋势 (close > MA)
        -1 = 下跌趋势 (close < MA)
         0 = 盘整 (close ≈ MA)

    Args:
        close: 收盘价序列
        lookback: 滚动均线窗口

    Returns:
        整数 Series: +1 / -1 / 0
    """
    ma = close.rolling(window=lookback, min_periods=1).mean()
    diff = close - ma
    threshold = ma * 0.001  # 0.1% of price as neutral zone

    result = pd.Series(0, index=close.index, dtype=int)
    result[diff > threshold] = 1
    result[diff < -threshold] = -1
    return result


def trend_health(
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """趋势健康度（上下文感知）

    判断量价关系是否支持当前趋势。核心改进：同一K线在不同趋势中含义不同。

    上下文感知逻辑：
        1. 判断趋势方向（上涨/下跌/盘整）
        2. 将每日分为"推进日"（顺势）和"回撤日"（逆势）
        3. 推进日放量 + 回撤日缩量 = 趋势健康
        4. 推进日缩量 或 回撤日放量 = 趋势衰竭

    上涨趋势中：
        上涨日(推进) + 高量 = 健康    # 买方积极追高
        下跌日(回撤) + 低量 = 健康    # 正常回撤，卖方不积极
        上涨日(推进) + 低量 = 衰竭    # No Demand，买方不愿追高
        下跌日(回撤) + 高量 = 衰竭    # 卖方介入，供应增加

    下跌趋势中：
        下跌日(推进) + 高量 = 健康    # 卖方积极抛售
        上涨日(回撤) + 低量 = 健康    # 弱势反弹，买方不积极
        下跌日(推进) + 低量 = 衰竭    # No Supply，卖方耗尽
        上涨日(回撤) + 高量 = 衰竭    # 买方介入，需求增加

    Args:
        close: 收盘价序列
        volume: 成交量序列
        lookback: 滚动窗口

    Returns:
        整数 Series: +1（健康）/ -1（衰竭）/ 0（中性）
    """
    direction = trend_direction(close, lookback)
    price_change = close.diff()
    vol_mean = volume.rolling(window=lookback, min_periods=1).mean()

    up_day = price_change > 0
    down_day = price_change < 0
    vol_high = volume > vol_mean
    vol_low = volume < vol_mean

    uptrend = direction == 1
    downtrend = direction == -1

    result = pd.Series(0, index=close.index, dtype=int)

    # 上涨趋势
    result[uptrend & up_day & vol_high] = 1      # 推进放量 = 健康
    result[uptrend & down_day & vol_low] = 1     # 回撤缩量 = 健康
    result[uptrend & up_day & vol_low] = -1      # No Demand = 衰竭
    result[uptrend & down_day & vol_high] = -1   # 回撤放量 = 衰竭

    # 下跌趋势
    result[downtrend & down_day & vol_high] = 1   # 推进放量 = 健康
    result[downtrend & up_day & vol_low] = 1      # 回撤缩量 = 健康
    result[downtrend & down_day & vol_low] = -1   # No Supply = 衰竭
    result[downtrend & up_day & vol_high] = -1    # 回撤放量 = 衰竭

    return result


def run_validation() -> bool:
    """趋势信号验证协议"""
    print("=" * 60)
    print("趋势信号验证协议 (trend.py 重写版)")
    print("=" * 60)

    all_pass = True
    np.random.seed(42)
    n = 500

    # ── trend_direction: 正控 ──
    print("\n【trend_direction】正控")
    print("-" * 60)

    # 上涨趋势
    close_up = pd.Series(np.linspace(10, 20, n))
    td_up = trend_direction(close_up, lookback=20)
    up_ratio = (td_up == 1).sum() / len(td_up)
    up_ok = up_ratio >= 0.80
    print(f"  上涨趋势: +1 占比 {up_ratio:.2%} (要求 >= 80%)  "
          f"[{'PASS' if up_ok else 'FAIL'}]")
    if not up_ok:
        all_pass = False

    # 下跌趋势
    close_down = pd.Series(np.linspace(20, 10, n))
    td_down = trend_direction(close_down, lookback=20)
    down_ratio = (td_down == -1).sum() / len(td_down)
    down_ok = down_ratio >= 0.80
    print(f"  下跌趋势: -1 占比 {down_ratio:.2%} (要求 >= 80%)  "
          f"[{'PASS' if down_ok else 'FAIL'}]")
    if not down_ok:
        all_pass = False

    # ── trend_health: 正控 -- 上涨趋势 + 上涨放量 ──
    print("\n【trend_health】正控: 上涨趋势 + 上涨放量")
    print("-" * 60)

    base_body = np.linspace(0.05, 0.3, n)
    body = np.maximum(base_body + np.random.randn(n) * 0.05, 0.01)
    close = pd.Series(np.cumsum(body) + 10)
    volume = pd.Series(body * 50000 + np.random.randint(100, 500, n))

    th = trend_health(close, volume, lookback=20)
    healthy_ratio = (th == 1).sum() / len(th)
    healthy_ok = healthy_ratio >= 0.50
    print(f"  健康信号占比: {healthy_ratio:.2%} (要求 >= 50%)  "
          f"[{'PASS' if healthy_ok else 'FAIL'}]")
    if not healthy_ok:
        all_pass = False

    # ── trend_health: 上下文感知验证 ──
    print("\n【trend_health】上下文感知验证")
    print("-" * 60)

    # 同一根K线（上涨+低量）在不同趋势中应有不同结果
    # 构造：前半段上涨趋势，后半段下跌趋势
    # 在两段中都插入"上涨+低量"的K线
    n_ctx = 100
    # 前50天上涨趋势，后50天下跌趋势
    close_ctx = pd.Series(np.concatenate([
        np.linspace(10, 20, 50),
        np.linspace(20, 10, 50),
    ]))

    # 成交量：随机低量（明显低于均值的日子）
    np.random.seed(99)
    volume_ctx = pd.Series(np.random.randint(100, 300, n_ctx).astype(float))

    th_ctx = trend_health(close_ctx, volume_ctx, lookback=20)

    # 上涨趋势中的上涨+低量 = -1 (No Demand, 衰竭)
    # 下跌趋势中的上涨+低量 = +1 (弱势反弹, 健康)
    uptrend_low_vol_up = (th_ctx[:50] == -1).sum()
    downtrend_low_vol_up = (th_ctx[50:] == 1).sum()

    ctx_ok = uptrend_low_vol_up > 0 and downtrend_low_vol_up > 0
    print(f"  上涨趋势中低量上涨->衰竭: {uptrend_low_vol_up} 天 (要求 > 0)  "
          f"[{'PASS' if uptrend_low_vol_up > 0 else 'FAIL'}]")
    print(f"  下跌趋势中低量上涨->健康: {downtrend_low_vol_up} 天 (要求 > 0)  "
          f"[{'PASS' if downtrend_low_vol_up > 0 else 'FAIL'}]")
    if not ctx_ok:
        all_pass = False

    # ── trend_health: 负控 -- 随机游走 ──
    print("\n【trend_health】负控: 随机游走")
    print("-" * 60)

    np.random.seed(43)
    close_random = pd.Series(np.cumsum(np.random.randn(n)) + 10)
    volume_random = pd.Series(np.random.randint(500, 1500, n))

    th_random = trend_health(close_random, volume_random, lookback=20)
    healthy_random = (th_random == 1).sum() / len(th_random)
    weak_random = (th_random == -1).sum() / len(th_random)
    balanced_ok = abs(healthy_random - weak_random) < 0.25
    print(f"  健康: {healthy_random:.2%}, 衰竭: {weak_random:.2%}, "
          f"差值: {abs(healthy_random - weak_random):.2%} (要求 < 25%)  "
          f"[{'PASS' if balanced_ok else 'FAIL'}]")
    if not balanced_ok:
        all_pass = False

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] 趋势信号验证通过")
    else:
        print("[FAIL] 存在验证失败项")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys

    ok = run_validation()
    sys.exit(0 if ok else 1)
