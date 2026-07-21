"""
rules.py — A股交易规则与约束
==============================

将A股特有的交易限制封装为简单函数,
供回测和策略模块调用。

用法:
    from data.rules import round_to_lot, transaction_cost, next_trade_date
"""

import pandas as pd

# ============================================================
# 基本参数 (可按需修改)
# ============================================================

# 每手股数
LOT_SIZE = 100

# 费率
COMMISSION_RATE = 0.00025   # 佣金 万2.5 (双边)
MIN_COMMISSION = 5.0        # 最低佣金 5元
STAMP_TAX_RATE = 0.0005     # 印花税 万5 (仅卖出, 2023年8月后)
SLIPPAGE_RATE = 0.001       # 滑点估计 千1 (可选)


# ============================================================
# 核心函数
# ============================================================

def round_to_lot(shares: float, lot: int = LOT_SIZE) -> int:
    """
    将股数取整到整数手 (A股最小交易单位)。

    参数:
        shares: 理论股数 (可以是小数)
        lot:    每手股数, 默认100

    返回:
        整数股数 (向下取整到lot的倍数)

    示例:
        >>> round_to_lot(1234)
        1200
        >>> round_to_lot(99)
        0
    """
    return int(shares // lot) * lot


def transaction_cost(amount: float, direction: str,
                     include_slippage: bool = False) -> float:
    """
    计算单笔交易成本。

    参数:
        amount:           成交金额 (元)
        direction:        "buy" 或 "sell"
        include_slippage: 是否包含滑点估计

    返回:
        总交易成本 (元)

    示例:
        >>> transaction_cost(100000, "buy")   # 买入10万
        25.0
        >>> transaction_cost(100000, "sell")  # 卖出10万
        75.0
    """
    # 佣金 (双边, 有最低限制)
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)

    # 印花税 (仅卖出)
    stamp = amount * STAMP_TAX_RATE if direction == "sell" else 0.0

    # 滑点 (可选)
    slippage = amount * SLIPPAGE_RATE if include_slippage else 0.0

    return commission + stamp + slippage


def next_trade_date(signal_date, trade_dates: pd.DatetimeIndex) -> pd.Timestamp:
    """
    T+1规则: 给定信号日, 返回下一个可交易日。

    参数:
        signal_date: 信号产生日期 (pd.Timestamp 或可转换类型)
        trade_dates: 可用交易日序列 (从数据中获取)

    返回:
        下一个交易日 (pd.Timestamp)
        如果信号日是最后一个交易日, 返回信号日本身 (无法执行)

    示例:
        >>> dates = pd.date_range("2024-01-01", periods=10, freq="B")
        >>> next_trade_date("2024-01-03", dates)
        Timestamp('2024-01-04 00:00:00')
    """
    signal_date = pd.Timestamp(signal_date)
    # 找到信号日在交易日序列中的位置
    mask = trade_dates > signal_date
    if mask.any():
        return trade_dates[mask][0]
    else:
        # 信号日是最后一天, 无法执行
        return signal_date


def is_tradable(date, price_data: pd.DataFrame) -> bool:
    """
    判断某日是否可交易 (非停牌、非涨跌停)。

    简化版: 仅检查该日是否有数据 (停牌日无数据)。
    涨跌停判断需要前一日收盘价, 此处暂不实现。

    参数:
        date:       日期
        price_data: 含 open/high/low/close 的 DataFrame

    返回:
        True = 可交易
    """
    date = pd.Timestamp(date)
    if date not in price_data.index:
        return False  # 停牌
    row = price_data.loc[date]
    # 成交量为0也视为不可交易 (集合竞价未成交)
    return row["volume"] != 0


def check_price_limit(today_close: float, prev_close: float,
                      board: str = "main") -> dict:
    """
    检查是否触及涨跌停。

    参数:
        today_close: 今日收盘价
        prev_close:  昨日收盘价
        board:       板块类型
                     "main"      = 主板 (±10%)
                     "chinext"   = 创业板 (±20%)
                     "star"      = 科创板 (±20%)
                     "st"        = ST股 (±5%)
                     "bse"       = 北交所 (±30%)

    返回:
        dict: {"limit_up": bool, "limit_down": bool,
               "upper": float, "lower": float}

    示例:
        >>> check_price_limit(11.0, 10.0, "main")
        {'limit_up': True, 'limit_down': False, 'upper': 11.0, 'lower': 9.0}
    """
    pct_map = {"main": 0.10, "chinext": 0.20, "star": 0.20, "st": 0.05, "bse": 0.30}
    pct = pct_map.get(board, 0.10)

    upper = round(prev_close * (1 + pct), 2)
    lower = round(prev_close * (1 - pct), 2)

    return {
        "limit_up": today_close >= upper,
        "limit_down": today_close <= lower,
        "upper": upper,
        "lower": lower,
    }


# ============================================================
# 直接运行: 快速验证
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("A股交易规则验证")
    print("=" * 60)

    print(f"\n[1] 整数手: 1234股 -> {round_to_lot(1234)}股")
    print(f"    整数手: 99股  -> {round_to_lot(99)}股")
    print(f"    整数手: 500股 -> {round_to_lot(500)}股")

    print("\n[2] 交易成本 (10万元):")
    print(f"    买入: {transaction_cost(100000, 'buy'):.2f} 元")
    print(f"    卖出: {transaction_cost(100000, 'sell'):.2f} 元")
    print(f"    卖出(含滑点): {transaction_cost(100000, 'sell', True):.2f} 元")

    print("\n[3] 涨跌停检查:")
    result = check_price_limit(11.0, 10.0, "main")
    print(f"    主板 10.0->11.0: {result}")
    result2 = check_price_limit(12.0, 10.0, "chinext")
    print(f"    创业板 10.0->12.0: {result2}")

    print("\n[4] T+1 下一交易日:")
    dates = pd.bdate_range("2024-01-01", periods=5)
    print(f"    交易日序列: {[d.date() for d in dates]}")
    print(f"    1月3日信号 -> 执行日: {next_trade_date('2024-01-03', dates).date()}")
