"""
fetcher.py — 通达信 .day 二进制日线数据读取
============================================

数据源: 通达信盘后下载数据包
路径:   D:\\new_tdx64\\vipdoc\\{market}\\lday\\{symbol}.day

二进制格式 (每条记录 32 字节, little-endian):
    字段        类型      说明
    ----        ----      ----
    date        uint32    日期, 格式 YYYYMMDD
    open        uint32    开盘价 × 100 (整数存储)
    high        uint32    最高价 × 100
    low         uint32    最低价 × 100
    close       uint32    收盘价 × 100
    amount      float32   成交额 (元)
    volume      uint32    成交量 (股)
    reserved    uint32    保留字段 (忽略)

用法:
    from data.fetcher import read_day, read_symbols, list_symbols

    # 读取单只
    df = read_day("sh600000")

    # 批量读取
    dfs = read_symbols(["sh600000", "sz000001", "sh510050"])

    # 列出某市场所有可用代码
    codes = list_symbols("sh")
"""

import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 配置
# ============================================================

# 通达信数据根目录
# 优先读取环境变量 QLAB_TDX_ROOT，未设置则使用默认路径
TDX_ROOT = Path(os.environ.get("QLAB_TDX_ROOT", r"D:\new_tdx64\vipdoc"))

# 每条记录的字节数和struct格式
RECORD_SIZE = 32
RECORD_FORMAT = "<IIIIIfII"  # little-endian: 5×uint32 + 1×float32 + 2×uint32


# ============================================================
# 核心读取函数
# ============================================================

def read_day(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    读取单只标的的通达信日线数据。

    参数:
        symbol: 标的代码, 格式为 "{market}{code}"
                例: "sh600000", "sz000001", "sh510050", "sz159915"
                market: sh=上海, sz=深圳, bj=北京
        start:  起始日期 (含), 格式 "YYYYMMDD" 或 "YYYY-MM-DD", 可选
        end:    结束日期 (含), 格式同上, 可选

    返回:
        DataFrame, 列: date, open, high, low, close, amount, volume
        - date:   datetime64 类型
        - 价格:   float64, 单位 元
        - amount: float64, 单位 元
        - volume: int64,   单位 股

    示例:
        >>> df = read_day("sh600000")
        >>> df = read_day("sz000001", start="20200101", end="20231231")
    """
    # 解析市场和代码
    market, code = _parse_symbol(symbol)

    # 构造文件路径
    filepath = TDX_ROOT / market / "lday" / f"{market}{code}.day"
    if not filepath.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {filepath}\n"
            f"请确认通达信已下载该标的的日线数据。"
        )

    # 读取二进制
    raw = filepath.read_bytes()
    n_records = len(raw) // RECORD_SIZE
    if n_records == 0:
        return _empty_df()

    # 解包所有记录
    records = []
    for i in range(n_records):
        chunk = raw[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]
        fields = struct.unpack(RECORD_FORMAT, chunk)
        records.append(fields)

    # 转为 DataFrame
    df = pd.DataFrame(
        records,
        columns=["date_raw", "open_raw", "high_raw", "low_raw",
                 "close_raw", "amount", "volume", "_reserved"],
    )

    # 类型转换
    df["date"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y%m%d")
    df["open"] = df["open_raw"] / 100.0
    df["high"] = df["high_raw"] / 100.0
    df["low"] = df["low_raw"] / 100.0
    df["close"] = df["close_raw"] / 100.0
    df["amount"] = df["amount"].astype(np.float64)
    df["volume"] = df["volume"].astype(np.int64)

    # 只保留需要的列
    df = df[["date", "open", "high", "low", "close", "amount", "volume"]]
    df = df.set_index("date").sort_index()

    # 日期过滤
    if start is not None:
        start_dt = pd.to_datetime(_normalize_date(start))
        df = df[df.index >= start_dt]
    if end is not None:
        end_dt = pd.to_datetime(_normalize_date(end))
        df = df[df.index <= end_dt]

    return df


def read_symbols(symbols: list, start: str | None = None, end: str | None = None) -> dict:
    """
    批量读取多只标的。

    参数:
        symbols: 代码列表, 例 ["sh600000", "sz000001", "sh510050"]
        start, end: 同 read_day

    返回:
        dict, key=代码, value=DataFrame
        读取失败的标的会打印警告并跳过。

    示例:
        >>> dfs = read_symbols(["sh510050", "sh510300", "sz159915"])
        >>> dfs["sh510050"].tail()
    """
    result = {}
    for sym in symbols:
        try:
            result[sym] = read_day(sym, start=start, end=end)
        except FileNotFoundError as e:
            print(f"[跳过] {sym}: {e}")
    return result


def list_symbols(market: str = "sh") -> list:
    """
    列出某市场下所有可用的日线数据代码。

    参数:
        market: "sh" | "sz" | "bj"

    返回:
        代码列表 (含市场前缀), 例 ["sh000001", "sh600000", ...]

    示例:
        >>> etfs = [s for s in list_symbols("sh") if s.startswith("sh51")]
    """
    lday_dir = TDX_ROOT / market / "lday"
    if not lday_dir.exists():
        raise FileNotFoundError(f"目录不存在: {lday_dir}")

    codes = []
    for f in sorted(lday_dir.iterdir()):
        if f.suffix == ".day":
            codes.append(f.stem)  # e.g. "sh600000"
    return codes


# ============================================================
# 辅助函数
# ============================================================

def _parse_symbol(symbol: str) -> tuple:
    """
    解析标的代码为 (market, code)。

    支持格式:
        "sh600000"  -> ("sh", "600000")
        "sz000001"  -> ("sz", "000001")
        "600000"    -> ("sh", "600000")  (自动推断市场)
        "000001"    -> ("sz", "000001")  (自动推断市场)
    """
    symbol = symbol.strip().lower()

    if symbol.startswith(("sh", "sz", "bj")):
        return symbol[:2], symbol[2:]

    # 纯数字: 根据代码段推断市场
    code = symbol
    if code.startswith(("6", "5", "9", "11", "13", "20")):
        return "sh", code
    elif code.startswith(("0", "1", "2", "3")):
        return "sz", code
    elif code.startswith(("4", "8")):
        return "bj", code
    else:
        raise ValueError(f"无法推断市场, 请使用完整代码如 'sh{code}' 或 'sz{code}'")


def _normalize_date(date_str: str) -> str:
    """将 'YYYY-MM-DD' 或 'YYYYMMDD' 统一为 'YYYYMMDD'。"""
    return date_str.replace("-", "")


def _empty_df() -> pd.DataFrame:
    """返回空的日线DataFrame。"""
    df = pd.DataFrame(columns=["open", "high", "low", "close", "amount", "volume"])
    df.index.name = "date"
    return df


# ============================================================
# 直接运行: 快速验证
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("通达信日线数据读取测试")
    print("=" * 60)

    # 测试1: 读取上证指数
    print("\n[1] 上证指数 sh000001:")
    df = read_day("sh000001")
    print(f"    记录数: {len(df)}")
    print(f"    时间范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(df.tail(3).to_string())

    # 测试2: 读取浦发银行
    print("\n[2] 浦发银行 sh600000:")
    df2 = read_day("sh600000", start="20240101")
    print(f"    2024年以来记录数: {len(df2)}")
    print(df2.tail(3).to_string())

    # 测试3: 读取ETF
    print("\n[3] 50ETF sh510050:")
    df3 = read_day("sh510050")
    print(f"    记录数: {len(df3)}")
    print(df3.tail(3).to_string())

    # 测试4: 列出上海ETF
    print("\n[4] 上海ETF列表 (前10):")
    all_sh = list_symbols("sh")
    etfs = [s for s in all_sh if s[2:].startswith("51")]
    print(f"    共 {len(etfs)} 只ETF, 前10: {etfs[:10]}")
