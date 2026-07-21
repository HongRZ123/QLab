"""
TDX 数据源实现

包装现有的 data.fetcher 模块，提供 OHLCVSource 接口。
"""
import pandas as pd

from data import fetcher


class TDXSource:
    """通达信数据源实现"""

    def get_ohlcv(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        从通达信读取 OHLCV 数据

        Args:
            symbol: 标的代码（如 'sh000001'）
            start: 开始日期（YYYYMMDD 格式）
            end: 结束日期（YYYYMMDD 格式）

        Returns:
            DataFrame，包含 date, open, high, low, close, amount, volume 列
        """
        return fetcher.read_day(symbol, start, end)

    def list_symbols(self, market: str = "sh") -> list[str]:
        """
        列出指定市场的所有标的代码

        Args:
            market: 市场代码（'sh' 或 'sz'）

        Returns:
            标的代码列表
        """
        return fetcher.list_symbols(market)
