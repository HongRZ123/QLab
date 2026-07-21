"""
数据源抽象层 - OHLCVSource Protocol

定义数据源的统一接口，支持不同数据提供者（TDX、CSV、数据库等）。
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class OHLCVSource(Protocol):
    """OHLCV 数据源协议"""

    def get_ohlcv(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        获取 OHLCV 数据

        Args:
            symbol: 标的代码（如 'sh000001'）
            start: 开始日期（YYYYMMDD 格式）
            end: 结束日期（YYYYMMDD 格式）

        Returns:
            DataFrame，包含 date, open, high, low, close, amount, volume 列
        """
        ...

    def list_symbols(self, market: str = "sh") -> list[str]:
        """
        列出指定市场的所有标的代码

        Args:
            market: 市场代码（'sh' 或 'sz'）

        Returns:
            标的代码列表
        """
        ...
