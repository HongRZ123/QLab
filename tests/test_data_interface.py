"""OHLCVSource Protocol 单元测试"""

import pandas as pd

from data.interface import OHLCVSource


class MockSource:
    """实现 OHLCVSource 的模拟类"""

    def get_ohlcv(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "close": [10.5] * 5,
                "amount": [1000000.0] * 5,
                "volume": [100000] * 5,
            }
        )

    def list_symbols(self, market: str = "sh") -> list[str]:
        return ["sh000001", "sh600000"]


class IncompleteSource:
    """缺少 list_symbols 方法的不完整实现"""

    def get_ohlcv(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame()


def test_complete_source_satisfies_protocol():
    """完整实现应满足 Protocol"""
    source = MockSource()
    assert isinstance(source, OHLCVSource)


def test_incomplete_source_fails_protocol():
    """不完整实现不应满足 Protocol"""
    source = IncompleteSource()
    assert not isinstance(source, OHLCVSource)


def test_protocol_methods_callable():
    """Protocol 方法应可调用"""
    source = MockSource()
    df = source.get_ohlcv("sh000001")
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns

    symbols = source.list_symbols("sh")
    assert isinstance(symbols, list)
    assert len(symbols) > 0
