"""TDXSource 单元测试"""
import pandas as pd
import pytest

from data import fetcher
from data.interface import OHLCVSource
from data.sources.tdx import TDXSource

# 检查 TDX 数据是否可用
TDX_AVAILABLE = fetcher.TDX_ROOT.exists()


@pytest.mark.skipif(not TDX_AVAILABLE, reason="TDX data not available")
def test_tdx_source_satisfies_protocol():
    """TDXSource 应满足 OHLCVSource Protocol"""
    source = TDXSource()
    assert isinstance(source, OHLCVSource)


@pytest.mark.skipif(not TDX_AVAILABLE, reason="TDX data not available")
def test_get_ohlcv_returns_dataframe():
    """get_ohlcv 应返回包含 close 列的 DataFrame"""
    source = TDXSource()
    df = source.get_ohlcv("sh000001")
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert len(df) > 0


@pytest.mark.skipif(not TDX_AVAILABLE, reason="TDX data not available")
def test_list_symbols_returns_list():
    """list_symbols 应返回非空列表"""
    source = TDXSource()
    symbols = source.list_symbols("sh")
    assert isinstance(symbols, list)
    assert len(symbols) > 0


def test_tdx_source_without_data():
    """即使没有 TDX 数据，TDXSource 也应满足 Protocol"""
    source = TDXSource()
    assert isinstance(source, OHLCVSource)
