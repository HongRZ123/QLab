"""
QLab 数据模块
=============
读取通达信盘后数据，提供A股交易规则约束。

用法:
    from data.fetcher import read_day, read_symbols
    from data.rules import round_to_lot, transaction_cost
    from data.dividend import detect_ex_dividend, adjust_close_prices
"""

from data.dividend import adjust_close_prices as adjust_close_prices
from data.dividend import detect_ex_dividend as detect_ex_dividend
from data.dividend import filter_ex_dividend_returns as filter_ex_dividend_returns
from data.fetcher import list_symbols as list_symbols
from data.fetcher import read_day as read_day
from data.fetcher import read_symbols as read_symbols
from data.interface import OHLCVSource as OHLCVSource
from data.rules import next_trade_date as next_trade_date
from data.rules import round_to_lot as round_to_lot
from data.rules import transaction_cost as transaction_cost
from data.sources.tdx import TDXSource as TDXSource
