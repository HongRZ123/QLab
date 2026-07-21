# QLab 数据模块 (data/)

## 概述

本模块负责从**通达信盘后数据包**读取A股日线数据，并提供A股交易规则约束函数。

不依赖任何在线API，直接读取本地二进制文件。

---

## 数据源

| 项目 | 说明 |
|------|------|
| 软件 | 通达信（盘后数据下载） |
| 路径 | 环境变量 `QLAB_TDX_ROOT`（默认 `D:\new_tdx64\vipdoc\`） |
| 格式 | `.day` 二进制文件，每条记录 32 字节 |
| 市场 | `sh/`（上海）、`sz/`（深圳）、`bj/`（北京） |
| 周期 | 日线（`lday/` 目录） |

### 目录结构

```
D:\new_tdx64\vipdoc\
├── sh\lday\          # 上海日线
│   ├── sh000001.day  # 上证指数
│   ├── sh600000.day  # 浦发银行
│   ├── sh510050.day  # 50ETF
│   └── ...
├── sz\lday\          # 深圳日线
│   ├── sz000001.day  # 平安银行
│   ├── sz002594.day  # 比亚迪
│   ├── sz159915.day  # 创业板ETF
│   └── ...
└── bj\               # 北京交易所
```

### 二进制格式（32字节/条）

```
偏移  类型      字段      说明
0     uint32    date      日期 YYYYMMDD
4     uint32    open      开盘价 × 100
8     uint32    high      最高价 × 100
12    uint32    low       最低价 × 100
16    uint32    close     收盘价 × 100
20    float32   amount    成交额（元）
24    uint32    volume    成交量（股）
28    uint32    reserved  保留（忽略）
```

---

## 使用方法

### 1. 读取单只标的

```python
from data.fetcher import read_day

# 读取浦发银行全部日线
df = read_day("sh600000")

# 读取指定日期范围
df = read_day("sz000001", start="20200101", end="20241231")

# 也支持不带市场前缀（自动推断）
df = read_day("600000")   # 6开头 → 上海
df = read_day("000001")   # 0开头 → 深圳
```

返回的 DataFrame：

```
            open   high    low  close        amount     volume
date
2024-01-02  9.45   9.52   9.40   9.48  1.23e+09   130000000
2024-01-03  9.47   9.55   9.43   9.51  1.15e+09   121000000
...
```

- 索引：`date`（datetime64）
- 价格：float64，单位**元**（已除以100）
- 成交额：float64，单位**元**
- 成交量：int64，单位**股**

### 2. 批量读取

```python
from data.fetcher import read_symbols

dfs = read_symbols(["sh510050", "sh510300", "sz159915"])
# 返回 dict: {"sh510050": DataFrame, "sh510300": DataFrame, ...}
```

### 3. 列出可用代码

```python
from data.fetcher import list_symbols

# 列出上海所有标的
all_sh = list_symbols("sh")

# 筛选ETF（51开头）
etfs = [s for s in all_sh if s[2:].startswith("51")]

# 筛选可转债（113/118开头）
bonds = [s for s in all_sh if s[2:].startswith(("113", "118"))]
```

### 4. 交易规则

```python
from data.rules import round_to_lot, transaction_cost, next_trade_date

# 整数手
shares = round_to_lot(1234)  # → 1200

# 交易成本
cost = transaction_cost(100000, "buy")   # 买入: 25元佣金
cost = transaction_cost(100000, "sell")  # 卖出: 25元佣金 + 50元印花税

# T+1: 信号日的下一个交易日
exec_date = next_trade_date(signal_date, trade_dates)
```

### 5. 除权除息检测与价格调整

通达信 `.day` 数据为**不复权**价格，分红除权会导致价格跳空，影响 Z-Score 和收益率计算。

```python
from data.dividend import detect_ex_dividend, adjust_close_prices, filter_ex_dividend_returns

df = read_day("sh510050")

# 检测除权除息日 (隔夜跳空 < -3%)
mask = detect_ex_dividend(df["close"], df["open"], threshold=-0.03)

# 后复权调整: 消除除权跳空, 使价格序列连续
close_adj = adjust_close_prices(df["close"], df["open"], mask)

# 将除权日收益率置零 (避免虚假信号)
ret = close_adj.pct_change().fillna(0.0)
ret_clean = filter_ex_dividend_returns(ret, mask)
```

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `detect_ex_dividend` | close, open, threshold | `pd.Series[bool]` | 隔夜跳空 < threshold → 除权日 |
| `adjust_close_prices` | close, open, mask | `pd.Series[float]` | 后复权, 向前累乘调整因子 |
| `filter_ex_dividend_returns` | returns, mask | `pd.Series[float]` | 除权日收益率置零 |

---

## 代码段识别

| 代码前缀 | 市场 | 类型 |
|---------|------|------|
| `sh000xxx` | 上海 | 指数 |
| `sh600xxx` / `sh601xxx` / `sh603xxx` / `sh605xxx` | 上海 | 主板股票 |
| `sh688xxx` | 上海 | 科创板 |
| `sh510xxx` ~ `sh518xxx` | 上海 | ETF |
| `sh588xxx` | 上海 | 科创板ETF |
| `sh110xxx` / `sh113xxx` / `sh118xxx` | 上海 | 可转债 |
| `sz000xxx` / `sz001xxx` | 深圳 | 主板股票 |
| `sz002xxx` / `sz003xxx` | 深圳 | 中小板 |
| `sz300xxx` / `sz301xxx` | 深圳 | 创业板 |
| `sz159xxx` | 深圳 | ETF |
| `sz12xxxx` | 深圳 | 可转债 |

---

## 注意事项

1. **数据更新**：通达信盘后下载后，`.day` 文件自动追加新数据。重新运行 `read_day()` 即可获取最新数据。

2. **复权问题**：通达信 `.day` 文件存储的是**不复权**价格。如需前复权/后复权，需额外处理（本模块暂不处理，后续按需添加）。

3. **停牌处理**：停牌日无记录（文件中不存在该日期的条目），读取后 DataFrame 自然缺失该日。

4. **成交量单位**：
   - 股票：单位为**股**
   - 指数：单位为**手**（1手=100股），使用时注意区分

5. **路径配置**：设置环境变量 `QLAB_TDX_ROOT` 指向通达信 `vipdoc` 目录。未设置时默认 `D:\new_tdx64\vipdoc`。

---

## 文件清单

```
data/
├── __init__.py     # 模块入口, 导出常用函数
├── fetcher.py      # 通达信 .day 文件读取
├── rules.py        # A股交易规则 (T+1, 整数手, 成本, 涨跌停)
├── dividend.py     # 除权除息检测与后复权价格调整
└── README.md       # 本文件
```

---

## 依赖

```
numpy
pandas
```

无需安装 akshare、tushare 等在线数据源。
