# strategies/Tech -- 技术分析策略

基于技术指标（均线、RSI、量价分析等）生成交易决策。与均值回归策略不同，技术分析策略不依赖统计假设（平稳性、协整等），而是直接从量价数据中提取交易信号。

## 策略清单

| 策略 | 模块 | 信号来源 | num_units | 注册表 |
|------|------|----------|-----------|--------|
| MA 均线交叉 | `ma_crossover.py` | 价格序列（内联 SMA） | {0, 1} | ✅ `ma_crossover` |

---

## MA Crossover - 均线金叉死叉 `ma_crossover.py`

### 含义

最简单的趋势跟踪策略。短期均线上穿长期均线（金叉）买入，下穿（死叉）卖出。在 QLab 中主要用于管道测试和作为均值回归策略的基准对照。

### 公式

```
SMA_short(t) = mean(prices[t-short_window+1:t+1])
SMA_long(t)  = mean(prices[t-long_window+1:t+1])

SMA_short 上穿 SMA_long -> num_units = 1 (金叉买入)
SMA_short 下穿 SMA_long -> num_units = 0 (死叉卖出)
num_units ∈ {0, 1}
```

### 接口

```python
from strategies.Tech.ma_crossover import ma_crossover

result = ma_crossover(
    prices: pd.Series,        # 日价格序列
    short_window: int = 5,    # 短期均线窗口
    long_window: int = 20,    # 长期均线窗口, 必须 > short_window
) -> dict
```

### 返回

| 键 | 类型 | 说明 |
|----|------|------|
| `sma_short` | pd.Series | 短期均线 |
| `sma_long` | pd.Series | 长期均线 |
| `num_units` | pd.Series | 持仓 ∈ {0, 1} |
| `signals` | pd.Series | 交叉信号 (1=金叉, -1=死叉, 0=无) |
| `pnl` | pd.Series | 理论每日盈亏 |
| `ret` | pd.Series | 理论每日收益率 |
| `n_trades` | int | 往返交易次数 |

### 用法

```python
from data import read_day
from strategies.Tech.ma_crossover import ma_crossover
from backtest import run_backtest, performance_summary

df = read_day("sh512670")
result = ma_crossover(df["close"], short_window=5, long_window=20)
bt = run_backtest(df["close"], result["num_units"])
stats = performance_summary(bt["ret"])
```

### 验证

```bash
python -m strategies.Tech.ma_crossover    # 趋势序列交叉 + 恒定价格无交叉
```
