# explore — 数据探索与策略运行脚本

独立可执行脚本，用于 ETF 平稳性扫描和端到端策略回测。

## 依赖关系

```
explore/
├── 内部依赖: data (read_day, read_symbols, list_symbols, detect_ex_dividend, adjust_close_prices)
├── 内部依赖: tests (run_adf, hurst_exponent, estimate_half_life)
├── 内部依赖: strategies (kalman_hedge; linear_mr/bollinger_mr/linear_portfolio 由 walk_forward 内部调用)
├── 内部依赖: backtest (run_backtest, performance_summary, walk_forward_*)
└── 外部依赖: numpy, pandas, statsmodels
```

```
data ──→ explore/scan_stationarity.py (加载价格 → 运行检验)
data + tests + strategies + backtest ──→ explore/run_strategy.py (端到端)
```

## 文件结构

```
explore/
├── scan_stationarity.py     # 38只ETF平稳性扫描
├── scan_ma_crossover.py     # 全市场 MA 金叉死叉扫描
└── run_strategy.py          # 端到端 walk-forward 回测
```

---

## scan_stationarity.py — ETF 平稳性扫描

### 用途

对 38 只 A 股 ETF 运行多层平稳性检验，筛选适合均值回归策略的标的。

### 运行方式

```bash
cd D:\EtLab\QLab
python explore/scan_stationarity.py
```

### 输出

控制台打印三层检验结果表：

**第一层 — 单资产检验**:

| 检验 | 函数 | 通过条件 |
|------|------|----------|
| S1 ADF | `tests.run_adf` | p-value < 0.05 |
| S2 Hurst | `tests.hurst_exponent` | H < 0.5 |
| S3 Half-life | `tests.estimate_half_life` | HL < 60 且 HL > 0 |

**第二层 — 滚动窗口稳健性**:

| 检验 | 函数 | 通过条件 |
|------|------|----------|
| 滚动 ADF | `tests.run_adf` | 窗口内 p-value < 0.05 比例 |
| 滚动 Hurst | `tests.hurst_exponent` | 窗口内 H < 0.5 比例 |
| 滚动 Half-life | `tests.estimate_half_life` | 窗口内 HL 稳定 |

**第三层 — 多重检验校正**:

- Holm-Bonferroni 校正，控制族错误率 (FWER) < 0.05

### 标的列表

38 只 ETF，覆盖宽基、行业、主题、跨境、商品等类别。具体列表见脚本内 `SYMBOLS` 常量。

### 依赖

- `data.read_day`: 加载日线数据
- `data.list_symbols`: 列出可用标的
- `tests.run_adf`, `tests.hurst_exponent`, `tests.estimate_half_life`: 单资产检验

---

## scan_ma_crossover.py — 全市场 MA 交叉扫描

### 用途

对所有 A 股标的运行均线金叉死叉扫描，输出近期出现金叉的标的列表。

### 运行方式

```bash
cd D:\EtLab\QLab
python explore/scan_ma_crossover.py
```

### 输出

- 控制台打印扫描进度与汇总统计
- `output/ma_crossover_scan.csv`: 全市场扫描结果

### 依赖

- `data.list_symbols`, `data.read_day`: 加载全市场日线数据
- `strategies.Tech.ma_crossover`: 生成金叉/死叉信号

---

## run_strategy.py — 端到端策略回测

### 用途

对指定 ETF 对（默认 sh512670/sh512760）运行完整的 Walk-Forward 回测流程。

### 运行方式

```bash
cd D:\EtLab\QLab
python explore/run_strategy.py
```

### 流程

```
1. 加载数据 (data.read_day) + 除权除息调整 (data.dividend)
2. 平稳性预检 (tests.run_adf, tests.hurst_exponent)
3. Walk-Forward 滚动重估 + 样本外信号
   ├── walk_forward_linear_mr (S4)
   ├── walk_forward_bollinger (S8)
   └── walk_forward_portfolio (S7)
4. 回测 (backtest.run_backtest) + 绩效汇总 (backtest.performance_summary)
5. 卡尔曼对冲 (strategies.kalman_hedge, S9)
6. 输出 CSV + 打印结果
```

### 输出

控制台打印：
- 各策略的样本外 Sharpe、年化收益、最大回撤
- 最优参数组合
- 交易次数

### 默认标的

- `sh512670` — 银行ETF
- `sh512760` — 芯片ETF

### 依赖

- `data`: 数据加载
- `tests`: 预检验
- `strategies`: 信号生成（由 walk_forward 内部调用）
- `backtest`: 回测引擎 + Walk-Forward + 绩效指标

## 注意事项

- 两个脚本均为**独立入口**，直接 `python explore/xxx.py` 运行
- 需要 `D:\new_tdx64\vipdoc\` 下有对应标的的 TDX 日线数据
- 扫描结果依赖数据时间范围，不同时期结论可能不同
- `run_strategy.py` 的 Walk-Forward 耗时较长（滚动重估），38 只 ETF 全扫描约需数分钟
