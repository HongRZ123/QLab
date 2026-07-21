# templates/ - 研究模板

复制即用的研究脚本模板，覆盖最常见的量化研究场景。

## 模板列表

| 模板 | 用途 | 适合谁 |
|------|------|--------|
| `run_single_asset.py` | 单资产均值回归研究 | **新用户首选** |
| `run_pair_trade.py` | 配对交易（CADF + S9） | 研究两个标的的对冲关系 |
| `run_portfolio.py` | 多资产协整组合（Johansen + S7） | 研究 3+ 标的的组合策略 |
| `custom_strategy.py` | 自定义策略开发 | 开发新策略并接入注册表 |
| `research_workflow.py` | 完整研究工作流 | 系统性评估一个标的 |

## 怎么用

### 方式一：复制到根目录改参数

```bash
# 复制模板
cp templates/run_single_asset.py my_research.py

# 编辑 CONFIG 区域的参数
# (用你喜欢的编辑器打开 my_research.py)

# 运行
python my_research.py
```

### 方式二：直接在 templates 目录运行

```bash
python templates/run_single_asset.py
```

> 模板自带 `sys.path` 设置，在任何位置都能运行。

## 模板详解

### run_single_asset.py

最常用的模板。流程：

```
读数据 -> ADF/Hurst/半衰期检验 -> 跑策略 -> 回测 -> 绩效报告
```

只需改两个参数：
- `SYMBOL`: 标的代码（如 `sh512670`）
- `STRATEGY_NAME`: 策略名（`linear_mr` / `bollinger_mr` / `ma_crossover`）

### run_pair_trade.py

研究两个标的的配对交易。流程：

```
读两个标的数据 -> CADF 协整检验 -> S9 卡尔曼动态对冲 -> 绩效
```

需要改：
- `SYMBOL_Y`: 因变量标的
- `SYMBOL_X`: 自变量标的

### run_portfolio.py

研究多资产协整组合。流程：

```
读多个标的数据 -> Johansen 协整检验 -> S7 Walk-Forward 组合策略 -> 绩效
```

需要改：
- `SYMBOLS`: 标的列表（至少 2 个）

### custom_strategy.py

开发自定义策略的模板。包含：

1. 示例策略（RSI 均值回归）的完整实现
2. `run_validation()` 验证协议（正控 + 负控 + 不变式）
3. Registry 注册代码（注释状态，取消注释即生效）

开发新策略的步骤：
1. 复制此文件到 `strategies/` 目录
2. 替换策略函数
3. 实现验证协议
4. 取消注释注册代码
5. 运行 `python -m strategies.your_module` 验证

### research_workflow.py

最完整的模板，一次性跑完所有流程：

1. 数据加载 + 除权除息
2. 三项统计检验（ADF / Hurst / 半衰期）+ 评分
3. 所有已注册策略对比
4. Walk-Forward 滚动回测（S4 + S8）
5. 总结 + 下一步建议

## 常见问题

### 标的代码怎么写？

格式：`{市场}{代码}`

| 市场 | 前缀 | 示例 |
|------|------|------|
| 上海 | `sh` | `sh512670` (银行ETF) |
| 深圳 | `sz` | `sz159915` (芯片ETF) |
| 北京 | `bj` | - |

### 数据从哪来？

通达信盘后数据，默认路径 `D:\new_tdx64\vipdoc`。

修改路径：编辑 `data/fetcher.py` 中的 `TDX_ROOT`。

### 策略亏钱怎么办？

正常。说明：
1. 标的不适合均值回归（看统计检验评分）
2. 或者市场处于趋势阶段

换标的的方法：`python explore/scan_stationarity.py` 扫描全市场找合适的。
