# strategies - 量化策略模块

按 alpha 寻找视角分类的策略集合。所有策略为**纯函数**，输入价格序列（或信号），输出含 `num_units` 的 dict。

> **重要**: 策略内置的 `pnl`/`ret` 为**理论值**（无成本、无 T+1、无手数取整）。生产回测请使用 `backtest.run_backtest()`。

## 目录结构

```
strategies/
├── MR/                     # 均值回归策略（价格偏离均值后回归）
│   ├── s4_linear.py            # S4: 线性均值回归 (单资产)
│   ├── s7_linear_portfolio.py  # S7: 线性均值回归 (组合)
│   ├── s8_bollinger.py         # S8: 布林带均值回归
│   └── s9_kalman_hedge.py      # S9: 卡尔曼滤波动态对冲
├── MM/                     # 做市策略（提供流动性赚价差）
│   └── s10_kalman_mm.py        # S10: 卡尔曼做市模型
├── Tech/                   # 技术分析策略（均线、量价等）
│   ├── ma_crossover.py         # MA 金叉死叉
│   ├── vpa_trend.py            # VPA 量价确认趋势跟踪
│   ├── vpa_reversal.py         # VPA 反转形态
│   └── vpa_breakout.py         # VPA 放量突破
├── experimental/           # 实验策略（草稿，不保证稳定性）
│   ├── s11_rsi_draft.py        # RSI 均值回归草案
│   └── s12_vpa_draft.py        # VPA 量价分析草案
├── registry.py             # 策略注册表
└── __init__.py             # 公开 API
```

## 分类原则

strategies/ 的子目录按 **alpha 寻找视角** 分类，与 signals/ 按**技术概念**组织不同：

| strategies/ 子目录 | alpha 类型 | 含义 |
|---------------------|-----------|------|
| `MR/` | 均值回归 | 价格偏离均衡后回归 |
| `MM/` | 做市 | 提供流动性赚价差 |
| `Tech/` | 技术分析 | 量价指标驱动 |

一个 `signals/kalman` 信号可以被 `strategies/MR/` 的策略消费（做均值回归），也可以被 `strategies/MM/` 的策略消费（做市）。信号分类和策略分类是两个正交维度。

## 公开 API

```python
from strategies import (
    linear_mr,           # S4  -> strategies.MR.s4_linear
    linear_portfolio,    # S7  -> strategies.MR.s7_linear_portfolio
    bollinger_mr,        # S8  -> strategies.MR.s8_bollinger
    bollinger_portfolio, # S8 组合封装
    kalman_hedge,        # S9  -> strategies.MR.s9_kalman_hedge
    kalman_mm,           # S10 -> strategies.MM.s10_kalman_mm
    ma_crossover,        # MA  -> strategies.Tech.ma_crossover
)

# 通过注册表
from strategies.registry import list_names
print(list_names())
# 端到端运行: python run/run_linear_mr.py
```

## 各子模块详细文档

- [MR/README.md](MR/README.md) -- 均值回归策略（S4, S7, S8, S9）
- [MM/README.md](MM/README.md) -- 做市策略（S10）
- [Tech/README.md](Tech/README.md) -- 技术分析策略（MA Crossover, VPA）

## 策略清单

| 策略 | 模块 | 类型 | 信号来源 | num_units | 注册表 |
|------|------|------|----------|-----------|--------|
| S4 线性均值回归 | `strategies.MR.s4_linear` | 单资产 | 价格序列 | ≥ 0 (连续) | ✅ `linear_mr` |
| S7 线性组合 MR | `strategies.MR.s7_linear_portfolio` | 多资产 | 价格矩阵 | ≥ 0 (连续) | ❌ |
| S8 布林带 MR | `strategies.MR.s8_bollinger` | 单资产 | 价格序列 | {0, 1} | ✅ `bollinger_mr` |
| S8 布林带组合 | `strategies.MR.s8_bollinger` | 组合 | 组合净值 | {0, 1} | ❌ |
| S9 卡尔曼对冲 | `strategies.MR.s9_kalman_hedge` | 配对 | `signals.kalman` | {0, 1} | ❌ |
| S10 卡尔曼做市 | `strategies.MM.s10_kalman_mm` | 做市 | 价格+成交量 | 无信号 | ❌ |
| MA Crossover | `strategies.Tech.ma_crossover` | 单资产 | 价格序列 | {0, 1} | ✅ `ma_crossover` |
| VPA 趋势跟踪 | `strategies.Tech.vpa_trend` | 单资产 | `signals.vpa` + `signals.trend` | {0, 0.5, 1} | ✅ `vpa_trend` |
| VPA 反转形态 | `strategies.Tech.vpa_reversal` | 单资产 | `signals.vpa` | {0, 1} | ✅ `vpa_reversal` |
| VPA 放量突破 | `strategies.Tech.vpa_breakout` | 单资产 | `signals.pivot` | {0, 1} | ✅ `vpa_breakout` |
| RSI 草案 | `strategies.experimental.s11_rsi_draft` | 单资产 | 价格序列 | {0, 1} | ❌ (实验) |
| VPA 草案 | `strategies.experimental.s12_vpa_draft` | 单资产 | `signals.vpa` | {0, 1} | ❌ (实验) |

## 依赖关系

```
signals.kalman             ──-> strategies.MR.s9_kalman_hedge
signals.vpa                ──-> strategies.Tech.vpa_trend
signals.vpa                ──-> strategies.Tech.vpa_reversal
signals.vpa                ──-> strategies.experimental.s12_vpa_draft
signals.pivot              ──-> strategies.Tech.vpa_breakout

tests.s3_half_life  ──-> strategies.MR.s4_linear, s7_linear_portfolio, s8_bollinger (半衰期 -> lookback)
tests.s6_johansen   ──-> strategies.MR.s7_linear_portfolio (验证协议中的协整序列生成)
```

## 理论 PnL vs 生产回测

| | 策略内置 PnL | backtest 引擎 PnL |
|---|---|---|
| T+1 执行 | ❌ | ✅ |
| 手数取整 | ❌ 连续仓位 | ✅ round_to_lot |
| 佣金/印花税/滑点 | ❌ | ✅ |
| 用途 | 验证协议 (正控/负控) | 生产回测 |

**正确用法**: 策略输出 `num_units` -> 传给 `backtest.run_backtest()` 做含成本的回测。

## 验证协议

```bash
python -m strategies.MR.s4_linear              # OU 正控 + GBM 负控
python -m strategies.MR.s7_linear_portfolio     # 协整组合正控
python -m strategies.MR.s8_bollinger           # OU 正控 + GBM 负控
python -m strategies.MR.s9_kalman_hedge        # 线性关系 β 收敛 + 独立 GBM 负控
python -m strategies.MM.s10_kalman_mm          # 恒定价格收敛 + 大单 K=1 + 小单 K≈0
python -m strategies.Tech.ma_crossover         # 趋势序列交叉 + 恒定价格无交叉
python -m strategies.Tech.vpa_trend            # 量价确认正控 + 随机游走负控
python -m strategies.Tech.vpa_reversal         # 锤头线买入 + 随机游走稀疏信号
python -m strategies.Tech.vpa_breakout         # 放量突破买入 + 缩量突破无信号
```
