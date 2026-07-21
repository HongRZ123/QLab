# experiments — 独立实验脚本

独立研究实验，**不依赖主项目模块**（不 import data/、tests/、strategies/、backtest/）。每个脚本自包含数据加载和计算逻辑。

## 依赖关系

```
experiments/
├── 无内部依赖（完全独立于主项目）
└── 外部依赖: numpy, pandas, statsmodels, matplotlib (部分)
```

## 文件结构

```
experiments/
├── kalman_mu_test.py        # 卡尔曼 μ 估计基础测试
├── kalman_mu_v2.py          # 卡尔曼 μ 估计 v2 改进
├── kalman_mu_returns.py     # 基于收益率的卡尔曼 μ 估计
├── kalman_mu_longshort.py   # 卡尔曼 μ 多空信号实验
├── kalman_mu_futures.py     # 卡尔曼 μ 期货数据实验
├── kalman_mu_walkforward.py # 卡尔曼 μ Walk-Forward 验证
├── chan_s4_futures.py       # Chan S4 策略期货适配
├── hurst_switch.py          # Hurst 指数体制切换实验
├── trend_mr_composite.py    # 趋势+均值回归复合策略
└── trend_mr_portfolio.py    # 趋势+均值回归组合策略
```

---

## 卡尔曼 μ 系列 (6 个)

研究均值回归速度参数 μ（= 1 - φ，φ 为 AR(1) 系数）的卡尔曼滤波在线估计。

| 脚本 | 内容 | 数据源 |
|------|------|--------|
| `kalman_mu_test.py` | 基础框架验证：固定 μ vs 卡尔曼估计 μ | 模拟数据 / ETF |
| `kalman_mu_v2.py` | 改进：自适应噪声参数、收敛速度优化 | ETF 日线 |
| `kalman_mu_returns.py` | 用收益率（非价格）估计 μ，消除趋势干扰 | ETF 日线 |
| `kalman_mu_longshort.py` | 基于动态 μ 的多空信号生成 | ETF 日线 |
| `kalman_mu_futures.py` | 期货市场 μ 估计（含做空） | 期货日线 |
| `kalman_mu_walkforward.py` | Walk-Forward 验证动态 μ 策略的样本外表现 | ETF 日线 |

### 核心公式

```
AR(1): r_t = μ·θ + (1-μ)·r_{t-1} + ε_t
卡尔曼状态: μ_t = μ_{t-1} + η_t
观测: r_t = μ_t·(θ - r_{t-1}) + r_{t-1} + ε_t
```

### 运行

```bash
python experiments/kalman_mu_test.py
python experiments/kalman_mu_v2.py
# ... 每个脚本独立运行
```

---

## chan_s4_futures.py — Chan S4 期货适配

将 Chan 教程 S4 线性均值回归策略适配到期货市场（含做空、保证金、合约乘数）。

- 数据源：期货日线（TDX 格式或 CSV）
- 与主项目 `strategies.linear_mr` 的区别：期货版含做空逻辑和保证金计算

---

## hurst_switch.py — Hurst 体制切换

研究 Hurst 指数随时间的变化，检测趋势/均值回归体制切换。

- 滚动窗口计算 Hurst 指数
- 当 H 从 < 0.5 切换到 > 0.5（或反之）时触发策略切换
- 输出：H 时间序列 + 体制标注

---

## trend_mr_composite.py — 趋势+均值回归复合

将趋势跟踪（动量）和均值回归信号复合：

```
signal = α · trend_signal + (1-α) · mr_signal
```

- 根据 Hurst 指数动态调整 α
- H > 0.5 → 偏趋势，H < 0.5 → 偏均值回归

---

## trend_mr_portfolio.py — 趋势+均值回归组合

多资产版本的趋势+均值回归复合策略：

- 对每个资产分别计算复合信号
- 按波动率倒数加权合成组合
- 含再平衡逻辑

---

## 注意事项

- 所有脚本**独立运行**，不 import 主项目任何模块
- 数据路径硬编码在各脚本内（`D:\new_tdx64\vipdoc\` 或 `D:\new_tdx64\` 下期货数据）
- 部分脚本使用 matplotlib 绘图，需要图形环境
- 实验结论记录在各脚本顶部 docstring 中
- 这些脚本是**研究草稿**，不保证代码质量与主项目一致
