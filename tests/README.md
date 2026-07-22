# tests/ — 单元测试

pytest 测试用例。

> 统计函数 (ADF/Hurst/CADF/Johansen) 在 `signals/stats.py` 和 `signals/stats_cointegration.py` 中。
> `from signals.stats import run_adf, hurst_exponent, estimate_half_life`

## 测试文件

| 文件 | 测试内容 |
|------|----------|
| `test_vpa.py` | VPA 信号 (17 tests) |
| `test_vpa_strategies.py` | VPA 策略输出 |
| `test_vpa_strategy.py` | VPA draft 策略 |
| `test_pivot.py` | 价格结构信号 |
| `test_trend.py` | 趋势健康度 |
| `test_kalman_spread.py` | 卡尔曼 spread |
| `test_engine_snapshot.py` | 回测引擎快照 (6 场景) |
| `test_backtest_core.py` | run_core vs run_backtest |
| `test_constraints.py` | 约束 + 成本模型 |
| `test_data_interface.py` | OHLCVSource 协议 |
| `test_tdx_source.py` | TDXSource |

## 运行

```bash
python -m pytest tests/ -q     # 109 passed
```
