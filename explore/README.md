# explore/ — 全市场扫描脚本

独立可执行扫描脚本。平稳性扫描已迁移到 `stats/scan.py`。

## 文件

| 脚本 | 用途 |
|------|------|
| `scan_ma_crossover.py` | 全市场 MA 金叉死叉扫描 |

> 平稳性扫描: `python stats/scan.py`

## scan_ma_crossover.py

对所有 A 股标的运行均线金叉死叉扫描，输出近期出现金叉的标的列表。

```bash
python explore/scan_ma_crossover.py
```

输出 `output/ma_crossover_scan.csv`。
