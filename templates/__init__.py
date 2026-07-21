"""
templates 包 - 研究模板集合
============================

每个模板都是一个独立可运行的脚本，复制到项目根目录改参数即可使用。

模板列表:
    run_single_asset.py  - 单资产均值回归研究（最常用）
    run_pair_trade.py    - 配对交易研究（CADF + S9）
    run_portfolio.py     - 多资产协整组合研究（Johansen + S7）
    custom_strategy.py   - 自定义策略开发模板
    research_workflow.py - 完整研究工作流（检验 -> 策略 -> 回测 -> 对比）
"""
