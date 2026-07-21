"""
strategies/experimental — 实验策略子模块
========================================

存放半完成 / 正在探索中的策略。

约定:
    - 每个文件导出一个可调用的策略函数 (接受 prices, 返回 dict 含 "num_units")
    - 实验策略不强制通过 run_validation() 协议
    - 不在此处注册到 strategies.registry — 由使用方按需导入
"""
