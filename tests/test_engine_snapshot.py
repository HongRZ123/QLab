"""
快照测试: run_backtest() 输出回归锁

在引擎拆分前捕获 run_backtest() 的精确输出,
作为后续重构的回归基准。每个场景断言
pnl, equity_curve, total_cost, n_trades, positions
与硬编码期望值匹配 (浮点容差 1e-6)。
"""

import pandas as pd
import pytest

from backtest import run_backtest


# ------------------------------------------------------------------ helpers
def _series_close(actual: pd.Series, expected: list[float], tol: float = 1e-6) -> bool:
    """逐元素浮点比较 (容差 tol)。"""
    if len(actual) != len(expected):
        return False
    return all(abs(float(actual.iloc[i]) - expected[i]) < tol for i in range(len(expected)))


def _series_eq_int(actual: pd.Series, expected: list[int]) -> bool:
    """整数序列精确比较。"""
    return actual.tolist() == expected


# ============================================================
# (a) 恒定价格 + 恒定持仓 → PnL = 0
# ============================================================
class TestConstantPriceConstantPosition:
    """价格不变时, 持有仓位不产生盈亏, 权益仅因首日买入成本下降。"""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5)
        p_const = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=dates)
        u_const = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)
        self.result = run_backtest(p_const, u_const, dynamic_sizing=False)

    def test_pnl_all_zero(self) -> None:
        assert _series_close(self.result["pnl"], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_equity_curve(self) -> None:
        # 首日无仓位 (T+1), t=1 买入扣成本 1250.0
        assert _series_close(
            self.result["equity_curve"],
            [1_000_000.0, 998_750.0, 998_750.0, 998_750.0, 998_750.0],
        )

    def test_total_cost(self) -> None:
        assert abs(self.result["total_cost"] - 1250.0) < 1e-6

    def test_n_trades(self) -> None:
        assert self.result["n_trades"] == 1

    def test_positions(self) -> None:
        # T+1: t=0 信号在 t=1 执行, t=0 持仓为 0
        assert _series_eq_int(
            self.result["positions"], [0, 100_000, 100_000, 100_000, 100_000]
        )


# ============================================================
# (b) 已知价格序列 + 已知信号
# ============================================================
class TestKnownPriceKnownSignal:
    """从 engine.py run_validation() 测试 2 复用的已知序列。"""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4)
        p_known = pd.Series([10.0, 11.0, 9.0, 12.0], index=dates)
        u_known = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates)
        self.result = run_backtest(
            p_known, u_known,
            initial_capital=1_000_000, dynamic_sizing=False, check_limits=False,
        )

    def test_pnl(self) -> None:
        # t=3: shares[2]=111100, pnl = 111100 * (12-9) = 333300
        assert _series_close(self.result["pnl"], [0.0, 0.0, 0.0, 333_300.0])

    def test_equity_curve(self) -> None:
        # t=2: 1_000_000 - 1249.875 (买入成本) = 998_750.125
        # t=3: 998_750.125 + 333_300 - 583.8 (卖出成本) = 1_331_466.325
        assert _series_close(
            self.result["equity_curve"],
            [1_000_000.0, 1_000_000.0, 998_750.125, 1_331_466.325],
        )

    def test_total_cost(self) -> None:
        # 买入成本 1249.875 + 卖出成本 583.8 = 1833.675
        assert abs(self.result["total_cost"] - 1833.675) < 1e-6

    def test_n_trades(self) -> None:
        assert self.result["n_trades"] == 2

    def test_positions(self) -> None:
        # t=2: round(1_000_000/9, 100)=111100;  t=3: round(1_000_000/12, 100)=83300
        assert _series_eq_int(self.result["positions"], [0, 0, 111_100, 83_300])


# ============================================================
# (c) 动态仓位 vs 固定仓位
# ============================================================
class TestDynamicVsFixedSizing:
    """验证 dynamic_sizing=True/False 的不同行为。"""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6)
        p = pd.Series([10.0, 10.0, 10.0, 20.0, 20.0, 21.0], index=dates)
        u = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=dates)
        self.res_dyn = run_backtest(
            p, u, initial_capital=1000, lot_size=1,
            commission_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0,
            dynamic_sizing=True, check_limits=False,
        )
        self.res_fix = run_backtest(
            p, u, initial_capital=1000, lot_size=1,
            commission_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0,
            dynamic_sizing=False, check_limits=False,
        )

    # --- 动态仓位 ---
    def test_dynamic_equity_curve(self) -> None:
        assert _series_close(
            self.res_dyn["equity_curve"],
            [1000.0, 1000.0, 995.0, 1990.0, 1985.0, 2079.0],
        )

    def test_dynamic_positions(self) -> None:
        assert _series_eq_int(self.res_dyn["positions"], [0, 0, 100, 49, 99, 94])

    def test_dynamic_n_trades(self) -> None:
        assert self.res_dyn["n_trades"] == 4

    def test_dynamic_total_cost(self) -> None:
        assert abs(self.res_dyn["total_cost"] - 20.0) < 1e-6

    # --- 固定仓位 ---
    def test_fixed_equity_curve(self) -> None:
        assert _series_close(
            self.res_fix["equity_curve"],
            [1000.0, 1000.0, 995.0, 1990.0, 1990.0, 2035.0],
        )

    def test_fixed_positions(self) -> None:
        assert _series_eq_int(self.res_fix["positions"], [0, 0, 100, 50, 50, 47])

    def test_fixed_n_trades(self) -> None:
        assert self.res_fix["n_trades"] == 3

    def test_fixed_total_cost(self) -> None:
        assert abs(self.res_fix["total_cost"] - 15.0) < 1e-6

    # --- 对比 ---
    def test_dynamic_beats_fixed(self) -> None:
        """复利效应: 动态最终权益 > 固定最终权益。"""
        dyn_final = float(self.res_dyn["equity_curve"].iloc[-1])
        fix_final = float(self.res_fix["equity_curve"].iloc[-1])
        assert abs(dyn_final - 2079.0) < 1e-6
        assert abs(fix_final - 2035.0) < 1e-6
        assert dyn_final > fix_final


# ============================================================
# (d) 手数取整
# ============================================================
class TestLotSizeRounding:
    """所有持仓股数必须是 lot_size 的整数倍。"""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        p_lot = pd.Series([10.0, 10.5, 10.3], index=dates)
        u_lot = pd.Series([0.0, 1.5, 0.8], index=dates)
        self.result = run_backtest(
            p_lot, u_lot, lot_size=100, initial_capital=500_000, dynamic_sizing=False,
        )

    def test_all_positions_are_lot_multiples(self) -> None:
        for pos in self.result["positions"]:
            assert int(pos) % 100 == 0, f"持仓 {pos} 不是 100 的倍数"

    def test_positions(self) -> None:
        # t=2: round(1.5 * 500_000 / 10.3, 100) = 72800
        assert _series_eq_int(self.result["positions"], [0, 0, 72_800])

    def test_equity_curve(self) -> None:
        assert _series_close(
            self.result["equity_curve"], [500_000.0, 500_000.0, 499_062.7],
        )

    def test_total_cost(self) -> None:
        assert abs(self.result["total_cost"] - 937.3) < 1e-6

    def test_n_trades(self) -> None:
        assert self.result["n_trades"] == 1


# ============================================================
# (e) 涨停阻断买入
# ============================================================
class TestLimitUpBlocksBuy:
    """主板涨停 (+10%) 日不可买入, 持仓保持为 0。"""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4)
        # 10→10→11(+10%=涨停)→11
        p_limit = pd.Series([10.0, 10.0, 11.0, 11.0], index=dates)
        u_limit = pd.Series([0.0, 1.0, 0.0, 0.0], index=dates)
        self.result = run_backtest(
            p_limit, u_limit,
            initial_capital=1_000_000,
            dynamic_sizing=False, board="main", check_limits=True,
        )

    def test_positions_all_zero(self) -> None:
        """涨停日 (t=2) 买入被阻断, 全部持仓为 0。"""
        assert _series_eq_int(self.result["positions"], [0, 0, 0, 0])

    def test_pnl_all_zero(self) -> None:
        assert _series_close(self.result["pnl"], [0.0, 0.0, 0.0, 0.0])

    def test_equity_curve(self) -> None:
        assert _series_close(
            self.result["equity_curve"],
            [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        )

    def test_total_cost(self) -> None:
        assert abs(self.result["total_cost"] - 0.0) < 1e-6

    def test_n_trades(self) -> None:
        assert self.result["n_trades"] == 0


# ============================================================
# (f) 停牌阻断交易
# ============================================================
class TestSuspensionBlocksTrade:
    """停牌日 (volume=0) 不执行交易, 沿用前一日持仓。"""

    @pytest.fixture(autouse=True)
    def _run(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5)
        p_susp = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=dates)
        u_susp = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=dates)
        self.price_data = pd.DataFrame(
            {
                "open": [10.0] * 5,
                "high": [10.0] * 5,
                "low": [10.0] * 5,
                "close": [10.0] * 5,
                "volume": [1000, 1000, 0, 1000, 1000],  # t=2 停牌
            },
            index=dates,
        )
        self.result = run_backtest(
            p_susp, u_susp,
            initial_capital=1_000_000, dynamic_sizing=False,
            check_limits=False, check_suspension=True,
            price_data=self.price_data,
        )

    def test_suspension_day_no_buy(self) -> None:
        """t=2 停牌日 volume=0, 买入被阻断, 持仓保持 0。"""
        assert int(self.result["positions"].iloc[2]) == 0

    def test_resumption_day_buy(self) -> None:
        """t=3 复牌后正常买入。"""
        assert int(self.result["positions"].iloc[3]) > 0

    def test_positions(self) -> None:
        assert _series_eq_int(
            self.result["positions"], [0, 0, 0, 100_000, 0],
        )

    def test_equity_curve(self) -> None:
        assert _series_close(
            self.result["equity_curve"],
            [1_000_000.0, 1_000_000.0, 1_000_000.0, 998_750.0, 997_000.0],
        )

    def test_total_cost(self) -> None:
        # 买入成本 + 卖出成本 = 1250 + 1750 = 3000
        assert abs(self.result["total_cost"] - 3000.0) < 1e-6

    def test_n_trades(self) -> None:
        assert self.result["n_trades"] == 2
