import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.execution.stress_testing import (
    CostSensitivityScenario,
    generate_cost_sensitivity_grid,
    evaluate_scenario_on_trades,
    run_cost_sensitivity_grid,
    load_baseline_execution_population,
)


class TestExecutionCostSensitivity(unittest.TestCase):
    """
    Test suite for Step 2 Cost Sensitivity Grid infrastructure.
    """

    @classmethod
    def setUpClass(cls):
        # Load baseline STRAT_09 trades and setup count
        cls.trades_df, cls.total_setups = load_baseline_execution_population()

    def test_cartesian_grid_count_is_exact(self):
        """1. Cartesian grid count is exact: 5 spreads x 4 slippages x 3 commissions = 60 scenarios."""
        scenarios = generate_cost_sensitivity_grid(
            spread_levels=(0.00, 0.05, 0.10, 0.20, 0.30),
            slippage_levels=(0.00, 0.02, 0.05, 0.10),
            commission_levels=(0.00, 0.02, 0.05),
        )
        self.assertEqual(len(scenarios), 60)

    def test_scenario_ids_are_unique(self):
        """2. Scenario IDs are unique across the entire Cartesian grid."""
        scenarios = generate_cost_sensitivity_grid()
        ids = [sc.scenario_id for sc in scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        # Verify first and last scenario format
        self.assertEqual(scenarios[0].scenario_id, "COST_S000_SL000_C000")
        self.assertEqual(scenarios[-1].scenario_id, "COST_S030_SL010_C005")

    def test_zero_cost_baseline_reproduction(self):
        """
        3. Zero-cost scenario exactly reproduces the validated baseline:
        gross_net_r == net_r, zero costs, baseline max drawdown, consecutive losses.
        """
        zero_sc = CostSensitivityScenario(
            scenario_id="ZERO_COST",
            spread_usd=0.0,
            entry_slippage_usd=0.0,
            exit_slippage_usd=0.0,
            commission_usd=0.0,
            commission_r_fixed=0.0,
        )
        res = evaluate_scenario_on_trades(zero_sc, self.trades_df, self.total_setups)

        # Baseline targets
        self.assertEqual(res["total_setups"], 629)
        self.assertEqual(res["filled_trades"], 398)
        self.assertAlmostEqual(res["gross_net_r"], 107.6593, places=3)
        self.assertAlmostEqual(res["net_r"], 107.6593, places=3)
        self.assertEqual(res["total_execution_cost_r"], 0.0)
        self.assertEqual(res["total_spread_cost_r"], 0.0)
        self.assertEqual(res["total_slippage_cost_r"], 0.0)
        self.assertEqual(res["total_commission_cost_r"], 0.0)
        self.assertEqual(res["baseline_relative_net_r"], 0.0)

        # Expectancy and EV
        self.assertAlmostEqual(res["gross_expectancy_per_filled_trade"], 0.2705, places=4)
        self.assertAlmostEqual(res["net_expectancy_per_filled_trade"], 0.2705, places=4)
        self.assertAlmostEqual(res["candidate_ev_net_r_per_setup"], 0.1712, places=4)

        # Risk metrics match baseline execution report
        self.assertAlmostEqual(res["net_win_rate"], 0.6357, places=4)
        self.assertEqual(res["net_profit_factor"], 1.75)
        self.assertEqual(res["maximum_drawdown_r"], 5.0)
        self.assertEqual(res["maximum_consecutive_losses"], 4)

    def test_monotonicity_spread(self):
        """4. Increasing spread cannot improve net R (holding other frictions constant)."""
        spreads = [0.00, 0.05, 0.10, 0.20, 0.30]
        net_rs = []
        for sp in spreads:
            sc = CostSensitivityScenario("SC", spread_usd=sp, entry_slippage_usd=0.02, exit_slippage_usd=0.02, commission_usd=0.02)
            res = evaluate_scenario_on_trades(sc, self.trades_df, self.total_setups)
            net_rs.append(res["net_r"])

        for i in range(len(net_rs) - 1):
            self.assertGreater(net_rs[i], net_rs[i + 1])

    def test_monotonicity_slippage(self):
        """5. Increasing slippage cannot improve net R (holding other frictions constant)."""
        slips = [0.00, 0.02, 0.05, 0.10]
        net_rs = []
        for sl in slips:
            sc = CostSensitivityScenario("SC", spread_usd=0.10, entry_slippage_usd=sl, exit_slippage_usd=sl, commission_usd=0.02)
            res = evaluate_scenario_on_trades(sc, self.trades_df, self.total_setups)
            net_rs.append(res["net_r"])

        for i in range(len(net_rs) - 1):
            self.assertGreater(net_rs[i], net_rs[i + 1])

    def test_monotonicity_commission(self):
        """6. Increasing commission cannot improve net R (holding other frictions constant)."""
        comms = [0.00, 0.02, 0.05, 0.10]
        net_rs = []
        for cm in comms:
            sc = CostSensitivityScenario("SC", spread_usd=0.10, entry_slippage_usd=0.02, exit_slippage_usd=0.02, commission_usd=cm)
            res = evaluate_scenario_on_trades(sc, self.trades_df, self.total_setups)
            net_rs.append(res["net_r"])

        for i in range(len(net_rs) - 1):
            self.assertGreater(net_rs[i], net_rs[i + 1])

    def test_gross_trade_outcomes_remain_unchanged(self):
        """7. Gross trade outcomes remain unchanged across all cost scenarios."""
        grid_df = run_cost_sensitivity_grid(self.trades_df, self.total_setups)
        # Every row must have identical gross_net_r and filled_trades
        unique_gross_net_r = grid_df["gross_net_r"].unique()
        unique_filled_trades = grid_df["filled_trades"].unique()
        self.assertEqual(len(unique_gross_net_r), 1)
        self.assertEqual(len(unique_filled_trades), 1)
        self.assertAlmostEqual(unique_gross_net_r[0], 107.6593, places=3)
        self.assertEqual(unique_filled_trades[0], 398)

    def test_total_cost_aggregation_identity(self):
        """8. Total execution costs equal spread + slippage + commission at scenario level."""
        grid_df = run_cost_sensitivity_grid(self.trades_df, self.total_setups)
        for _, row in grid_df.iterrows():
            expected_cost = round(row["total_spread_cost_r"] + row["total_slippage_cost_r"] + row["total_commission_cost_r"], 4)
            self.assertAlmostEqual(row["total_execution_cost_r"], expected_cost, places=4)

    def test_net_r_identity(self):
        """9. Net R equals gross_net_r - total_execution_cost_r."""
        grid_df = run_cost_sensitivity_grid(self.trades_df, self.total_setups)
        for _, row in grid_df.iterrows():
            expected_net = round(row["gross_net_r"] - row["total_execution_cost_r"], 4)
            self.assertAlmostEqual(row["net_r"], expected_net, places=4)
            self.assertAlmostEqual(row["baseline_relative_net_r"], round(row["net_r"] - row["gross_net_r"], 4), places=4)

    def test_candidate_ev_denominator(self):
        """10. Candidate EV uses total setup count (629) as denominator."""
        grid_df = run_cost_sensitivity_grid(self.trades_df, self.total_setups)
        for _, row in grid_df.iterrows():
            expected_ev = round(row["net_r"] / self.total_setups, 4)
            self.assertAlmostEqual(row["candidate_ev_net_r_per_setup"], expected_ev, places=4)

    def test_net_equity_metrics_computed_from_net_r(self):
        """
        11. Net equity metrics are computed from net_realized_r, not gross outcome labels.
        Verify with synthetic trade where gross +1.0R becomes net negative.
        """
        # Create 2 synthetic trades: Trade 1 is +1.0R win (range 1.0 USD), Trade 2 is -1.0R loss (range 1.0 USD)
        synthetic_trades = pd.DataFrame([
            {"trade_id": "T1", "entry_timestamp": "2020-01-01 10:00", "realized_r": 1.0, "ob_range": 1.0},
            {"trade_id": "T2", "entry_timestamp": "2020-01-02 10:00", "realized_r": -1.0, "ob_range": 1.0},
        ])

        # Apply extreme friction: total friction = 1.20 USD / 1.0 range = 1.20 R
        # Trade 1: Gross +1.0 R -> Net -0.20 R (turns into loss!)
        # Trade 2: Gross -1.0 R -> Net -2.20 R
        heavy_cost = CostSensitivityScenario(
            scenario_id="HEAVY",
            spread_usd=0.60,
            entry_slippage_usd=0.30,
            exit_slippage_usd=0.30,
            commission_usd=0.0,
        )
        res = evaluate_scenario_on_trades(heavy_cost, synthetic_trades, total_setups=2)

        # Both trades are net negative! Net win rate must be 0.0, not 0.50
        self.assertEqual(res["net_win_rate"], 0.0)
        # Consecutive losses must be 2 (since T1 turned into a loss, followed by T2)
        self.assertEqual(res["maximum_consecutive_losses"], 2)
        # Net profit factor is 0.0 because there are zero positive net gains
        self.assertEqual(res["net_profit_factor"], 0.0)

    def test_input_trade_dataframe_not_mutated(self):
        """12. Input trade DataFrame is not mutated by scenario evaluation."""
        original_cols = list(self.trades_df.columns)
        original_vals = self.trades_df["realized_r"].tolist()

        sc = CostSensitivityScenario("SC", spread_usd=0.10, entry_slippage_usd=0.02, exit_slippage_usd=0.02, commission_usd=0.02)
        _ = evaluate_scenario_on_trades(sc, self.trades_df, self.total_setups)

        self.assertEqual(list(self.trades_df.columns), original_cols)
        self.assertEqual(self.trades_df["realized_r"].tolist(), original_vals)


if __name__ == "__main__":
    unittest.main()
