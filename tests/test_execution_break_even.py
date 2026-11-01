import unittest
import math
import pandas as pd
import numpy as np

from src.execution.stress_testing import (
    compute_break_even_analysis,
    run_break_even_analysis,
    generate_break_even_summary_df,
    compute_scenario_break_even_margins,
    run_cost_sensitivity_grid,
    load_baseline_execution_population,
    BreakEvenAnalysisResult,
)


class TestExecutionBreakEven(unittest.TestCase):
    """
    Test suite for Step 3 Break-Even Cost Analysis.
    """

    @classmethod
    def setUpClass(cls):
        cls.trades_df, cls.total_setups = load_baseline_execution_population()
        cls.be_result = compute_break_even_analysis(cls.trades_df, cls.total_setups)
        cls.cost_grid_df = run_cost_sensitivity_grid(cls.trades_df, cls.total_setups)

    def test_average_r_break_even_identity(self):
        """1. Average R break-even identity: break_even_cost_per_trade == gross_net_r / filled_trades."""
        res = self.be_result
        expected = round(res.gross_net_r / res.filled_trades, 4)
        self.assertEqual(res.break_even_average_cost_r_per_filled_trade, expected)
        self.assertEqual(res.gross_expectancy_per_filled_trade, expected)
        self.assertAlmostEqual(res.break_even_average_cost_r_per_filled_trade, 0.2705, places=4)

    def test_candidate_setup_break_even_identity(self):
        """2. Candidate setup break-even identity: break_even_cost_per_setup == gross_net_r / total_setups."""
        res = self.be_result
        expected = round(res.gross_net_r / res.total_setups, 4)
        self.assertEqual(res.break_even_average_cost_r_per_setup, expected)
        self.assertEqual(res.gross_candidate_ev_r_per_setup, expected)
        self.assertAlmostEqual(res.break_even_average_cost_r_per_setup, 0.1712, places=4)

    def test_exact_constant_usd_break_even_yields_zero_net_r(self):
        """
        3. Exact constant USD break-even produces approximately zero aggregate net R
        when applied across the empirical ob_range distribution.
        """
        res = self.be_result
        c_star = res.exact_break_even_constant_usd_per_oz
        self.assertAlmostEqual(c_star, 0.4460, places=4)

        # Apply C* trade by trade
        costs_r = c_star / self.trades_df["ob_range"].astype(float)
        net_r_sum = self.trades_df["realized_r"].astype(float).sum() - costs_r.sum()

        # Tolerance within 0.05 R across 398 trades due to 4 decimal place rounding on C*
        self.assertAlmostEqual(net_r_sum, 0.0, delta=0.05)

    def test_constant_usd_friction_below_break_even_is_profitable(self):
        """4. Constant USD friction below break-even produces positive aggregate net R."""
        c_sub = self.be_result.exact_break_even_constant_usd_per_oz * 0.90
        costs_r = c_sub / self.trades_df["ob_range"].astype(float)
        net_r_sum = self.trades_df["realized_r"].astype(float).sum() - costs_r.sum()
        self.assertGreater(net_r_sum, 0.0)

    def test_constant_usd_friction_above_break_even_is_negative(self):
        """5. Constant USD friction above break-even produces negative aggregate net R."""
        c_super = self.be_result.exact_break_even_constant_usd_per_oz * 1.10
        costs_r = c_super / self.trades_df["ob_range"].astype(float)
        net_r_sum = self.trades_df["realized_r"].astype(float).sum() - costs_r.sum()
        self.assertLess(net_r_sum, 0.0)

    def test_exact_reciprocal_distribution_vs_arithmetic_mean_mandatory_synthetic(self):
        """
        6. Mandatory synthetic heterogeneous test proving why mean(ob_range) FAILS
        and exact reciprocal distribution sum(1/ob_range_i) is mathematically REQUIRED.
        """
        # Synthetic dataset:
        # Trade 1: range = 1.0 USD, gross = +1.0 R
        # Trade 2: range = 4.0 USD, gross = +1.0 R
        # Total Gross R = 2.0 R, N = 2
        synth_df = pd.DataFrame([
            {"realized_r": 1.0, "ob_range": 1.0},
            {"realized_r": 1.0, "ob_range": 4.0},
        ])

        synth_res = compute_break_even_analysis(synth_df, total_setups=2)

        # 1/1.0 + 1/4.0 = 1.0 + 0.25 = 1.25
        self.assertEqual(synth_res.reciprocal_ob_range_sum, 1.25)

        # Exact break-even C* = 2.0 / 1.25 = 1.60 USD
        self.assertEqual(synth_res.exact_break_even_constant_usd_per_oz, 1.60)

        # Arithmetic mean range = (1.0 + 4.0) / 2 = 2.50 USD
        self.assertEqual(synth_res.arithmetic_mean_ob_range, 2.50)

        # Naive arithmetic mean approx = 1.0 R * 2.50 USD = 2.50 USD
        self.assertEqual(synth_res.approximate_break_even_usd_from_mean_range, 2.50)

        # PROOF 1: Exact C* = 1.60 yields EXACTLY 0.0 Net R
        trade1_net_exact = 1.0 - (1.60 / 1.0)  # 1.0 - 1.60 = -0.60
        trade2_net_exact = 1.0 - (1.60 / 4.0)  # 1.0 - 0.40 = +0.60
        total_net_exact = trade1_net_exact + trade2_net_exact
        self.assertAlmostEqual(total_net_exact, 0.0, places=12)

        # PROOF 2: Naive C_approx = 2.50 FAILS completely (deeply negative net R!)
        trade1_net_approx = 1.0 - (2.50 / 1.0)  # 1.0 - 2.50 = -1.50
        trade2_net_approx = 1.0 - (2.50 / 4.0)  # 1.0 - 0.625 = +0.375
        total_net_approx = trade1_net_approx + trade2_net_approx
        self.assertEqual(total_net_approx, -1.125)  # Severe loss, not break-even!

    def test_arithmetic_mean_labelled_approximation_only(self):
        """7. Arithmetic-mean USD conversion is labelled as approximation only."""
        summary_df = generate_break_even_summary_df(self.be_result)
        self.assertIn("approximate_break_even_usd_from_mean_range", summary_df.columns)
        self.assertIn("exact_break_even_constant_usd_per_oz", summary_df.columns)

        # Verify exact and approx are distinctly different for STRAT_09
        exact_val = summary_df["exact_break_even_constant_usd_per_oz"].iloc[0]
        approx_val = summary_df["approximate_break_even_usd_from_mean_range"].iloc[0]
        self.assertNotEqual(exact_val, approx_val)
        self.assertAlmostEqual(exact_val, 0.4460, places=4)
        self.assertAlmostEqual(approx_val, 0.7235, places=4)

    def test_scenario_cost_utilization_fraction(self):
        """8. Scenario cost utilization equals total_execution_cost_r / gross_net_r."""
        margin_df = compute_scenario_break_even_margins(self.cost_grid_df)
        self.assertEqual(len(margin_df), 60)

        for _, row in margin_df.iterrows():
            expected_util = round(row["total_execution_cost_r"] / row["gross_net_r"], 4)
            self.assertAlmostEqual(row["cost_utilization_fraction"], expected_util, places=4)

    def test_scenario_remaining_edge_identity(self):
        """9. Scenario remaining edge equals net_r (unclipped)."""
        margin_df = compute_scenario_break_even_margins(self.cost_grid_df)
        for _, row in margin_df.iterrows():
            self.assertEqual(row["remaining_edge_r"], row["net_r"])
            expected_fraction = round(row["net_r"] / row["gross_net_r"], 4)
            self.assertAlmostEqual(row["remaining_edge_fraction"], expected_fraction, places=4)

    def test_nonzero_scenario_break_even_multiplier(self):
        """10. Non-zero scenario break-even multiplier identity: m = gross_net_r / total_execution_cost_r."""
        margin_df = compute_scenario_break_even_margins(self.cost_grid_df)
        non_zero_rows = margin_df[margin_df["total_execution_cost_r"] > 0]

        for _, row in non_zero_rows.iterrows():
            expected_m = round(row["gross_net_r"] / row["total_execution_cost_r"], 4)
            self.assertAlmostEqual(row["break_even_cost_multiplier"], expected_m, places=4)
            self.assertAlmostEqual(row["cost_headroom_multiplier"], round(expected_m - 1.0, 4), places=4)

    def test_zero_cost_division_handling_is_deterministic(self):
        """11. Zero-cost scenario division handling is deterministic (inf)."""
        margin_df = compute_scenario_break_even_margins(self.cost_grid_df)
        zero_cost_row = margin_df[margin_df["total_execution_cost_r"] == 0.0].iloc[0]

        self.assertTrue(math.isinf(zero_cost_row["break_even_cost_multiplier"]))
        self.assertTrue(math.isinf(zero_cost_row["cost_headroom_multiplier"]))
        self.assertEqual(zero_cost_row["cost_utilization_fraction"], 0.0)
        self.assertEqual(zero_cost_row["remaining_edge_fraction"], 1.0)

    def test_input_dataframes_not_mutated(self):
        """12. Input trade data and existing cost sensitivity DataFrame are not mutated."""
        orig_trades_cols = list(self.trades_df.columns)
        orig_grid_cols = list(self.cost_grid_df.columns)

        _ = compute_break_even_analysis(self.trades_df, self.total_setups)
        _ = compute_scenario_break_even_margins(self.cost_grid_df)

        self.assertEqual(list(self.trades_df.columns), orig_trades_cols)
        self.assertEqual(list(self.cost_grid_df.columns), orig_grid_cols)


if __name__ == "__main__":
    unittest.main()
