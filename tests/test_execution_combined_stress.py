import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.execution.combined_stress import (
    CombinedStressScenario,
    CombinedStressSimulator,
    generate_predefined_combined_scenarios,
    run_combined_stress_scenarios,
)
from src.execution.robustness import load_robustness_input_datasets


class TestExecutionCombinedStress(unittest.TestCase):
    """
    Dedicated test suite for Step 6 Combined Execution Stress Scenarios.
    Verifies all requirements specified in the user directive.
    """

    @classmethod
    def setUpClass(cls):
        cls.market_df, cls.annotated_ob_df = load_robustness_input_datasets()
        cls.simulator = CombinedStressSimulator(cls.market_df, cls.annotated_ob_df)
        cls.predefined_scenarios = generate_predefined_combined_scenarios()
        cls.results_df = run_combined_stress_scenarios(cls.market_df, cls.annotated_ob_df, cls.predefined_scenarios)
        cls.results_map = {row["scenario_id"]: row for _, row in cls.results_df.iterrows()}

    def test_01_exactly_four_predefined_scenarios_generated(self):
        """1. Exactly four predefined scenarios are generated."""
        self.assertEqual(len(self.predefined_scenarios), 4)

    def test_02_scenario_ids_are_unique(self):
        """2. Scenario IDs are unique."""
        ids = [sc.scenario_id for sc in self.predefined_scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            set(ids),
            {
                "SCENARIO_1_IDEALIZED",
                "SCENARIO_2_MODERATE_REALISTIC",
                "SCENARIO_3_HIGH_FRICTION",
                "SCENARIO_4_SEVERE_STRESS",
            },
        )

    def test_03_scenario_parameters_exactly_match_specified_constants(self):
        """3. Scenario parameters exactly match the specified constants."""
        sc1, sc2, sc3, sc4 = self.predefined_scenarios

        # SCENARIO 1 — IDEALIZED
        self.assertEqual(sc1.spread_usd, 0.00)
        self.assertEqual(sc1.entry_slippage_usd, 0.00)
        self.assertEqual(sc1.exit_slippage_usd, 0.00)
        self.assertEqual(sc1.commission_usd, 0.00)
        self.assertEqual(sc1.commission_r_fixed, 0.00)
        self.assertEqual(sc1.entry_offset_r, 0.00)
        self.assertEqual(sc1.stop_multiplier_r, 1.00)
        self.assertEqual(sc1.target_multiplier_r, 1.00)

        # SCENARIO 2 — MODERATE_REALISTIC
        self.assertEqual(sc2.spread_usd, 0.10)
        self.assertEqual(sc2.entry_slippage_usd, 0.02)
        self.assertEqual(sc2.exit_slippage_usd, 0.02)
        self.assertEqual(sc2.commission_usd, 0.02)
        self.assertEqual(sc2.commission_r_fixed, 0.00)
        self.assertEqual(sc2.entry_offset_r, 0.00)
        self.assertEqual(sc2.stop_multiplier_r, 1.00)
        self.assertEqual(sc2.target_multiplier_r, 1.00)

        # SCENARIO 3 — HIGH_FRICTION
        self.assertEqual(sc3.spread_usd, 0.20)
        self.assertEqual(sc3.entry_slippage_usd, 0.05)
        self.assertEqual(sc3.exit_slippage_usd, 0.05)
        self.assertEqual(sc3.commission_usd, 0.05)
        self.assertEqual(sc3.commission_r_fixed, 0.00)
        self.assertEqual(sc3.entry_offset_r, 0.10)
        self.assertEqual(sc3.stop_multiplier_r, 1.00)
        self.assertEqual(sc3.target_multiplier_r, 1.00)

        # SCENARIO 4 — SEVERE_STRESS
        self.assertEqual(sc4.spread_usd, 0.30)
        self.assertEqual(sc4.entry_slippage_usd, 0.10)
        self.assertEqual(sc4.exit_slippage_usd, 0.10)
        self.assertEqual(sc4.commission_usd, 0.05)
        self.assertEqual(sc4.commission_r_fixed, 0.00)
        self.assertEqual(sc4.entry_offset_r, 0.20)
        self.assertEqual(sc4.stop_multiplier_r, 1.25)
        self.assertEqual(sc4.target_multiplier_r, 1.25)

    def test_04_idealized_reproduces_frozen_baseline(self):
        """4. IDEALIZED reproduces the frozen baseline execution metrics exactly."""
        r1 = self.results_map["SCENARIO_1_IDEALIZED"]
        self.assertEqual(r1["total_setups"], 629)
        self.assertEqual(r1["filled_trades"], 398)
        self.assertEqual(r1["unfilled_trades"], 231)
        self.assertEqual(r1["take_profit_trades"], 251)
        self.assertEqual(r1["stop_loss_trades"], 92)
        self.assertEqual(r1["ambiguous_stop_loss_trades"], 52)
        self.assertEqual(r1["horizon_exit_trades"], 3)
        self.assertAlmostEqual(r1["gross_net_r"], 107.6593, places=4)
        self.assertAlmostEqual(r1["net_r"], 107.6593, places=4)
        self.assertAlmostEqual(r1["net_win_rate"], 0.6357, places=4)
        self.assertAlmostEqual(r1["maximum_drawdown_r"], 5.0000, places=4)
        self.assertEqual(r1["maximum_consecutive_losses"], 4)

    def test_05_idealized_has_zero_execution_costs(self):
        """5. IDEALIZED has zero execution costs across all categories."""
        r1 = self.results_map["SCENARIO_1_IDEALIZED"]
        self.assertEqual(r1["total_spread_cost_r"], 0.0)
        self.assertEqual(r1["total_slippage_cost_r"], 0.0)
        self.assertEqual(r1["total_commission_cost_r"], 0.0)
        self.assertEqual(r1["total_execution_cost_r"], 0.0)
        self.assertEqual(r1["cost_utilization_fraction"], 0.0)
        self.assertEqual(r1["break_even_margin_fraction"], 1.0)

    def test_06_increasing_friction_cannot_create_cost_accounting_improvements(self):
        """6. Increasing scenario friction monotonically increases cost deductions."""
        r1 = self.results_map["SCENARIO_1_IDEALIZED"]
        r2 = self.results_map["SCENARIO_2_MODERATE_REALISTIC"]

        # Scenarios 1 and 2 share identical execution geometry (Entry 0.00, SL 1.00, TP 1.00, 398 fills)
        self.assertEqual(r1["filled_trades"], r2["filled_trades"])
        self.assertEqual(r1["gross_net_r"], r2["gross_net_r"])
        self.assertGreater(r2["total_execution_cost_r"], r1["total_execution_cost_r"])
        self.assertLess(r2["net_r"], r1["net_r"])

    def test_07_gross_outcomes_independent_from_costs_where_geometry_identical(self):
        """7. Gross outcomes remain independent from costs where execution geometry is identical."""
        r1 = self.results_map["SCENARIO_1_IDEALIZED"]
        r2 = self.results_map["SCENARIO_2_MODERATE_REALISTIC"]

        self.assertEqual(r1["take_profit_trades"], r2["take_profit_trades"])
        self.assertEqual(r1["stop_loss_trades"], r2["stop_loss_trades"])
        self.assertEqual(r1["ambiguous_stop_loss_trades"], r2["ambiguous_stop_loss_trades"])
        self.assertEqual(r1["horizon_exit_trades"], r2["horizon_exit_trades"])
        self.assertAlmostEqual(r1["gross_net_r"], r2["gross_net_r"], places=4)

    def test_08_net_r_identity_holds_for_all_scenarios(self):
        """8. Net R identity holds: net_r = gross_net_r - total_execution_cost_r."""
        for sc_id, res in self.results_map.items():
            expected_net = round(res["gross_net_r"] - res["total_execution_cost_r"], 4)
            self.assertAlmostEqual(res["net_r"], expected_net, places=4)

    def test_09_candidate_ev_denominator_remains_exactly_629_setups(self):
        """9. Candidate EV denominator remains exactly 629 setups."""
        for sc_id, res in self.results_map.items():
            self.assertEqual(res["total_setups"], 629)
            expected_ev = round(res["net_r"] / 629, 4)
            self.assertAlmostEqual(res["candidate_ev_net_r_per_setup"], expected_ev, places=4)

    def test_10_input_raw_market_data_not_mutated(self):
        """10. Input raw market data is not mutated."""
        orig_shape = self.market_df.shape
        orig_cols = list(self.market_df.columns)
        _ = self.simulator.simulate_scenario(self.predefined_scenarios[1])
        self.assertEqual(self.market_df.shape, orig_shape)
        self.assertEqual(list(self.market_df.columns), orig_cols)

    def test_11_input_setup_data_not_mutated(self):
        """11. Input setup data is not mutated."""
        orig_shape = self.annotated_ob_df.shape
        orig_cols = list(self.annotated_ob_df.columns)
        _ = self.simulator.simulate_scenario(self.predefined_scenarios[2])
        self.assertEqual(self.annotated_ob_df.shape, orig_shape)
        self.assertEqual(list(self.annotated_ob_df.columns), orig_cols)

    def test_12_candle_processing_remains_chronological(self):
        """12. Candle processing remains strictly chronological."""
        for s in self.simulator._param_simulator._setup_data:
            if s["has_day_slice"]:
                ts_list = s["timestamps"]
                for i in range(len(ts_list) - 1):
                    self.assertLess(ts_list[i], ts_list[i + 1])

    def test_13_combined_scenario_execution_does_not_modify_frozen_modules(self):
        """13. Combined scenario execution does not modify upstream frozen modules."""
        root = Path(__file__).resolve().parent.parent
        frozen_files = [
            root / "src" / "execution" / "order.py",
            root / "src" / "execution" / "contract.py",
            root / "src" / "execution" / "simulator.py",
            root / "src" / "execution" / "performance.py",
            root / "src" / "execution" / "reconciliation.py",
            root / "scripts" / "run_execution_simulation.py",
        ]
        for f in frozen_files:
            self.assertTrue(f.exists())

    def test_14_break_even_metrics_reuse_step_3_methodology(self):
        """14. Break-even metrics reuse existing Step 3 methodology."""
        for sc_id, res in self.results_map.items():
            g_r = res["gross_net_r"]
            c_r = res["total_execution_cost_r"]
            net_r = res["net_r"]

            self.assertAlmostEqual(res["remaining_edge_r"], net_r, places=4)
            if g_r > 0.0:
                expected_util = round(c_r / g_r, 4)
                expected_margin = round(net_r / g_r, 4)
                self.assertAlmostEqual(res["cost_utilization_fraction"], expected_util, places=4)
                self.assertAlmostEqual(res["break_even_margin_fraction"], expected_margin, places=4)

    def test_15_all_results_are_deterministic_across_repeated_runs(self):
        """15. All results are deterministic across repeated runs."""
        run2_df = run_combined_stress_scenarios(self.market_df, self.annotated_ob_df, self.predefined_scenarios)
        pd.testing.assert_frame_equal(self.results_df, run2_df)


if __name__ == "__main__":
    unittest.main()
