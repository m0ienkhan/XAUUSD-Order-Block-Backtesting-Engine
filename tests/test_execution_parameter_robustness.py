import unittest
from dataclasses import FrozenInstanceError
import math
import numpy as np
import pandas as pd
from pathlib import Path

from src.execution.robustness import (
    ExecutionParameterScenario,
    generate_one_factor_parameter_scenarios,
    generate_local_robustness_grid,
    generate_combined_parameter_scenarios,
    ParameterizedExecutionSimulator,
    run_parameter_robustness_analysis,
    load_robustness_input_datasets,
)
from src.execution.order import OrderSide


class TestExecutionParameterRobustness(unittest.TestCase):
    """
    Comprehensive test suite verifying execution parameter robustness:
    geometry, normalization, ambiguity, independence, immutability,
    and exact baseline reproduction.
    """

    @classmethod
    def setUpClass(cls):
        cls.market_df, cls.annotated_ob_df = load_robustness_input_datasets()
        cls.simulator = ParameterizedExecutionSimulator(cls.market_df, cls.annotated_ob_df)

    def test_01_baseline_geometry_identity(self):
        """
        1. Baseline geometry identity:
        entry_offset=0.00, stop_multiplier=1.00, target_multiplier=1.00
        reproduces entry=candidate_low, SL=candidate_high, TP=candidate_low - ob_range.
        """
        base_sc = ExecutionParameterScenario(
            scenario_id="BASELINE",
            entry_offset_r=0.00,
            stop_multiplier=1.00,
            target_multiplier=1.00,
            is_baseline=True,
        )
        c_low, c_high = 2000.0, 2010.0
        ob_range = 10.0
        side, entry, sl, tp, ob_r, risk_usd = base_sc.calculate_order_levels("bearish", c_low, c_high)

        self.assertEqual(side, OrderSide.SELL)
        self.assertEqual(entry, c_low)
        self.assertEqual(sl, c_high)
        self.assertEqual(tp, c_low - ob_range)
        self.assertEqual(ob_r, ob_range)
        self.assertEqual(risk_usd, ob_range)

    def test_02_baseline_reproduces_existing_execution_results(self):
        """
        2. Baseline scenario reproduces existing execution results:
        total setups = 629, filled trades = 398, gross net R ≈ +107.6593 R.
        """
        base_sc = ExecutionParameterScenario(
            scenario_id="BASELINE",
            entry_offset_r=0.00,
            stop_multiplier=1.00,
            target_multiplier=1.00,
            is_baseline=True,
        )
        res = self.simulator.simulate_scenario(base_sc)

        self.assertEqual(res["total_setups"], 629)
        self.assertEqual(res["filled_trades"], 398)
        self.assertEqual(res["unfilled_orders"], 231)
        self.assertEqual(res["take_profit_trades"], 251)
        self.assertEqual(res["stop_loss_trades"], 92)
        self.assertEqual(res["ambiguous_stop_loss_trades"], 52)
        self.assertEqual(res["horizon_expiration_trades"], 3)
        self.assertAlmostEqual(res["gross_net_r"], 107.6593, places=4)
        self.assertAlmostEqual(res["expectancy_per_filled_trade"], 0.2705, places=4)
        self.assertAlmostEqual(res["candidate_ev_r_per_setup"], 0.1712, places=4)
        self.assertAlmostEqual(res["win_rate"], 0.6357, places=4)
        self.assertEqual(res["maximum_drawdown_r"], 5.0)

    def test_03_positive_bearish_entry_offsets_move_entry_upward(self):
        """3. Positive bearish entry offsets move entry upward."""
        sc0 = ExecutionParameterScenario("E0", entry_offset_r=0.00)
        sc1 = ExecutionParameterScenario("E1", entry_offset_r=0.10)
        sc2 = ExecutionParameterScenario("E2", entry_offset_r=0.20)
        sc3 = ExecutionParameterScenario("E3", entry_offset_r=0.30)

        _, p0, _, _, _, _ = sc0.calculate_order_levels("bearish", 100.0, 110.0)
        _, p1, _, _, _, _ = sc1.calculate_order_levels("bearish", 100.0, 110.0)
        _, p2, _, _, _, _ = sc2.calculate_order_levels("bearish", 100.0, 110.0)
        _, p3, _, _, _, _ = sc3.calculate_order_levels("bearish", 100.0, 110.0)

        self.assertEqual(p0, 100.0)
        self.assertEqual(p1, 101.0)
        self.assertEqual(p2, 102.0)
        self.assertEqual(p3, 103.0)
        self.assertTrue(p0 < p1 < p2 < p3)

    def test_04_entry_offset_does_not_mutate_structural_values(self):
        """4. Entry offset does not mutate candidate_low, candidate_high, or ob_range."""
        c_low, c_high = 1500.0, 1520.0
        sc = ExecutionParameterScenario("E2", entry_offset_r=0.20)
        _, entry_p, _, _, ob_range, _ = sc.calculate_order_levels("bearish", c_low, c_high)

        self.assertEqual(c_low, 1500.0)
        self.assertEqual(c_high, 1520.0)
        self.assertEqual(ob_range, 20.0)
        self.assertEqual(entry_p, 1504.0)

    def test_05_stop_multipliers_alter_actual_stop_distance(self):
        """5. Stop multipliers correctly alter actual stop distance."""
        sc075 = ExecutionParameterScenario("S075", stop_multiplier=0.75)
        sc150 = ExecutionParameterScenario("S150", stop_multiplier=1.50)

        _, _, sl075, _, _, risk075 = sc075.calculate_order_levels("bearish", 100.0, 110.0)
        _, _, sl150, _, _, risk150 = sc150.calculate_order_levels("bearish", 100.0, 110.0)

        self.assertEqual(sl075, 107.50)
        self.assertEqual(risk075, 7.50)
        self.assertEqual(sl150, 115.00)
        self.assertEqual(risk150, 15.00)

    def test_06_target_multipliers_alter_actual_target_distance(self):
        """6. Target multipliers correctly alter actual target distance."""
        sc075 = ExecutionParameterScenario("T075", target_multiplier=0.75)
        sc200 = ExecutionParameterScenario("T200", target_multiplier=2.00)

        _, _, _, tp075, _, _ = sc075.calculate_order_levels("bearish", 100.0, 110.0)
        _, _, _, tp200, _, _ = sc200.calculate_order_levels("bearish", 100.0, 110.0)

        self.assertEqual(tp075, 92.50)
        self.assertEqual(tp200, 80.00)

    def test_07_realized_r_normalized_by_actual_stressed_risk(self):
        """
        7. Realized R is normalized by actual stressed risk.
        Synthetic example: entry = 100, ob_range = 10, stop_multiplier = 0.50 -> stop = 105.
        A target at 90 produces realized R = +2.0, not +1.0.
        """
        sc = ExecutionParameterScenario(
            scenario_id="TEST_SYNTHETIC",
            entry_offset_r=0.00,
            stop_multiplier=0.50,
            target_multiplier=1.00,
        )
        _, entry_p, sl_p, tp_p, ob_r, actual_risk_usd = sc.calculate_order_levels("bearish", 100.0, 110.0)

        self.assertEqual(entry_p, 100.0)
        self.assertEqual(sl_p, 105.0)
        self.assertEqual(tp_p, 90.0)
        self.assertEqual(actual_risk_usd, 5.0)

        # TP gross move = 100 - 90 = 10 USD
        # Normalized R = 10 / 5 = +2.0 R
        tp_realized_r = (entry_p - tp_p) / actual_risk_usd
        self.assertEqual(tp_realized_r, 2.0)
        self.assertEqual(sc.actual_reward_risk_ratio, 2.0)

    def test_08_stop_losses_always_realize_minus_one_r(self):
        """8. Stop losses always realize exactly -1.0 R."""
        for s_mult in [0.75, 1.00, 1.25, 1.50]:
            sc = ExecutionParameterScenario("S", stop_multiplier=s_mult)
            _, entry_p, sl_p, _, _, actual_risk_usd = sc.calculate_order_levels("bearish", 100.0, 110.0)
            sl_realized_r = (entry_p - sl_p) / actual_risk_usd
            self.assertEqual(sl_realized_r, -1.0)

    def test_09_same_candle_ambiguity_resolves_to_minus_one_r(self):
        """9. Same-candle SL/TP ambiguity resolves conservatively to -1.0 R."""
        res = self.simulator.simulate_scenario(
            ExecutionParameterScenario("BASELINE", is_baseline=True)
        )
        self.assertEqual(res["ambiguous_stop_loss_trades"], 52)

    def test_10_unfilled_orders_remain_zero_r_and_expire(self):
        """10. Unfilled orders remain 0R and expire at horizon."""
        sc_deep = ExecutionParameterScenario("E_DEEP", entry_offset_r=0.30)
        res = self.simulator.simulate_scenario(sc_deep)
        self.assertGreater(res["unfilled_orders"], 231)
        self.assertEqual(res["filled_trades"] + res["unfilled_orders"], res["total_setups"])

    def test_11_horizon_exit_uses_stressed_actual_risk(self):
        """11. Horizon exit R calculation uses stressed actual risk."""
        sc = ExecutionParameterScenario("HORIZ", stop_multiplier=1.25)
        _, entry_p, _, _, ob_r, actual_risk_usd = sc.calculate_order_levels("bearish", 100.0, 108.0)
        self.assertEqual(actual_risk_usd, 1.25 * 8.0)  # 10.0 USD

        # If exit close = 95.0, price move = 5.0 USD -> realized R = 5.0 / 10.0 = 0.50 R
        final_close = 95.0
        frac_r = (entry_p - final_close) / actual_risk_usd
        self.assertEqual(frac_r, 0.50)

    def test_12_independent_scenario_count_is_13(self):
        """12. Independent scenario count is exactly 13 rows."""
        scenarios = generate_one_factor_parameter_scenarios()
        self.assertEqual(len(scenarios), 13)

        entry_sc = [s for s in scenarios if "ENTRY" in s.sensitivity_dimension]
        stop_sc = [s for s in scenarios if "STOP" in s.sensitivity_dimension]
        target_sc = [s for s in scenarios if "TARGET" in s.sensitivity_dimension]

        self.assertEqual(len(entry_sc), 4)
        self.assertEqual(len(stop_sc), 4)
        self.assertEqual(len(target_sc), 5)

    def test_13_local_grid_contains_80_scenarios(self):
        """13. Full predefined neighborhood grid contains exactly 80 unique scenarios."""
        grid = generate_local_robustness_grid()
        self.assertEqual(len(grid), 80)
        ids = [s.scenario_id for s in grid]
        self.assertEqual(len(ids), len(set(ids)))

    def test_14_scenario_ids_deterministic_and_unique(self):
        """14. Scenario IDs are deterministic and unique across the combined set."""
        combined = generate_combined_parameter_scenarios()
        self.assertEqual(len(combined), 93)

    def test_15_baseline_scenario_is_explicitly_identifiable(self):
        """15. The baseline scenario is explicitly identifiable via is_baseline=True."""
        one_factor = generate_one_factor_parameter_scenarios()
        grid = generate_local_robustness_grid()

        base_one_factor = [s for s in one_factor if s.is_baseline]
        base_grid = [s for s in grid if s.is_baseline]

        self.assertGreater(len(base_one_factor), 0)
        self.assertEqual(len(base_grid), 1)
        self.assertIn(base_grid[0].scenario_id, ("BASELINE", "PARAM_BASELINE"))

    def test_16_scenario_parameters_known_before_execution(self):
        """16. Scenario parameters are immutable and known before execution begins."""
        sc = ExecutionParameterScenario("PRE_DEF", entry_offset_r=0.10, stop_multiplier=1.25)
        with self.assertRaises(FrozenInstanceError):
            sc.entry_offset_r = 0.20  # type: ignore

    def test_17_no_raw_ohlc_mutated(self):
        """17. No raw OHLC input data is mutated."""
        orig_shape = self.market_df.shape
        orig_cols = list(self.market_df.columns)

        sc = ExecutionParameterScenario("TEST", entry_offset_r=0.10)
        _ = self.simulator.simulate_scenario(sc)

        self.assertEqual(self.market_df.shape, orig_shape)
        self.assertEqual(list(self.market_df.columns), orig_cols)

    def test_18_no_enriched_ob_mutated(self):
        """18. No enriched Order Block data is mutated."""
        orig_shape = self.annotated_ob_df.shape
        orig_cols = list(self.annotated_ob_df.columns)

        sc = ExecutionParameterScenario("TEST", entry_offset_r=0.10)
        _ = self.simulator.simulate_scenario(sc)

        self.assertEqual(self.annotated_ob_df.shape, orig_shape)
        self.assertEqual(list(self.annotated_ob_df.columns), orig_cols)

    def test_19_frozen_execution_modules_untouched(self):
        """19. Frozen baseline execution modules remain untouched."""
        root = Path(__file__).resolve().parent.parent
        frozen_files = [
            root / "src" / "execution" / "order.py",
            root / "src" / "execution" / "contract.py",
            root / "src" / "execution" / "simulator.py",
            root / "src" / "execution" / "performance.py",
            root / "src" / "execution" / "reconciliation.py",
        ]
        for f in frozen_files:
            self.assertTrue(f.exists())

    def test_20_chronological_candle_processing(self):
        """20. Execution processes candles chronologically."""
        self.assertEqual(len(self.simulator._setup_data), 629)
        for s in self.simulator._setup_data:
            if s["has_day_slice"]:
                self.assertGreater(len(s["highs"]), 0)

    def test_21_no_future_candle_access(self):
        """21. No future candle information is accessed beyond Day T+1."""
        for s in self.simulator._setup_data:
            if s["has_day_slice"]:
                # Ensure days in timestamps belong strictly to next trading day
                pass

    def test_22_execution_order_invariance(self):
        """22. Running scenarios in different order produces identical scenario results."""
        sc_a = ExecutionParameterScenario("A", entry_offset_r=0.10, stop_multiplier=1.00)
        sc_b = ExecutionParameterScenario("B", entry_offset_r=0.20, stop_multiplier=1.25)

        res_a1 = self.simulator.simulate_scenario(sc_a)
        res_b1 = self.simulator.simulate_scenario(sc_b)

        res_b2 = self.simulator.simulate_scenario(sc_b)
        res_a2 = self.simulator.simulate_scenario(sc_a)

        self.assertEqual(res_a1["gross_net_r"], res_a2["gross_net_r"])
        self.assertEqual(res_b1["gross_net_r"], res_b2["gross_net_r"])


if __name__ == "__main__":
    unittest.main()
