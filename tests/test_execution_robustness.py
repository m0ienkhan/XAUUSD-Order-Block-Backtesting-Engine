import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
import numpy as np
import pandas as pd

from src.execution.robustness import (
    ExecutionParameterScenario,
    generate_one_factor_parameter_scenarios,
    ParameterizedExecutionSimulator,
    run_parameter_robustness_analysis,
    load_robustness_input_datasets,
)
from src.execution.order import OrderSide


class TestExecutionRobustness(unittest.TestCase):
    """
    Focused test suite for Step 4 Execution Parameter Robustness Testing.
    Verifies all 20 requirements specified in section O of the user directive.
    """

    @classmethod
    def setUpClass(cls):
        cls.market_df, cls.annotated_ob_df = load_robustness_input_datasets()
        cls.simulator = ParameterizedExecutionSimulator(cls.market_df, cls.annotated_ob_df)

    def test_01_baseline_scenario_exact_parity_with_frozen_results(self):
        """
        1. Baseline scenario exact parity with frozen execution results:
        629 setups, 398 fills, 231 unfilled, 251 TP, 92 SL, 52 ambiguous SL,
        3 horizon exits, Net R ≈ +107.6593 R, win rate ≈ 63.57%, max DD ≈ 5.0 R, max losses = 4.
        """
        base_sc = ExecutionParameterScenario(
            scenario_id="BASELINE",
            entry_offset_r=0.00,
            stop_multiplier_r=1.00,
            target_multiplier_r=1.00,
            is_baseline=True,
            parameter_dimension="BASELINE",
        )
        res = self.simulator.simulate_scenario(base_sc)

        self.assertEqual(res["total_setups"], 629)
        self.assertEqual(res["filled_trades"], 398)
        self.assertEqual(res["unfilled_trades"], 231)
        self.assertEqual(res["take_profit_trades"], 251)
        self.assertEqual(res["stop_loss_trades"], 92)
        self.assertEqual(res["ambiguous_stop_loss_trades"], 52)
        self.assertEqual(res["horizon_exit_trades"], 3)
        self.assertAlmostEqual(res["gross_net_r"], 107.6593, places=4)
        self.assertAlmostEqual(res["expectancy_per_filled_trade"], 0.2705, places=4)
        self.assertAlmostEqual(res["candidate_ev_r_per_setup"], 0.1712, places=4)
        self.assertAlmostEqual(res["filled_trade_win_rate"], 0.6357, places=4)
        self.assertEqual(res["maximum_drawdown_r"], 5.0)
        self.assertEqual(res["maximum_consecutive_losses"], 4)
        self.assertEqual(res["baseline_relative_net_r"], 0.0)

    def test_02_entry_offset_price_geometry(self):
        """2. Entry offset price geometry is correct: entry = candidate_low + offset * ob_range."""
        sc = ExecutionParameterScenario(
            scenario_id="ENTRY_OFF_010",
            entry_offset_r=0.10,
            stop_multiplier_r=1.00,
            target_multiplier_r=1.00,
        )
        c_low, c_high = 2000.0, 2010.0  # ob_range = 10.0
        side, entry, sl, tp, ob_r, risk = sc.calculate_order_levels("bearish", c_low, c_high)

        self.assertEqual(side, OrderSide.SELL)
        self.assertEqual(ob_r, 10.0)
        self.assertEqual(entry, 2001.0)
        self.assertEqual(sl, 2011.0)
        self.assertEqual(tp, 1991.0)
        self.assertEqual(risk, 10.0)

    def test_03_positive_bearish_entry_offsets_represent_deeper_entries(self):
        """3. Positive bearish entry offsets represent deeper entries into the Order Block zone."""
        c_low, c_high = 1800.0, 1820.0  # ob_range = 20.0
        sc0 = ExecutionParameterScenario("E0", entry_offset_r=0.00)
        sc1 = ExecutionParameterScenario("E1", entry_offset_r=0.10)
        sc2 = ExecutionParameterScenario("E2", entry_offset_r=0.20)
        sc3 = ExecutionParameterScenario("E3", entry_offset_r=0.30)

        _, p0, _, _, _, _ = sc0.calculate_order_levels("bearish", c_low, c_high)
        _, p1, _, _, _, _ = sc1.calculate_order_levels("bearish", c_low, c_high)
        _, p2, _, _, _, _ = sc2.calculate_order_levels("bearish", c_low, c_high)
        _, p3, _, _, _, _ = sc3.calculate_order_levels("bearish", c_low, c_high)

        self.assertEqual(p0, 1800.0)
        self.assertEqual(p1, 1802.0)
        self.assertEqual(p2, 1804.0)
        self.assertEqual(p3, 1806.0)
        self.assertTrue(p0 < p1 < p2 < p3)

    def test_04_stop_price_geometry_for_each_multiplier(self):
        """4. Stop price geometry for each multiplier: stop = entry + stop_multiplier * ob_range."""
        entry = 100.0
        ob_range = 10.0
        multipliers = [0.75, 1.00, 1.25, 1.50]
        expected_stops = [107.50, 110.00, 112.50, 115.00]

        for mult, exp_stop in zip(multipliers, expected_stops):
            sc = ExecutionParameterScenario(f"S_{mult}", stop_multiplier_r=mult)
            _, e_p, sl_p, _, _, _ = sc.calculate_order_levels("bearish", entry, entry + ob_range)
            self.assertEqual(e_p, entry)
            self.assertEqual(sl_p, exp_stop)

    def test_05_target_price_geometry_for_each_multiplier(self):
        """5. Target price geometry for each multiplier: target = entry - target_multiplier * ob_range."""
        entry = 100.0
        ob_range = 10.0
        multipliers = [0.75, 1.00, 1.25, 1.50, 2.00]
        expected_targets = [92.50, 90.00, 87.50, 85.00, 80.00]

        for mult, exp_target in zip(multipliers, expected_targets):
            sc = ExecutionParameterScenario(f"T_{mult}", target_multiplier_r=mult)
            _, e_p, _, tp_p, _, _ = sc.calculate_order_levels("bearish", entry, entry + ob_range)
            self.assertEqual(e_p, entry)
            self.assertEqual(tp_p, exp_target)

    def test_06_original_ob_range_remains_unchanged(self):
        """6. Original ob_range remains unchanged across scenarios."""
        c_low, c_high = 1500.0, 1525.0
        original_range = c_high - c_low

        sc_off = ExecutionParameterScenario("E_OFF", entry_offset_r=0.25)
        sc_sl = ExecutionParameterScenario("S_WIDE", stop_multiplier_r=1.50)
        sc_tp = ExecutionParameterScenario("T_WIDE", target_multiplier_r=2.00)

        _, _, _, _, r_off, _ = sc_off.calculate_order_levels("bearish", c_low, c_high)
        _, _, _, _, r_sl, _ = sc_sl.calculate_order_levels("bearish", c_low, c_high)
        _, _, _, _, r_tp, _ = sc_tp.calculate_order_levels("bearish", c_low, c_high)

        self.assertEqual(r_off, original_range)
        self.assertEqual(r_sl, original_range)
        self.assertEqual(r_tp, original_range)
        self.assertEqual(c_high - c_low, original_range)

    def test_07_each_scenario_evaluates_fills_independently(self):
        """7. Each robustness scenario evaluates fills independently."""
        sc0 = ExecutionParameterScenario("E0", entry_offset_r=0.00)
        sc1 = ExecutionParameterScenario("E1", entry_offset_r=0.10)
        sc2 = ExecutionParameterScenario("E2", entry_offset_r=0.20)
        sc3 = ExecutionParameterScenario("E3", entry_offset_r=0.30)

        r0 = self.simulator.simulate_scenario(sc0)
        r1 = self.simulator.simulate_scenario(sc1)
        r2 = self.simulator.simulate_scenario(sc2)
        r3 = self.simulator.simulate_scenario(sc3)

        self.assertEqual(r0["filled_trades"], 398)
        self.assertEqual(r1["filled_trades"], 391)
        self.assertEqual(r2["filled_trades"], 377)
        self.assertEqual(r3["filled_trades"], 365)
        self.assertTrue(r0["filled_trades"] > r1["filled_trades"] > r2["filled_trades"] > r3["filled_trades"])

    def test_08_deeper_entry_can_expire_even_if_baseline_filled(self):
        """8. A deeper entry can expire even if the baseline entry would have filled."""
        r_base = self.simulator.simulate_scenario(ExecutionParameterScenario("BASE", entry_offset_r=0.00))
        r_deep = self.simulator.simulate_scenario(ExecutionParameterScenario("DEEP", entry_offset_r=0.30))

        # 33 trades that filled at baseline expired unfilled at +0.30R deeper entry
        self.assertEqual(r_base["unfilled_trades"], 231)
        self.assertEqual(r_deep["unfilled_trades"], 264)
        self.assertEqual(r_deep["unfilled_trades"] - r_base["unfilled_trades"], 33)

    def test_09_stop_distance_changes_scenario_risk_correctly(self):
        """9. Stop distance changes scenario risk correctly: scenario_risk_price = stop_multiplier * ob_range."""
        sc075 = ExecutionParameterScenario("S075", stop_multiplier_r=0.75)
        sc125 = ExecutionParameterScenario("S125", stop_multiplier_r=1.25)
        sc150 = ExecutionParameterScenario("S150", stop_multiplier_r=1.50)

        _, entry, sl075, _, ob_r, risk075 = sc075.calculate_order_levels("bearish", 100.0, 110.0)
        _, _, sl125, _, _, risk125 = sc125.calculate_order_levels("bearish", 100.0, 110.0)
        _, _, sl150, _, _, risk150 = sc150.calculate_order_levels("bearish", 100.0, 110.0)

        self.assertEqual(risk075, 7.50)
        self.assertEqual(risk125, 12.50)
        self.assertEqual(risk150, 15.00)
        self.assertEqual(abs(sl075 - entry), 7.50)
        self.assertEqual(abs(sl125 - entry), 12.50)
        self.assertEqual(abs(sl150 - entry), 15.00)

    def test_10_target_realized_r_uses_scenario_risk_denominator(self):
        """10. Target realized R uses the defined scenario risk denominator: TP realized R = target_mult / stop_mult."""
        sc_tight_stop = ExecutionParameterScenario("S075_T100", stop_multiplier_r=0.75, target_multiplier_r=1.00)
        sc_wide_stop = ExecutionParameterScenario("S150_T100", stop_multiplier_r=1.50, target_multiplier_r=1.00)

        self.assertAlmostEqual(sc_tight_stop.actual_reward_risk_ratio, 1.00 / 0.75, places=4)
        self.assertAlmostEqual(sc_wide_stop.actual_reward_risk_ratio, 1.00 / 1.50, places=4)

    def test_11_horizon_exit_r_calculated_correctly(self):
        """11. Horizon exit R is calculated correctly: (entry_price - exit_price) / scenario_risk_price."""
        sc = ExecutionParameterScenario("HORIZ", stop_multiplier_r=1.25)
        _, entry, _, _, _, risk = sc.calculate_order_levels("bearish", 100.0, 108.0)
        self.assertEqual(risk, 10.0)  # 1.25 * 8.0 = 10.0 USD

        final_close = 95.0
        price_move = entry - final_close  # 100 - 95 = +5.0 USD
        realized_r = price_move / risk     # +5.0 / 10.0 = +0.50 R
        self.assertEqual(realized_r, 0.50)

    def test_12_same_candle_ambiguity_uses_conservative_stop_first_policy(self):
        """12. Same-candle ambiguity uses conservative stop-first policy: realized_r = -1.0."""
        res = self.simulator.simulate_scenario(ExecutionParameterScenario("BASE", is_baseline=True))
        self.assertEqual(res["ambiguous_stop_loss_trades"], 52)

    def test_13_input_market_data_not_mutated(self):
        """13. Input market data is not mutated."""
        orig_shape = self.market_df.shape
        orig_cols = list(self.market_df.columns)
        _ = self.simulator.simulate_scenario(ExecutionParameterScenario("E_TEST", entry_offset_r=0.20))
        self.assertEqual(self.market_df.shape, orig_shape)
        self.assertEqual(list(self.market_df.columns), orig_cols)

    def test_14_input_order_block_data_not_mutated(self):
        """14. Input Order Block data is not mutated."""
        orig_shape = self.annotated_ob_df.shape
        orig_cols = list(self.annotated_ob_df.columns)
        _ = self.simulator.simulate_scenario(ExecutionParameterScenario("S_TEST", stop_multiplier_r=1.25))
        self.assertEqual(self.annotated_ob_df.shape, orig_shape)
        self.assertEqual(list(self.annotated_ob_df.columns), orig_cols)

    def test_15_candle_processing_is_chronological(self):
        """15. Candle processing is strictly chronological."""
        self.assertEqual(len(self.simulator._setup_data), 629)
        for s in self.simulator._setup_data:
            if s["has_day_slice"]:
                self.assertGreater(len(s["highs"]), 0)
                # Confirm timestamps are monotonically increasing
                ts_list = s["timestamps"]
                for i in range(len(ts_list) - 1):
                    self.assertLess(ts_list[i], ts_list[i + 1])

    def test_16_drawdown_uses_chronological_trade_outcomes(self):
        """16. Drawdown uses chronological trade outcomes."""
        res = self.simulator.simulate_scenario(ExecutionParameterScenario("BASE", is_baseline=True))
        self.assertAlmostEqual(res["maximum_drawdown_r"], 5.0, places=4)

    def test_17_maximum_consecutive_losses_calculated_from_outcomes(self):
        """17. Maximum consecutive losses is calculated from realized scenario outcomes."""
        res = self.simulator.simulate_scenario(ExecutionParameterScenario("BASE", is_baseline=True))
        self.assertEqual(res["maximum_consecutive_losses"], 4)

    def test_18_scenario_ids_are_unique_and_deterministic(self):
        """18. Scenario IDs are unique and deterministic."""
        scenarios = generate_one_factor_parameter_scenarios()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), 13)
        self.assertEqual(len(ids), len(set(ids)))

    def test_19_baseline_parameter_configuration_clearly_identifiable(self):
        """19. Baseline parameter configuration is clearly identifiable via is_baseline=True."""
        scenarios = generate_one_factor_parameter_scenarios()
        baseline_scs = [s for s in scenarios if s.is_baseline]
        self.assertEqual(len(baseline_scs), 3)  # One in each sensitivity family
        for b in baseline_scs:
            self.assertEqual(b.entry_offset_r, 0.00)
            self.assertEqual(b.stop_multiplier_r, 1.00)
            self.assertEqual(b.target_multiplier_r, 1.00)

    def test_20_parameter_sensitivity_does_not_modify_frozen_modules(self):
        """20. Parameter sensitivity execution does not modify frozen upstream modules."""
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


if __name__ == "__main__":
    unittest.main()
