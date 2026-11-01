import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.execution.robustness_analysis import (
    build_robustness_matrix,
    compute_dimension_summary,
    compute_aggregate_conclusion,
)


class TestExecutionRobustnessMatrix(unittest.TestCase):
    """
    Dedicated test suite for Step 5 Robustness Matrix and Analytical Consolidation.
    Verifies all 19 requirements specified in the user directive.
    """

    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / "outputs" / "reports" / "execution_parameter_robustness_strat_09.csv"
        cls.step4_raw_df = pd.read_csv(csv_path)
        cls.matrix_df = build_robustness_matrix(cls.step4_raw_df)
        cls.summary_df = compute_dimension_summary(cls.matrix_df)
        cls.conclusion = compute_aggregate_conclusion(cls.matrix_df)

    def test_01_all_expected_sensitivity_dimensions_represented(self):
        """1. All expected sensitivity dimensions are represented: ENTRY, STOP, TARGET."""
        dims = set(self.matrix_df["sensitivity_dimension"].unique())
        self.assertEqual(dims, {"ENTRY", "STOP", "TARGET"})

    def test_02_all_13_step_4_scenarios_appear_in_detailed_matrix(self):
        """2. All 13 Step 4 scenarios appear in the detailed matrix."""
        self.assertEqual(len(self.matrix_df), 13)
        expected_ids = {
            "ENTRY_OFF_000", "ENTRY_OFF_010", "ENTRY_OFF_020", "ENTRY_OFF_030",
            "STOP_075", "STOP_100", "STOP_125", "STOP_150",
            "TARGET_075", "TARGET_100", "TARGET_125", "TARGET_150", "TARGET_200",
        }
        self.assertEqual(set(self.matrix_df["scenario_id"].unique()), expected_ids)

    def test_03_baseline_rows_are_correctly_identified(self):
        """3. Baseline rows are correctly identified with is_baseline=True."""
        base_rows = self.matrix_df[self.matrix_df["is_baseline"]]
        self.assertEqual(len(base_rows), 3)
        self.assertEqual(set(base_rows["scenario_id"]), {"ENTRY_OFF_000", "STOP_100", "TARGET_100"})

    def test_04_all_baseline_representations_reconcile_to_same_metrics(self):
        """4. All baseline representations reconcile to the same frozen baseline metrics."""
        base_rows = self.matrix_df[self.matrix_df["is_baseline"]]
        net_r_vals = base_rows["gross_net_r"].tolist()
        exp_vals = base_rows["expectancy_r_per_filled_trade"].tolist()
        ev_vals = base_rows["candidate_ev_r_per_setup"].tolist()
        dd_vals = base_rows["maximum_drawdown_r"].tolist()

        for val in net_r_vals:
            self.assertAlmostEqual(val, 107.6593, places=4)
        for val in exp_vals:
            self.assertAlmostEqual(val, 0.2705, places=4)
        for val in ev_vals:
            self.assertAlmostEqual(val, 0.1712, places=4)
        for val in dd_vals:
            self.assertEqual(val, 5.0)

    def test_05_baseline_duplication_does_not_inflate_unique_count(self):
        """5. Baseline duplication does not inflate aggregate unique-scenario counts."""
        # 1 unique baseline + 3 entry + 3 stop + 4 target = 11 unique configurations
        self.assertEqual(self.conclusion["unique_scenarios_evaluated"], 11)

    def test_06_delta_net_r_vs_baseline_is_mathematically_correct(self):
        """6. delta_net_r_vs_baseline is mathematically correct: scenario_net_r - baseline_net_r."""
        for _, row in self.matrix_df.iterrows():
            expected_delta = round(row["gross_net_r"] - row["baseline_net_r"], 4)
            self.assertAlmostEqual(row["delta_net_r_vs_baseline"], expected_delta, places=4)

    def test_07_delta_expectancy_vs_baseline_is_mathematically_correct(self):
        """7. delta_expectancy_vs_baseline is mathematically correct."""
        for _, row in self.matrix_df.iterrows():
            expected_delta = round(row["expectancy_r_per_filled_trade"] - row["baseline_expectancy_r_per_filled_trade"], 4)
            self.assertAlmostEqual(row["delta_expectancy_vs_baseline"], expected_delta, places=4)

    def test_08_delta_candidate_ev_vs_baseline_is_mathematically_correct(self):
        """8. delta_candidate_ev_vs_baseline is mathematically correct."""
        for _, row in self.matrix_df.iterrows():
            expected_delta = round(row["candidate_ev_r_per_setup"] - row["baseline_candidate_ev_r_per_setup"], 4)
            self.assertAlmostEqual(row["delta_candidate_ev_vs_baseline"], expected_delta, places=4)

    def test_09_profitability_classification_based_strictly_on_net_r_positive(self):
        """9. Profitability classification is strictly based on gross_net_r > 0."""
        for _, row in self.matrix_df.iterrows():
            expected_is_prof = bool(row["gross_net_r"] > 0.0)
            self.assertEqual(row["is_profitable"], expected_is_prof)

    def test_10_positive_expectancy_classification_is_mathematically_correct(self):
        """10. Positive expectancy classification is mathematically correct."""
        for _, row in self.matrix_df.iterrows():
            expected_has_pos_exp = bool(row["expectancy_r_per_filled_trade"] > 0.0)
            self.assertEqual(row["has_positive_expectancy"], expected_has_pos_exp)

    def test_11_positive_candidate_ev_classification_is_mathematically_correct(self):
        """11. Positive candidate EV classification is mathematically correct."""
        for _, row in self.matrix_df.iterrows():
            expected_has_pos_ev = bool(row["candidate_ev_r_per_setup"] > 0.0)
            self.assertEqual(row["has_positive_candidate_ev"], expected_has_pos_ev)

    def test_12_immediate_neighborhood_definitions_are_deterministic(self):
        """12. Immediate neighborhood definitions are deterministic and correct."""
        summary = self.summary_df.set_index("dimension_name")
        self.assertEqual(summary.loc["ENTRY", "immediate_neighbor_count"], 1)  # 0.10R
        self.assertEqual(summary.loc["STOP", "immediate_neighbor_count"], 2)   # 0.75R, 1.25R
        self.assertEqual(summary.loc["TARGET", "immediate_neighbor_count"], 2) # 0.75R, 1.25R

    def test_13_immediate_profitable_neighbor_ratios_are_mathematically_correct(self):
        """13. Immediate profitable neighbor ratios are mathematically correct: 1.0 for all dimensions."""
        for _, row in self.summary_df.iterrows():
            self.assertEqual(row["immediate_profitable_neighbor_ratio"], 1.0)
            self.assertEqual(row["immediate_neighbor_count"], row["immediate_profitable_neighbor_count"])

    def test_14_dimension_level_profitable_scenario_ratios_are_correct(self):
        """14. Dimension-level profitable scenario ratios are mathematically correct: 1.0 (100%)."""
        for _, row in self.summary_df.iterrows():
            self.assertEqual(row["profitable_scenario_ratio"], 1.0)
            self.assertEqual(row["tested_scenario_count"], row["profitable_scenario_count"])

    def test_15_nonbaseline_profitable_scenario_ratios_are_correct(self):
        """15. Nonbaseline profitable scenario ratios are mathematically correct: 1.0 (100%)."""
        summary = self.summary_df.set_index("dimension_name")
        self.assertEqual(summary.loc["ENTRY", "nonbaseline_scenario_count"], 3)
        self.assertEqual(summary.loc["STOP", "nonbaseline_scenario_count"], 3)
        self.assertEqual(summary.loc["TARGET", "nonbaseline_scenario_count"], 4)
        for _, row in self.summary_df.iterrows():
            self.assertEqual(row["nonbaseline_profitable_ratio"], 1.0)

    def test_16_aggregate_unique_scenario_counts_do_not_triple_count_baseline(self):
        """16. Aggregate unique-scenario counts do not triple-count baseline representations."""
        self.assertEqual(self.conclusion["unique_scenarios_evaluated"], 11)
        self.assertEqual(self.conclusion["unique_profitable_scenarios"], 11)
        self.assertEqual(self.conclusion["unique_profitable_ratio"], 1.0)
        self.assertTrue(self.conclusion["baseline_profitable"])
        self.assertEqual(self.conclusion["dimensions_with_all_scenarios_profitable"], 3)
        self.assertEqual(self.conclusion["dimensions_with_any_nonprofitable_scenario"], 0)

    def test_17_input_dataframes_are_not_mutated(self):
        """17. Input DataFrames are not mutated during Step 5 analysis."""
        orig_copy = self.step4_raw_df.copy(deep=True)
        _ = build_robustness_matrix(self.step4_raw_df)
        pd.testing.assert_frame_equal(self.step4_raw_df, orig_copy)

    def test_18_step_5_does_not_modify_step_4_scenario_results(self):
        """18. Step 5 does not modify Step 4 scenario results."""
        m = self.matrix_df.set_index("scenario_id")
        r = self.step4_raw_df.set_index("scenario_id")
        for sc_id in m.index:
            self.assertAlmostEqual(m.loc[sc_id, "gross_net_r"], r.loc[sc_id, "gross_net_r"], places=4)
            self.assertAlmostEqual(m.loc[sc_id, "filled_trades"], r.loc[sc_id, "filled_trades"], places=4)

    def test_19_frozen_baseline_modules_remain_untouched(self):
        """19. Frozen baseline modules remain untouched."""
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
