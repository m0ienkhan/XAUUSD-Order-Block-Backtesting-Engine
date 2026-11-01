import unittest
from pathlib import Path
import pandas as pd

from src.execution.contract import load_strat_09_contract
from src.execution.simulator import ExecutionSimulator
from src.execution.reconciliation import perform_reconciliation_audit
from src.analytics.walk_forward_validation import (
    compute_training_feature_boundaries,
    apply_frozen_feature_boundaries,
    split_datasets_by_period,
)


class TestExecutionReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        cls.enriched_csv = project_root / "outputs" / "reports" / "full_enriched_order_blocks_2016_2026.csv"
        cls.path_csv = project_root / "outputs" / "reports" / "full_path_outcomes_2016_2026.csv"
        cls.market_csv = project_root / "data" / "raw" / "XAUUSD_15min_2016_2026.csv"


        if not cls.enriched_csv.exists() or not cls.path_csv.exists() or not cls.market_csv.exists():
            raise unittest.SkipTest("Required historical datasets not found for reconciliation test.")

        # Load market data
        cls.market_df = pd.read_csv(cls.market_csv)
        cls.market_df["timestamp"] = pd.to_datetime(cls.market_df["timestamp"], utc=True)
        cls.market_df = cls.market_df.set_index("timestamp").sort_index()

        # Assign trading day using canonical rollover
        from src.data.trading_day import assign_trading_day
        cls.market_df = assign_trading_day(cls.market_df)

        # Load enriched OBs and apply frozen training boundaries
        cls.enriched_df = pd.read_csv(cls.enriched_csv)
        cls.path_df = pd.read_csv(cls.path_csv)

        train_df, _, _ = split_datasets_by_period(cls.enriched_df)
        boundaries = compute_training_feature_boundaries(train_df)
        cls.annotated_df = apply_frozen_feature_boundaries(cls.enriched_df, boundaries)

        cls.contract = load_strat_09_contract()

    def test_full_dataset_reconciliation_parity(self):
        """
        Assert 100% 1-to-1 parity between Track B simulator and path_analysis.py across all 629 STRAT_09 OBs.
        """
        sim = ExecutionSimulator(self.market_df, self.annotated_df, self.contract)
        sim_a, sim_b = sim.run_dual_track_simulation()

        audit_res, report_text = perform_reconciliation_audit(
            sim_track_a=sim_a,
            sim_track_b=sim_b,
            path_outcomes_df=self.path_df,
            enriched_df=self.annotated_df,
        )

        # Invariant 1: Total audited candidates must equal 629
        self.assertEqual(audit_res.total_audited_candidates, 629)

        # Invariant 2: Track B must achieve 100% exact parity with path_analysis.py
        self.assertEqual(audit_res.mismatches_track_b, 0)
        self.assertEqual(audit_res.match_rate_track_b_pct, 100.0)
        self.assertTrue(audit_res.is_fully_reconciled)

        # Invariant 3: Category counts must match exactly
        self.assertEqual(audit_res.track_b_matches_by_category["target_first"], 251)
        self.assertEqual(audit_res.track_b_matches_by_category["stop_first"], 92)
        self.assertEqual(audit_res.track_b_matches_by_category["not_tapped"], 231)
        self.assertEqual(audit_res.track_b_matches_by_category["ambiguous_same_candle"], 52)
        self.assertEqual(audit_res.track_b_matches_by_category["no_resolution_within_horizon"], 3)

        # Invariant 4: Track A conversions must reflect conservative execution
        self.assertEqual(audit_res.track_a_conversions["target_first_to_tp"], 251)
        self.assertEqual(audit_res.track_a_conversions["stop_first_to_sl"], 92)
        self.assertEqual(audit_res.track_a_conversions["ambiguous_to_conservative_sl"], 52)
        self.assertEqual(audit_res.track_a_conversions["unresolved_to_fractional_r"], 3)
        self.assertEqual(audit_res.track_a_conversions["not_tapped_to_expired"], 231)


if __name__ == "__main__":
    unittest.main()
