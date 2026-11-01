import unittest
import pandas as pd
import numpy as np

from src.analytics.walk_forward_validation import (
    FrozenFeatureBoundaries,
    FrozenStrategyDefinition,
    compute_training_feature_boundaries,
    apply_frozen_feature_boundaries,
    split_datasets_by_period,
    evaluate_strategy_on_period,
    discover_and_freeze_strategies,
    filter_validation_survivors,
    evaluate_holdout_and_classify,
    compute_equity_curve_and_drawdown,
    compute_walk_forward_breakdown_tables,
)


class TestWalkForwardValidation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create dataset spanning 2016 to 2026
        train_ts = pd.date_range("2016-01-04", "2022-12-31", freq="24h", tz="UTC")
        val_ts = pd.date_range("2023-01-01", "2024-12-31", freq="24h", tz="UTC")
        holdout_ts = pd.date_range("2025-01-01", "2026-08-31", freq="24h", tz="UTC")

        all_ts = train_ts.union(val_ts).union(holdout_ts)
        n = len(all_ts)

        self.df = pd.DataFrame({
            "trading_day": [ts.strftime("%Y-%m-%d 00:00:00") for ts in all_ts],
            "candidate_position": range(n),
            "candidate_timestamp": all_ts,
            "ob_direction": ["bullish" if i % 2 == 0 else "bearish" for i in range(n)],
            "day_close": np.random.uniform(1200, 2000, n),
            "ob_range": np.random.uniform(0.5, 4.0, n),
            "distance_from_day_close": np.random.uniform(0.5, 15.0, n),
            "fvg_size": np.random.uniform(0.2, 2.0, n),
            "candles_between_ob_and_fvg": [2] * n,
            "has_associated_fvg": [True] * n,
            "tap_entry_boundary": np.random.uniform(1200, 2000, n),
            "was_tapped_in_horizon": [True] * n,
            "path_1r_status": ["target_first" if (i % 3 != 0) else "stop_first" for i in range(n)],
            "path_2r_status": ["target_first" if (i % 2 == 0) else "stop_first" for i in range(n)],
            "path_3r_status": ["target_first" if (i % 4 == 0) else "stop_first" for i in range(n)],
        })

    def test_strict_zero_lookahead_invariant(self):
        """
        Verify that perturbing Validation and Holdout data has ZERO effect on Training boundaries and strategy discovery.
        """
        train_df1, val_df1, holdout_df1 = split_datasets_by_period(self.df)
        boundaries1 = compute_training_feature_boundaries(train_df1)

        # Mutate validation and holdout drastically
        perturbed_df = self.df.copy()
        val_mask = pd.to_datetime(perturbed_df["candidate_timestamp"], utc=True) >= pd.Timestamp("2023-01-01", tz="UTC")
        perturbed_df.loc[val_mask, "distance_from_day_close"] *= 1000.0
        perturbed_df.loc[val_mask, "ob_range"] *= 500.0

        train_df2, val_df2, holdout_df2 = split_datasets_by_period(perturbed_df)
        boundaries2 = compute_training_feature_boundaries(train_df2)

        # Invariant 1: Training boundaries are strictly identical
        self.assertEqual(boundaries1.dist_q33, boundaries2.dist_q33)
        self.assertEqual(boundaries1.dist_q66, boundaries2.dist_q66)
        self.assertEqual(boundaries1.size_q25, boundaries2.size_q25)
        self.assertEqual(boundaries1.size_q50, boundaries2.size_q50)

        # Invariant 2: Discovered strategies on training partition are identical
        train_ann1 = apply_frozen_feature_boundaries(train_df1, boundaries1)
        train_ann2 = apply_frozen_feature_boundaries(train_df2, boundaries2)

        strats1, _ = discover_and_freeze_strategies(train_ann1, min_samples=10, top_n=5)
        strats2, _ = discover_and_freeze_strategies(train_ann2, min_samples=10, top_n=5)

        self.assertEqual(len(strats1), len(strats2))
        for s1, s2 in zip(strats1, strats2):
            self.assertEqual(s1.strategy_name, s2.strategy_name)
            self.assertEqual(s1.discovery_score, s2.discovery_score)

    def test_quantile_boundary_freezing(self):
        train_df, val_df, _ = split_datasets_by_period(self.df)
        boundaries = compute_training_feature_boundaries(train_df)

        val_ann = apply_frozen_feature_boundaries(val_df, boundaries)

        # Check that Near_Close on Validation is determined strictly by train_df's dist_q33
        near_close_val = val_ann[val_ann["dist_bucket"] == "Near_Close"]
        self.assertTrue((near_close_val["distance_from_day_close"] <= boundaries.dist_q33).all())

    def test_equity_curve_and_drawdown(self):
        mock_trades = pd.DataFrame({
            "candidate_timestamp": pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC"),
            "path_1r_status": ["target_first", "target_first", "stop_first", "stop_first", "stop_first", "target_first"],
        })
        eq_curve, max_dd, max_w, max_l, net_r = compute_equity_curve_and_drawdown(mock_trades)

        # Returns: +1, +1, -1, -1, -1, +1
        # Cum: 1, 2, 1, 0, -1, 0
        # Peak: 2 -> Trough: -1 -> Max DD: 3.0 R
        self.assertEqual(eq_curve, [1.0, 2.0, 1.0, 0.0, -1.0, 0.0])
        self.assertEqual(max_dd, 3.0)
        self.assertEqual(max_w, 2)
        self.assertEqual(max_l, 3)
        self.assertEqual(net_r, 0.0)

    def test_two_stage_validation_and_holdout_classification(self):
        train_df, val_df, holdout_df = split_datasets_by_period(self.df)
        boundaries = compute_training_feature_boundaries(train_df)

        train_ann = apply_frozen_feature_boundaries(train_df, boundaries)
        val_ann = apply_frozen_feature_boundaries(val_df, boundaries)
        holdout_ann = apply_frozen_feature_boundaries(holdout_df, boundaries)

        candidate_strategies, funnel = discover_and_freeze_strategies(train_ann, min_samples=10, top_n=5)
        self.assertGreater(len(candidate_strategies), 0)

        # Evaluate Train
        train_perfs = {
            s.strategy_id: evaluate_strategy_on_period(train_ann, s, "Train_2016_2022", min_samples=10)
            for s in candidate_strategies
        }

        # Stage 2: Validation Filtering
        survivors, val_perfs, val_statuses, deg_flags = filter_validation_survivors(
            val_ann,
            candidate_strategies,
            train_perfs,
            min_validation_samples=10,
            degradation_warning_threshold=15.0,
        )

        # Stage 4: Holdout Evaluation
        results, holdout_perfs = evaluate_holdout_and_classify(
            holdout_ann,
            candidate_strategies,
            survivors,
            train_perfs,
            val_perfs,
            val_statuses,
            deg_flags,
            min_holdout_samples=20,
            min_holdout_partial_samples=10,
        )

        self.assertEqual(len(results), len(candidate_strategies))
        for res in results:
            self.assertIn("strategy_id", res.strategy.__dict__)
            self.assertIn(res.overall_classification, [
                "VERIFIED_OOS_EDGE",
                "POSITIVE_BUT_INSUFFICIENT_OOS_SAMPLE",
                "INSUFFICIENT_HOLDOUT_SAMPLE",
                "FAILED_IN_HOLDOUT",
            ] + [f"REJECTED_IN_VALIDATION ({v})" for v in ["INSUFFICIENT_VALIDATION_SAMPLE", "FAILED_NEGATIVE_EXPECTANCY", "FAILED_AMBIGUITY_VULNERABILITY"]])

    def test_breakdown_tables(self):
        train_df, _, _ = split_datasets_by_period(self.df)
        boundaries = compute_training_feature_boundaries(train_df)
        ann_df = apply_frozen_feature_boundaries(self.df, boundaries)

        strats, _ = discover_and_freeze_strategies(ann_df, min_samples=10, top_n=2)
        yearly_df, monthly_df, equity_df = compute_walk_forward_breakdown_tables(ann_df, strats)

        self.assertIn("year", yearly_df.columns)
        self.assertIn("net_r", yearly_df.columns)
        self.assertIn("year_month", monthly_df.columns)
        self.assertIn("cumulative_equity_r", equity_df.columns)


if __name__ == "__main__":
    unittest.main()
