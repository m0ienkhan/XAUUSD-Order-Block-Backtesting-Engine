import unittest
import pandas as pd
import numpy as np

from src.analytics.conditional_edge_analysis import (
    derive_creation_features,
    prepare_conditional_dataset,
    evaluate_conditional_group,
    analyze_all_feature_combinations,
    test_chronological_robustness,
    rank_feature_interactions,
    extract_top_hypotheses,
)


class TestConditionalEdgeAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset of 100 enriched order blocks and path outcomes
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2016-01-04 00:00:00", periods=n, freq="4h", tz="UTC")
        trading_days = [ts.strftime("%Y-%m-%d 00:00:00") for ts in timestamps]

        self.enriched_df = pd.DataFrame({
            "trading_day": trading_days,
            "candidate_position": range(n),
            "candidate_timestamp": timestamps,
            "ob_direction": ["bullish" if i % 2 == 0 else "bearish" for i in range(n)],
            "day_close": np.random.uniform(1100, 1300, n),
            "candidate_open": np.random.uniform(1100, 1300, n),
            "candidate_high": np.random.uniform(1100, 1300, n),
            "candidate_low": np.random.uniform(1100, 1300, n),
            "candidate_close": np.random.uniform(1100, 1300, n),
            "ob_range": np.random.uniform(0.5, 5.0, n),
            "distance_from_day_close": np.random.uniform(0.5, 20.0, n),
            "fvg_size": np.random.uniform(0.2, 2.0, n),
            "candles_between_ob_and_fvg": np.random.randint(1, 4, n),
            "has_associated_fvg": [True] * n,
            "tap_entry_boundary": np.random.uniform(1100, 1300, n),
        })

        statuses_1r = ["target_first"] * 45 + ["stop_first"] * 25 + ["ambiguous_same_candle"] * 10 + ["not_tapped"] * 20
        statuses_2r = ["target_first"] * 30 + ["stop_first"] * 40 + ["ambiguous_same_candle"] * 10 + ["not_tapped"] * 20
        statuses_3r = ["target_first"] * 20 + ["stop_first"] * 50 + ["ambiguous_same_candle"] * 10 + ["not_tapped"] * 20

        self.path_df = pd.DataFrame({
            "trading_day": trading_days,
            "candidate_position": range(n),
            "candidate_timestamp": timestamps,
            "ob_direction": ["bullish" if i % 2 == 0 else "bearish" for i in range(n)],
            "was_tapped_in_horizon": [s != "not_tapped" for s in statuses_1r],
            "path_1r_status": statuses_1r,
            "path_1r_candles_to_resolution": [5] * n,
            "path_2r_status": statuses_2r,
            "path_2r_candles_to_resolution": [8] * n,
            "path_3r_status": statuses_3r,
            "path_3r_candles_to_resolution": [12] * n,
        })

    def test_derive_creation_features(self):
        derived = derive_creation_features(self.enriched_df)
        self.assertIn("dist_bucket", derived.columns)
        self.assertIn("session", derived.columns)
        self.assertIn("size_bucket", derived.columns)
        self.assertIn("fvg_ratio_bucket", derived.columns)
        self.assertIn("period", derived.columns)

        # Check valid categories
        self.assertTrue(set(derived["dist_bucket"].unique()).issubset({"Near_Close", "Medium_Dist", "Far_Close"}))
        self.assertTrue(set(derived["session"].unique()).issubset({"1_Asian", "2_London", "3_NY_Morning", "4_NY_Late"}))
        self.assertTrue(set(derived["size_bucket"].unique()).issubset({"Q1_Small", "Q2_Medium", "Q3_Medium", "Q4_Large"}))

    def test_prepare_conditional_dataset(self):
        prepared = prepare_conditional_dataset(self.enriched_df, self.path_df)
        self.assertEqual(len(prepared), 100)
        self.assertIn("path_1r_status", prepared.columns)
        self.assertIn("dist_bucket", prepared.columns)

    def test_evaluate_conditional_group_metrics(self):
        prepared = prepare_conditional_dataset(self.enriched_df, self.path_df)
        res = evaluate_conditional_group(
            prepared,
            group_name="All",
            combination_type="1_Way",
            feature_dims="All",
            min_samples=20,
        )

        self.assertEqual(res["total_obs"], 100)
        self.assertEqual(res["tapped_obs"], 80)
        self.assertEqual(res["tap_rate_pct"], 80.0)

        # 1R checks: 45 target_first, 25 stop_first, 10 ambiguous
        self.assertEqual(res["target_first_1r"], 45)
        self.assertEqual(res["stop_first_1r"], 25)
        self.assertEqual(res["ambig_1r"], 10)
        self.assertEqual(res["clean_resolved_1r"], 70)

        # Clean win rate = 45 / 70 = 64.29%
        self.assertAlmostEqual(res["clean_win_rate_1r_pct"], 64.29, places=2)
        # Expectancy 1R = 0.642857 * 1 - (1 - 0.642857) * 1 = 0.2857
        self.assertAlmostEqual(res["expectancy_clean_1r"], 0.2857, places=3)
        # Total win rate = 45 / 100 = 45.0%
        self.assertEqual(res["total_win_rate_1r_pct"], 45.0)

    def test_ambiguity_bounds(self):
        prepared = prepare_conditional_dataset(self.enriched_df, self.path_df)
        res = evaluate_conditional_group(
            prepared,
            group_name="All",
            combination_type="1_Way",
            feature_dims="All",
            min_samples=20,
        )

        opt_wr = res["opt_win_rate_1r_pct"]
        clean_wr = res["clean_win_rate_1r_pct"]
        cons_wr = res["cons_win_rate_1r_pct"]

        # 45/80 (cons = 56.25%) <= 45/70 (clean = 64.29%) <= 55/80 (opt = 68.75%)
        self.assertLessEqual(cons_wr, clean_wr)
        self.assertLessEqual(clean_wr, opt_wr)

    def test_sample_size_protection(self):
        tiny_df = self.enriched_df.head(10).copy()
        tiny_path = self.path_df.head(10).copy()
        prepared = prepare_conditional_dataset(tiny_df, tiny_path)

        res = evaluate_conditional_group(
            prepared,
            group_name="Tiny",
            combination_type="1_Way",
            feature_dims="Tiny",
            min_samples=20,
        )

        self.assertTrue(res["insufficient_sample"])

    def test_chronological_robustness(self):
        # Create dataset spanning all 3 periods
        ts_p1 = pd.date_range("2016-06-01", periods=30, freq="D", tz="UTC")
        ts_p2 = pd.date_range("2020-06-01", periods=30, freq="D", tz="UTC")
        ts_p3 = pd.date_range("2024-06-01", periods=30, freq="D", tz="UTC")
        all_ts = ts_p1.union(ts_p2).union(ts_p3)
        n = len(all_ts)

        e_df = pd.DataFrame({
            "trading_day": [ts.strftime("%Y-%m-%d 00:00:00") for ts in all_ts],
            "candidate_position": range(n),
            "candidate_timestamp": all_ts,
            "ob_direction": ["bullish"] * n,
            "day_close": [1200.0] * n,
            "ob_range": [1.5] * n,
            "distance_from_day_close": [3.0] * n,
            "fvg_size": [0.5] * n,
        })
        p_df = pd.DataFrame({
            "trading_day": [ts.strftime("%Y-%m-%d 00:00:00") for ts in all_ts],
            "candidate_position": range(n),
            "candidate_timestamp": all_ts,
            "ob_direction": ["bullish"] * n,
            "was_tapped_in_horizon": [True] * n,
            "path_1r_status": ["target_first"] * 20 + ["stop_first"] * 10 + ["target_first"] * 20 + ["stop_first"] * 10 + ["target_first"] * 20 + ["stop_first"] * 10,
        })

        prep = prepare_conditional_dataset(e_df, p_df)
        comb_df = analyze_all_feature_combinations(prep, min_samples=5)
        rob_df = test_chronological_robustness(prep, comb_df, min_period_samples=5)

        self.assertIn("positive_periods_count", rob_df.columns)
        self.assertIn("is_consistently_profitable", rob_df.columns)

    def test_ranking_and_tiering(self):
        prepared = prepare_conditional_dataset(self.enriched_df, self.path_df)
        comb_df = analyze_all_feature_combinations(prepared, min_samples=10)
        rob_df = test_chronological_robustness(prepared, comb_df, min_period_samples=5)
        ranked = rank_feature_interactions(comb_df, rob_df)

        self.assertIn("composite_edge_score", ranked.columns)
        self.assertIn("tier", ranked.columns)
        self.assertTrue((ranked["composite_edge_score"].diff().dropna() <= 0).all())


if __name__ == "__main__":
    unittest.main()
