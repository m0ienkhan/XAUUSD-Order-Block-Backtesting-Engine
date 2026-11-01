import unittest
import pandas as pd
import numpy as np

from src.analytics.metrics import OBDirection
from src.analytics.path_analysis import (
    evaluate_single_target_path,
    scan_order_block_path_outcomes,
    batch_scan_path_dependent_outcomes,
)
from src.analytics.statistical_analysis import (
    CREATION_TIME_FEATURES,
    compute_summary_stats,
    analyze_by_direction,
    analyze_by_ob_size,
    analyze_by_time_and_session,
    analyze_by_fvg_characteristics,
    analyze_by_distance_features,
    analyze_unconditional_targets,
    summarize_path_dependent_results,
    analyze_mfe_mae_relationship,
)


class TestPathDependentTargetAnalysis(unittest.TestCase):

    def test_bullish_target_before_stop(self):
        # Bullish OB: entry = 1000.0, range = 10.0, target 2R = 1020.0, stop 1R = 990.0
        # Candle 0: H=1005, L=995 (no target/stop)
        # Candle 1: H=1022, L=998 (hits target 1020, does not touch stop 990)
        ts = pd.date_range("2026-01-01 00:00", periods=2, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1005.0, 1022.0],
            "low": [995.0, 998.0],
            "close": [1002.0, 1021.0],
        }, index=ts)

        res = evaluate_single_target_path(
            direction="bullish",
            tap_entry_boundary=1000.0,
            ob_range=10.0,
            target_r=2.0,
            stop_r=1.0,
            post_tap_candles=df_post,
        )

        self.assertEqual(res["status"], "target_first")
        self.assertEqual(res["resolution_timestamp"], ts[1])
        self.assertEqual(res["candles_to_resolution"], 1)
        self.assertEqual(res["target_price"], 1020.0)
        self.assertEqual(res["stop_price"], 990.0)

    def test_bullish_stop_before_target(self):
        # Bullish OB: entry = 1000.0, range = 10.0, target 2R = 1020.0, stop 1R = 990.0
        # Candle 0: H=1005, L=988 (hits stop 990, does not touch target 1020)
        ts = pd.date_range("2026-01-01 00:00", periods=1, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1005.0],
            "low": [988.0],
            "close": [992.0],
        }, index=ts)

        res = evaluate_single_target_path(
            direction="bullish",
            tap_entry_boundary=1000.0,
            ob_range=10.0,
            target_r=2.0,
            stop_r=1.0,
            post_tap_candles=df_post,
        )

        self.assertEqual(res["status"], "stop_first")
        self.assertEqual(res["resolution_timestamp"], ts[0])
        self.assertEqual(res["candles_to_resolution"], 0)

    def test_bearish_target_before_stop(self):
        # Bearish OB: entry = 1000.0, range = 10.0, target 2R = 980.0, stop 1R = 1010.0
        # Candle 0: H=1002, L=978 (hits target 980, does not touch stop 1010)
        ts = pd.date_range("2026-01-01 00:00", periods=1, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1002.0],
            "low": [978.0],
            "close": [982.0],
        }, index=ts)

        res = evaluate_single_target_path(
            direction="bearish",
            tap_entry_boundary=1000.0,
            ob_range=10.0,
            target_r=2.0,
            stop_r=1.0,
            post_tap_candles=df_post,
        )

        self.assertEqual(res["status"], "target_first")
        self.assertEqual(res["resolution_timestamp"], ts[0])
        self.assertEqual(res["target_price"], 980.0)
        self.assertEqual(res["stop_price"], 1010.0)

    def test_ambiguous_same_candle(self):
        # Both target and stop touched in the same candle
        # Bullish OB: entry = 1000, target 2R = 1020, stop 1R = 990
        # Candle 0: H=1025, L=985 (wide bar touching BOTH levels)
        ts = pd.date_range("2026-01-01 00:00", periods=1, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1025.0],
            "low": [985.0],
            "close": [1005.0],
        }, index=ts)

        res = evaluate_single_target_path(
            direction="bullish",
            tap_entry_boundary=1000.0,
            ob_range=10.0,
            target_r=2.0,
            stop_r=1.0,
            post_tap_candles=df_post,
        )

        self.assertEqual(res["status"], "ambiguous_same_candle")
        self.assertEqual(res["resolution_timestamp"], ts[0])

    def test_neither_level_reached(self):
        # Bullish OB: entry = 1000, target 2R = 1020, stop 1R = 990
        # Candle 0: H=1005, L=995
        # Candle 1: H=1010, L=998
        ts = pd.date_range("2026-01-01 00:00", periods=2, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1005.0, 1010.0],
            "low": [995.0, 998.0],
            "close": [1002.0, 1008.0],
        }, index=ts)

        res = evaluate_single_target_path(
            direction="bullish",
            tap_entry_boundary=1000.0,
            ob_range=10.0,
            target_r=2.0,
            stop_r=1.0,
            post_tap_candles=df_post,
        )

        self.assertEqual(res["status"], "no_resolution_within_horizon")
        self.assertTrue(pd.isna(res["resolution_timestamp"]))
        self.assertTrue(pd.isna(res["candles_to_resolution"]))


class TestStatisticalAnalysisModules(unittest.TestCase):

    def setUp(self):
        # Synthetic enriched dataset
        self.df_enriched = pd.DataFrame([
            {
                "trading_day": "2026-01-01",
                "candidate_timestamp": pd.Timestamp("2026-01-01 08:30:00", tz="UTC"),
                "candidate_position": 10,
                "ob_direction": "bullish",
                "extreme_type": "daily_low",
                "candidate_open": 1050.0,
                "candidate_high": 1055.0,
                "candidate_low": 1045.0,
                "candidate_close": 1052.0,
                "ob_range": 10.0,
                "day_close": 1060.0,
                "distance_from_day_close": 5.0,
                "tap_entry_boundary": 1055.0,
                "has_associated_fvg": True,
                "fvg_size": 4.0,
                "candles_between_ob_and_fvg": 2,
                "was_future_tapped": True,
                "mfe_r": 2.5,
                "mae_r": 0.5,
                "mfe_price": 25.0,
                "mae_price": 5.0,
            },
            {
                "trading_day": "2026-01-02",
                "candidate_timestamp": pd.Timestamp("2026-01-02 14:00:00", tz="UTC"),
                "candidate_position": 40,
                "ob_direction": "bearish",
                "extreme_type": "daily_high",
                "candidate_open": 1100.0,
                "candidate_high": 1105.0,
                "candidate_low": 1095.0,
                "candidate_close": 1098.0,
                "ob_range": 10.0,
                "day_close": 1085.0,
                "distance_from_day_close": 10.0,
                "tap_entry_boundary": 1095.0,
                "has_associated_fvg": True,
                "fvg_size": 8.0,
                "candles_between_ob_and_fvg": 1,
                "was_future_tapped": True,
                "mfe_r": 1.2,
                "mae_r": 1.5,
                "mfe_price": 12.0,
                "mae_price": 15.0,
            },
            {
                "trading_day": "2026-01-03",
                "candidate_timestamp": pd.Timestamp("2026-01-03 04:00:00", tz="UTC"),
                "candidate_position": 70,
                "ob_direction": "bullish",
                "extreme_type": "daily_low",
                "candidate_open": 1040.0,
                "candidate_high": 1045.0,
                "candidate_low": 1035.0,
                "candidate_close": 1042.0,
                "ob_range": 10.0,
                "day_close": 1050.0,
                "distance_from_day_close": 5.0,
                "tap_entry_boundary": 1045.0,
                "has_associated_fvg": True,
                "fvg_size": 2.0,
                "candles_between_ob_and_fvg": 3,
                "was_future_tapped": False,
                "mfe_r": np.nan,
                "mae_r": np.nan,
                "mfe_price": np.nan,
                "mae_price": np.nan,
            }
        ])

    def test_summary_stats(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_summary_stats(s, min_samples=3)
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["median"], 3.0)
        self.assertFalse(stats["insufficient_sample"])

    def test_analyze_by_direction(self):
        df_res = analyze_by_direction(self.df_enriched, min_samples=1)
        self.assertEqual(len(df_res), 3)  # all, bullish, bearish
        bullish_row = df_res[df_res["segment"] == "bullish"].iloc[0]
        self.assertEqual(bullish_row["total_obs"], 2)
        self.assertEqual(bullish_row["tapped_obs"], 1)
        self.assertEqual(bullish_row["tap_rate_pct"], 50.0)

    def test_unconditional_targets(self):
        df_targets = analyze_unconditional_targets(self.df_enriched, target_multiples=[1.0, 2.0, 3.0])
        self.assertEqual(len(df_targets), 3)
        # Tapped obs = 2 (mfe_r: 2.5, 1.2) -> both >= 1.0 -> 100%
        # >= 2.0 -> 1 of 2 -> 50%
        # >= 3.0 -> 0 of 2 -> 0%
        self.assertEqual(df_targets.loc[df_targets["target_multiple"] == "1.0R", "reached_count"].iloc[0], 2)
        self.assertEqual(df_targets.loc[df_targets["target_multiple"] == "2.0R", "reached_count"].iloc[0], 1)
        self.assertEqual(df_targets.loc[df_targets["target_multiple"] == "3.0R", "reached_count"].iloc[0], 0)

    def test_data_leakage_prevention(self):
        """Verify that CREATION_TIME_FEATURES contains only pre-future variables."""
        forbidden_terms = ["mfe", "mae", "tapped", "future", "penetration", "resolution"]
        for feature in CREATION_TIME_FEATURES:
            for term in forbidden_terms:
                self.assertNotIn(term, feature.lower(), f"Potential data leakage feature found: {feature}")


if __name__ == "__main__":
    unittest.main()
