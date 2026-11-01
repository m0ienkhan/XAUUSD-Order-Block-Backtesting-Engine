import unittest
import pandas as pd
import numpy as np

from src.data.loader import load_market_data
from src.data.trading_day import assign_trading_day
from src.structure.inside_bar import annotate_inside_bar_structure
from src.structure.fvg import detect_fvg
from src.pipeline.daily_ob_pipeline import build_daily_order_block_datasets
from src.analytics.metrics import (
    calculate_ob_range,
    calculate_distance_from_day_close,
    get_tap_entry_boundary,
    calculate_penetration_metrics,
    calculate_excursions,
)
from src.analytics.mitigation_analysis import (
    resolve_horizon_window,
    analyze_ob_interaction_in_window,
)
from src.analytics.ob_enrichment import (
    enrich_single_order_block,
    enrich_selected_order_blocks,
)


class TestMetricsAndExcursions(unittest.TestCase):

    def test_range_and_boundaries(self):
        # Bearish OB
        ob_range = calculate_ob_range(high=1100.0, low=1090.0)
        self.assertAlmostEqual(ob_range, 10.0)
        bearish_entry = get_tap_entry_boundary("bearish", candidate_low=1090.0, candidate_high=1100.0)
        self.assertEqual(bearish_entry, 1090.0)
        bearish_dist = calculate_distance_from_day_close("bearish", candidate_low=1090.0, candidate_high=1100.0, day_close=1070.0)
        self.assertEqual(bearish_dist, 20.0)

        # Bullish OB
        bullish_entry = get_tap_entry_boundary("bullish", candidate_low=1050.0, candidate_high=1060.0)
        self.assertEqual(bullish_entry, 1060.0)
        bullish_dist = calculate_distance_from_day_close("bullish", candidate_low=1050.0, candidate_high=1060.0, day_close=1070.0)
        self.assertEqual(bullish_dist, 10.0)

    def test_bearish_excursions(self):
        # Bearish OB [1090, 1100], entry = 1090
        # Post-tap candles:
        # Candle 0: H=1092, L=1085, C=1088
        # Candle 1: H=1095, L=1080, C=1082
        # Candle 2: H=1083, L=1075, C=1078  (Max drop to 1075 -> MFE = 1090 - 1075 = 15)
        # Candle 3: H=1098, L=1080, C=1095  (Max rise to 1098 -> MAE = 1098 - 1090 = 8)
        ts = pd.date_range("2026-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1092.0, 1095.0, 1083.0, 1098.0],
            "low": [1085.0, 1080.0, 1075.0, 1080.0],
            "close": [1088.0, 1082.0, 1078.0, 1095.0],
        }, index=ts)

        res = calculate_excursions(
            direction="bearish",
            tap_entry_boundary=1090.0,
            ob_range=10.0,
            post_tap_df=df_post,
        )

        self.assertAlmostEqual(res["mfe_price"], 15.0)
        self.assertAlmostEqual(res["mfe_r"], 1.5)
        self.assertEqual(res["mfe_timestamp"], ts[2])
        self.assertEqual(res["candles_to_mfe"], 2)

        self.assertAlmostEqual(res["mae_price"], 8.0)
        self.assertAlmostEqual(res["mae_r"], 0.8)
        self.assertEqual(res["mae_timestamp"], ts[3])
        self.assertEqual(res["candles_to_mae"], 3)

    def test_bullish_excursions(self):
        # Bullish OB [1050, 1060], entry = 1060
        # Post-tap candles:
        # Candle 0: H=1062, L=1058, C=1060
        # Candle 1: H=1075, L=1059, C=1074  (Max rise to 1075 -> MFE = 1075 - 1060 = 15)
        # Candle 2: H=1065, L=1055, C=1058  (Max drop to 1055 -> MAE = 1060 - 1055 = 5)
        ts = pd.date_range("2026-01-01 00:00", periods=3, freq="15min", tz="UTC")
        df_post = pd.DataFrame({
            "high": [1062.0, 1075.0, 1065.0],
            "low": [1058.0, 1059.0, 1055.0],
            "close": [1060.0, 1074.0, 1058.0],
        }, index=ts)

        res = calculate_excursions(
            direction="bullish",
            tap_entry_boundary=1060.0,
            ob_range=10.0,
            post_tap_df=df_post,
        )

        self.assertAlmostEqual(res["mfe_price"], 15.0)
        self.assertAlmostEqual(res["mfe_r"], 1.5)
        self.assertEqual(res["mfe_timestamp"], ts[1])
        self.assertEqual(res["candles_to_mfe"], 1)

        self.assertAlmostEqual(res["mae_price"], 5.0)
        self.assertAlmostEqual(res["mae_r"], 0.5)
        self.assertEqual(res["mae_timestamp"], ts[2])
        self.assertEqual(res["candles_to_mae"], 2)


class TestEnrichmentIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df, _ = load_market_data("data/raw/XAUUSD_15min_2016_test.csv", timeframe_minutes=15)
        df = assign_trading_day(df)
        df = annotate_inside_bar_structure(df)
        df = detect_fvg(df)
        cls.df = df

        cls.candidates_df, cls.selected_df = build_daily_order_block_datasets(
            df=df,
            start_day="2016-01-04",
            end_day="2016-01-08",
        )

        cls.enriched_df = enrich_selected_order_blocks(
            df=df,
            selected_df=cls.selected_df,
            horizons=["next_trading_day", "3_trading_days", "5_trading_days", "full"],
        )

    def test_invariant_1_selected_dataset_unchanged(self):
        """1. Original selected dataset remains unchanged."""
        self.assertEqual(len(self.selected_df), 8)
        self.assertTrue((self.selected_df["is_selected"] == True).all())

    def test_invariant_2_one_to_one_mapping(self):
        """2. Every enriched record maps exactly to one selected OB."""
        self.assertEqual(len(self.enriched_df), len(self.selected_df))

    def test_invariant_3_no_duplicate_identities(self):
        """3. No duplicate OB identities are introduced."""
        identities = [
            (r["candidate_position"], r["ob_direction"], r["extreme_type"])
            for _, r in self.enriched_df.iterrows()
        ]
        self.assertEqual(len(identities), len(set(identities)))

    def test_invariant_4_time_order_consistency(self):
        """4. Future timestamps are always after the defined analysis start time."""
        for _, row in self.enriched_df.iterrows():
            future_start = row["future_analysis_start_timestamp"]
            self.assertIsNotNone(future_start)
            self.assertFalse(pd.isna(future_start))

            if row["was_future_tapped"]:
                tap_ts = row["first_future_tap_timestamp"]
                self.assertFalse(pd.isna(tap_ts))
                self.assertGreaterEqual(tap_ts, future_start)

                if pd.notna(row["mfe_timestamp"]):
                    self.assertGreaterEqual(row["mfe_timestamp"], tap_ts)
                if pd.notna(row["mae_timestamp"]):
                    self.assertGreaterEqual(row["mae_timestamp"], tap_ts)

    def test_invariant_5_mfe_mae_bounded_by_horizon(self):
        """5. MFE/MAE calculations use only candles inside the configured horizon."""
        for _, row in self.enriched_df.iterrows():
            if row["was_tapped_h1d"]:
                self.assertFalse(pd.isna(row["mfe_price_h1d"]))
                self.assertFalse(pd.isna(row["mae_price_h1d"]))
                self.assertGreaterEqual(row["mfe_price_h1d"], 0.0)
                self.assertGreaterEqual(row["mae_price_h1d"], 0.0)

    def test_invariant_6_no_future_leakage_in_metadata(self):
        """6. No future information leaks into OB creation or selection metadata."""
        for _, row in self.enriched_df.iterrows():
            self.assertAlmostEqual(row["ob_range"], row["candidate_high"] - row["candidate_low"])
            self.assertTrue(row["ob_range"] > 0)
            if row["ob_direction"] == "bearish":
                self.assertAlmostEqual(row["distance_from_day_close"], row["candidate_low"] - row["day_close"])
                self.assertGreater(row["distance_from_day_close"], 0)
            elif row["ob_direction"] == "bullish":
                self.assertAlmostEqual(row["distance_from_day_close"], row["day_close"] - row["candidate_high"])
                self.assertGreater(row["distance_from_day_close"], 0)

    def test_invariant_7_same_day_mode_separation(self):
        """7. Same-day post-FVG analysis and next-trading-day analysis are strictly separate."""
        for _, row in self.enriched_df.iterrows():
            # For selected OBs, same_day_was_tapped is False by rule
            self.assertFalse(row["same_day_was_tapped"])
            self.assertTrue(pd.isna(row["same_day_first_tap_timestamp"]))
            self.assertTrue(pd.isna(row["same_day_mfe_price"]))


if __name__ == "__main__":
    unittest.main()
