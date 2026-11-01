import unittest
import pandas as pd
import numpy as np

from src.execution.contract import ExecutionContract
from src.execution.order import OrderSide, OrderStatus, TradeResolution
from src.execution.simulator import ExecutionSimulator


class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        # Create synthetic 15m price bars for Day 2020-01-05 (Day T) and Day 2020-01-06 (Day T+1)
        day_t_ts = pd.date_range("2020-01-05 00:00:00", "2020-01-05 23:45:00", freq="15min", tz="UTC")
        day_t1_ts = pd.date_range("2020-01-06 00:00:00", "2020-01-06 23:45:00", freq="15min", tz="UTC")

        all_ts = day_t_ts.union(day_t1_ts)
        n = len(all_ts)

        self.market_df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100.0] * n,
            "trading_day": ["2020-01-05 00:00:00"] * len(day_t_ts) + ["2020-01-06 00:00:00"] * len(day_t1_ts),
        }, index=all_ts)

    def test_bearish_take_profit_execution(self):
        """Bearish OB: Limit Sell @ 100.0, SL @ 102.0, TP @ 98.0. Price taps 100.0 and drops to 97.5."""
        ob_df = pd.DataFrame([{
            "trading_day": "2020-01-05 00:00:00",
            "candidate_position": 10,
            "candidate_timestamp": pd.Timestamp("2020-01-05 10:00:00", tz="UTC"),
            "ob_direction": "bearish",
            "candidate_low": 100.0,
            "candidate_high": 102.0,
            "ob_range": 2.0,
            "dist_bucket": "Medium_Dist",
        }])

        # Modify Day T+1 bars: Bar 0 taps 100.2 (fill), Bar 1 drops to 97.8 (TP hit)
        m_df = self.market_df.copy()
        t1_indices = m_df[m_df["trading_day"] == "2020-01-06 00:00:00"].index
        m_df.loc[t1_indices[0], "high"] = 100.5
        m_df.loc[t1_indices[0], "low"] = 99.8
        m_df.loc[t1_indices[1], "low"] = 97.5  # Below TP (98.0)

        contract = ExecutionContract(
            strategy_id="TEST_BEARISH_TP",
            strategy_name="test_bearish",
            filter_dict={"dist_bucket": "Medium_Dist", "ob_direction": "bearish"},
        )
        sim = ExecutionSimulator(m_df, ob_df, contract)
        res = sim.run_simulation(track="TRACK_A_CONSERVATIVE")

        self.assertEqual(len(res.trades), 1)
        trade = res.trades[0]
        self.assertEqual(trade.resolution, TradeResolution.TAKE_PROFIT)
        self.assertEqual(trade.realized_r, 1.0)
        self.assertTrue(trade.is_win)
        self.assertEqual(trade.entry_price, 100.0)
        self.assertEqual(trade.exit_price, 98.0)

    def test_bullish_stop_loss_execution(self):
        """Bullish OB: Limit Buy @ 100.0, SL @ 98.0, TP @ 102.0. Price drops to 99.8 (fill) then 97.5 (SL hit)."""
        ob_df = pd.DataFrame([{
            "trading_day": "2020-01-05 00:00:00",
            "candidate_position": 10,
            "candidate_timestamp": pd.Timestamp("2020-01-05 10:00:00", tz="UTC"),
            "ob_direction": "bullish",
            "candidate_low": 98.0,
            "candidate_high": 100.0,
            "ob_range": 2.0,
            "dist_bucket": "Medium_Dist",
        }])

        m_df = self.market_df.copy()
        t1_indices = m_df[m_df["trading_day"] == "2020-01-06 00:00:00"].index
        m_df.loc[t1_indices[0], "low"] = 99.5  # Below entry (100.0)
        m_df.loc[t1_indices[1], "low"] = 97.5  # Below SL (98.0)

        contract = ExecutionContract(
            strategy_id="TEST_BULLISH_SL",
            strategy_name="test_bullish",
            filter_dict={"dist_bucket": "Medium_Dist", "ob_direction": "bullish"},
        )
        sim = ExecutionSimulator(m_df, ob_df, contract)
        res = sim.run_simulation(track="TRACK_A_CONSERVATIVE")

        self.assertEqual(len(res.trades), 1)
        trade = res.trades[0]
        self.assertEqual(trade.resolution, TradeResolution.STOP_LOSS)
        self.assertEqual(trade.realized_r, -1.0)
        self.assertTrue(trade.is_loss)

    def test_conservative_ambiguity_enforcement(self):
        """Same-candle SL and TP breach must resolve as STOP_LOSS_FIRST (-1.0R) in Track A."""
        ob_df = pd.DataFrame([{
            "trading_day": "2020-01-05 00:00:00",
            "candidate_position": 10,
            "candidate_timestamp": pd.Timestamp("2020-01-05 10:00:00", tz="UTC"),
            "ob_direction": "bearish",
            "candidate_low": 100.0,
            "candidate_high": 102.0,
            "ob_range": 2.0,
            "dist_bucket": "Medium_Dist",
        }])

        m_df = self.market_df.copy()
        t1_indices = m_df[m_df["trading_day"] == "2020-01-06 00:00:00"].index
        # Bar 0 spans both SL (102.0) and TP (98.0)
        m_df.loc[t1_indices[0], "high"] = 103.0
        m_df.loc[t1_indices[0], "low"] = 97.0

        contract = ExecutionContract(
            strategy_id="TEST_AMBIG",
            strategy_name="test_ambig",
            filter_dict={"dist_bucket": "Medium_Dist", "ob_direction": "bearish"},
        )
        sim = ExecutionSimulator(m_df, ob_df, contract)
        res_a = sim.run_simulation(track="TRACK_A_CONSERVATIVE")
        res_b = sim.run_simulation(track="TRACK_B_RESEARCH")

        # Track A must force -1.0 R Loss
        self.assertEqual(res_a.trades[0].resolution, TradeResolution.AMBIGUOUS_STOPPED_OUT)
        self.assertEqual(res_a.trades[0].realized_r, -1.0)
        self.assertTrue(res_a.trades[0].is_loss)

        # Track B must mark ambiguous without booking clean PnL
        self.assertEqual(res_b.trades[0].is_ambiguous, True)
        self.assertTrue(np.isnan(res_b.trades[0].realized_r))

    def test_horizon_expiration_fractional_r(self):
        """Unresolved position closed on final candle of Day T+1 with fractional realized R."""
        ob_df = pd.DataFrame([{
            "trading_day": "2020-01-05 00:00:00",
            "candidate_position": 10,
            "candidate_timestamp": pd.Timestamp("2020-01-05 10:00:00", tz="UTC"),
            "ob_direction": "bearish",
            "candidate_low": 100.0,
            "candidate_high": 102.0,
            "ob_range": 2.0,
            "dist_bucket": "Medium_Dist",
        }])

        m_df = self.market_df.copy()
        t1_indices = m_df[m_df["trading_day"] == "2020-01-06 00:00:00"].index
        # Fill on Bar 0
        m_df.loc[t1_indices[0], "high"] = 100.5
        # Set final bar close to 99.0 (floating +1.0 USD = +0.5 R)
        m_df.loc[t1_indices[-1], "close"] = 99.0

        contract = ExecutionContract(
            strategy_id="TEST_HORIZON",
            strategy_name="test_horizon",
            filter_dict={"dist_bucket": "Medium_Dist", "ob_direction": "bearish"},
        )
        sim = ExecutionSimulator(m_df, ob_df, contract)
        res = sim.run_simulation(track="TRACK_A_CONSERVATIVE")

        self.assertEqual(len(res.trades), 1)
        trade = res.trades[0]
        self.assertEqual(trade.resolution, TradeResolution.HORIZON_EXPIRATION)
        self.assertEqual(trade.realized_r, 0.5)  # (100.0 - 99.0) / 2.0 = +0.5 R
        self.assertTrue(trade.is_win)

    def test_unfilled_order_expiration(self):
        """Order never touched during Day T+1 must expire with EXPIRED_UNFILLED and 0.0 R."""
        ob_df = pd.DataFrame([{
            "trading_day": "2020-01-05 00:00:00",
            "candidate_position": 10,
            "candidate_timestamp": pd.Timestamp("2020-01-05 10:00:00", tz="UTC"),
            "ob_direction": "bearish",
            "candidate_low": 150.0,  # Far away, price will never touch
            "candidate_high": 152.0,
            "ob_range": 2.0,
            "dist_bucket": "Medium_Dist",
        }])

        contract = ExecutionContract(
            strategy_id="TEST_UNFILLED",
            strategy_name="test_unfilled",
            filter_dict={"dist_bucket": "Medium_Dist", "ob_direction": "bearish"},
        )
        sim = ExecutionSimulator(self.market_df, ob_df, contract)
        res = sim.run_simulation(track="TRACK_A_CONSERVATIVE")

        self.assertEqual(len(res.trades), 0)
        self.assertEqual(res.summary.unfilled_orders, 1)
        self.assertEqual(res.summary.filled_trades, 0)
        self.assertEqual(res.summary.total_realized_r, 0.0)


if __name__ == "__main__":
    unittest.main()
