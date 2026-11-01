import unittest
import math
import numpy as np

from src.execution.cost_model import (
    ExecutionCostModel,
    TradeCostBreakdown,
    compute_trade_cost_breakdown,
)


class TestExecutionCosts(unittest.TestCase):
    """
    Test suite verifying deterministic transaction cost accounting, component R conversion,
    identity invariants, monotonicity, and input validation.
    """

    def test_zero_cost_invariant(self):
        """
        TEST 1 — ZERO COST INVARIANT
        When all frictions are 0, all cost components must equal 0.0,
        and net_realized_r must equal gross_realized_r exactly across multiple outcomes.
        """
        zero_cost = ExecutionCostModel(
            spread_usd=0.0,
            entry_slippage_usd=0.0,
            exit_slippage_usd=0.0,
            commission_usd=0.0,
            commission_r_fixed=0.0,
        )

        test_cases = [
            (+1.0, 2.5),    # 1.0R Take Profit
            (-1.0, 3.0),    # -1.0R Stop Loss
            (+0.5234, 1.8), # Fractional positive horizon exit
            (-0.3512, 4.2), # Fractional negative horizon exit
            (0.0, 2.0),     # Scratch trade
        ]

        for gross_r, ob_range in test_cases:
            breakdown = zero_cost.compute_trade_cost_breakdown(gross_r, ob_range)
            self.assertEqual(breakdown.spread_cost_r, 0.0)
            self.assertEqual(breakdown.slippage_cost_r, 0.0)
            self.assertEqual(breakdown.commission_cost_r, 0.0)
            self.assertEqual(breakdown.total_cost_r, 0.0)
            self.assertEqual(breakdown.net_realized_r, gross_r)
            self.assertEqual(breakdown.gross_realized_r, gross_r)
            self.assertEqual(breakdown.ob_range, ob_range)

    def test_exact_cost_conversion(self):
        """
        TEST 2 — EXACT COST CONVERSION
        Deterministic synthetic example:
        ob_range = 2.00, spread = 0.20, entry_slippage = 0.05, exit_slippage = 0.05, commission = 0.10.
        Expected:
        spread_cost_r = 0.10, slippage_cost_r = 0.05, commission_cost_r = 0.05, total_cost_r = 0.20.
        """
        cost_model = ExecutionCostModel(
            spread_usd=0.20,
            entry_slippage_usd=0.05,
            exit_slippage_usd=0.05,
            commission_usd=0.10,
            commission_r_fixed=0.0,
        )

        ob_range = 2.00
        gross_r = 1.00  # +1.0R Win

        breakdown = cost_model.compute_trade_cost_breakdown(
            gross_realized_r=gross_r,
            ob_range=ob_range,
            direction="bearish",
            nominal_entry_price=100.0,
            nominal_exit_price=98.0,
        )

        self.assertAlmostEqual(breakdown.spread_cost_r, 0.10, places=9)
        self.assertAlmostEqual(breakdown.slippage_cost_r, 0.05, places=9)
        self.assertAlmostEqual(breakdown.commission_cost_r, 0.05, places=9)
        self.assertAlmostEqual(breakdown.total_cost_r, 0.20, places=9)
        self.assertAlmostEqual(breakdown.net_realized_r, 0.80, places=9)

        # Also verify for loss trade (-1.0R)
        loss_breakdown = cost_model.compute_trade_cost_breakdown(
            gross_realized_r=-1.00,
            ob_range=ob_range,
        )
        self.assertAlmostEqual(loss_breakdown.total_cost_r, 0.20, places=9)
        self.assertAlmostEqual(loss_breakdown.net_realized_r, -1.20, places=9)

    def test_net_r_identity(self):
        """
        TEST 3 — NET R IDENTITY
        Verify net_realized_r = gross_realized_r - (spread_cost_r + slippage_cost_r + commission_cost_r)
        with high precision across varied parameters.
        """
        cost_model = ExecutionCostModel(
            spread_usd=0.15,
            entry_slippage_usd=0.03,
            exit_slippage_usd=0.02,
            commission_usd=0.04,
            commission_r_fixed=0.005,
        )

        rng = np.random.default_rng(42)
        for _ in range(50):
            gross_r = float(rng.uniform(-1.5, 1.5))
            ob_range = float(rng.uniform(0.5, 10.0))

            b = cost_model.compute_trade_cost_breakdown(gross_r, ob_range)

            # High precision assertion: net == gross - total_cost
            expected_total = b.spread_cost_r + b.slippage_cost_r + b.commission_cost_r
            self.assertAlmostEqual(b.total_cost_r, expected_total, places=12)
            self.assertAlmostEqual(b.net_realized_r, gross_r - expected_total, places=12)

    def test_cost_monotonicity(self):
        """
        TEST 4 — COST MONOTONICITY
        For identical gross trade and ob_range:
        zero_cost < moderate_cost < high_cost must imply:
        net_r_zero >= net_r_moderate >= net_r_high.
        Increasing transaction costs must never improve net realized R.
        """
        zero_cost = ExecutionCostModel()
        moderate_cost = ExecutionCostModel(spread_usd=0.12, entry_slippage_usd=0.02, exit_slippage_usd=0.02, commission_usd=0.02)
        high_cost = ExecutionCostModel(spread_usd=0.30, entry_slippage_usd=0.05, exit_slippage_usd=0.05, commission_usd=0.05)

        for gross_r in [+1.0, -1.0, +0.3, -0.7]:
            for ob_r in [1.0, 2.5, 5.0]:
                b_zero = zero_cost.compute_trade_cost_breakdown(gross_r, ob_r)
                b_mod = moderate_cost.compute_trade_cost_breakdown(gross_r, ob_r)
                b_high = high_cost.compute_trade_cost_breakdown(gross_r, ob_r)

                self.assertGreater(b_zero.net_realized_r, b_mod.net_realized_r)
                self.assertGreater(b_mod.net_realized_r, b_high.net_realized_r)

    def test_break_even_basic_identity(self):
        """
        TEST 5 — BREAK-EVEN BASIC IDENTITY
        For a trade with gross_realized_r = +0.30 and total_cost_r = +0.30,
        verify net_realized_r == 0.0.
        For total_cost_r > gross_realized_r, verify net_realized_r < 0.
        """
        # Exactly 0.30 R total cost when ob_range = 1.00
        cost_be = ExecutionCostModel(spread_usd=0.20, entry_slippage_usd=0.05, exit_slippage_usd=0.05)
        b_be = cost_be.compute_trade_cost_breakdown(gross_realized_r=0.30, ob_range=1.00)
        self.assertAlmostEqual(b_be.net_realized_r, 0.0, places=9)

        # Friction exceeds gross return -> net negative
        cost_over = ExecutionCostModel(spread_usd=0.25, entry_slippage_usd=0.05, exit_slippage_usd=0.05)
        b_over = cost_over.compute_trade_cost_breakdown(gross_realized_r=0.30, ob_range=1.00)
        self.assertLess(b_over.net_realized_r, 0.0)

    def test_input_validation(self):
        """
        TEST 6 — INPUT VALIDATION
        Verify exceptions for invalid inputs:
        ob_range <= 0, negative spread/slippage/commission, NaN, infinity.
        """
        # Invalid ExecutionCostModel parameters
        with self.assertRaises(ValueError):
            ExecutionCostModel(spread_usd=-0.1)
        with self.assertRaises(ValueError):
            ExecutionCostModel(entry_slippage_usd=-0.05)
        with self.assertRaises(ValueError):
            ExecutionCostModel(exit_slippage_usd=-0.05)
        with self.assertRaises(ValueError):
            ExecutionCostModel(commission_usd=-0.02)
        with self.assertRaises(ValueError):
            ExecutionCostModel(commission_r_fixed=-0.01)
        with self.assertRaises(ValueError):
            ExecutionCostModel(spread_usd=float("nan"))
        with self.assertRaises(ValueError):
            ExecutionCostModel(spread_usd=float("inf"))

        # Invalid trade computation inputs
        valid_model = ExecutionCostModel(spread_usd=0.10)
        with self.assertRaises(ValueError):
            valid_model.compute_trade_cost_breakdown(gross_realized_r=1.0, ob_range=0.0)
        with self.assertRaises(ValueError):
            valid_model.compute_trade_cost_breakdown(gross_realized_r=1.0, ob_range=-2.0)
        with self.assertRaises(ValueError):
            valid_model.compute_trade_cost_breakdown(gross_realized_r=1.0, ob_range=float("nan"))
        with self.assertRaises(ValueError):
            valid_model.compute_trade_cost_breakdown(gross_realized_r=1.0, ob_range=float("inf"))
        with self.assertRaises(ValueError):
            valid_model.compute_trade_cost_breakdown(gross_realized_r=float("nan"), ob_range=2.0)
        with self.assertRaises(ValueError):
            valid_model.compute_trade_cost_breakdown(gross_realized_r=float("inf"), ob_range=2.0)

    def test_effective_prices_informational_audit(self):
        """
        TEST 7 — EFFECTIVE PRICES (INFORMATIONAL AUDIT)
        Verify that directional effective prices are correctly computed for bearish and bullish
        without altering the underlying nominal levels or nominal PnL logic.
        """
        model = ExecutionCostModel(spread_usd=0.20, entry_slippage_usd=0.05, exit_slippage_usd=0.05)

        # Bearish (Short Sell @ 100.0, Buy to Cover @ 98.0)
        # Half-spread = 0.10. Effective entry = 100.0 - 0.10 - 0.05 = 99.85
        # Effective exit = 98.0 + 0.10 + 0.05 = 98.15
        b_bearish = model.compute_trade_cost_breakdown(
            gross_realized_r=1.0,
            ob_range=2.0,
            direction="bearish",
            nominal_entry_price=100.0,
            nominal_exit_price=98.0,
        )
        self.assertEqual(b_bearish.effective_entry_price, 99.85)
        self.assertEqual(b_bearish.effective_exit_price, 98.15)

        # Price difference = 99.85 - 98.15 = 1.70 USD.
        # In R = 1.70 / 2.0 = 0.85 R (Gross 1.0 R - (Spread 0.10 R + Slippage 0.05 R) = 0.85 R)
        self.assertEqual(b_bearish.net_realized_r, 0.85)

        # Bullish (Long Buy @ 100.0, Sell to Close @ 102.0)
        # Effective entry = 100.0 + 0.10 + 0.05 = 100.15
        # Effective exit = 102.0 - 0.10 - 0.05 = 101.85
        b_bullish = model.compute_trade_cost_breakdown(
            gross_realized_r=1.0,
            ob_range=2.0,
            direction="bullish",
            nominal_entry_price=100.0,
            nominal_exit_price=102.0,
        )
        self.assertEqual(b_bullish.effective_entry_price, 100.15)
        self.assertEqual(b_bullish.effective_exit_price, 101.85)
        self.assertEqual(b_bullish.net_realized_r, 0.85)


if __name__ == "__main__":
    unittest.main()
