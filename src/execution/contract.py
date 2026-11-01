from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from src.analytics.metrics import OBDirection
from src.execution.order import OrderSide


@dataclass(frozen=True)
class ExecutionContract:
    """
    Formal execution contract governing strategy eligibility, order placement,
    SL/TP levels, horizon boundaries, and ambiguity resolution policies.
    """
    strategy_id: str
    strategy_name: str
    target_r: float = 1.0
    stop_r: float = 1.0
    filter_dict: dict[str, str] = field(default_factory=dict)
    enforce_conservative_ambiguity: bool = True
    close_unresolved_at_horizon: bool = True
    allow_concurrent_positions: bool = True
    zero_friction: bool = True

    def is_ob_eligible(self, ob_row: pd.Series | dict[str, Any]) -> bool:
        """Check if an annotated Order Block record satisfies all contract filters."""
        for col, expected_val in self.filter_dict.items():
            if col not in ob_row:
                return False
            if str(ob_row[col]) != expected_val:
                return False
        return True

    def calculate_order_levels(
        self,
        direction: OBDirection,
        candidate_low: float,
        candidate_high: float,
    ) -> tuple[OrderSide, float, float, float, float]:
        """
        Calculate limit entry price, stop loss price, take profit price, and OB range.

        Returns
        -------
        tuple of (side, entry_price, stop_loss_price, take_profit_price, ob_range)
        """
        ob_range = max(candidate_high - candidate_low, 1e-8)

        if direction == "bearish":
            side = OrderSide.SELL
            entry_price = float(candidate_low)
            stop_loss_price = float(candidate_low + (self.stop_r * ob_range))  # candidate_high
            take_profit_price = float(candidate_low - (self.target_r * ob_range))
        elif direction == "bullish":
            side = OrderSide.BUY
            entry_price = float(candidate_high)
            stop_loss_price = float(candidate_high - (self.stop_r * ob_range))  # candidate_low
            take_profit_price = float(candidate_high + (self.target_r * ob_range))
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return side, entry_price, stop_loss_price, take_profit_price, ob_range


def load_strat_09_contract(
    summary_csv: Optional[Path] = None,
) -> ExecutionContract:
    """
    Load and validate the formal STRAT_09 execution contract from the walk-forward summary artifact.
    """
    if summary_csv is not None and summary_csv.exists():
        df = pd.read_csv(summary_csv)
        strat_rows = df[df["strategy_id"] == "STRAT_09"]
        if strat_rows.empty:
            raise ValueError("STRAT_09 not found in walk-forward summary CSV.")
        row = strat_rows.iloc[0]
        if row["overall_classification"] != "VERIFIED_OOS_EDGE":
            raise ValueError(f"STRAT_09 does not have VERIFIED_OOS_EDGE status: {row['overall_classification']}")

    # Canonical STRAT_09 frozen definition
    return ExecutionContract(
        strategy_id="STRAT_09",
        strategy_name="dist_bucket=Medium_Dist | ob_direction=bearish",
        target_r=1.0,
        stop_r=1.0,
        filter_dict={
            "dist_bucket": "Medium_Dist",
            "ob_direction": "bearish",
        },
        enforce_conservative_ambiguity=True,
        close_unresolved_at_horizon=True,
        allow_concurrent_positions=True,
        zero_friction=True,
    )
