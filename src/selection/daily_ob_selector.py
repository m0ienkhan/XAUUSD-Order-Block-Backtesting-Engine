from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from src.structure.order_block import OrderBlockCandidate


OBDirection = Literal["bullish", "bearish"]


@dataclass
class DailyOBSelectionResult:
    """
    Final Order Block selection result for one trading day.

    At most two Order Blocks can be selected:

    - one bearish OB above the daily close
    - one bullish OB below the daily close
    """

    trading_day: object
    day_close: float

    selected_bearish_ob: Optional[OrderBlockCandidate]
    selected_bullish_ob: Optional[OrderBlockCandidate]

    valid_high_candidates: list[OrderBlockCandidate]
    valid_low_candidates: list[OrderBlockCandidate]


def get_day_close(
    df: pd.DataFrame,
    trading_day,
) -> float:
    """
    Return the closing price of the final candle
    belonging to the specified trading day.
    """

    day_rows = df[
        df["trading_day"] == trading_day
    ]

    if day_rows.empty:
        raise ValueError(
            f"No candles found for trading day: "
            f"{trading_day}"
        )

    return float(
        day_rows.iloc[-1]["close"]
    )


def select_extreme_candidate(
    candidates: list[OrderBlockCandidate],
    direction: OBDirection,
) -> Optional[OrderBlockCandidate]:
    """
    Select the most extreme candidate.

    Bearish:
        highest candidate high

    Bullish:
        lowest candidate low
    """

    if not candidates:
        return None

    if direction == "bearish":
        return max(
            candidates,
            key=lambda candidate: (
                candidate.candidate_high,
                -candidate.candidate_position,
            ),
        )

    if direction == "bullish":
        return min(
            candidates,
            key=lambda candidate: (
                candidate.candidate_low,
                candidate.candidate_position,
            ),
        )

    raise ValueError(
        f"Unsupported direction: {direction}"
    )


def select_daily_order_blocks(
    valid_high_candidates: list[OrderBlockCandidate],
    valid_low_candidates: list[OrderBlockCandidate],
    day_close: float,
    trading_day=None,
) -> DailyOBSelectionResult:
    """
    Select final Order Blocks for one trading day.

    Rules:

    - Select at most one bearish OB above day close.
    - Select at most one bullish OB below day close.
    - Bearish selection = highest valid bearish candidate.
    - Bullish selection = lowest valid bullish candidate.
    """

    valid_high_candidates = list(
        valid_high_candidates
    )

    valid_low_candidates = list(
        valid_low_candidates
    )

    bearish_candidates_above_close = [
        candidate
        for candidate in valid_high_candidates
        if (
            candidate.direction == "bearish"
            and candidate.candidate_low > day_close
        )
    ]

    bullish_candidates_below_close = [
        candidate
        for candidate in valid_low_candidates
        if (
            candidate.direction == "bullish"
            and candidate.candidate_high < day_close
        )
    ]

    selected_bearish_ob = select_extreme_candidate(
        candidates=bearish_candidates_above_close,
        direction="bearish",
    )

    selected_bullish_ob = select_extreme_candidate(
        candidates=bullish_candidates_below_close,
        direction="bullish",
    )

    return DailyOBSelectionResult(
        trading_day=trading_day,
        day_close=float(day_close),
        selected_bearish_ob=selected_bearish_ob,
        selected_bullish_ob=selected_bullish_ob,
        valid_high_candidates=valid_high_candidates,
        valid_low_candidates=valid_low_candidates,
    )
