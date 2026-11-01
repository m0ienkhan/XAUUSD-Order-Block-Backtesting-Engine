from __future__ import annotations

from typing import Optional

import pandas as pd

from src.structure.daily_high import calculate_daily_extremes
from src.structure.order_block import (
    OrderBlockCandidate,
    scan_order_block_candidates,
    get_valid_same_day_candidates,
)
from src.selection.daily_ob_selector import (
    get_day_close,
    select_daily_order_blocks,
)


def order_block_to_record(
    candidate: OrderBlockCandidate,
    trading_day,
    day_close: float,
) -> dict:
    """
    Convert an OrderBlockCandidate into a flat dictionary.

    This record preserves candidate-level metadata for
    future research and statistical analysis.
    """

    return {
        # Trading day context
        "trading_day": trading_day,
        "day_close": day_close,

        # Candidate identity
        "candidate_position": candidate.candidate_position,
        "candidate_timestamp": candidate.candidate_timestamp,

        # Candidate OHLC
        "candidate_open": candidate.candidate_open,
        "candidate_high": candidate.candidate_high,
        "candidate_low": candidate.candidate_low,
        "candidate_close": candidate.candidate_close,

        # Extreme context
        "extreme_type": candidate.extreme_type,
        "ob_direction": candidate.direction,

        # Inside bar metadata
        "is_inside_bar": candidate.is_inside_bar,
        "mother_candle_position": candidate.mother_candle_position,
        "mother_candle_timestamp": candidate.mother_candle_timestamp,

        # FVG metadata
        "has_associated_fvg": candidate.has_associated_fvg,
        "fvg_position": candidate.fvg_position,
        "fvg_timestamp": candidate.fvg_timestamp,
        "fvg_type": candidate.fvg_type,
        "fvg_lower": candidate.fvg_lower,
        "fvg_upper": candidate.fvg_upper,
        "fvg_size": candidate.fvg_size,

        # Same-day tap analysis
        "was_tapped_same_day": candidate.was_tapped_same_day,
        "first_tap_position": candidate.first_tap_position,
        "first_tap_timestamp": candidate.first_tap_timestamp,

        # Candidate validity
        "is_valid_same_day": candidate.is_valid_same_day,

        # Selection metadata
        "is_selected": False,
        "selection_side": None,
        "selection_reason": None,
    }



def get_candidate_identity(
    candidate: OrderBlockCandidate,
) -> tuple:
    """
    Return the identity of an OrderBlockCandidate.

    Must match get_record_identity().
    """

    return (
        candidate.candidate_position,
        candidate.direction,
        candidate.extreme_type,
    )


def get_record_identity(
    record: dict,
) -> tuple:
    """
    Return the identity of a candidate record.

    Must match get_candidate_identity().
    """

    return (
        record["candidate_position"],
        record["ob_direction"],
        record["extreme_type"],
    )


def get_candidate_side(
    candidate: OrderBlockCandidate,
    day_close: float,
) -> Optional[str]:
    """
    Determine where the candidate sits relative to
    the trading day's close.

    Returns:
        "above_close"
        "below_close"
        None
    """

    if candidate.candidate_low > day_close:
        return "above_close"

    if candidate.candidate_high < day_close:
        return "below_close"

    return None


def mark_candidate_selection(
    records: list[dict],
    selected_bearish: Optional[OrderBlockCandidate],
    selected_bullish: Optional[OrderBlockCandidate],
) -> None:
    """
    Mark only the exact selected candidates.

    Matching uses the complete candidate identity:

        candidate_position
        direction
        extreme_type

    This prevents multiple candidates at the same candle
    position from being incorrectly marked as selected.
    """

    selected_bearish_identity = None
    selected_bullish_identity = None

    if selected_bearish is not None:
        selected_bearish_identity = (
            get_candidate_identity(
                selected_bearish
            )
        )

    if selected_bullish is not None:
        selected_bullish_identity = (
            get_candidate_identity(
                selected_bullish
            )
        )

    for record in records:

        record_identity = (
            get_record_identity(
                record
            )
        )

        if (
            selected_bearish_identity is not None
            and record_identity
            == selected_bearish_identity
        ):

            record["is_selected"] = True

            record["selection_side"] = (
                "bearish"
            )

            record["selection_reason"] = (
                "selected_bearish_ob_above_day_close"
            )

        elif (
            selected_bullish_identity is not None
            and record_identity
            == selected_bullish_identity
        ):

            record["is_selected"] = True

            record["selection_side"] = (
                "bullish"
            )

            record["selection_reason"] = (
                "selected_bullish_ob_below_day_close"
            )


def assign_candidate_rejection_reasons(
    records: list[dict],
    day_close: float,
) -> None:
    """
    Assign a detailed rejection reason to every
    non-selected candidate.

    Rejection priority:

    1. No associated FVG
    2. Tapped same day
    3. Candidate invalid for another reason
    4. Candidate not fully above/below day close
    5. Valid candidate but not the extreme selected candidate
    """

    for record in records:

        if record["is_selected"]:
            continue

        # --------------------------------------------------
        # No associated FVG
        # --------------------------------------------------

        if not record["has_associated_fvg"]:

            record["selection_reason"] = (
                "rejected_no_associated_fvg"
            )

            continue

        # --------------------------------------------------
        # Tapped same day
        # --------------------------------------------------

        if record["was_tapped_same_day"]:

            record["selection_reason"] = (
                "rejected_tapped_same_day"
            )

            continue

        # --------------------------------------------------
        # Invalid candidate for another reason
        # --------------------------------------------------

        if not record["is_valid_same_day"]:

            record["selection_reason"] = (
                "rejected_invalid_same_day"
            )

            continue

        direction = record["ob_direction"]

        candidate_low = float(
            record["candidate_low"]
        )

        candidate_high = float(
            record["candidate_high"]
        )

        # --------------------------------------------------
        # Bearish OB must remain completely above day close
        # --------------------------------------------------

        if direction == "bearish":

            if candidate_low <= day_close:

                record["selection_reason"] = (
                    "rejected_not_fully_above_day_close"
                )

                continue

        # --------------------------------------------------
        # Bullish OB must remain completely below day close
        # --------------------------------------------------

        elif direction == "bullish":

            if candidate_high >= day_close:

                record["selection_reason"] = (
                    "rejected_not_fully_below_day_close"
                )

                continue

        else:

            record["selection_reason"] = (
                "rejected_invalid_direction"
            )

            continue

        # --------------------------------------------------
        # Valid but not selected as extreme candidate
        # --------------------------------------------------

        record["selection_reason"] = (
            "rejected_not_extreme_selected_candidate"
        )


def process_single_trading_day(
    df: pd.DataFrame,
    extremes: pd.DataFrame,
    trading_day,
    max_fvg_lookahead: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Process one trading day.

    Returns
    -------
    tuple[list[dict], list[dict]]

    First:
        All candidate records.

    Second:
        Final selected Order Block records.
    """

    day_close = get_day_close(
        df=df,
        trading_day=trading_day,
    )

    high_timestamp = extremes.loc[
        trading_day,
        "high_timestamp"
    ]

    low_timestamp = extremes.loc[
        trading_day,
        "low_timestamp"
    ]

    high_position = df.index.get_loc(
        high_timestamp
    )

    low_position = df.index.get_loc(
        low_timestamp
    )

    # --------------------------------------------------
    # Scan ALL candidates from daily high
    # --------------------------------------------------

    high_candidates = scan_order_block_candidates(
        df=df,
        start_position=high_position,
        extreme_type="daily_high",
        trading_day=trading_day,
        max_fvg_lookahead=max_fvg_lookahead,
    )

    # --------------------------------------------------
    # Scan ALL candidates from daily low
    # --------------------------------------------------

    low_candidates = scan_order_block_candidates(
        df=df,
        start_position=low_position,
        extreme_type="daily_low",
        trading_day=trading_day,
        max_fvg_lookahead=max_fvg_lookahead,
    )

    # Preserve all candidates.
    all_candidates = (
        high_candidates
        + low_candidates
    )

    # --------------------------------------------------
    # Build candidate-level records
    # --------------------------------------------------

    candidate_records = []

    for candidate in all_candidates:

        record = order_block_to_record(
            candidate=candidate,
            trading_day=trading_day,
            day_close=day_close,
        )

        candidate_records.append(
            record
        )

    # --------------------------------------------------
    # Get valid candidates
    # --------------------------------------------------

    valid_high_candidates = (
        get_valid_same_day_candidates(
            high_candidates
        )
    )

    valid_low_candidates = (
        get_valid_same_day_candidates(
            low_candidates
        )
    )

    # --------------------------------------------------
    # Daily selection
    # --------------------------------------------------

    selection_result = select_daily_order_blocks(
        valid_high_candidates=valid_high_candidates,
        valid_low_candidates=valid_low_candidates,
        day_close=day_close,
    )

    selected_bearish = (
        selection_result.selected_bearish_ob
    )

    selected_bullish = (
        selection_result.selected_bullish_ob
    )

    # --------------------------------------------------
    # Mark selected candidates
    # --------------------------------------------------

    mark_candidate_selection(
        records=candidate_records,
        selected_bearish=selected_bearish,
        selected_bullish=selected_bullish,
    )

    # --------------------------------------------------
    # Mark detailed rejection reasons
    # --------------------------------------------------

    assign_candidate_rejection_reasons(
        records=candidate_records,
        day_close=day_close,
    )

    # --------------------------------------------------
    # Build final selected records
    # --------------------------------------------------

    selected_records = []

    for candidate in [
        selected_bearish,
        selected_bullish,
    ]:

        if candidate is None:
            continue

        record = order_block_to_record(
            candidate=candidate,
            trading_day=trading_day,
            day_close=day_close,
        )

        record["is_selected"] = True

        if candidate.direction == "bearish":

            record["selection_side"] = (
                "bearish"
            )

            record["selection_reason"] = (
                "selected_bearish_ob_above_day_close"
            )

        elif candidate.direction == "bullish":

            record["selection_side"] = (
                "bullish"
            )

            record["selection_reason"] = (
                "selected_bullish_ob_below_day_close"
            )

        selected_records.append(
            record
        )

    return (
        candidate_records,
        selected_records,
    )


def build_daily_order_block_datasets(
    df: pd.DataFrame,
    max_fvg_lookahead: int = 5,
    start_day=None,
    end_day=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two datasets.

    Returns
    -------
    candidate_dataset:
        Contains ALL detected Order Block candidates.

    selected_dataset:
        Contains only final selected daily Order Blocks.

    The candidate dataset is intended for future
    detailed statistical and structural analysis.
    """

    extremes = calculate_daily_extremes(
        df
    )

    # --------------------------------------------------
    # Date filtering
    # --------------------------------------------------

    if start_day is not None:

        start_timestamp = pd.Timestamp(
            start_day
        )

        extremes = extremes.loc[
            extremes.index >= start_timestamp
        ]

    if end_day is not None:

        end_timestamp = pd.Timestamp(
            end_day
        )

        extremes = extremes.loc[
            extremes.index <= end_timestamp
        ]

    all_candidate_records = []

    all_selected_records = []

    # --------------------------------------------------
    # Process every trading day
    # --------------------------------------------------

    for trading_day in extremes.index:

        (
            candidate_records,
            selected_records,
        ) = process_single_trading_day(
            df=df,
            extremes=extremes,
            trading_day=trading_day,
            max_fvg_lookahead=max_fvg_lookahead,
        )

        all_candidate_records.extend(
            candidate_records
        )

        all_selected_records.extend(
            selected_records
        )

    # --------------------------------------------------
    # Convert to DataFrames
    # --------------------------------------------------

    candidate_dataset = pd.DataFrame(
        all_candidate_records
    )

    selected_dataset = pd.DataFrame(
        all_selected_records
    )

    # --------------------------------------------------
    # Sort datasets
    # --------------------------------------------------

    if not candidate_dataset.empty:

        candidate_dataset = (
            candidate_dataset.sort_values(
                [
                    "trading_day",
                    "candidate_timestamp",
                ]
            )
            .reset_index(drop=True)
        )

    if not selected_dataset.empty:

        selected_dataset = (
            selected_dataset.sort_values(
                [
                    "trading_day",
                    "candidate_timestamp",
                ]
            )
            .reset_index(drop=True)
        )

    return (
        candidate_dataset,
        selected_dataset,
    )


def build_daily_order_block_dataset(
    df: pd.DataFrame,
    max_fvg_lookahead: int = 5,
    start_day=None,
    end_day=None,
) -> pd.DataFrame:
    """
    Backward-compatible helper.

    Returns only the final selected dataset.

    For detailed analysis use:

        build_daily_order_block_datasets()
    """

    (
        _,
        selected_dataset,
    ) = build_daily_order_block_datasets(
        df=df,
        max_fvg_lookahead=max_fvg_lookahead,
        start_day=start_day,
        end_day=end_day,
    )

    return selected_dataset