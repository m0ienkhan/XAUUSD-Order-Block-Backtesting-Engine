from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


OBDirection = Literal["bullish", "bearish"]
ExtremeType = Literal["daily_high", "daily_low"]


@dataclass
class OrderBlockCandidate:
    """
    Represents one Order Block candidate.

    Candidate lifecycle:

    1. Candidate candle is selected.
    2. Relevant directional FVG is searched.
    3. Same-day tap status is evaluated.
    4. Candidate metadata is retained for later analysis.
    """

    # --------------------------------------------------
    # Candidate identity
    # --------------------------------------------------

    candidate_position: int
    candidate_timestamp: pd.Timestamp

    # --------------------------------------------------
    # Candidate OHLC
    # --------------------------------------------------

    candidate_open: float
    candidate_high: float
    candidate_low: float
    candidate_close: float

    # --------------------------------------------------
    # Structural metadata
    # --------------------------------------------------

    extreme_type: ExtremeType
    direction: OBDirection

    is_inside_bar: bool

    mother_candle_position: Optional[int]
    mother_candle_timestamp: Optional[pd.Timestamp]

    # --------------------------------------------------
    # FVG metadata
    # --------------------------------------------------

    has_associated_fvg: bool

    fvg_position: Optional[int]
    fvg_timestamp: Optional[pd.Timestamp]

    fvg_type: Optional[str]

    fvg_lower: Optional[float]
    fvg_upper: Optional[float]

    fvg_size: Optional[float]

    # --------------------------------------------------
    # Same-day interaction metadata
    # --------------------------------------------------

    was_tapped_same_day: bool

    first_tap_position: Optional[int]
    first_tap_timestamp: Optional[pd.Timestamp]

    # --------------------------------------------------
    # Final candidate validity
    # --------------------------------------------------

    is_valid_same_day: bool


def get_candidate_direction(
    extreme_type: ExtremeType,
) -> OBDirection:
    """
    Determine expected Order Block direction.

    Daily High -> bearish Order Block
    Daily Low  -> bullish Order Block
    """

    if extreme_type == "daily_high":
        return "bearish"

    if extreme_type == "daily_low":
        return "bullish"

    raise ValueError(
        f"Unsupported extreme_type: {extreme_type}"
    )


def find_associated_fvg(
    df: pd.DataFrame,
    candidate_position: int,
    direction: OBDirection,
    max_fvg_lookahead: int = 5,
) -> Optional[int]:
    """
    Find the first directionally relevant FVG after a candidate.

    Rules:

    - FVG must occur after candidate candle.
    - FVG must occur within max_fvg_lookahead candles.
    - FVG direction must match candidate direction.

    Returns
    -------
    Optional[int]
        Positional index of the first valid FVG.
        None when no valid FVG exists.
    """

    start_position = candidate_position + 1

    end_position = min(
        candidate_position + max_fvg_lookahead,
        len(df) - 1,
    )

    for position in range(
        start_position,
        end_position + 1,
    ):

        fvg_type = df.iloc[position][
            "fvg_type"
        ]

        if pd.isna(fvg_type):
            continue

        if fvg_type == direction:
            return position

    return None


def does_candle_tap_zone(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
) -> bool:
    """
    Return True when candle range overlaps the OB zone.

    OB zone:
        [zone_low, zone_high]

    Candle range:
        [candle_low, candle_high]

    Any overlap counts as a tap.
    """

    return (
        candle_low <= zone_high
        and candle_high >= zone_low
    )


def find_first_same_day_tap(
    df: pd.DataFrame,
    candidate_position: int,
    fvg_position: Optional[int],
    trading_day,
) -> Optional[int]:
    """
    Find the first same-day candle that taps the OB zone.

    Important:
    Tap checking begins AFTER the associated FVG.

    This avoids treating candles involved in formation of
    the displacement/FVG as a revisit.

    Returns
    -------
    Optional[int]
        Position of first tap candle.

        None if price does not revisit the OB before
        the trading day ends.
    """

    if fvg_position is None:
        return None

    candidate = df.iloc[
        candidate_position
    ]

    zone_low = float(
        candidate["low"]
    )

    zone_high = float(
        candidate["high"]
    )

    start_position = (
        fvg_position + 1
    )

    for position in range(
        start_position,
        len(df),
    ):

        row = df.iloc[position]

        # Stop when trading day changes.
        if row["trading_day"] != trading_day:
            break

        candle_low = float(
            row["low"]
        )

        candle_high = float(
            row["high"]
        )

        tapped = does_candle_tap_zone(
            candle_low=candle_low,
            candle_high=candle_high,
            zone_low=zone_low,
            zone_high=zone_high,
        )

        if tapped:
            return position

    return None


def is_candidate_tapped_same_day(
    df: pd.DataFrame,
    candidate_position: int,
    fvg_position: Optional[int],
    trading_day,
) -> bool:
    """
    Return True when the candidate is tapped later
    during the same trading day.
    """

    tap_position = find_first_same_day_tap(
        df=df,
        candidate_position=candidate_position,
        fvg_position=fvg_position,
        trading_day=trading_day,
    )

    return tap_position is not None


def build_order_block_candidate(
    df: pd.DataFrame,
    candidate_position: int,
    extreme_type: ExtremeType,
    trading_day,
    max_fvg_lookahead: int = 5,
) -> OrderBlockCandidate:
    """
    Build one Order Block candidate.

    Process:

    1. Determine expected OB direction.
    2. Search for directional FVG.
    3. Check whether FVG belongs to same trading day.
    4. Check whether OB is tapped after FVG.
    5. Preserve all metadata.
    """

    direction = get_candidate_direction(
        extreme_type
    )

    row = df.iloc[
        candidate_position
    ]

    # --------------------------------------------------
    # Inside-bar metadata
    # --------------------------------------------------

    is_inside_bar = bool(
        row.get(
            "is_inside_bar",
            False,
        )
    )

    mother_position = row.get(
        "mother_candle_position",
        pd.NA,
    )

    mother_timestamp = row.get(
        "mother_candle_timestamp",
        pd.NaT,
    )

    if pd.isna(
        mother_position
    ):
        mother_position = None

    else:
        mother_position = int(
            mother_position
        )

    if pd.isna(
        mother_timestamp
    ):
        mother_timestamp = None

    # --------------------------------------------------
    # Find associated FVG
    # --------------------------------------------------

    fvg_position = find_associated_fvg(
        df=df,
        candidate_position=candidate_position,
        direction=direction,
        max_fvg_lookahead=max_fvg_lookahead,
    )

    # --------------------------------------------------
    # No FVG
    # --------------------------------------------------

    if fvg_position is None:

        return OrderBlockCandidate(
            candidate_position=candidate_position,

            candidate_timestamp=df.index[
                candidate_position
            ],

            candidate_open=float(
                row["open"]
            ),

            candidate_high=float(
                row["high"]
            ),

            candidate_low=float(
                row["low"]
            ),

            candidate_close=float(
                row["close"]
            ),

            extreme_type=extreme_type,

            direction=direction,

            is_inside_bar=is_inside_bar,

            mother_candle_position=
            mother_position,

            mother_candle_timestamp=
            mother_timestamp,

            has_associated_fvg=False,

            fvg_position=None,

            fvg_timestamp=None,

            fvg_type=None,

            fvg_lower=None,

            fvg_upper=None,

            fvg_size=None,

            was_tapped_same_day=False,

            first_tap_position=None,

            first_tap_timestamp=None,

            is_valid_same_day=False,
        )

    # --------------------------------------------------
    # Validate FVG trading day
    # --------------------------------------------------

    fvg_row = df.iloc[
        fvg_position
    ]

    fvg_same_day = (
        fvg_row["trading_day"]
        == trading_day
    )

    # --------------------------------------------------
    # FVG crosses into next trading day
    # --------------------------------------------------

    if not fvg_same_day:

        return OrderBlockCandidate(
            candidate_position=candidate_position,

            candidate_timestamp=df.index[
                candidate_position
            ],

            candidate_open=float(
                row["open"]
            ),

            candidate_high=float(
                row["high"]
            ),

            candidate_low=float(
                row["low"]
            ),

            candidate_close=float(
                row["close"]
            ),

            extreme_type=extreme_type,

            direction=direction,

            is_inside_bar=is_inside_bar,

            mother_candle_position=
            mother_position,

            mother_candle_timestamp=
            mother_timestamp,

            has_associated_fvg=False,

            fvg_position=None,

            fvg_timestamp=None,

            fvg_type=None,

            fvg_lower=None,

            fvg_upper=None,

            fvg_size=None,

            was_tapped_same_day=False,

            first_tap_position=None,

            first_tap_timestamp=None,

            is_valid_same_day=False,
        )

    # --------------------------------------------------
    # Find first same-day tap
    # --------------------------------------------------

    first_tap_position = (
        find_first_same_day_tap(
            df=df,
            candidate_position=
            candidate_position,

            fvg_position=fvg_position,

            trading_day=trading_day,
        )
    )

    was_tapped_same_day = (
        first_tap_position
        is not None
    )

    if was_tapped_same_day:

        first_tap_timestamp = (
            df.index[
                first_tap_position
            ]
        )

    else:

        first_tap_timestamp = None

    # --------------------------------------------------
    # Build candidate
    # --------------------------------------------------

    return OrderBlockCandidate(
        candidate_position=candidate_position,

        candidate_timestamp=df.index[
            candidate_position
        ],

        candidate_open=float(
            row["open"]
        ),

        candidate_high=float(
            row["high"]
        ),

        candidate_low=float(
            row["low"]
        ),

        candidate_close=float(
            row["close"]
        ),

        extreme_type=extreme_type,

        direction=direction,

        is_inside_bar=is_inside_bar,

        mother_candle_position=
        mother_position,

        mother_candle_timestamp=
        mother_timestamp,

        has_associated_fvg=True,

        fvg_position=fvg_position,

        fvg_timestamp=df.index[
            fvg_position
        ],

        fvg_type=str(
            fvg_row["fvg_type"]
        ),

        fvg_lower=float(
            fvg_row["fvg_lower"]
        ),

        fvg_upper=float(
            fvg_row["fvg_upper"]
        ),

        fvg_size=float(
            fvg_row["fvg_size"]
        ),

        was_tapped_same_day=
        was_tapped_same_day,

        first_tap_position=
        first_tap_position,

        first_tap_timestamp=
        first_tap_timestamp,

        is_valid_same_day=
        not was_tapped_same_day,
    )


def scan_order_block_candidates(
    df: pd.DataFrame,
    start_position: int,
    extreme_type: ExtremeType,
    trading_day,
    max_fvg_lookahead: int = 5,
    include_unconfirmed: bool = True,
) -> list[OrderBlockCandidate]:
    """
    Scan Order Block candidates from a starting position
    until the current trading day ends.

    Parameters
    ----------
    df:
        Market dataframe.

    start_position:
        Position where scanning begins.

    extreme_type:
        "daily_high" or "daily_low".

    trading_day:
        Trading day being analysed.

    max_fvg_lookahead:
        Maximum number of candles after a candidate in
        which an associated directional FVG may appear.

    include_unconfirmed:
        If True:
            Keep candidates without FVG metadata.

        If False:
            Only return candidates with confirmed FVGs.

    Returns
    -------
    list[OrderBlockCandidate]

    Notes
    -----
    All same-day tap information is retained.

    A candidate with a valid FVG can still be returned even
    if tapped. This allows detailed research later.

    Use candidate.is_valid_same_day for final filtering.
    """

    candidates: list[
        OrderBlockCandidate
    ] = []

    for position in range(
        start_position,
        len(df),
    ):

        row = df.iloc[
            position
        ]

        # --------------------------------------------------
        # Stop at next trading day
        # --------------------------------------------------

        if (
            row["trading_day"]
            != trading_day
        ):
            break

        # --------------------------------------------------
        # Build candidate
        # --------------------------------------------------

        candidate = (
            build_order_block_candidate(
                df=df,

                candidate_position=
                position,

                extreme_type=
                extreme_type,

                trading_day=
                trading_day,

                max_fvg_lookahead=
                max_fvg_lookahead,
            )
        )

        # --------------------------------------------------
        # Optional filtering
        # --------------------------------------------------

        if (
            not include_unconfirmed
            and not candidate.has_associated_fvg
        ):
            continue

        candidates.append(
            candidate
        )

    return candidates


def get_valid_same_day_candidates(
    candidates: list[
        OrderBlockCandidate
    ],
) -> list[OrderBlockCandidate]:
    """
    Return candidates that:

    1. Have an associated directional FVG.
    2. Were not tapped during the same trading day.
    """

    return [
        candidate
        for candidate in candidates
        if (
            candidate.has_associated_fvg
            and candidate.is_valid_same_day
        )
    ]