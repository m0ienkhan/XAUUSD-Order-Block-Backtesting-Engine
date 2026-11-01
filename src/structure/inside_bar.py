from __future__ import annotations

import pandas as pd


def is_inside_bar(
    mother_high: float,
    mother_low: float,
    current_high: float,
    current_low: float,
) -> bool:
    """
    Return True when the current candle is completely contained
    within the mother candle range.
    """

    return (
        current_high <= mother_high
        and current_low >= mother_low
    )


def find_mother_candle(
    df: pd.DataFrame,
    position: int,
) -> int | None:
    """
    Find the original mother candle for a given candle position.

    If the previous candle is itself an inside bar, the original
    mother candle of that inside-bar cluster is returned.

    Returns
    -------
    int | None
        Position of the mother candle, or None when the candle
        is not an inside bar.
    """

    if position <= 0 or position >= len(df):
        return None

    previous_position = position - 1

    previous_high = df.iloc[previous_position]["high"]
    previous_low = df.iloc[previous_position]["low"]

    current_high = df.iloc[position]["high"]
    current_low = df.iloc[position]["low"]

    inside_previous = is_inside_bar(
        mother_high=previous_high,
        mother_low=previous_low,
        current_high=current_high,
        current_low=current_low,
    )

    if not inside_previous:
        return None

    if (
        "is_inside_bar" in df.columns
        and "mother_candle_position" in df.columns
        and bool(df.iloc[previous_position]["is_inside_bar"])
    ):
        mother_position = df.iloc[
            previous_position
        ]["mother_candle_position"]

        if pd.notna(mother_position):
            return int(mother_position)

    return previous_position


def resolve_inside_bar_cluster(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve inside-bar relationships for every candle.

    Adds the following columns:

        is_inside_bar
        mother_candle_position
        mother_candle_timestamp

    Consecutive inside bars inherit the original mother candle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:

            high
            low

        The index should represent candle timestamps.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with inside-bar annotations.
    """

    if "high" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'high' column."
        )

    if "low" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'low' column."
        )

    result = df.copy()

    result["is_inside_bar"] = False

    result["mother_candle_position"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result["mother_candle_timestamp"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype=result.index.dtype,
    )

    is_inside_col = result.columns.get_loc(
        "is_inside_bar"
    )

    mother_position_col = result.columns.get_loc(
        "mother_candle_position"
    )

    mother_timestamp_col = result.columns.get_loc(
        "mother_candle_timestamp"
    )

    for position in range(1, len(result)):

        mother_position = find_mother_candle(
            result,
            position,
        )

        if mother_position is None:
            continue

        result.iat[
            position,
            is_inside_col,
        ] = True

        result.iat[
            position,
            mother_position_col,
        ] = mother_position

        result.iat[
            position,
            mother_timestamp_col,
        ] = result.index[mother_position]

    return result


def annotate_inside_bar_structure(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Public entry point for inside-bar structural annotation.

    This function exists so downstream structure and order-block
    modules have a stable API.
    """

    return resolve_inside_bar_cluster(df)