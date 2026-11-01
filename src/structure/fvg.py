from __future__ import annotations

import pandas as pd


def detect_fvg(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect bullish and bearish Fair Value Gaps (FVGs).

    A bullish FVG exists when:

        candle_1.high < candle_3.low

    A bearish FVG exists when:

        candle_1.low > candle_3.high

    The FVG is associated with the middle candle (candle_2).

    Added columns
    -------------
    fvg_type
        'bullish', 'bearish', or None.

    fvg_lower
        Lower boundary of the FVG price range.

    fvg_upper
        Upper boundary of the FVG price range.

    fvg_size
        Absolute size of the FVG.

    fvg_start_timestamp
        Timestamp of candle_1.

    fvg_end_timestamp
        Timestamp of candle_3.

    Notes
    -----
    This function only detects FVGs using information available
    at candle_3 close.

    It does not use future candles and does not determine whether
    the FVG is later tested or filled.
    """

    required_columns = {"high", "low"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    result["fvg_type"] = pd.Series(
        None,
        index=result.index,
        dtype="object",
    )

    result["fvg_lower"] = pd.Series(
        float("nan"),
        index=result.index,
        dtype="float64",
    )

    result["fvg_upper"] = pd.Series(
        float("nan"),
        index=result.index,
        dtype="float64",
    )

    result["fvg_size"] = pd.Series(
        float("nan"),
        index=result.index,
        dtype="float64",
    )

    result["fvg_start_timestamp"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype=result.index.dtype,
    )

    result["fvg_end_timestamp"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype=result.index.dtype,
    )

    fvg_type_col = result.columns.get_loc("fvg_type")
    fvg_lower_col = result.columns.get_loc("fvg_lower")
    fvg_upper_col = result.columns.get_loc("fvg_upper")
    fvg_size_col = result.columns.get_loc("fvg_size")
    fvg_start_col = result.columns.get_loc(
        "fvg_start_timestamp"
    )
    fvg_end_col = result.columns.get_loc(
        "fvg_end_timestamp"
    )

    for position in range(2, len(result)):

        candle_1_position = position - 2
        candle_3_position = position

        candle_1_high = result.iloc[
            candle_1_position
        ]["high"]

        candle_1_low = result.iloc[
            candle_1_position
        ]["low"]

        candle_3_high = result.iloc[
            candle_3_position
        ]["high"]

        candle_3_low = result.iloc[
            candle_3_position
        ]["low"]

        fvg_type = None
        fvg_lower = None
        fvg_upper = None

        # Bullish FVG:
        # Candle 1 High < Candle 3 Low
        if candle_1_high < candle_3_low:

            fvg_type = "bullish"

            fvg_lower = candle_1_high
            fvg_upper = candle_3_low

        # Bearish FVG:
        # Candle 1 Low > Candle 3 High
        elif candle_1_low > candle_3_high:

            fvg_type = "bearish"

            fvg_lower = candle_3_high
            fvg_upper = candle_1_low

        else:
            continue

        fvg_size = fvg_upper - fvg_lower

        result.iat[
            position,
            fvg_type_col,
        ] = fvg_type

        result.iat[
            position,
            fvg_lower_col,
        ] = fvg_lower

        result.iat[
            position,
            fvg_upper_col,
        ] = fvg_upper

        result.iat[
            position,
            fvg_size_col,
        ] = fvg_size

        result.iat[
            position,
            fvg_start_col,
        ] = result.index[
            candle_1_position
        ]

        result.iat[
            position,
            fvg_end_col,
        ] = result.index[
            candle_3_position
        ]

    return result


def get_detected_fvgs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only candles where an FVG was detected.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by detect_fvg().

    Returns
    -------
    pd.DataFrame
        DataFrame containing only detected FVG rows.
    """

    required_columns = {
        "fvg_type",
        "fvg_lower",
        "fvg_upper",
        "fvg_size",
    }

    missing_columns = required_columns - set(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            "FVG columns are missing. "
            "Run detect_fvg() first. Missing: "
            f"{sorted(missing_columns)}"
        )

    return df[
        df["fvg_type"].notna()
    ].copy()


def is_price_inside_fvg(
    price: float,
    fvg_lower: float,
    fvg_upper: float,
) -> bool:
    """
    Return True when a price lies inside an FVG range.

    Boundaries are inclusive.
    """

    return (
        fvg_lower <= price <= fvg_upper
    )