from __future__ import annotations

import pandas as pd


NEW_YORK_TIMEZONE = "America/New_York"

ROLLOVER_HOUR = 17


def assign_trading_day(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign a trading_day column using the New York
    5 PM daily rollover.

    Rules:
        New York time < 17:00
            trading_day = current calendar date

        New York time >= 17:00
            trading_day = next calendar date

    The DataFrame index must be a timezone-aware
    DatetimeIndex.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex."
        )

    if df.index.tz is None:
        raise ValueError(
            "DatetimeIndex must be timezone-aware."
        )

    result = df.copy()

    # Convert UTC timestamps to New York time.
    new_york_index = result.index.tz_convert(
        NEW_YORK_TIMEZONE
    )

    # Convert timestamps to normalized local dates.
    trading_day = new_york_index.normalize()

    # After 5 PM New York time, the next trading day begins.
    after_rollover = (
        new_york_index.hour >= ROLLOVER_HOUR
    )

    # DatetimeIndex operations are vectorized.
    trading_day = (
        trading_day
        + pd.to_timedelta(
            after_rollover.astype(int),
            unit="D",
        )
    )

    # Store timezone-naive date because this represents
    # a trading-day label, not an actual timestamp.
    result["trading_day"] = (
        trading_day.tz_localize(None)
    )

    return result


def get_trading_day(
    timestamp: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return the trading day for a single timezone-aware
    timestamp using the New York 5 PM rollover.
    """

    if timestamp.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware."
        )

    new_york_time = timestamp.tz_convert(
        NEW_YORK_TIMEZONE
    )

    trading_day = new_york_time.normalize()

    if new_york_time.hour >= ROLLOVER_HOUR:
        trading_day += pd.Timedelta(days=1)

    # Return only the trading-day label.
    return trading_day.tz_localize(None)