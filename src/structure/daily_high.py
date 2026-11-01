from __future__ import annotations

import pandas as pd


def calculate_daily_extremes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the high and low for every trading day.

    Required columns:
        high
        low
        trading_day

    Returns a DataFrame with one row per trading day:

        trading_day
        day_high
        high_timestamp
        day_low
        low_timestamp
        candle_count
    """

    required_columns = {
        "high",
        "low",
        "trading_day",
    }

    missing_columns = required_columns.difference(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex."
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "day_high",
                "high_timestamp",
                "day_low",
                "low_timestamp",
                "candle_count",
            ]
        )

    grouped = df.groupby(
        "trading_day",
        sort=True,
    )

    day_high = grouped["high"].max()

    day_low = grouped["low"].min()

    high_timestamp = grouped["high"].idxmax()

    low_timestamp = grouped["low"].idxmin()

    candle_count = grouped.size()

    result = pd.DataFrame(
        {
            "day_high": day_high,
            "high_timestamp": high_timestamp,
            "day_low": day_low,
            "low_timestamp": low_timestamp,
            "candle_count": candle_count,
        }
    )

    return result