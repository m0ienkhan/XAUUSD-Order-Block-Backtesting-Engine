from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass
class ValidationReport:
    total_rows: int
    duplicate_timestamps: int
    null_values: dict[str, int]
    invalid_high_count: int
    invalid_low_count: int
    timestamp_ordered: bool

    total_large_gaps: int
    expected_daily_breaks: int
    holiday_or_long_gaps: int
    unexpected_intraday_gaps: int

    unexpected_gap_details: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "duplicate_timestamps": self.duplicate_timestamps,
            "null_values": self.null_values,
            "invalid_high_count": self.invalid_high_count,
            "invalid_low_count": self.invalid_low_count,
            "timestamp_ordered": self.timestamp_ordered,
            "total_large_gaps": self.total_large_gaps,
            "expected_daily_breaks": self.expected_daily_breaks,
            "holiday_or_long_gaps": self.holiday_or_long_gaps,
            "unexpected_intraday_gaps": self.unexpected_intraday_gaps,
        }


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_timestamp_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a pandas DatetimeIndex."
        )


def count_invalid_highs(df: pd.DataFrame) -> int:
    invalid_high = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    )

    return int(invalid_high.sum())


def count_invalid_lows(df: pd.DataFrame) -> int:
    invalid_low = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )

    return int(invalid_low.sum())


def analyze_time_gaps(
    df: pd.DataFrame,
    timeframe_minutes: int,
) -> dict[str, Any]:

    expected_interval = pd.Timedelta(
        minutes=timeframe_minutes
    )

    working_df = df.sort_index().copy()

    gaps = working_df.index.to_series().diff()

    large_gap_mask = gaps > expected_interval

    large_gaps = pd.DataFrame(
        {
            "timestamp": working_df.index[large_gap_mask],
            "gap": gaps[large_gap_mask].values,
        }
    )

    if large_gaps.empty:
        return {
            "total_large_gaps": 0,
            "expected_daily_breaks": 0,
            "holiday_or_long_gaps": 0,
            "unexpected_intraday_gaps": 0,
            "unexpected_gap_details": large_gaps,
        }

    expected_daily_mask = (
        large_gaps["timestamp"].dt.hour.isin([22, 23])
        & (
            large_gaps["gap"]
            <= pd.Timedelta(hours=2)
        )
    )

    long_gap_mask = (
        large_gaps["gap"]
        >= pd.Timedelta(hours=2)
    )

    unexpected_mask = (
        ~expected_daily_mask
        & ~long_gap_mask
    )

    expected_daily_breaks = large_gaps[
        expected_daily_mask
    ]

    holiday_or_long_gaps = large_gaps[
        long_gap_mask
    ]

    unexpected_intraday_gaps = large_gaps[
        unexpected_mask
    ]

    return {
        "total_large_gaps": len(large_gaps),
        "expected_daily_breaks": len(
            expected_daily_breaks
        ),
        "holiday_or_long_gaps": len(
            holiday_or_long_gaps
        ),
        "unexpected_intraday_gaps": len(
            unexpected_intraday_gaps
        ),
        "unexpected_gap_details": (
            unexpected_intraday_gaps.reset_index(
                drop=True
            )
        ),
    }


def validate_market_data(
    df: pd.DataFrame,
    timeframe_minutes: int,
) -> ValidationReport:

    validate_required_columns(df)

    validate_timestamp_index(df)

    duplicate_timestamps = int(
        df.index.duplicated().sum()
    )

    null_values = {
        column: int(count)
        for column, count
        in df.isnull().sum().items()
    }

    invalid_high_count = count_invalid_highs(df)

    invalid_low_count = count_invalid_lows(df)

    timestamp_ordered = bool(
        df.index.is_monotonic_increasing
    )

    gap_report = analyze_time_gaps(
        df=df,
        timeframe_minutes=timeframe_minutes,
    )

    return ValidationReport(
        total_rows=len(df),
        duplicate_timestamps=duplicate_timestamps,
        null_values=null_values,
        invalid_high_count=invalid_high_count,
        invalid_low_count=invalid_low_count,
        timestamp_ordered=timestamp_ordered,

        total_large_gaps=gap_report[
            "total_large_gaps"
        ],

        expected_daily_breaks=gap_report[
            "expected_daily_breaks"
        ],

        holiday_or_long_gaps=gap_report[
            "holiday_or_long_gaps"
        ],

        unexpected_intraday_gaps=gap_report[
            "unexpected_intraday_gaps"
        ],

        unexpected_gap_details=gap_report[
            "unexpected_gap_details"
        ],
    )