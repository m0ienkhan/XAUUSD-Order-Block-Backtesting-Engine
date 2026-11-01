from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.validator import (
    ValidationReport,
    validate_market_data,
)


def load_market_data(
    file_path: str | Path,
    timeframe_minutes: int,
    validate: bool = True,
) -> tuple[pd.DataFrame, ValidationReport | None]:
    """
    Load OHLCV market data from a CSV file.

    Expected CSV structure:
        timestamp, open, high, low, close, volume

    The function:
        1. Reads the CSV
        2. Parses timestamps
        3. Sets timestamp as DatetimeIndex
        4. Sorts the data
        5. Optionally runs validation

    Returns:
        tuple:
            (
                standardized DataFrame,
                ValidationReport or None
            )
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Market data file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    if "timestamp" not in df.columns:
        raise ValueError(
            "CSV must contain a 'timestamp' column."
        )

    df = df.set_index("timestamp")

    df = df.sort_index()

    validation_report = None

    if validate:
        validation_report = validate_market_data(
            df=df,
            timeframe_minutes=timeframe_minutes,
        )

    return df, validation_report