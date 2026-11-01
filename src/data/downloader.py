"""
Reusable historical market data downloader.

Currently uses Dukascopy as the data provider.
This module is independent from strategy logic and can be reused
for different instruments, timeframes, and projects.
"""

from pathlib import Path
import pandas as pd
import dukascopy_python as dp


TIMEFRAME_MAP = {
    "1min": dp.INTERVAL_MIN_1,
    "5min": dp.INTERVAL_MIN_5,
    "10min": dp.INTERVAL_MIN_10,
    "15min": dp.INTERVAL_MIN_15,
    "30min": dp.INTERVAL_MIN_30,
    "1hour": dp.INTERVAL_HOUR_1,
    "4hour": dp.INTERVAL_HOUR_4,
    "1day": dp.INTERVAL_DAY_1,
}


INSTRUMENT_MAP = {
    "XAUUSD": "XAU/USD",
}


def validate_download_request(symbol, timeframe, start_date, end_date):
    """Validate download request parameters."""

    if symbol not in INSTRUMENT_MAP:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Supported symbols: {list(INSTRUMENT_MAP.keys())}"
        )

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported timeframes: {list(TIMEFRAME_MAP.keys())}"
        )

    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date.")


def resolve_instrument(symbol):
    """Convert generic symbol to provider-specific symbol."""

    return INSTRUMENT_MAP[symbol]


def resolve_interval(timeframe):
    """Convert generic timeframe to Dukascopy interval."""

    return TIMEFRAME_MAP[timeframe]


def fetch_data(
    symbol,
    timeframe,
    start_date,
    end_date,
    offer_side=dp.OFFER_SIDE_BID,
):
    """Download historical market data from Dukascopy."""

    validate_download_request(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    instrument = resolve_instrument(symbol)
    interval = resolve_interval(timeframe)

    df = dp.fetch(
        instrument=instrument,
        interval=interval,
        offer_side=offer_side,
        start=start_date,
        end=end_date,
    )

    if df is None or df.empty:
        raise ValueError(
            "No data was returned for the requested period."
        )

    return df


def filter_date_range(df, start_date, end_date):
    """Strictly filter data to the requested date range."""

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    if start_timestamp.tzinfo is None:
        start_timestamp = start_timestamp.tz_localize("UTC")

    if end_timestamp.tzinfo is None:
        end_timestamp = end_timestamp.tz_localize("UTC")

    filtered_df = df.loc[
        (df.index >= start_timestamp)
        & (df.index < end_timestamp)
    ].copy()

    return filtered_df


def normalize_dataframe(df):
    """Normalize dataframe for the rest of the project."""

    normalized_df = df.copy()
    normalized_df = normalized_df.sort_index()
    normalized_df.index.name = "timestamp"

    return normalized_df


def save_data(df, output_path):
    """Save dataframe as CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)

    return output_path


def download_data(
    symbol,
    timeframe,
    start_date,
    end_date,
    output_path,
    offer_side=dp.OFFER_SIDE_BID,
):
    """
    Complete download pipeline.

    1. Validate request
    2. Download data
    3. Filter exact date range
    4. Normalize dataframe
    5. Save CSV
    """

    df = fetch_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        offer_side=offer_side,
    )

    df = filter_date_range(
        df=df,
        start_date=start_date,
        end_date=end_date,
    )

    df = normalize_dataframe(df)

    save_data(
        df=df,
        output_path=output_path,
    )

    return df



def download_data_in_chunks(
    symbol,
    timeframe,
    start_date,
    end_date,
    output_path,
    chunk_years=1,
    offer_side=dp.OFFER_SIDE_BID,
):
    """
    Download historical data in yearly chunks and combine it.

    Parameters
    ----------
    symbol : str
        Generic instrument symbol, for example "XAUUSD".

    timeframe : str
        Candle timeframe, for example "15min".

    start_date : datetime
        Start of requested historical period.

    end_date : datetime
        End of requested historical period.

    output_path : str or Path
        Final combined CSV output path.

    chunk_years : int
        Number of years per download chunk.

    offer_side : str
        Dukascopy offer side.
    """

    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date.")

    if chunk_years < 1:
        raise ValueError("chunk_years must be at least 1.")

    chunks = []
    chunk_start = start_date

    while chunk_start < end_date:

        chunk_end = chunk_start.replace(
            year=chunk_start.year + chunk_years
        )

        if chunk_end > end_date:
            chunk_end = end_date

        print(
            f"Downloading {symbol} {timeframe}: "
            f"{chunk_start.date()} -> {chunk_end.date()}"
        )

        chunk_df = fetch_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=chunk_start,
            end_date=chunk_end,
            offer_side=offer_side,
        )

        chunk_df = filter_date_range(
            df=chunk_df,
            start_date=chunk_start,
            end_date=chunk_end,
        )

        chunk_df = normalize_dataframe(chunk_df)

        print(f"Rows downloaded: {len(chunk_df)}")

        chunks.append(chunk_df)

        chunk_start = chunk_end

    if not chunks:
        raise ValueError("No data chunks were downloaded.")

    combined_df = pd.concat(chunks)

    combined_df = combined_df[
        ~combined_df.index.duplicated(keep="first")
    ]

    combined_df = normalize_dataframe(combined_df)

    save_data(
        df=combined_df,
        output_path=output_path,
    )

    print(f"\nFinal rows: {len(combined_df)}")
    print(f"Saved to: {output_path}")

    return combined_df