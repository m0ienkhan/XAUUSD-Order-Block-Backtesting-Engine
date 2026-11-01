"""
Data configuration for the XAUUSD Order Block Backtesting Engine.
"""

DATA_CONFIG = {
    # Expected OHLC columns
    "timestamp_column": "timestamp",
    "open_column": "open",
    "high_column": "high",
    "low_column": "low",
    "close_column": "close",

    # Expected timeframe
    "expected_timeframe": "15min",

    # Data quality
    "check_duplicates": True,
    "check_missing_values": True,
    "sort_by_timestamp": True,
}