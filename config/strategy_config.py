"""
Strategy configuration for the XAUUSD Order Block Backtesting Engine.
"""

STRATEGY_CONFIG = {
    # Instrument and timeframe
    "instrument": "XAUUSD",
    "timeframe": "15min",

    # Daily order block selection
    "one_ob_per_day": True,
    "search_reference": "daily_high",
    "selection_rule": "first_valid_ob",

    # Inside bar handling
    "inside_bar_mode": "mother_candle",

    # Fair Value Gap requirements
    "require_associated_fvg": True,
    "track_fvg_lifecycle": True,

    # Trading rules
    "allow_same_day_trade": False,

    # Risk / Reward
    "stop_distance_usd": 10.0,
    "target_distance_usd": 30.0,

    # Research behaviour
    "store_all_fvg_statuses": True,
    "analyze_mitigation_status": True,
}