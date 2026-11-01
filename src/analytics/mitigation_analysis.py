from __future__ import annotations

from typing import Any, Literal, Optional, Union
import pandas as pd
import numpy as np

from src.analytics.metrics import (
    OBDirection,
    calculate_ob_range,
    get_tap_entry_boundary,
    calculate_penetration_metrics,
    calculate_excursions,
)


HorizonType = Union[Literal["same_day_post_fvg", "next_trading_day", "1_trading_day", "3_trading_days", "5_trading_days", "full"], int]


def resolve_horizon_window(
    df: pd.DataFrame,
    ob_trading_day: object,
    fvg_position: Optional[int],
    horizon: HorizonType,
) -> tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[int]]:
    """
    Resolve the exact candle slice for a specified analysis horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Market data indexed by DatetimeIndex with a 'trading_day' column.
    ob_trading_day : object
        Trading day on which the OB formed.
    fvg_position : Optional[int]
        Positional index of the associated FVG (required for same_day_post_fvg).
    horizon : HorizonType
        - 'same_day_post_fvg': candles from (fvg_position + 1) to end of formation trading day.
        - 'next_trading_day' / '1_trading_day': all candles of the single next trading day.
        - '3_trading_days': all candles across the next 3 subsequent trading days.
        - '5_trading_days': all candles across the next 5 subsequent trading days.
        - 'full': all candles from the start of the next trading day to dataset end.
        - int: exactly N candles starting from the first candle of next trading day.

    Returns
    -------
    tuple of (df_slice, start_timestamp, start_position)
        If no candles exist for the horizon, returns (empty DataFrame, NaT, None).
    """
    if horizon == "same_day_post_fvg":
        if fvg_position is None or fvg_position + 1 >= len(df):
            return pd.DataFrame(), None, None

        start_pos = fvg_position + 1
        start_ts = df.index[start_pos]

        # Slices from start_pos until trading_day changes
        same_day_mask = df.iloc[start_pos:]["trading_day"] == ob_trading_day
        if not same_day_mask.any():
            return pd.DataFrame(), None, None

        # Take contiguous same-day candles
        count = int(same_day_mask.cumprod().sum())
        df_slice = df.iloc[start_pos : start_pos + count].copy()

        if df_slice.empty:
            return pd.DataFrame(), None, None

        return df_slice, start_ts, start_pos

    # Next-trading-day modes
    future_mask = df["trading_day"] > ob_trading_day
    if not future_mask.any():
        return pd.DataFrame(), None, None

    future_df = df[future_mask]
    start_ts = future_df.index[0]
    start_pos = int(df.index.get_loc(start_ts))

    future_days = future_df["trading_day"].drop_duplicates().tolist()
    if not future_days:
        return pd.DataFrame(), None, None

    if horizon in ("next_trading_day", "1_trading_day"):
        target_days = future_days[:1]
        df_slice = future_df[future_df["trading_day"].isin(target_days)].copy()

    elif horizon == "3_trading_days":
        target_days = future_days[:3]
        df_slice = future_df[future_df["trading_day"].isin(target_days)].copy()

    elif horizon == "5_trading_days":
        target_days = future_days[:5]
        df_slice = future_df[future_df["trading_day"].isin(target_days)].copy()

    elif horizon == "full":
        df_slice = future_df.copy()

    elif isinstance(horizon, int):
        if horizon <= 0:
            return pd.DataFrame(), None, None
        df_slice = df.iloc[start_pos : start_pos + horizon].copy()

    else:
        raise ValueError(f"Unsupported horizon: {horizon}")

    return df_slice, start_ts, start_pos


def analyze_ob_interaction_in_window(
    df_window: pd.DataFrame,
    direction: OBDirection,
    ob_low: float,
    ob_high: float,
    global_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze OB mitigation, tap metrics, penetration, and excursions over a candle window.

    Parameters
    ----------
    df_window : pd.DataFrame
        The specific horizon candle slice.
    direction : OBDirection
        'bullish' or 'bearish'
    ob_low : float
        Candidate low
    ob_high : float
        Candidate high
    global_df : pd.DataFrame
        Full dataset for absolute positional index resolution.

    Returns
    -------
    dict with tap status, entry boundary, penetration details, and MFE/MAE excursions.
    """
    ob_range = calculate_ob_range(high=ob_high, low=ob_low)
    tap_entry_boundary = get_tap_entry_boundary(
        direction=direction,
        candidate_low=ob_low,
        candidate_high=ob_high,
    )

    if df_window.empty:
        return {
            "was_tapped": False,
            "first_tap_timestamp": pd.NaT,
            "first_tap_position": pd.NA,
            "candles_until_first_tap": pd.NA,
            "tap_entry_boundary": float(tap_entry_boundary),
            "first_tap_candle_low": float("nan"),
            "first_tap_candle_high": float("nan"),
            "first_tap_candle_close": float("nan"),
            "entered_partially": False,
            "fully_crossed": False,
            "closed_beyond": False,
            "max_penetration_depth_usd": 0.0,
            "max_penetration_pct": 0.0,
            "mfe_price": float("nan"),
            "mfe_r": float("nan"),
            "mfe_timestamp": pd.NaT,
            "candles_to_mfe": pd.NA,
            "mae_price": float("nan"),
            "mae_r": float("nan"),
            "mae_timestamp": pd.NaT,
            "candles_to_mae": pd.NA,
        }

    # Identify first tap candle where candle range overlaps OB zone
    # [candle_low, candle_high] overlaps [ob_low, ob_high]
    tap_mask = (df_window["low"] <= ob_high) & (df_window["high"] >= ob_low)

    if not tap_mask.any():
        # No tap in window
        penetration = calculate_penetration_metrics(
            direction=direction,
            ob_low=ob_low,
            ob_high=ob_high,
            highs=df_window["high"],
            lows=df_window["low"],
            closes=df_window["close"],
        )
        return {
            "was_tapped": False,
            "first_tap_timestamp": pd.NaT,
            "first_tap_position": pd.NA,
            "candles_until_first_tap": pd.NA,
            "tap_entry_boundary": float(tap_entry_boundary),
            "first_tap_candle_low": float("nan"),
            "first_tap_candle_high": float("nan"),
            "first_tap_candle_close": float("nan"),
            "entered_partially": penetration["entered_partially"],
            "fully_crossed": penetration["fully_crossed"],
            "closed_beyond": penetration["closed_beyond"],
            "max_penetration_depth_usd": penetration["max_penetration_depth_usd"],
            "max_penetration_pct": penetration["max_penetration_pct"],
            "mfe_price": float("nan"),
            "mfe_r": float("nan"),
            "mfe_timestamp": pd.NaT,
            "candles_to_mfe": pd.NA,
            "mae_price": float("nan"),
            "mae_r": float("nan"),
            "mae_timestamp": pd.NaT,
            "candles_to_mae": pd.NA,
        }

    # First tap found
    first_tap_loc = int(np.argmax(tap_mask.values))
    tap_row = df_window.iloc[first_tap_loc]
    first_tap_ts = df_window.index[first_tap_loc]
    first_tap_pos = int(global_df.index.get_loc(first_tap_ts))

    # Slice from tap candle onwards within the horizon window
    post_tap_df = df_window.iloc[first_tap_loc:].copy()

    # Calculate excursions from tap
    excursions = calculate_excursions(
        direction=direction,
        tap_entry_boundary=tap_entry_boundary,
        ob_range=ob_range,
        post_tap_df=post_tap_df,
    )

    # Calculate penetration over the post-tap / window candles
    penetration = calculate_penetration_metrics(
        direction=direction,
        ob_low=ob_low,
        ob_high=ob_high,
        highs=post_tap_df["high"],
        lows=post_tap_df["low"],
        closes=post_tap_df["close"],
    )

    return {
        "was_tapped": True,
        "first_tap_timestamp": pd.Timestamp(first_tap_ts),
        "first_tap_position": first_tap_pos,
        "candles_until_first_tap": first_tap_loc,
        "tap_entry_boundary": float(tap_entry_boundary),
        "first_tap_candle_low": float(tap_row["low"]),
        "first_tap_candle_high": float(tap_row["high"]),
        "first_tap_candle_close": float(tap_row["close"]),
        "entered_partially": penetration["entered_partially"],
        "fully_crossed": penetration["fully_crossed"],
        "closed_beyond": penetration["closed_beyond"],
        "max_penetration_depth_usd": penetration["max_penetration_depth_usd"],
        "max_penetration_pct": penetration["max_penetration_pct"],
        "mfe_price": excursions["mfe_price"],
        "mfe_r": excursions["mfe_r"],
        "mfe_timestamp": excursions["mfe_timestamp"],
        "candles_to_mfe": excursions["candles_to_mfe"],
        "mae_price": excursions["mae_price"],
        "mae_r": excursions["mae_r"],
        "mae_timestamp": excursions["mae_timestamp"],
        "candles_to_mae": excursions["candles_to_mae"],
    }
