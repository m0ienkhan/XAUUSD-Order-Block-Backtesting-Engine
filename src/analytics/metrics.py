from __future__ import annotations

from typing import Any, Literal, Optional
import pandas as pd
import numpy as np


OBDirection = Literal["bullish", "bearish"]


def calculate_ob_range(
    high: float,
    low: float,
) -> float:
    """
    Calculate the absolute price range of an Order Block candle.
    """
    return float(high - low)


def get_tap_entry_boundary(
    direction: OBDirection,
    candidate_low: float,
    candidate_high: float,
) -> float:
    """
    Return the primary price entry boundary for the Order Block.

    Bullish OB: candidate_high (top of the OB zone)
    Bearish OB: candidate_low (bottom of the OB zone)
    """
    if direction == "bullish":
        return float(candidate_high)
    elif direction == "bearish":
        return float(candidate_low)
    else:
        raise ValueError(f"Unsupported direction: {direction}")


def calculate_distance_from_day_close(
    direction: OBDirection,
    candidate_low: float,
    candidate_high: float,
    day_close: float,
) -> float:
    """
    Calculate the distance from the day close to the nearest boundary of the OB.

    For bearish OB (sits above day close): candidate_low - day_close
    For bullish OB (sits below day close): day_close - candidate_high
    """
    if direction == "bearish":
        return float(candidate_low - day_close)
    elif direction == "bullish":
        return float(day_close - candidate_high)
    else:
        raise ValueError(f"Unsupported direction: {direction}")


def does_candle_overlap_zone(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
) -> bool:
    """
    Return True if candle [low, high] overlaps the zone [zone_low, zone_high].
    """
    return bool(candle_low <= zone_high and candle_high >= zone_low)


def calculate_penetration_metrics(
    direction: OBDirection,
    ob_low: float,
    ob_high: float,
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
) -> dict[str, Any]:
    """
    Calculate penetration depth and structural breach flags over a candle window.

    Returns
    -------
    dict containing:
        - is_touched: bool
        - entered_partially: bool (touched zone without breaching protective extreme)
        - fully_crossed: bool (wick exceeded protective extreme)
        - closed_beyond: bool (close exceeded protective extreme)
        - max_penetration_depth_usd: float (depth in price units)
        - max_penetration_pct: float (depth as percentage of OB range)
    """
    ob_range = max(ob_high - ob_low, 1e-8)

    if highs.empty or lows.empty:
        return {
            "is_touched": False,
            "entered_partially": False,
            "fully_crossed": False,
            "closed_beyond": False,
            "max_penetration_depth_usd": 0.0,
            "max_penetration_pct": 0.0,
        }

    max_high = float(highs.max())
    min_low = float(lows.min())

    if direction == "bearish":
        # Protective extreme is ob_high, entry is ob_low
        is_touched = bool(max_high >= ob_low and min_low <= ob_high)
        if not is_touched and max_high < ob_low:
            return {
                "is_touched": False,
                "entered_partially": False,
                "fully_crossed": False,
                "closed_beyond": False,
                "max_penetration_depth_usd": 0.0,
                "max_penetration_pct": 0.0,
            }

        depth_usd = max(0.0, max_high - ob_low)
        depth_pct = (depth_usd / ob_range) * 100.0
        fully_crossed = bool(max_high > ob_high)
        entered_partially = bool(is_touched and not fully_crossed)
        closed_beyond = bool((closes > ob_high).any())

        return {
            "is_touched": is_touched,
            "entered_partially": entered_partially,
            "fully_crossed": fully_crossed,
            "closed_beyond": closed_beyond,
            "max_penetration_depth_usd": float(depth_usd),
            "max_penetration_pct": float(depth_pct),
        }

    elif direction == "bullish":
        # Protective extreme is ob_low, entry is ob_high
        is_touched = bool(min_low <= ob_high and max_high >= ob_low)
        if not is_touched and min_low > ob_high:
            return {
                "is_touched": False,
                "entered_partially": False,
                "fully_crossed": False,
                "closed_beyond": False,
                "max_penetration_depth_usd": 0.0,
                "max_penetration_pct": 0.0,
            }

        depth_usd = max(0.0, ob_high - min_low)
        depth_pct = (depth_usd / ob_range) * 100.0
        fully_crossed = bool(min_low < ob_low)
        entered_partially = bool(is_touched and not fully_crossed)
        closed_beyond = bool((closes < ob_low).any())

        return {
            "is_touched": is_touched,
            "entered_partially": entered_partially,
            "fully_crossed": fully_crossed,
            "closed_beyond": closed_beyond,
            "max_penetration_depth_usd": float(depth_usd),
            "max_penetration_pct": float(depth_pct),
        }

    else:
        raise ValueError(f"Unsupported direction: {direction}")


def calculate_excursions(
    direction: OBDirection,
    tap_entry_boundary: float,
    ob_range: float,
    post_tap_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate MFE and MAE from the tap candle to the end of the post-tap window.

    Reference price: tap_entry_boundary

    Bearish OB:
        MFE = max(0, tap_entry_boundary - min(lows))
        MAE = max(0, max(highs) - tap_entry_boundary)

    Bullish OB:
        MFE = max(0, max(highs) - tap_entry_boundary)
        MAE = max(0, tap_entry_boundary - min(lows))

    Parameters
    ----------
    direction : OBDirection
        "bullish" or "bearish"
    tap_entry_boundary : float
        Reference price level (candidate_high for bullish, candidate_low for bearish)
    ob_range : float
        Candidate candle range (high - low)
    post_tap_df : pd.DataFrame
        DataFrame of candles from the tap candle (inclusive) onward.

    Returns
    -------
    dict with:
        mfe_price, mfe_r, mfe_timestamp, candles_to_mfe,
        mae_price, mae_r, mae_timestamp, candles_to_mae
    """
    if post_tap_df.empty:
        return {
            "mfe_price": float("nan"),
            "mfe_r": float("nan"),
            "mfe_timestamp": pd.NaT,
            "candles_to_mfe": pd.NA,
            "mae_price": float("nan"),
            "mae_r": float("nan"),
            "mae_timestamp": pd.NaT,
            "candles_to_mae": pd.NA,
        }

    highs = post_tap_df["high"]
    lows = post_tap_df["low"]
    timestamps = post_tap_df.index

    safe_ob_range = ob_range if ob_range > 1e-8 else float("nan")

    if direction == "bearish":
        # Favorable = price drops below entry
        min_low = float(lows.min())
        mfe_price = max(0.0, tap_entry_boundary - min_low)
        mfe_timestamp = lows.idxmin()
        candles_to_mfe = int(timestamps.get_loc(mfe_timestamp))

        # Adverse = price rises above entry
        max_high = float(highs.max())
        mae_price = max(0.0, max_high - tap_entry_boundary)
        mae_timestamp = highs.idxmax()
        candles_to_mae = int(timestamps.get_loc(mae_timestamp))

    elif direction == "bullish":
        # Favorable = price rises above entry
        max_high = float(highs.max())
        mfe_price = max(0.0, max_high - tap_entry_boundary)
        mfe_timestamp = highs.idxmax()
        candles_to_mfe = int(timestamps.get_loc(mfe_timestamp))

        # Adverse = price drops below entry
        min_low = float(lows.min())
        mae_price = max(0.0, tap_entry_boundary - min_low)
        mae_timestamp = lows.idxmin()
        candles_to_mae = int(timestamps.get_loc(mae_timestamp))

    else:
        raise ValueError(f"Unsupported direction: {direction}")

    mfe_r = mfe_price / safe_ob_range if pd.notna(safe_ob_range) else float("nan")
    mae_r = mae_price / safe_ob_range if pd.notna(safe_ob_range) else float("nan")

    return {
        "mfe_price": float(mfe_price),
        "mfe_r": float(mfe_r),
        "mfe_timestamp": pd.Timestamp(mfe_timestamp),
        "candles_to_mfe": candles_to_mfe,
        "mae_price": float(mae_price),
        "mae_r": float(mae_r),
        "mae_timestamp": pd.Timestamp(mae_timestamp),
        "candles_to_mae": candles_to_mae,
    }
