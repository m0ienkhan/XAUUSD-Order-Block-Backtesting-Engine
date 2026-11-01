from __future__ import annotations

from typing import Any, Literal, Optional, Sequence
import pandas as pd
import numpy as np

from src.analytics.metrics import OBDirection, calculate_ob_range, get_tap_entry_boundary
from src.analytics.mitigation_analysis import HorizonType, resolve_horizon_window


PathResolutionStatus = Literal[
    "target_first",
    "stop_first",
    "ambiguous_same_candle",
    "no_resolution_within_horizon",
    "not_tapped",
]


def evaluate_single_target_path(
    direction: OBDirection,
    tap_entry_boundary: float,
    ob_range: float,
    target_r: float,
    stop_r: float,
    post_tap_candles: pd.DataFrame,
) -> dict[str, Any]:
    """
    Perform exact sequential candle-by-candle scan to determine whether target
    or stop was hit first in the post-tap candle sequence.

    Parameters
    ----------
    direction : OBDirection
        'bullish' or 'bearish'
    tap_entry_boundary : float
        Entry reference price level
    ob_range : float
        Order block price range
    target_r : float
        Favorable target distance in R-multiples (e.g. 1.0, 2.0, 3.0, 5.0)
    stop_r : float
        Adverse stop distance in R-multiples (default 1.0)
    post_tap_candles : pd.DataFrame
        DataFrame of candles starting from the tap candle (inclusive).

    Returns
    -------
    dict with:
        - status: PathResolutionStatus
        - target_price: float
        - stop_price: float
        - resolution_timestamp: pd.Timestamp | NaT
        - candles_to_resolution: int | NA
    """
    safe_ob_range = max(ob_range, 1e-8)

    if direction == "bullish":
        target_price = float(tap_entry_boundary + (target_r * safe_ob_range))
        stop_price = float(tap_entry_boundary - (stop_r * safe_ob_range))
    elif direction == "bearish":
        target_price = float(tap_entry_boundary - (target_r * safe_ob_range))
        stop_price = float(tap_entry_boundary + (stop_r * safe_ob_range))
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    if post_tap_candles.empty:
        return {
            "status": "no_resolution_within_horizon",
            "target_price": target_price,
            "stop_price": stop_price,
            "resolution_timestamp": pd.NaT,
            "candles_to_resolution": pd.NA,
        }

    for idx in range(len(post_tap_candles)):
        candle = post_tap_candles.iloc[idx]
        c_high = float(candle["high"])
        c_low = float(candle["low"])
        c_ts = post_tap_candles.index[idx]

        if direction == "bullish":
            hit_target = c_high >= target_price
            hit_stop = c_low <= stop_price
        else:  # bearish
            hit_target = c_low <= target_price
            hit_stop = c_high >= stop_price

        # Intrabar ambiguity: both target and stop touched in the same candle
        if hit_target and hit_stop:
            return {
                "status": "ambiguous_same_candle",
                "target_price": target_price,
                "stop_price": stop_price,
                "resolution_timestamp": pd.Timestamp(c_ts),
                "candles_to_resolution": idx,
            }
        elif hit_target:
            return {
                "status": "target_first",
                "target_price": target_price,
                "stop_price": stop_price,
                "resolution_timestamp": pd.Timestamp(c_ts),
                "candles_to_resolution": idx,
            }
        elif hit_stop:
            return {
                "status": "stop_first",
                "target_price": target_price,
                "stop_price": stop_price,
                "resolution_timestamp": pd.Timestamp(c_ts),
                "candles_to_resolution": idx,
            }

    return {
        "status": "no_resolution_within_horizon",
        "target_price": target_price,
        "stop_price": stop_price,
        "resolution_timestamp": pd.NaT,
        "candles_to_resolution": pd.NA,
    }


def scan_order_block_path_outcomes(
    df: pd.DataFrame,
    ob_record: pd.Series | dict,
    horizon: HorizonType = "next_trading_day",
    target_multiples: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    stop_r: float = 1.0,
) -> dict[str, Any]:
    """
    Scan future candles sequentially for one selected Order Block and compute
    path-dependent outcomes for all requested target multiples.

    Parameters
    ----------
    df : pd.DataFrame
        Full market dataframe with DatetimeIndex and 'trading_day' column.
    ob_record : pd.Series | dict
        Enriched or selected Order Block record.
    horizon : HorizonType
        Target analysis horizon window.
    target_multiples : Sequence[float]
        List of target multiples to test (e.g. 1.0, 2.0, 3.0, 5.0).
    stop_r : float
        Adverse stop threshold in R multiples (default 1.0 R).

    Returns
    -------
    dict[str, Any]
        Dictionary of path outcomes keyed by target multiple.
    """
    row = dict(ob_record)
    trading_day = row["trading_day"]
    direction: OBDirection = row["ob_direction"]
    c_low = float(row["candidate_low"])
    c_high = float(row["candidate_high"])
    ob_range = calculate_ob_range(high=c_high, low=c_low)
    tap_entry_boundary = get_tap_entry_boundary(
        direction=direction, candidate_low=c_low, candidate_high=c_high
    )

    fvg_pos = row.get("fvg_position")
    fvg_int_pos = int(fvg_pos) if pd.notna(fvg_pos) else None

    # Resolve the exact candle slice for this horizon
    df_window, _, _ = resolve_horizon_window(
        df=df,
        ob_trading_day=trading_day,
        fvg_position=fvg_int_pos,
        horizon=horizon,
    )

    results: dict[str, Any] = {
        "trading_day": trading_day,
        "candidate_position": int(row["candidate_position"]),
        "candidate_timestamp": pd.Timestamp(row["candidate_timestamp"]),
        "ob_direction": direction,
        "horizon": str(horizon),
        "was_tapped_in_horizon": False,
    }

    if df_window.empty:
        for k in target_multiples:
            k_tag = f"path_{int(k)}r" if k.is_integer() else f"path_{k}r"
            results[f"{k_tag}_status"] = "not_tapped"
            results[f"{k_tag}_resolution_timestamp"] = pd.NaT
            results[f"{k_tag}_candles_to_resolution"] = pd.NA
        return results

    # Check for tap in window
    tap_mask = (df_window["low"] <= c_high) & (df_window["high"] >= c_low)
    if not tap_mask.any():
        for k in target_multiples:
            k_tag = f"path_{int(k)}r" if k.is_integer() else f"path_{k}r"
            results[f"{k_tag}_status"] = "not_tapped"
            results[f"{k_tag}_resolution_timestamp"] = pd.NaT
            results[f"{k_tag}_candles_to_resolution"] = pd.NA
        return results

    # Tapped in window
    results["was_tapped_in_horizon"] = True
    first_tap_loc = int(np.argmax(tap_mask.values))
    post_tap_candles = df_window.iloc[first_tap_loc:].copy()

    for k in target_multiples:
        k_tag = f"path_{int(k)}r" if k.is_integer() else f"path_{k}r"
        path_res = evaluate_single_target_path(
            direction=direction,
            tap_entry_boundary=tap_entry_boundary,
            ob_range=ob_range,
            target_r=k,
            stop_r=stop_r,
            post_tap_candles=post_tap_candles,
        )
        results[f"{k_tag}_status"] = path_res["status"]
        results[f"{k_tag}_target_price"] = path_res["target_price"]
        results[f"{k_tag}_stop_price"] = path_res["stop_price"]
        results[f"{k_tag}_resolution_timestamp"] = path_res["resolution_timestamp"]
        results[f"{k_tag}_candles_to_resolution"] = path_res["candles_to_resolution"]

    return results


def batch_scan_path_dependent_outcomes(
    df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    horizon: HorizonType = "next_trading_day",
    target_multiples: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    stop_r: float = 1.0,
) -> pd.DataFrame:
    """
    Perform batch path-dependent target scan across all enriched Order Blocks.
    """
    if enriched_df.empty:
        return pd.DataFrame()

    records = []
    for _, ob_row in enriched_df.iterrows():
        rec = scan_order_block_path_outcomes(
            df=df,
            ob_record=ob_row,
            horizon=horizon,
            target_multiples=target_multiples,
            stop_r=stop_r,
        )
        records.append(rec)

    return pd.DataFrame(records)
