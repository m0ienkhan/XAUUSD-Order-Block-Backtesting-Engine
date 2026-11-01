from __future__ import annotations

from typing import Any, Optional, Sequence
import pandas as pd
import numpy as np

from src.analytics.metrics import (
    OBDirection,
    calculate_ob_range,
    calculate_distance_from_day_close,
    get_tap_entry_boundary,
)
from src.analytics.mitigation_analysis import (
    HorizonType,
    resolve_horizon_window,
    analyze_ob_interaction_in_window,
)


DEFAULT_HORIZONS: list[HorizonType] = [
    "next_trading_day",
    "3_trading_days",
    "5_trading_days",
    "full",
]


def enrich_single_order_block(
    df: pd.DataFrame,
    ob_record: pd.Series | dict,
    horizons: Sequence[HorizonType] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """
    Enrich one selected Order Block record with creation-time metadata,
    FVG formation metrics, same-day post-FVG behavior, and next-trading-day
    multi-horizon future price behavior.

    Parameters
    ----------
    df : pd.DataFrame
        Full market dataframe indexed by DatetimeIndex with 'trading_day' column.
    ob_record : pd.Series | dict
        Selected Order Block record from selected_dataset.
    horizons : Sequence[HorizonType]
        List of future horizons to evaluate.

    Returns
    -------
    dict[str, Any]
        Flat enriched dictionary for this Order Block.
    """
    row = dict(ob_record)

    trading_day = row["trading_day"]
    candidate_pos = int(row["candidate_position"])
    candidate_ts = pd.Timestamp(row["candidate_timestamp"])
    direction: OBDirection = row["ob_direction"]
    extreme_type = row["extreme_type"]

    c_open = float(row["candidate_open"])
    c_high = float(row["candidate_high"])
    c_low = float(row["candidate_low"])
    c_close = float(row["candidate_close"])

    ob_range = calculate_ob_range(high=c_high, low=c_low)
    day_close = float(row["day_close"])
    distance_from_close = calculate_distance_from_day_close(
        direction=direction,
        candidate_low=c_low,
        candidate_high=c_high,
        day_close=day_close,
    )

    is_inside_bar = bool(row.get("is_inside_bar", False))
    mother_pos = row.get("mother_candle_position", pd.NA)
    mother_ts = row.get("mother_candle_timestamp", pd.NaT)

    has_fvg = bool(row.get("has_associated_fvg", False))
    fvg_pos = row.get("fvg_position")
    fvg_ts = row.get("fvg_timestamp")
    fvg_type = row.get("fvg_type")
    fvg_lower = row.get("fvg_lower")
    fvg_upper = row.get("fvg_upper")
    fvg_size = row.get("fvg_size")

    fvg_int_pos = int(fvg_pos) if pd.notna(fvg_pos) else None
    candles_between_ob_fvg = (
        int(fvg_int_pos - candidate_pos) if fvg_int_pos is not None else pd.NA
    )

    tap_entry_boundary = get_tap_entry_boundary(
        direction=direction,
        candidate_low=c_low,
        candidate_high=c_high,
    )

    # ---------------------------------------------------------
    # Base enriched dictionary
    # ---------------------------------------------------------
    enriched: dict[str, Any] = {
        # Trading day & selection context
        "trading_day": trading_day,
        "day_close": day_close,
        "is_selected": bool(row.get("is_selected", True)),
        "selection_side": row.get("selection_side", direction),
        "selection_reason": row.get("selection_reason"),

        # Order Block Identity & OHLC
        "candidate_timestamp": candidate_ts,
        "candidate_position": candidate_pos,
        "ob_direction": direction,
        "extreme_type": extreme_type,
        "candidate_open": c_open,
        "candidate_high": c_high,
        "candidate_low": c_low,
        "candidate_close": c_close,
        "ob_range": ob_range,
        "distance_from_day_close": distance_from_close,
        "tap_entry_boundary": tap_entry_boundary,

        # Inside Bar Metadata
        "is_inside_bar": is_inside_bar,
        "mother_candle_position": mother_pos,
        "mother_candle_timestamp": mother_ts,

        # FVG Metadata
        "has_associated_fvg": has_fvg,
        "fvg_timestamp": fvg_ts,
        "fvg_position": fvg_pos,
        "fvg_type": fvg_type,
        "fvg_lower": fvg_lower,
        "fvg_upper": fvg_upper,
        "fvg_size": fvg_size,
        "candles_between_ob_and_fvg": candles_between_ob_fvg,
    }

    # ---------------------------------------------------------
    # MODE A: Same-Day Post-FVG Analysis
    # ---------------------------------------------------------
    same_day_df, same_day_start_ts, same_day_start_pos = resolve_horizon_window(
        df=df,
        ob_trading_day=trading_day,
        fvg_position=fvg_int_pos,
        horizon="same_day_post_fvg",
    )

    enriched["same_day_analysis_start_timestamp"] = (
        pd.Timestamp(same_day_start_ts) if same_day_start_ts is not None else pd.NaT
    )
    enriched["same_day_analysis_start_position"] = (
        same_day_start_pos if same_day_start_pos is not None else pd.NA
    )

    same_day_metrics = analyze_ob_interaction_in_window(
        df_window=same_day_df,
        direction=direction,
        ob_low=c_low,
        ob_high=c_high,
        global_df=df,
    )

    for k, v in same_day_metrics.items():
        enriched[f"same_day_{k}"] = v

    # ---------------------------------------------------------
    # MODE B: Next-Trading-Day Base Analysis & Multi-Horizons
    # ---------------------------------------------------------
    future_start_df, future_start_ts, future_start_pos = resolve_horizon_window(
        df=df,
        ob_trading_day=trading_day,
        fvg_position=fvg_int_pos,
        horizon="full",
    )

    enriched["future_analysis_start_timestamp"] = (
        pd.Timestamp(future_start_ts) if future_start_ts is not None else pd.NaT
    )
    enriched["future_analysis_start_position"] = (
        future_start_pos if future_start_pos is not None else pd.NA
    )

    # Primary Next-Trading-Day metrics (standard un-prefixed names)
    primary_df, _, _ = resolve_horizon_window(
        df=df,
        ob_trading_day=trading_day,
        fvg_position=fvg_int_pos,
        horizon="next_trading_day",
    )
    primary_metrics = analyze_ob_interaction_in_window(
        df_window=primary_df,
        direction=direction,
        ob_low=c_low,
        ob_high=c_high,
        global_df=df,
    )

    enriched["was_future_tapped"] = primary_metrics["was_tapped"]
    enriched["first_future_tap_timestamp"] = primary_metrics["first_tap_timestamp"]
    enriched["first_future_tap_position"] = primary_metrics["first_tap_position"]
    enriched["candles_until_first_tap"] = primary_metrics["candles_until_first_tap"]
    enriched["first_tap_candle_low"] = primary_metrics["first_tap_candle_low"]
    enriched["first_tap_candle_high"] = primary_metrics["first_tap_candle_high"]
    enriched["first_tap_candle_close"] = primary_metrics["first_tap_candle_close"]
    enriched["entered_partially"] = primary_metrics["entered_partially"]
    enriched["fully_crossed"] = primary_metrics["fully_crossed"]
    enriched["closed_beyond"] = primary_metrics["closed_beyond"]
    enriched["max_penetration_depth_usd"] = primary_metrics["max_penetration_depth_usd"]
    enriched["max_penetration_pct"] = primary_metrics["max_penetration_pct"]
    enriched["mfe_price"] = primary_metrics["mfe_price"]
    enriched["mfe_r"] = primary_metrics["mfe_r"]
    enriched["mfe_timestamp"] = primary_metrics["mfe_timestamp"]
    enriched["candles_to_mfe"] = primary_metrics["candles_to_mfe"]
    enriched["mae_price"] = primary_metrics["mae_price"]
    enriched["mae_r"] = primary_metrics["mae_r"]
    enriched["mae_timestamp"] = primary_metrics["mae_timestamp"]
    enriched["candles_to_mae"] = primary_metrics["candles_to_mae"]

    # Multi-horizon evaluations
    horizon_tag_map = {
        "next_trading_day": "h1d",
        "1_trading_day": "h1d",
        "3_trading_days": "h3d",
        "5_trading_days": "h5d",
        "full": "full",
    }

    for horizon in horizons:
        tag = horizon_tag_map.get(str(horizon), f"h_{horizon}")
        h_df, _, _ = resolve_horizon_window(
            df=df,
            ob_trading_day=trading_day,
            fvg_position=fvg_int_pos,
            horizon=horizon,
        )
        h_metrics = analyze_ob_interaction_in_window(
            df_window=h_df,
            direction=direction,
            ob_low=c_low,
            ob_high=c_high,
            global_df=df,
        )
        for metric_name, val in h_metrics.items():
            enriched[f"{metric_name}_{tag}"] = val

    return enriched


def enrich_selected_order_blocks(
    df: pd.DataFrame,
    selected_df: pd.DataFrame,
    horizons: Sequence[HorizonType] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Enrich all selected Order Blocks in selected_dataset with creation metadata,
    FVG formation metrics, same-day post-FVG metrics, and next-trading-day future behaviors.

    Parameters
    ----------
    df : pd.DataFrame
        Market data indexed by DatetimeIndex with 'trading_day' column.
    selected_df : pd.DataFrame
        DataFrame of selected Order Blocks produced by daily_ob_pipeline.
    horizons : Sequence[HorizonType]
        Configured future analysis horizons.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with one row per selected Order Block.
    """
    if selected_df.empty:
        return pd.DataFrame()

    enriched_records = []
    for _, ob_row in selected_df.iterrows():
        record = enrich_single_order_block(
            df=df,
            ob_record=ob_row,
            horizons=horizons,
        )
        enriched_records.append(record)

    enriched_df = pd.DataFrame(enriched_records)
    return enriched_df
