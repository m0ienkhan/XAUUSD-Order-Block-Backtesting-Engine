from __future__ import annotations

from typing import Any, Optional, Sequence, Union
import pandas as pd
import numpy as np

from src.analytics.metrics import OBDirection


CREATION_TIME_FEATURES: frozenset[str] = frozenset([
    "trading_day",
    "day_close",
    "candidate_timestamp",
    "candidate_position",
    "ob_direction",
    "extreme_type",
    "candidate_open",
    "candidate_high",
    "candidate_low",
    "candidate_close",
    "ob_range",
    "distance_from_day_close",
    "tap_entry_boundary",
    "is_inside_bar",
    "mother_candle_position",
    "mother_candle_timestamp",
    "has_associated_fvg",
    "fvg_timestamp",
    "fvg_position",
    "fvg_type",
    "fvg_lower",
    "fvg_upper",
    "fvg_size",
    "candles_between_ob_and_fvg",
])


def calculate_wilson_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Calculate Wilson score 95% confidence interval for a binomial proportion.
    Returns (ci_lower_pct, ci_upper_pct).
    """
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.95996  # 95% confidence
    p = successes / total
    denominator = 1.0 + (z**2 / total)
    centre_adj = p + (z**2 / (2 * total))
    spread = z * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    lower = max(0.0, (centre_adj - spread) / denominator) * 100.0
    upper = min(1.0, (centre_adj + spread) / denominator) * 100.0
    return float(round(lower, 2)), float(round(upper, 2))


def compute_summary_stats(
    series: pd.Series,
    min_samples: int = 3,
) -> dict[str, Any]:
    """
    Calculate summary statistics with sample-size gating and outlier resistance.
    """
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "insufficient_sample": True,
        }

    p25 = float(clean.quantile(0.25))
    p75 = float(clean.quantile(0.75))
    std = float(clean.std()) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0

    return {
        "count": n,
        "mean": float(clean.mean()),
        "std": std,
        "sem": sem,
        "median": float(clean.median()),
        "p25": p25,
        "p75": p75,
        "iqr": float(p75 - p25),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "insufficient_sample": bool(n < min_samples),
    }


def _get_column_names(horizon_tag: str) -> tuple[str, str, str, str, str]:
    """
    Helper to resolve target metric column names based on horizon tag.
    """
    tag = f"_{horizon_tag}" if horizon_tag else ""
    tapped_col = f"was_future_tapped{tag}" if horizon_tag else "was_future_tapped"
    if tapped_col not in ["was_future_tapped", "was_tapped_h1d", "was_tapped_h3d", "was_tapped_h5d", "was_tapped_full"]:
        # fallback to standard pattern
        if f"was_tapped_{horizon_tag}" in ["was_tapped_h1d", "was_tapped_h3d", "was_tapped_h5d", "was_tapped_full"]:
            tapped_col = f"was_tapped_{horizon_tag}"

    mfe_r_col = f"mfe_r{tag}"
    mae_r_col = f"mae_r{tag}"
    mfe_p_col = f"mfe_price{tag}"
    mae_p_col = f"mae_price{tag}"

    return tapped_col, mfe_r_col, mae_r_col, mfe_p_col, mae_p_col


def analyze_by_direction(
    enriched_df: pd.DataFrame,
    horizon_tag: str = "",
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Direction Analysis (Bullish vs Bearish):
    - Count, tap rate, MFE/MAE price & R distributions, asymmetry rate.
    """
    if enriched_df.empty:
        return pd.DataFrame()

    tapped_col, mfe_r_col, mae_r_col, mfe_p_col, mae_p_col = _get_column_names(horizon_tag)

    results = []
    for direction in ["all", "bullish", "bearish"]:
        sub = enriched_df if direction == "all" else enriched_df[enriched_df["ob_direction"] == direction]
        n_total = len(sub)
        if n_total == 0:
            continue

        # Tap rate
        tapped_sub = sub[sub[tapped_col] == True] if tapped_col in sub.columns else pd.DataFrame()
        n_tapped = len(tapped_sub)
        tap_rate = (n_tapped / n_total) * 100.0 if n_total > 0 else 0.0
        ci_l, ci_u = calculate_wilson_ci(n_tapped, n_total)

        # Tapped stats
        mfe_r_stats = compute_summary_stats(tapped_sub[mfe_r_col], min_samples=min_samples) if mfe_r_col in tapped_sub else {}
        mae_r_stats = compute_summary_stats(tapped_sub[mae_r_col], min_samples=min_samples) if mae_r_col in tapped_sub else {}

        asymmetry_rate = float("nan")
        if mfe_r_col in tapped_sub and mae_r_col in tapped_sub and n_tapped > 0:
            asymmetry_rate = float((tapped_sub[mfe_r_col] > tapped_sub[mae_r_col]).mean() * 100.0)

        results.append({
            "segment": direction,
            "total_obs": n_total,
            "tapped_obs": n_tapped,
            "tap_rate_pct": tap_rate,
            "tap_rate_ci95": f"[{ci_l}%, {ci_u}%]",
            "median_mfe_r": mfe_r_stats.get("median", float("nan")),
            "mean_mfe_r": mfe_r_stats.get("mean", float("nan")),
            "p75_mfe_r": mfe_r_stats.get("p75", float("nan")),
            "median_mae_r": mae_r_stats.get("median", float("nan")),
            "mean_mae_r": mae_r_stats.get("mean", float("nan")),
            "p75_mae_r": mae_r_stats.get("p75", float("nan")),
            "favorable_asymmetry_pct": asymmetry_rate,
            "insufficient_sample": bool(n_tapped < min_samples),
        })

    return pd.DataFrame(results)


def analyze_by_ob_size(
    enriched_df: pd.DataFrame,
    n_buckets: int = 3,
    horizon_tag: str = "",
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Order Block Size Analysis:
    Buckets OB range into quantile categories and compares tap rate, MFE_R, MAE_R.
    """
    if enriched_df.empty or "ob_range" not in enriched_df.columns:
        return pd.DataFrame()

    df_copy = enriched_df.copy()
    tapped_col, mfe_r_col, mae_r_col, _, _ = _get_column_names(horizon_tag)

    # Robust quantile bucketing with fallback for small samples
    if len(df_copy["ob_range"].unique()) >= n_buckets and len(df_copy) >= n_buckets:
        df_copy["size_bucket"] = pd.qcut(
            df_copy["ob_range"].rank(method="first"),
            q=n_buckets,
            labels=[f"Q{i+1}_Small" if i == 0 else f"Q{i+1}_Large" if i == n_buckets-1 else f"Q{i+1}_Medium" for i in range(n_buckets)],
        )
    else:
        df_copy["size_bucket"] = "All_Sizes"

    results = []
    for bucket, group in df_copy.groupby("size_bucket", observed=False):
        n_total = len(group)
        if n_total == 0:
            continue

        min_r = float(group["ob_range"].min())
        max_r = float(group["ob_range"].max())
        tapped_sub = group[group[tapped_col] == True] if tapped_col in group.columns else pd.DataFrame()
        n_tapped = len(tapped_sub)
        tap_rate = (n_tapped / n_total) * 100.0 if n_total > 0 else 0.0
        ci_l, ci_u = calculate_wilson_ci(n_tapped, n_total)

        mfe_r_stats = compute_summary_stats(tapped_sub[mfe_r_col], min_samples=min_samples) if mfe_r_col in tapped_sub else {}
        mae_r_stats = compute_summary_stats(tapped_sub[mae_r_col], min_samples=min_samples) if mae_r_col in tapped_sub else {}

        results.append({
            "size_bucket": str(bucket),
            "range_min_usd": min_r,
            "range_max_usd": max_r,
            "total_obs": n_total,
            "tapped_obs": n_tapped,
            "tap_rate_pct": tap_rate,
            "tap_rate_ci95": f"[{ci_l}%, {ci_u}%]",
            "median_mfe_r": mfe_r_stats.get("median", float("nan")),
            "median_mae_r": mae_r_stats.get("median", float("nan")),
            "p75_mfe_r": mfe_r_stats.get("p75", float("nan")),
            "insufficient_sample": bool(n_tapped < min_samples),
        })

    return pd.DataFrame(results)


def analyze_by_time_and_session(
    enriched_df: pd.DataFrame,
    horizon_tag: str = "",
    min_samples: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time & Session Analysis:
    - Session performance (Asian, London, NY Morning, NY Afternoon)
    - Day of week performance
    - Hour of formation performance
    """
    if enriched_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_copy = enriched_df.copy()
    timestamps = pd.to_datetime(df_copy["candidate_timestamp"])

    df_copy["hour"] = timestamps.dt.hour
    df_copy["day_of_week"] = timestamps.dt.day_name()

    def classify_session(hour: int) -> str:
        if 0 <= hour < 7:
            return "1_Asian (00-07 UTC)"
        elif 7 <= hour < 13:
            return "2_London (07-13 UTC)"
        elif 13 <= hour < 17:
            return "3_NY_Morning (13-17 UTC)"
        else:
            return "4_NY_Late (17-24 UTC)"

    df_copy["session"] = df_copy["hour"].apply(classify_session)
    tapped_col, mfe_r_col, mae_r_col, _, _ = _get_column_names(horizon_tag)

    def _aggregate(group_col: str) -> pd.DataFrame:
        rows = []
        for cat, group in df_copy.groupby(group_col, observed=False):
            n_total = len(group)
            if n_total == 0:
                continue
            tapped_sub = group[group[tapped_col] == True] if tapped_col in group.columns else pd.DataFrame()
            n_tapped = len(tapped_sub)
            tap_rate = (n_tapped / n_total) * 100.0 if n_total > 0 else 0.0
            ci_l, ci_u = calculate_wilson_ci(n_tapped, n_total)

            mfe_r = compute_summary_stats(tapped_sub[mfe_r_col], min_samples=min_samples) if mfe_r_col in tapped_sub else {}
            mae_r = compute_summary_stats(tapped_sub[mae_r_col], min_samples=min_samples) if mae_r_col in tapped_sub else {}

            rows.append({
                group_col: str(cat),
                "total_obs": n_total,
                "tapped_obs": n_tapped,
                "tap_rate_pct": tap_rate,
                "tap_rate_ci95": f"[{ci_l}%, {ci_u}%]",
                "median_mfe_r": mfe_r.get("median", float("nan")),
                "median_mae_r": mae_r.get("median", float("nan")),
                "insufficient_sample": bool(n_tapped < min_samples),
            })
        return pd.DataFrame(rows)

    session_df = _aggregate("session")
    dow_df = _aggregate("day_of_week")
    hour_df = _aggregate("hour")

    return session_df, dow_df, hour_df


def analyze_by_fvg_characteristics(
    enriched_df: pd.DataFrame,
    horizon_tag: str = "",
    min_samples: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    FVG Analysis:
    - Analyzes FVG size / FVG-to-OB ratio and candles between OB and FVG.
    """
    if enriched_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_copy = enriched_df.copy()
    tapped_col, mfe_r_col, mae_r_col, _, _ = _get_column_names(horizon_tag)

    # 1. Candles between OB and FVG
    rows_delay = []
    if "candles_between_ob_and_fvg" in df_copy.columns:
        for delay, group in df_copy.groupby("candles_between_ob_and_fvg", observed=False):
            n_total = len(group)
            if n_total == 0:
                continue
            tapped_sub = group[group[tapped_col] == True] if tapped_col in group.columns else pd.DataFrame()
            n_tapped = len(tapped_sub)
            tap_rate = (n_tapped / n_total) * 100.0 if n_total > 0 else 0.0
            ci_l, ci_u = calculate_wilson_ci(n_tapped, n_total)
            mfe_r = compute_summary_stats(tapped_sub[mfe_r_col], min_samples=min_samples) if mfe_r_col in tapped_sub else {}

            rows_delay.append({
                "candles_to_fvg": str(delay),
                "total_obs": n_total,
                "tapped_obs": n_tapped,
                "tap_rate_pct": tap_rate,
                "tap_rate_ci95": f"[{ci_l}%, {ci_u}%]",
                "median_mfe_r": mfe_r.get("median", float("nan")),
                "insufficient_sample": bool(n_tapped < min_samples),
            })
    delay_df = pd.DataFrame(rows_delay)

    # 2. FVG-to-OB ratio
    rows_ratio = []
    if "fvg_size" in df_copy.columns and "ob_range" in df_copy.columns:
        df_copy["fvg_ob_ratio"] = df_copy["fvg_size"] / df_copy["ob_range"].replace(0, np.nan)
        clean_ratio = df_copy.dropna(subset=["fvg_ob_ratio"])
        if len(clean_ratio) >= 3 and len(clean_ratio["fvg_ob_ratio"].unique()) >= 3:
            df_copy["fvg_ratio_bucket"] = pd.qcut(
                clean_ratio["fvg_ob_ratio"].rank(method="first"),
                q=3,
                labels=["Small_FVG", "Medium_FVG", "Large_FVG"],
            )
        else:
            df_copy["fvg_ratio_bucket"] = "All_FVG_Ratios"

        for bucket, group in df_copy.groupby("fvg_ratio_bucket", observed=False):
            n_total = len(group)
            if n_total == 0:
                continue
            tapped_sub = group[group[tapped_col] == True] if tapped_col in group.columns else pd.DataFrame()
            n_tapped = len(tapped_sub)
            tap_rate = (n_tapped / n_total) * 100.0 if n_total > 0 else 0.0
            ci_l, ci_u = calculate_wilson_ci(n_tapped, n_total)
            mfe_r = compute_summary_stats(tapped_sub[mfe_r_col], min_samples=min_samples) if mfe_r_col in tapped_sub else {}

            rows_ratio.append({
                "fvg_ratio_bucket": str(bucket),
                "total_obs": n_total,
                "tapped_obs": n_tapped,
                "tap_rate_pct": tap_rate,
                "tap_rate_ci95": f"[{ci_l}%, {ci_u}%]",
                "median_mfe_r": mfe_r.get("median", float("nan")),
                "insufficient_sample": bool(n_tapped < min_samples),
            })
    ratio_df = pd.DataFrame(rows_ratio)

    return delay_df, ratio_df


def analyze_by_distance_features(
    enriched_df: pd.DataFrame,
    horizon_tag: str = "",
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Distance Features Analysis:
    - Tests distance from day close and normalized distance / OB range.
    """
    if enriched_df.empty or "distance_from_day_close" not in enriched_df.columns:
        return pd.DataFrame()

    df_copy = enriched_df.copy()
    tapped_col, mfe_r_col, mae_r_col, _, _ = _get_column_names(horizon_tag)

    if len(df_copy["distance_from_day_close"].unique()) >= 3 and len(df_copy) >= 3:
        df_copy["dist_bucket"] = pd.qcut(
            df_copy["distance_from_day_close"].rank(method="first"),
            q=3,
            labels=["Near_Close", "Medium_Dist", "Far_Close"],
        )
    else:
        df_copy["dist_bucket"] = "All_Distances"

    results = []
    for bucket, group in df_copy.groupby("dist_bucket", observed=False):
        n_total = len(group)
        if n_total == 0:
            continue

        tapped_sub = group[group[tapped_col] == True] if tapped_col in group.columns else pd.DataFrame()
        n_tapped = len(tapped_sub)
        tap_rate = (n_tapped / n_total) * 100.0 if n_total > 0 else 0.0
        ci_l, ci_u = calculate_wilson_ci(n_tapped, n_total)

        mfe_r = compute_summary_stats(tapped_sub[mfe_r_col], min_samples=min_samples) if mfe_r_col in tapped_sub else {}
        mae_r = compute_summary_stats(tapped_sub[mae_r_col], min_samples=min_samples) if mae_r_col in tapped_sub else {}

        results.append({
            "distance_bucket": str(bucket),
            "min_dist_usd": float(group["distance_from_day_close"].min()),
            "max_dist_usd": float(group["distance_from_day_close"].max()),
            "total_obs": n_total,
            "tapped_obs": n_tapped,
            "tap_rate_pct": tap_rate,
            "tap_rate_ci95": f"[{ci_l}%, {ci_u}%]",
            "median_mfe_r": mfe_r.get("median", float("nan")),
            "median_mae_r": mae_r.get("median", float("nan")),
            "insufficient_sample": bool(n_tapped < min_samples),
        })

    return pd.DataFrame(results)


def analyze_unconditional_targets(
    enriched_df: pd.DataFrame,
    horizon_tag: str = "",
    target_multiples: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Unconditional Target Reach Probability:
    Computes P(MFE_R >= k) among tapped Order Blocks with Wilson 95% CI.
    """
    if enriched_df.empty:
        return pd.DataFrame()

    tapped_col, mfe_r_col, _, _, _ = _get_column_names(horizon_tag)
    tapped_sub = enriched_df[enriched_df[tapped_col] == True] if tapped_col in enriched_df.columns else pd.DataFrame()
    n_tapped = len(tapped_sub)

    results = []
    for k in target_multiples:
        if n_tapped > 0 and mfe_r_col in tapped_sub.columns:
            reached_count = int((tapped_sub[mfe_r_col] >= k).sum())
            reach_pct = (reached_count / n_tapped) * 100.0
            ci_l, ci_u = calculate_wilson_ci(reached_count, n_tapped)
        else:
            reached_count = 0
            reach_pct = float("nan")
            ci_l, ci_u = float("nan"), float("nan")

        results.append({
            "target_multiple": f"{k}R",
            "tapped_obs": n_tapped,
            "reached_count": reached_count,
            "unconditional_reach_pct": reach_pct,
            "reach_pct_ci95": f"[{ci_l}%, {ci_u}%]",
            "insufficient_sample": bool(n_tapped < min_samples),
        })

    return pd.DataFrame(results)


def summarize_path_dependent_results(
    path_df: pd.DataFrame,
    target_multiples: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Aggregate candle-by-candle path-dependent scan results with Wilson 95% CIs.
    """
    if path_df.empty:
        return pd.DataFrame()

    n_total = len(path_df)
    tapped_sub = path_df[path_df["was_tapped_in_horizon"] == True]
    n_tapped = len(tapped_sub)

    results = []
    for k in target_multiples:
        k_tag = f"path_{int(k)}r" if k.is_integer() else f"path_{k}r"
        status_col = f"{k_tag}_status"

        if status_col not in path_df.columns:
            continue

        target_first = int((path_df[status_col] == "target_first").sum())
        stop_first = int((path_df[status_col] == "stop_first").sum())
        ambiguous = int((path_df[status_col] == "ambiguous_same_candle").sum())
        no_res = int((path_df[status_col] == "no_resolution_within_horizon").sum())
        not_tapped = int((path_df[status_col] == "not_tapped").sum())

        resolved_unambiguous = target_first + stop_first
        clean_win_rate = (target_first / resolved_unambiguous * 100.0) if resolved_unambiguous > 0 else float("nan")
        total_win_rate = (target_first / n_tapped * 100.0) if n_tapped > 0 else float("nan")

        ci_l, ci_u = calculate_wilson_ci(target_first, resolved_unambiguous)

        results.append({
            "target_multiple": f"{k}R",
            "total_obs": n_total,
            "tapped_obs": n_tapped,
            "target_first": target_first,
            "stop_first": stop_first,
            "ambiguous_same_candle": ambiguous,
            "no_resolution": no_res,
            "not_tapped": not_tapped,
            "clean_win_rate_pct": clean_win_rate,
            "clean_win_rate_ci95": f"[{ci_l}%, {ci_u}%]",
            "total_win_rate_pct": total_win_rate,
            "insufficient_sample": bool(resolved_unambiguous < min_samples),
        })

    return pd.DataFrame(results)


def analyze_mfe_mae_relationship(
    enriched_df: pd.DataFrame,
    horizon_tag: str = "",
    min_samples: int = 3,
) -> dict[str, Any]:
    """
    MFE / MAE Relationship Analysis:
    - Excursion Ratio distributions and correlation.
    """
    if enriched_df.empty:
        return {}

    tapped_col, mfe_r_col, mae_r_col, _, _ = _get_column_names(horizon_tag)
    tapped_sub = enriched_df[enriched_df[tapped_col] == True] if tapped_col in enriched_df.columns else pd.DataFrame()

    if tapped_sub.empty or mfe_r_col not in tapped_sub or mae_r_col not in tapped_sub:
        return {"count": 0, "insufficient_sample": True}

    mfe_series = tapped_sub[mfe_r_col].dropna()
    mae_series = tapped_sub[mae_r_col].dropna()

    valid_mask = (mae_series > 1e-4) & (mfe_series.notna())
    ratio_series = (mfe_series[valid_mask] / mae_series[valid_mask])

    ratio_stats = compute_summary_stats(ratio_series, min_samples=min_samples)
    correlation = float(mfe_series.corr(mae_series)) if len(mfe_series) > 2 else float("nan")
    asymmetry_pct = float((mfe_series > mae_series).mean() * 100.0) if len(mfe_series) > 0 else float("nan")

    return {
        "tapped_count": len(mfe_series),
        "median_mfe_r": float(mfe_series.median()),
        "mean_mfe_r": float(mfe_series.mean()),
        "median_mae_r": float(mae_series.median()),
        "mean_mae_r": float(mae_series.mean()),
        "median_mfe_mae_ratio": ratio_stats.get("median", float("nan")),
        "mean_mfe_mae_ratio": ratio_stats.get("mean", float("nan")),
        "mfe_mae_correlation": correlation,
        "favorable_asymmetry_pct": asymmetry_pct,
        "insufficient_sample": bool(len(mfe_series) < min_samples),
    }
