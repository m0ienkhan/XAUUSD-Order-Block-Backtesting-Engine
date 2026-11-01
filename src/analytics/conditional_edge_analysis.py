from __future__ import annotations

from typing import Any, Optional, Sequence, Union
import pandas as pd
import numpy as np

from src.analytics.metrics import OBDirection
from src.analytics.statistical_analysis import CREATION_TIME_FEATURES, calculate_wilson_ci


def derive_creation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive strictly creation-time categorical features for conditional interaction analysis.
    Raises ValueError if non-creation features are used for predictive grouping.
    """
    out = df.copy()

    # Enforce creation time feature safety
    creation_cols = set(CREATION_TIME_FEATURES).intersection(out.columns)
    if "distance_from_day_close" not in creation_cols:
        raise ValueError("Missing mandatory creation feature: distance_from_day_close")

    # 1. Distance from day close tertiles
    # Use quantile bins based on empirical tertiles: Near (<$4.30), Medium ($4.30-$12.16), Far (>$12.16)
    dist = out["distance_from_day_close"]
    q33 = dist.quantile(0.3333)
    q66 = dist.quantile(0.6666)

    def _classify_dist(d: float) -> str:
        if pd.isna(d):
            return "Unknown"
        if d <= q33:
            return "Near_Close"
        elif d <= q66:
            return "Medium_Dist"
        else:
            return "Far_Close"

    out["dist_bucket"] = dist.apply(_classify_dist)

    # 2. Formation Session (UTC based)
    timestamps = pd.to_datetime(out["candidate_timestamp"], utc=True)
    hours = timestamps.dt.hour

    def _classify_session(h: int) -> str:
        if 0 <= h < 7:
            return "1_Asian"
        elif 7 <= h < 13:
            return "2_London"
        elif 13 <= h < 17:
            return "3_NY_Morning"
        else:
            return "4_NY_Late"

    out["session"] = hours.apply(_classify_session)

    # 3. Order Block Size Quartiles
    ob_r = out["ob_range"]
    q25 = ob_r.quantile(0.25)
    q50 = ob_r.quantile(0.50)
    q75 = ob_r.quantile(0.75)

    def _classify_size(r: float) -> str:
        if pd.isna(r):
            return "Unknown"
        if r <= q25:
            return "Q1_Small"
        elif r <= q50:
            return "Q2_Medium"
        elif r <= q75:
            return "Q3_Medium"
        else:
            return "Q4_Large"

    out["size_bucket"] = ob_r.apply(_classify_size)

    # 4. FVG to OB Ratio Tertiles
    fvg_ratio = out["fvg_size"] / np.maximum(out["ob_range"], 1e-6)
    f_q33 = fvg_ratio.quantile(0.3333)
    f_q66 = fvg_ratio.quantile(0.6666)

    def _classify_fvg_ratio(ratio: float) -> str:
        if pd.isna(ratio):
            return "Unknown"
        if ratio <= f_q33:
            return "Small_FVG"
        elif ratio <= f_q66:
            return "Medium_FVG"
        else:
            return "Large_FVG"

    out["fvg_ratio_bucket"] = fvg_ratio.apply(_classify_fvg_ratio)

    # 5. Chronological period for robustness testing
    def _classify_period(ts: pd.Timestamp) -> str:
        if ts < pd.Timestamp("2019-07-01", tz="UTC"):
            return "P1_2016_2019"
        elif ts < pd.Timestamp("2023-01-01", tz="UTC"):
            return "P2_2019_2022"
        else:
            return "P3_2023_2026"

    out["period"] = timestamps.apply(_classify_period)

    return out


def prepare_conditional_dataset(
    enriched_df: pd.DataFrame,
    path_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join enriched order blocks with candle-by-candle path outcomes and annotate
    creation-time conditional features.
    """
    match_keys = ["trading_day", "candidate_position", "candidate_timestamp", "ob_direction"]
    path_cols = [
        "trading_day",
        "candidate_position",
        "candidate_timestamp",
        "ob_direction",
        "was_tapped_in_horizon",
        "path_1r_status",
        "path_1r_candles_to_resolution",
        "path_2r_status",
        "path_2r_candles_to_resolution",
        "path_3r_status",
        "path_3r_candles_to_resolution",
    ]

    # Verify matching keys exist
    for k in match_keys:
        if k not in enriched_df.columns or k not in path_df.columns:
            raise KeyError(f"Matching key {k} missing in input datasets.")

    # Convert timestamps to string format for robust merging if needed
    e_df = enriched_df.copy()
    p_df = path_df[[c for c in path_cols if c in path_df.columns]].copy()

    merged = pd.merge(e_df, p_df, on=match_keys, how="inner")
    annotated = derive_creation_features(merged)
    return annotated


def evaluate_conditional_group(
    sub_df: pd.DataFrame,
    group_name: str,
    combination_type: str,
    feature_dims: str,
    min_samples: int = 20,
) -> dict[str, Any]:
    """
    Calculate full conditional performance, expectancy, and ambiguity sensitivity
    for a specific feature combination.
    """
    total_obs = len(sub_df)
    if total_obs == 0:
        return {
            "combination_type": combination_type,
            "feature_dims": feature_dims,
            "group_name": group_name,
            "total_obs": 0,
            "tapped_obs": 0,
            "tap_rate_pct": float("nan"),
            "tap_rate_ci95": "[nan, nan]",
            "insufficient_sample": True,
        }

    # Tap rate
    if "was_tapped_in_horizon" in sub_df.columns:
        tapped_series = sub_df["was_tapped_in_horizon"] == True
    elif "was_tapped_h1d" in sub_df.columns:
        tapped_series = sub_df["was_tapped_h1d"] == True
    else:
        tapped_series = sub_df["path_1r_status"] != "not_tapped"

    tapped_obs = int(tapped_series.sum())
    tap_rate_pct = float(round((tapped_obs / total_obs) * 100.0, 2))
    ci_tap_low, ci_tap_high = calculate_wilson_ci(tapped_obs, total_obs)

    row: dict[str, Any] = {
        "combination_type": combination_type,
        "feature_dims": feature_dims,
        "group_name": group_name,
        "total_obs": total_obs,
        "tapped_obs": tapped_obs,
        "tap_rate_pct": tap_rate_pct,
        "tap_rate_ci95": f"[{ci_tap_low}%, {ci_tap_high}%]",
    }

    # Evaluate multiples: 1.0R, 2.0R, 3.0R
    for k, mult_str in [(1.0, "1r"), (2.0, "2r"), (3.0, "3r")]:
        status_col = f"path_{mult_str}_status"
        if status_col not in sub_df.columns:
            continue

        statuses = sub_df[status_col].fillna("not_tapped")
        target_cnt = int((statuses == "target_first").sum())
        stop_cnt = int((statuses == "stop_first").sum())
        ambig_cnt = int((statuses == "ambiguous_same_candle").sum())
        no_res_cnt = int((statuses == "no_resolution_within_horizon").sum())
        not_tap_cnt = int((statuses == "not_tapped").sum())

        clean_resolved = target_cnt + stop_cnt
        total_resolved = target_cnt + stop_cnt + ambig_cnt + no_res_cnt

        # Clean Win Rate (Excluding Ambiguous)
        if clean_resolved > 0:
            clean_wr = float(round((target_cnt / clean_resolved) * 100.0, 2))
            ci_wr_low, ci_wr_high = calculate_wilson_ci(target_cnt, clean_resolved)
            ci_width = float(round(ci_wr_high - ci_wr_low, 2))
            clean_wr_prop = target_cnt / clean_resolved
            exp_clean = float(round(clean_wr_prop * k - (1.0 - clean_wr_prop) * 1.0, 4))
        else:
            clean_wr = float("nan")
            ci_wr_low, ci_wr_high = float("nan"), float("nan")
            ci_width = float("nan")
            exp_clean = float("nan")

        # Total Population Win Rate
        total_wr = float(round((target_cnt / total_obs) * 100.0, 2))

        # Ambiguity Sensitivity
        # Optimistic: Ambiguous counts as Win
        total_with_ambig = target_cnt + stop_cnt + ambig_cnt
        if total_with_ambig > 0:
            opt_wr = float(round(((target_cnt + ambig_cnt) / total_with_ambig) * 100.0, 2))
            cons_wr = float(round((target_cnt / total_with_ambig) * 100.0, 2))
            cons_wr_prop = target_cnt / total_with_ambig
            exp_cons = float(round(cons_wr_prop * k - (1.0 - cons_wr_prop) * 1.0, 4))
            ambig_pct = float(round((ambig_cnt / total_with_ambig) * 100.0, 2))
        else:
            opt_wr = float("nan")
            cons_wr = float("nan")
            exp_cons = float("nan")
            ambig_pct = 0.0

        # Candidate EV (R per selected candidate)
        cand_ev = float(round((target_cnt * k - stop_cnt * 1.0) / total_obs, 4))

        row[f"target_first_{mult_str}"] = target_cnt
        row[f"stop_first_{mult_str}"] = stop_cnt
        row[f"ambig_{mult_str}"] = ambig_cnt
        row[f"no_res_{mult_str}"] = no_res_cnt
        row[f"clean_resolved_{mult_str}"] = clean_resolved
        row[f"clean_win_rate_{mult_str}_pct"] = clean_wr
        row[f"clean_win_rate_{mult_str}_ci95"] = f"[{ci_wr_low}%, {ci_wr_high}%]"
        row[f"ci_width_{mult_str}"] = ci_width
        row[f"total_win_rate_{mult_str}_pct"] = total_wr
        row[f"expectancy_clean_{mult_str}"] = exp_clean
        row[f"expectancy_conservative_{mult_str}"] = exp_cons
        row[f"candidate_ev_{mult_str}"] = cand_ev
        row[f"opt_win_rate_{mult_str}_pct"] = opt_wr
        row[f"cons_win_rate_{mult_str}_pct"] = cons_wr
        row[f"ambig_rate_{mult_str}_pct"] = ambig_pct

    # Sample size protection
    clean_res_1r = row.get("clean_resolved_1r", 0)
    row["insufficient_sample"] = bool(total_obs < min_samples or clean_res_1r < min_samples)

    return row


def analyze_all_feature_combinations(
    df: pd.DataFrame,
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    Run systematic single, two-way, and limited three-way feature combination analyses.
    """
    records: list[dict[str, Any]] = []

    # 1. Base Single Features
    single_features = [
        ("dist_bucket", "dist_bucket"),
        ("session", "session"),
        ("size_bucket", "size_bucket"),
        ("ob_direction", "ob_direction"),
        ("fvg_ratio_bucket", "fvg_ratio_bucket"),
    ]

    for col, dim_name in single_features:
        if col not in df.columns:
            continue
        for val, sub in df.groupby(col, observed=True):
            r = evaluate_conditional_group(
                sub,
                group_name=f"{col}={val}",
                combination_type="1_Way",
                feature_dims=dim_name,
                min_samples=min_samples,
            )
            records.append(r)


    # 2. Two-Way Feature Combinations
    two_way_pairs = [
        (("dist_bucket", "session"), "dist x session"),
        (("dist_bucket", "size_bucket"), "dist x size"),
        (("dist_bucket", "ob_direction"), "dist x direction"),
        (("session", "size_bucket"), "session x size"),
        (("session", "ob_direction"), "session x direction"),
        (("size_bucket", "ob_direction"), "size x direction"),
        (("dist_bucket", "fvg_ratio_bucket"), "dist x fvg_ratio"),
        (("session", "fvg_ratio_bucket"), "session x fvg_ratio"),
    ]

    for (c1, c2), dim_name in two_way_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        for (v1, v2), sub in df.groupby([c1, c2], observed=True):
            g_name = f"{c1}={v1} | {c2}={v2}"
            r = evaluate_conditional_group(
                sub,
                group_name=g_name,
                combination_type="2_Way",
                feature_dims=dim_name,
                min_samples=min_samples,
            )
            records.append(r)

    # 3. Limited Three-Way Feature Combinations
    three_way_triplets = [
        (("dist_bucket", "session", "size_bucket"), "dist x session x size"),
        (("dist_bucket", "session", "ob_direction"), "dist x session x direction"),
        (("dist_bucket", "size_bucket", "ob_direction"), "dist x size x direction"),
        (("session", "size_bucket", "ob_direction"), "session x size x direction"),
        (("dist_bucket", "session", "fvg_ratio_bucket"), "dist x session x fvg_ratio"),
    ]

    for (c1, c2, c3), dim_name in three_way_triplets:
        if c1 not in df.columns or c2 not in df.columns or c3 not in df.columns:
            continue
        for (v1, v2, v3), sub in df.groupby([c1, c2, c3], observed=True):
            g_name = f"{c1}={v1} | {c2}={v2} | {c3}={v3}"
            r = evaluate_conditional_group(
                sub,
                group_name=g_name,
                combination_type="3_Way",
                feature_dims=dim_name,
                min_samples=min_samples,
            )
            records.append(r)

    results_df = pd.DataFrame(records)
    return results_df


def test_chronological_robustness(
    df: pd.DataFrame,
    combinations_df: pd.DataFrame,
    min_period_samples: int = 10,
) -> pd.DataFrame:
    """
    Test chronological stability of all feature combinations across three periods:
    P1 (2016-2019), P2 (2019-2022), P3 (2023-2026).
    """
    if "period" not in df.columns:
        raise KeyError("period column missing in dataset.")

    periods = [
        ("p1", "P1_2016_2019"),
        ("p2", "P2_2019_2022"),
        ("p3", "P3_2023_2026"),
    ]

    robustness_records: list[dict[str, Any]] = []

    for _, row in combinations_df.iterrows():
        g_name = row["group_name"]
        comb_type = row["combination_type"]
        is_insufficient = row.get("insufficient_sample", True)

        if is_insufficient:
            robustness_records.append({
                "group_name": g_name,
                "p1_resolved_1r": 0,
                "p1_win_rate_1r": float("nan"),
                "p1_exp_1r": float("nan"),
                "p2_resolved_1r": 0,
                "p2_win_rate_1r": float("nan"),
                "p2_exp_1r": float("nan"),
                "p3_resolved_1r": 0,
                "p3_win_rate_1r": float("nan"),
                "p3_exp_1r": float("nan"),
                "positive_periods_count": 0,
                "is_consistently_profitable": False,
                "win_rate_std": float("nan"),
                "robustness_score": 0.0,
            })
            continue

        # Parse filter from group_name
        # format: col1=val1 | col2=val2
        col_aliases = {
            "distance": "dist_bucket",
            "ob_size": "size_bucket",
            "direction": "ob_direction",
            "fvg_ratio": "fvg_ratio_bucket",
        }
        parts = [p.strip() for p in g_name.split("|")]
        mask = pd.Series(True, index=df.index)
        for part in parts:
            if "=" in part:
                col, val = part.split("=", 1)
                col, val = col.strip(), val.strip()
                col = col_aliases.get(col, col)
                if col in df.columns:
                    mask &= (df[col].astype(str) == val)


        sub_all = df[mask]

        period_stats: list[dict[str, Any]] = []
        for p_idx, p_val in periods:
            sub_p = sub_all[sub_all["period"] == p_val]
            st = sub_p["path_1r_status"].fillna("not_tapped")
            tgt = int((st == "target_first").sum())
            stp = int((st == "stop_first").sum())
            res = tgt + stp


            if res >= min_period_samples:
                wr = float(round((tgt / res) * 100.0, 2))
                wr_prop = tgt / res
                exp = float(round(wr_prop * 1.0 - (1.0 - wr_prop) * 1.0, 4))
            else:
                wr = float("nan")
                exp = float("nan")

            period_stats.append({
                f"{p_idx}_resolved_1r": res,
                f"{p_idx}_win_rate_1r": wr,
                f"{p_idx}_exp_1r": exp,
            })

        p_dict = {}
        for p in period_stats:
            p_dict.update(p)

        exps = [p_dict["p1_exp_1r"], p_dict["p2_exp_1r"], p_dict["p3_exp_1r"]]
        wrs = [p_dict["p1_win_rate_1r"], p_dict["p2_win_rate_1r"], p_dict["p3_win_rate_1r"]]

        valid_exps = [e for e in exps if not np.isnan(e)]
        valid_wrs = [w for w in wrs if not np.isnan(w)]

        pos_count = sum(1 for e in valid_exps if e > 0)
        is_consistent = (len(valid_exps) == 3 and pos_count == 3) or (len(valid_exps) >= 2 and pos_count == len(valid_exps))
        wr_std = float(round(float(np.std(valid_wrs)), 2)) if len(valid_wrs) >= 2 else float("nan")

        # Robustness score multiplier
        if pos_count == 3:
            rob_score = 1.0
        elif pos_count == 2:
            rob_score = 0.6
        elif pos_count == 1:
            rob_score = 0.2
        else:
            rob_score = 0.0

        p_dict["group_name"] = g_name
        p_dict["positive_periods_count"] = pos_count
        p_dict["is_consistently_profitable"] = is_consistent
        p_dict["win_rate_std"] = wr_std
        p_dict["robustness_score"] = rob_score

        robustness_records.append(p_dict)

    rob_df = pd.DataFrame(robustness_records)
    return rob_df


def rank_feature_interactions(
    results_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank all feature combinations using a transparent composite scoring model.
    Composite Score = Expectancy_1R * sqrt(min(Resolved_1R, 500)) * (1 - CI_Width / 200) * Robustness_Multiplier
    """
    merged = pd.merge(results_df, robustness_df, on="group_name", how="left")

    def _calc_score(row: pd.Series) -> float:
        if row.get("insufficient_sample", True):
            return -999.0
        exp_1r = row.get("expectancy_clean_1r", float("nan"))
        if pd.isna(exp_1r) or exp_1r <= 0:
            return float(exp_1r) if not pd.isna(exp_1r) else -100.0

        resolved = row.get("clean_resolved_1r", 0)
        ci_w = row.get("ci_width_1r", 20.0)
        if pd.isna(ci_w):
            ci_w = 20.0
        rob_mult = row.get("robustness_score", 0.5)

        # Dampened sample size factor
        n_factor = np.sqrt(min(resolved, 500))
        # Precision penalty (narrower CI = better)
        precision_factor = max(0.1, 1.0 - (ci_w / 200.0))

        score = exp_1r * n_factor * precision_factor * rob_mult
        return float(round(score, 3))

    merged["composite_edge_score"] = merged.apply(_calc_score, axis=1)

    # Assign Tier Classification
    def _classify_tier(row: pd.Series) -> str:
        if row.get("insufficient_sample", True):
            return "INSUFFICIENT_SAMPLE"
        exp_1r = row.get("expectancy_clean_1r", float("nan"))
        score = row.get("composite_edge_score", -999.0)
        is_robust = row.get("is_consistently_profitable", False)
        res_1r = row.get("clean_resolved_1r", 0)

        if pd.isna(exp_1r) or exp_1r <= 0:
            return "NEGATIVE_EXPECTANCY"
        if score >= 3.0 and res_1r >= 50 and is_robust:
            return "TIER_1_STRONG_EDGE"
        elif score >= 1.5 and is_robust:
            return "TIER_2_MODERATE_EDGE"
        elif exp_1r > 0:
            return "TIER_3_WEAK_OR_UNSTABLE"
        else:
            return "NEGATIVE_EXPECTANCY"

    merged["tier"] = merged.apply(_classify_tier, axis=1)
    merged = merged.sort_values(by="composite_edge_score", ascending=False).reset_index(drop=True)
    return merged


def extract_top_hypotheses(
    ranked_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract the top candidate strategy hypotheses meeting minimum robustness and sample size criteria.
    """
    valid = ranked_df[
        (~ranked_df["insufficient_sample"])
        & (ranked_df["tier"].isin(["TIER_1_STRONG_EDGE", "TIER_2_MODERATE_EDGE"]))
    ].copy()
    return valid.head(top_n)
