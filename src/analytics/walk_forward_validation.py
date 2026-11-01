from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union
import pandas as pd
import numpy as np

from src.analytics.metrics import OBDirection
from src.analytics.statistical_analysis import CREATION_TIME_FEATURES, calculate_wilson_ci
from src.analytics.conditional_edge_analysis import analyze_all_feature_combinations, evaluate_conditional_group


@dataclass(frozen=True)
class FrozenFeatureBoundaries:
    """
    Exact numerical quantile boundaries computed EXCLUSIVELY on the Training partition (2016-2022).
    Applied immutably across Validation and Holdout partitions to guarantee zero lookahead.
    """
    dist_q33: float
    dist_q66: float
    size_q25: float
    size_q50: float
    size_q75: float
    fvg_ratio_q33: float
    fvg_ratio_q66: float


@dataclass(frozen=True)
class FrozenStrategyDefinition:
    """
    Immutable representation of a strategy hypothesis discovered and frozen on Training data.
    """
    strategy_id: str
    strategy_name: str
    filter_dict: dict[str, str]
    combination_type: str
    discovery_score: float
    discovery_clean_win_rate_1r: float
    discovery_exp_1r: float
    discovery_resolved_1r: int


@dataclass
class StrategyPeriodPerformance:
    """
    Comprehensive path-dependent performance metrics for a strategy on a specific chronological period.
    """
    strategy_id: str
    strategy_name: str
    period_name: str
    total_obs: int
    tapped_obs: int
    tap_rate_pct: float
    tap_rate_ci95: str
    clean_resolved_1r: int
    target_first_1r: int
    stop_first_1r: int
    ambig_1r: int
    no_res_1r: int
    not_tapped_1r: int
    clean_win_rate_1r_pct: float
    clean_win_rate_1r_ci95: str
    opt_win_rate_1r_pct: float
    cons_win_rate_1r_pct: float
    ambig_rate_1r_pct: float
    expectancy_clean_1r: float
    expectancy_conservative_1r: float
    candidate_ev_1r: float
    clean_resolved_2r: int
    clean_win_rate_2r_pct: float
    expectancy_clean_2r: float
    clean_resolved_3r: int
    clean_win_rate_3r_pct: float
    expectancy_clean_3r: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown_r: float
    net_equity_r: float
    bullish_resolved_1r: int
    bullish_win_rate_1r_pct: float
    bearish_resolved_1r: int
    bearish_win_rate_1r_pct: float
    insufficient_sample: bool


@dataclass
class MultipleTestingFunnel:
    """
    Multiple testing audit tracking hypothesis counts through the discovery and validation funnel.
    """
    total_hypotheses_generated: int = 0
    passed_sample_size_filter: int = 0
    passed_positive_expectancy_filter: int = 0
    passed_wilson_ci_filter: int = 0
    removed_by_deduplication: int = 0
    candidate_pool_frozen_for_validation: int = 0
    validation_survivors: int = 0
    validation_insufficient_sample: int = 0
    validation_failed: int = 0
    holdout_verified_oos_edge: int = 0
    holdout_positive_insufficient_sample: int = 0
    holdout_insufficient_sample: int = 0
    holdout_failed: int = 0


@dataclass
class WalkForwardStrategyResult:
    """
    Final synthesized result for a strategy across Train, Validation, and Holdout stages.
    """
    strategy: FrozenStrategyDefinition
    train_perf: StrategyPeriodPerformance
    val_perf: StrategyPeriodPerformance
    holdout_perf: Optional[StrategyPeriodPerformance]
    validation_status: str
    holdout_status: str
    overall_classification: str
    validation_degradation_warning: bool
    win_rate_drop_val_pct: float
    win_rate_drop_holdout_pct: float
    notes: str


def compute_training_feature_boundaries(train_df: pd.DataFrame) -> FrozenFeatureBoundaries:
    """
    Compute feature quantile thresholds strictly and exclusively on the Training dataset.
    """
    if "distance_from_day_close" not in train_df.columns:
        raise ValueError("Missing distance_from_day_close in training data")
    if "ob_range" not in train_df.columns:
        raise ValueError("Missing ob_range in training data")
    if "fvg_size" not in train_df.columns:
        raise ValueError("Missing fvg_size in training data")

    dist = train_df["distance_from_day_close"]
    dist_q33 = float(dist.quantile(0.3333))
    dist_q66 = float(dist.quantile(0.6666))

    ob_r = train_df["ob_range"]
    size_q25 = float(ob_r.quantile(0.25))
    size_q50 = float(ob_r.quantile(0.50))
    size_q75 = float(ob_r.quantile(0.75))

    fvg_ratio = train_df["fvg_size"] / np.maximum(train_df["ob_range"], 1e-6)
    fvg_ratio_q33 = float(fvg_ratio.quantile(0.3333))
    fvg_ratio_q66 = float(fvg_ratio.quantile(0.6666))

    return FrozenFeatureBoundaries(
        dist_q33=dist_q33,
        dist_q66=dist_q66,
        size_q25=size_q25,
        size_q50=size_q50,
        size_q75=size_q75,
        fvg_ratio_q33=fvg_ratio_q33,
        fvg_ratio_q66=fvg_ratio_q66,
    )


def apply_frozen_feature_boundaries(
    df: pd.DataFrame,
    boundaries: FrozenFeatureBoundaries,
) -> pd.DataFrame:
    """
    Annotate creation-time categorical features using pre-frozen numerical thresholds.
    Does NOT compute any quantiles on the passed DataFrame.
    """
    out = df.copy()

    # 1. Distance bucket
    dist = out["distance_from_day_close"]
    def _classify_dist(d: float) -> str:
        if pd.isna(d):
            return "Unknown"
        if d <= boundaries.dist_q33:
            return "Near_Close"
        elif d <= boundaries.dist_q66:
            return "Medium_Dist"
        else:
            return "Far_Close"
    out["dist_bucket"] = dist.apply(_classify_dist)

    # 2. Session
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

    # 3. Size bucket
    ob_r = out["ob_range"]
    def _classify_size(r: float) -> str:
        if pd.isna(r):
            return "Unknown"
        if r <= boundaries.size_q25:
            return "Q1_Small"
        elif r <= boundaries.size_q50:
            return "Q2_Medium"
        elif r <= boundaries.size_q75:
            return "Q3_Medium"
        else:
            return "Q4_Large"
    out["size_bucket"] = ob_r.apply(_classify_size)

    # 4. FVG ratio bucket
    fvg_ratio = out["fvg_size"] / np.maximum(out["ob_range"], 1e-6)
    def _classify_fvg_ratio(ratio: float) -> str:
        if pd.isna(ratio):
            return "Unknown"
        if ratio <= boundaries.fvg_ratio_q33:
            return "Small_FVG"
        elif ratio <= boundaries.fvg_ratio_q66:
            return "Medium_FVG"
        else:
            return "Large_FVG"
    out["fvg_ratio_bucket"] = fvg_ratio.apply(_classify_fvg_ratio)

    return out


def split_datasets_by_period(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the full dataset into:
    - Training: 2016-01-01 through 2022-12-31
    - Validation: 2023-01-01 through 2024-12-31
    - Final Holdout: 2025-01-01 through latest available
    """
    timestamps = pd.to_datetime(df["candidate_timestamp"], utc=True)

    val_start = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    holdout_start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    train_mask = timestamps < val_start
    val_mask = (timestamps >= val_start) & (timestamps < holdout_start)
    holdout_mask = timestamps >= holdout_start

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    holdout_df = df[holdout_mask].copy().reset_index(drop=True)

    return train_df, val_df, holdout_df


def parse_group_name_to_filters(group_name: str) -> dict[str, str]:
    """
    Convert group_name (e.g. 'dist_bucket=Near_Close | size_bucket=Q1_Small') to a filter dict.
    """
    col_aliases = {
        "distance": "dist_bucket",
        "ob_size": "size_bucket",
        "direction": "ob_direction",
        "fvg_ratio": "fvg_ratio_bucket",
    }
    filters: dict[str, str] = {}
    for part in group_name.split("|"):
        if "=" in part:
            col, val = part.split("=", 1)
            col, val = col.strip(), val.strip()
            col = col_aliases.get(col, col)
            filters[col] = val
    return filters


def filter_dataset_by_strategy(
    df: pd.DataFrame,
    filter_dict: dict[str, str],
) -> pd.DataFrame:
    """Filter DataFrame rows matching all key-value conditions in filter_dict."""
    mask = pd.Series(True, index=df.index)
    for col, val in filter_dict.items():
        if col in df.columns:
            mask &= (df[col].astype(str) == val)
        else:
            return df.iloc[0:0].copy()
    return df[mask].copy()


def compute_equity_curve_and_drawdown(
    trades_df: pd.DataFrame,
) -> tuple[list[float], float, int, int, float]:
    """
    Compute sequential trade outcomes, cumulative R equity curve, max drawdown,
    and consecutive streak metrics.
    """
    if trades_df.empty:
        return [], 0.0, 0, 0, 0.0

    # Sort trades chronologically
    sorted_trades = trades_df.sort_values(by="candidate_timestamp").copy()
    status_series = sorted_trades["path_1r_status"].fillna("not_tapped")

    # Filter to resolved trades (target_first or stop_first)
    resolved = status_series[status_series.isin(["target_first", "stop_first"])]
    if resolved.empty:
        return [], 0.0, 0, 0, 0.0

    r_returns: list[float] = []
    equity_curve: list[float] = []
    cum_r = 0.0
    peak_r = 0.0
    max_dd = 0.0

    current_win_streak = 0
    max_win_streak = 0
    current_loss_streak = 0
    max_loss_streak = 0

    for st in resolved:
        if st == "target_first":
            ret = 1.0
            current_win_streak += 1
            current_loss_streak = 0
            if current_win_streak > max_win_streak:
                max_win_streak = current_win_streak
        else:  # stop_first
            ret = -1.0
            current_loss_streak += 1
            current_win_streak = 0
            if current_loss_streak > max_loss_streak:
                max_loss_streak = current_loss_streak

        cum_r += ret
        r_returns.append(ret)
        equity_curve.append(round(cum_r, 2))

        if cum_r > peak_r:
            peak_r = cum_r
        dd = peak_r - cum_r
        if dd > max_dd:
            max_dd = dd

    net_r = float(round(cum_r, 2))
    max_dd_r = float(round(max_dd, 2))

    return equity_curve, max_dd_r, max_win_streak, max_loss_streak, net_r


def evaluate_strategy_on_period(
    df: pd.DataFrame,
    strategy: FrozenStrategyDefinition,
    period_name: str,
    min_samples: int = 20,
) -> StrategyPeriodPerformance:
    """
    Evaluate a frozen strategy definition on a specific period partition.
    """
    sub_df = filter_dataset_by_strategy(df, strategy.filter_dict)
    total_obs = len(sub_df)

    if "was_tapped_in_horizon" in sub_df.columns:
        tapped_series = sub_df["was_tapped_in_horizon"] == True
    elif "was_tapped_h1d" in sub_df.columns:
        tapped_series = sub_df["was_tapped_h1d"] == True
    else:
        tapped_series = sub_df["path_1r_status"] != "not_tapped"

    tapped_obs = int(tapped_series.sum()) if total_obs > 0 else 0
    tap_rate = float(round((tapped_obs / total_obs) * 100.0, 2)) if total_obs > 0 else float("nan")
    ci_tap_low, ci_tap_high = calculate_wilson_ci(tapped_obs, total_obs)

    # 1R Evaluation
    st_1r = sub_df["path_1r_status"].fillna("not_tapped") if total_obs > 0 else pd.Series(dtype=str)
    tgt_1r = int((st_1r == "target_first").sum())
    stp_1r = int((st_1r == "stop_first").sum())
    amb_1r = int((st_1r == "ambiguous_same_candle").sum())
    no_res_1r = int((st_1r == "no_resolution_within_horizon").sum())
    not_tap_1r = int((st_1r == "not_tapped").sum())

    clean_res_1r = tgt_1r + stp_1r
    if clean_res_1r > 0:
        clean_wr_1r = float(round((tgt_1r / clean_res_1r) * 100.0, 2))
        ci_low_1r, ci_high_1r = calculate_wilson_ci(tgt_1r, clean_res_1r)
        p_clean = tgt_1r / clean_res_1r
        exp_clean_1r = float(round(p_clean * 1.0 - (1.0 - p_clean) * 1.0, 4))
    else:
        clean_wr_1r = float("nan")
        ci_low_1r, ci_high_1r = float("nan"), float("nan")
        exp_clean_1r = float("nan")

    total_with_amb_1r = tgt_1r + stp_1r + amb_1r
    if total_with_amb_1r > 0:
        opt_wr_1r = float(round(((tgt_1r + amb_1r) / total_with_amb_1r) * 100.0, 2))
        cons_wr_1r = float(round((tgt_1r / total_with_amb_1r) * 100.0, 2))
        p_cons = tgt_1r / total_with_amb_1r
        exp_cons_1r = float(round(p_cons * 1.0 - (1.0 - p_cons) * 1.0, 4))
        amb_rate_1r = float(round((amb_1r / total_with_amb_1r) * 100.0, 2))
    else:
        opt_wr_1r = float("nan")
        cons_wr_1r = float("nan")
        exp_cons_1r = float("nan")
        amb_rate_1r = float("nan")

    cand_ev_1r = float(round((tgt_1r * 1.0 - stp_1r * 1.0) / total_obs, 4)) if total_obs > 0 else float("nan")

    # 2R Evaluation
    st_2r = sub_df["path_2r_status"].fillna("not_tapped") if total_obs > 0 and "path_2r_status" in sub_df.columns else pd.Series(dtype=str)
    tgt_2r = int((st_2r == "target_first").sum())
    stp_2r = int((st_2r == "stop_first").sum())
    clean_res_2r = tgt_2r + stp_2r
    if clean_res_2r > 0:
        clean_wr_2r = float(round((tgt_2r / clean_res_2r) * 100.0, 2))
        p_clean_2r = tgt_2r / clean_res_2r
        exp_clean_2r = float(round(p_clean_2r * 2.0 - (1.0 - p_clean_2r) * 1.0, 4))
    else:
        clean_wr_2r = float("nan")
        exp_clean_2r = float("nan")

    # 3R Evaluation
    st_3r = sub_df["path_3r_status"].fillna("not_tapped") if total_obs > 0 and "path_3r_status" in sub_df.columns else pd.Series(dtype=str)
    tgt_3r = int((st_3r == "target_first").sum())
    stp_3r = int((st_3r == "stop_first").sum())
    clean_res_3r = tgt_3r + stp_3r
    if clean_res_3r > 0:
        clean_wr_3r = float(round((tgt_3r / clean_res_3r) * 100.0, 2))
        p_clean_3r = tgt_3r / clean_res_3r
        exp_clean_3r = float(round(p_clean_3r * 3.0 - (1.0 - p_clean_3r) * 1.0, 4))
    else:
        clean_wr_3r = float("nan")
        exp_clean_3r = float("nan")

    # Sequential Equity & Drawdown
    eq_curve, max_dd_r, max_w_streak, max_l_streak, net_r = compute_equity_curve_and_drawdown(sub_df)

    # Directional Breakdown
    if total_obs > 0 and "ob_direction" in sub_df.columns:
        bull_sub = sub_df[sub_df["ob_direction"] == "bullish"]
        bear_sub = sub_df[sub_df["ob_direction"] == "bearish"]

        bull_st = bull_sub["path_1r_status"].fillna("not_tapped")
        bull_tgt = int((bull_st == "target_first").sum())
        bull_stp = int((bull_st == "stop_first").sum())
        bull_res = bull_tgt + bull_stp
        bull_wr = float(round((bull_tgt / bull_res) * 100.0, 2)) if bull_res > 0 else float("nan")

        bear_st = bear_sub["path_1r_status"].fillna("not_tapped")
        bear_tgt = int((bear_st == "target_first").sum())
        bear_stp = int((bear_st == "stop_first").sum())
        bear_res = bear_tgt + bear_stp
        bear_wr = float(round((bear_tgt / bear_res) * 100.0, 2)) if bear_res > 0 else float("nan")
    else:
        bull_res, bull_wr = 0, float("nan")
        bear_res, bear_wr = 0, float("nan")

    is_insufficient = bool(clean_res_1r < min_samples)

    return StrategyPeriodPerformance(
        strategy_id=strategy.strategy_id,
        strategy_name=strategy.strategy_name,
        period_name=period_name,
        total_obs=total_obs,
        tapped_obs=tapped_obs,
        tap_rate_pct=tap_rate,
        tap_rate_ci95=f"[{ci_tap_low}%, {ci_tap_high}%]",
        clean_resolved_1r=clean_res_1r,
        target_first_1r=tgt_1r,
        stop_first_1r=stp_1r,
        ambig_1r=amb_1r,
        no_res_1r=no_res_1r,
        not_tapped_1r=not_tap_1r,
        clean_win_rate_1r_pct=clean_wr_1r,
        clean_win_rate_1r_ci95=f"[{ci_low_1r}%, {ci_high_1r}%]",
        opt_win_rate_1r_pct=opt_wr_1r,
        cons_win_rate_1r_pct=cons_wr_1r,
        ambig_rate_1r_pct=amb_rate_1r,
        expectancy_clean_1r=exp_clean_1r,
        expectancy_conservative_1r=exp_cons_1r,
        candidate_ev_1r=cand_ev_1r,
        clean_resolved_2r=clean_res_2r,
        clean_win_rate_2r_pct=clean_wr_2r,
        expectancy_clean_2r=exp_clean_2r,
        clean_resolved_3r=clean_res_3r,
        clean_win_rate_3r_pct=clean_wr_3r,
        expectancy_clean_3r=exp_clean_3r,
        max_consecutive_wins=max_w_streak,
        max_consecutive_losses=max_l_streak,
        max_drawdown_r=max_dd_r,
        net_equity_r=net_r,
        bullish_resolved_1r=bull_res,
        bullish_win_rate_1r_pct=bull_wr,
        bearish_resolved_1r=bear_res,
        bearish_win_rate_1r_pct=bear_wr,
        insufficient_sample=is_insufficient,
    )


def discover_and_freeze_strategies(
    train_annotated_df: pd.DataFrame,
    min_samples: int = 20,
    top_n: int = 15,
) -> tuple[list[FrozenStrategyDefinition], MultipleTestingFunnel]:
    """
    STAGE 1: Discover, filter, deduplicate, and rank strategy hypotheses using ONLY Training data (2016-2022).
    """
    funnel = MultipleTestingFunnel()

    # Generate all combinations on Training data
    all_comb_df = analyze_all_feature_combinations(train_annotated_df, min_samples=min_samples)
    funnel.total_hypotheses_generated = len(all_comb_df)

    # Filter 1: Min samples
    sample_mask = all_comb_df["clean_resolved_1r"] >= min_samples
    funnel.passed_sample_size_filter = int(sample_mask.sum())
    valid_df = all_comb_df[sample_mask].copy()

    # Filter 2: Positive expectancy
    exp_mask = valid_df["expectancy_clean_1r"] > 0.0
    funnel.passed_positive_expectancy_filter = int(exp_mask.sum())
    valid_df = valid_df[exp_mask].copy()

    # Filter 3: Wilson CI lower bound >= 50%
    def _ci_lower_ge_50(ci_str: str) -> bool:
        try:
            parts = str(ci_str).strip("[]").split(",")
            low = float(parts[0].strip(" %"))
            return low >= 50.0
        except Exception:
            return False


    ci_mask = valid_df["clean_win_rate_1r_ci95"].apply(_ci_lower_ge_50)
    funnel.passed_wilson_ci_filter = int(ci_mask.sum())
    valid_df = valid_df[ci_mask].copy()

    # Calculate Discovery Score for ranking
    def _calc_discovery_score(row: pd.Series) -> float:
        exp_1r = row["expectancy_clean_1r"]
        n_res = row["clean_resolved_1r"]
        ci_w = row.get("ci_width_1r", 20.0)
        if pd.isna(ci_w):
            ci_w = 20.0
        score = exp_1r * np.sqrt(min(n_res, 500)) * max(0.1, 1.0 - (ci_w / 200.0))
        return float(round(score, 3))

    valid_df["discovery_score"] = valid_df.apply(_calc_discovery_score, axis=1)
    valid_df = valid_df.sort_values(by="discovery_score", ascending=False).reset_index(drop=True)

    # Filter 4: Deduplication of nested / equivalent hypotheses
    # Deduplicate by trade overlap: if a 3-way combination has >=95% trade overlap with an existing 2-way rule, drop it
    frozen_strategies: list[FrozenStrategyDefinition] = []
    selected_trade_indices: list[set[int]] = []
    dedup_dropped_count = 0

    for idx, row in valid_df.iterrows():
        g_name = row["group_name"]
        filter_d = parse_group_name_to_filters(g_name)
        sub = filter_dataset_by_strategy(train_annotated_df, filter_d)
        current_indices = set(sub.index)

        # Check Jaccard overlap with already selected strategies to remove redundant near-identical rules
        is_duplicate = False
        for prev_indices in selected_trade_indices:
            intersection = len(current_indices.intersection(prev_indices))
            union = len(current_indices.union(prev_indices))
            if union > 0 and (intersection / union) >= 0.85:
                is_duplicate = True
                break

        if is_duplicate:
            dedup_dropped_count += 1
            continue

        strat_id = f"STRAT_{len(frozen_strategies) + 1:02d}"
        strat = FrozenStrategyDefinition(
            strategy_id=strat_id,
            strategy_name=g_name,
            filter_dict=filter_d,
            combination_type=row["combination_type"],
            discovery_score=row["discovery_score"],
            discovery_clean_win_rate_1r=row["clean_win_rate_1r_pct"],
            discovery_exp_1r=row["expectancy_clean_1r"],
            discovery_resolved_1r=int(row["clean_resolved_1r"]),
        )
        frozen_strategies.append(strat)
        selected_trade_indices.append(current_indices)

        if len(frozen_strategies) >= top_n:
            break


    funnel.removed_by_deduplication = dedup_dropped_count
    funnel.candidate_pool_frozen_for_validation = len(frozen_strategies)

    return frozen_strategies, funnel


def filter_validation_survivors(
    val_annotated_df: pd.DataFrame,
    candidate_strategies: list[FrozenStrategyDefinition],
    train_performances: dict[str, StrategyPeriodPerformance],
    min_validation_samples: int = 20,
    degradation_warning_threshold: float = 15.0,
) -> tuple[list[FrozenStrategyDefinition], dict[str, StrategyPeriodPerformance], dict[str, str], dict[str, bool]]:
    """
    STAGE 2: Evaluate frozen Training candidate strategies on 2023-2024 Validation data
    and enforce predefined validation survival criteria.
    """
    survivors: list[FrozenStrategyDefinition] = []
    val_perfs: dict[str, StrategyPeriodPerformance] = {}
    val_statuses: dict[str, str] = {}
    degradation_flags: dict[str, bool] = {}

    for strat in candidate_strategies:
        s_id = strat.strategy_id
        val_perf = evaluate_strategy_on_period(val_annotated_df, strat, "Validation_2023_2024", min_samples=min_validation_samples)
        val_perfs[s_id] = val_perf

        train_perf = train_performances[s_id]
        train_wr = train_perf.clean_win_rate_1r_pct
        val_wr = val_perf.clean_win_rate_1r_pct

        # Check win-rate degradation diagnostic flag
        drop = train_wr - val_wr if not (pd.isna(train_wr) or pd.isna(val_wr)) else float("nan")
        has_degradation_warning = bool(not pd.isna(drop) and drop > degradation_warning_threshold)
        degradation_flags[s_id] = has_degradation_warning

        # Predefined Validation Survival Rules:
        # 1. Sample size
        if val_perf.clean_resolved_1r < min_validation_samples:
            status = "INSUFFICIENT_VALIDATION_SAMPLE"
        # 2. Positive expectancy
        elif val_perf.expectancy_clean_1r <= 0.0 or val_perf.clean_win_rate_1r_pct <= 50.0:
            status = "FAILED_NEGATIVE_EXPECTANCY"
        # 3. Conservative ambiguity resilience
        elif val_perf.expectancy_conservative_1r <= -0.10:
            status = "FAILED_AMBIGUITY_VULNERABILITY"
        else:
            status = "VALIDATION_SURVIVOR"
            survivors.append(strat)

        val_statuses[s_id] = status

    return survivors, val_perfs, val_statuses, degradation_flags


def evaluate_holdout_and_classify(
    holdout_annotated_df: pd.DataFrame,
    candidate_strategies: list[FrozenStrategyDefinition],
    validation_survivors: list[FrozenStrategyDefinition],
    train_perfs: dict[str, StrategyPeriodPerformance],
    val_perfs: dict[str, StrategyPeriodPerformance],
    val_statuses: dict[str, str],
    degradation_flags: dict[str, bool],
    min_holdout_samples: int = 20,
    min_holdout_partial_samples: int = 10,
) -> tuple[list[WalkForwardStrategyResult], dict[str, StrategyPeriodPerformance]]:
    """
    STAGE 4: Perform one-time final out-of-sample evaluation on 2025-2026 Holdout partition.
    """
    results: list[WalkForwardStrategyResult] = []
    holdout_perfs: dict[str, StrategyPeriodPerformance] = {}
    survivor_ids = {s.strategy_id for s in validation_survivors}

    for strat in candidate_strategies:
        s_id = strat.strategy_id
        train_p = train_perfs[s_id]
        val_p = val_perfs[s_id]
        val_status = val_statuses[s_id]
        has_warning = degradation_flags[s_id]

        # Evaluate Holdout for all candidates (reporting completeness)
        holdout_p = evaluate_strategy_on_period(holdout_annotated_df, strat, "Holdout_2025_2026", min_samples=min_holdout_partial_samples)
        holdout_perfs[s_id] = holdout_p

        train_wr = train_p.clean_win_rate_1r_pct
        val_wr = val_p.clean_win_rate_1r_pct
        holdout_wr = holdout_p.clean_win_rate_1r_pct

        drop_val = float(round(train_wr - val_wr, 2)) if not (pd.isna(train_wr) or pd.isna(val_wr)) else float("nan")
        drop_holdout = float(round(train_wr - holdout_wr, 2)) if not (pd.isna(train_wr) or pd.isna(holdout_wr)) else float("nan")

        # Classify Holdout Status
        if s_id not in survivor_ids:
            # Did not pass validation stage
            holdout_status = "NOT_ADVANCED_TO_HOLDOUT"
            overall_class = f"REJECTED_IN_VALIDATION ({val_status})"
            notes = f"Failed Stage 2 Validation ({val_status})."
        else:
            # Passed Stage 2 Validation, evaluate Holdout survival
            h_res = holdout_p.clean_resolved_1r
            h_exp = holdout_p.expectancy_clean_1r

            if pd.isna(h_exp) or h_exp <= 0.0 or holdout_p.clean_win_rate_1r_pct <= 50.0:
                holdout_status = "FAILED_IN_HOLDOUT"
                overall_class = "FAILED_IN_HOLDOUT"
                notes = f"Passed Validation but negative expectancy in Holdout ({h_exp:.4f} R)."
            elif h_res >= min_holdout_samples:
                holdout_status = "VERIFIED_OOS_EDGE"
                overall_class = "VERIFIED_OOS_EDGE"
                notes = "Passed Train, Validation, and Holdout with full sample (N >= 20)."
            elif h_res >= min_holdout_partial_samples:
                holdout_status = "POSITIVE_BUT_INSUFFICIENT_OOS_SAMPLE"
                overall_class = "POSITIVE_BUT_INSUFFICIENT_OOS_SAMPLE"
                notes = f"Positive Holdout expectancy ({h_exp:.4f} R) with 10 <= N < 20 trades."
            else:
                holdout_status = "INSUFFICIENT_HOLDOUT_SAMPLE"
                overall_class = "INSUFFICIENT_HOLDOUT_SAMPLE"
                notes = f"Positive Holdout expectancy but N < 10 trades ({h_res} trades)."

        res = WalkForwardStrategyResult(
            strategy=strat,
            train_perf=train_p,
            val_perf=val_p,
            holdout_perf=holdout_p,
            validation_status=val_status,
            holdout_status=holdout_status,
            overall_classification=overall_class,
            validation_degradation_warning=has_warning,
            win_rate_drop_val_pct=drop_val,
            win_rate_drop_holdout_pct=drop_holdout,
            notes=notes,
        )
        results.append(res)

    return results, holdout_perfs


def compute_walk_forward_breakdown_tables(
    full_annotated_df: pd.DataFrame,
    strategies: list[FrozenStrategyDefinition],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate yearly performance, monthly performance, and cumulative equity curve tables
    for all candidate strategies across the full 10-year span.
    """
    yearly_records: list[dict[str, Any]] = []
    monthly_records: list[dict[str, Any]] = []
    equity_curve_records: list[dict[str, Any]] = []

    full_df = full_annotated_df.copy()
    timestamps = pd.to_datetime(full_df["candidate_timestamp"], utc=True)
    full_df["year"] = timestamps.dt.year
    full_df["year_month"] = timestamps.dt.strftime("%Y-%m")

    for strat in strategies:
        s_id = strat.strategy_id
        s_name = strat.strategy_name
        sub = filter_dataset_by_strategy(full_df, strat.filter_dict)

        if sub.empty:
            continue

        # Overall equity curve
        sorted_sub = sub.sort_values(by="candidate_timestamp")
        cum_r = 0.0
        trade_num = 0
        for _, tr in sorted_sub.iterrows():
            st = tr["path_1r_status"]
            if st in ["target_first", "stop_first"]:
                trade_num += 1
                ret = 1.0 if st == "target_first" else -1.0
                cum_r += ret
                equity_curve_records.append({
                    "strategy_id": s_id,
                    "trade_num": trade_num,
                    "timestamp": tr["candidate_timestamp"],
                    "status": st,
                    "return_r": ret,
                    "cumulative_equity_r": round(cum_r, 2),
                })

        # Yearly breakdown
        for yr, yr_df in sub.groupby("year"):
            st = yr_df["path_1r_status"].fillna("not_tapped")
            tgt = int((st == "target_first").sum())
            stp = int((st == "stop_first").sum())
            amb = int((st == "ambiguous_same_candle").sum())
            res = tgt + stp
            wr = float(round((tgt / res) * 100.0, 2)) if res > 0 else float("nan")
            net_r = float(round(tgt * 1.0 - stp * 1.0, 2))
            yearly_records.append({
                "strategy_id": s_id,
                "strategy_name": s_name,
                "year": yr,
                "total_obs": len(yr_df),
                "resolved_trades": res,
                "target_first": tgt,
                "stop_first": stp,
                "ambiguous": amb,
                "clean_win_rate_pct": wr,
                "net_r": net_r,
            })

        # Monthly breakdown
        for ym, ym_df in sub.groupby("year_month"):
            st = ym_df["path_1r_status"].fillna("not_tapped")
            tgt = int((st == "target_first").sum())
            stp = int((st == "stop_first").sum())
            amb = int((st == "ambiguous_same_candle").sum())
            res = tgt + stp
            wr = float(round((tgt / res) * 100.0, 2)) if res > 0 else float("nan")
            net_r = float(round(tgt * 1.0 - stp * 1.0, 2))
            monthly_records.append({
                "strategy_id": s_id,
                "strategy_name": s_name,
                "year_month": ym,
                "total_obs": len(ym_df),
                "resolved_trades": res,
                "target_first": tgt,
                "stop_first": stp,
                "ambiguous": amb,
                "clean_win_rate_pct": wr,
                "net_r": net_r,
            })

    yearly_df = pd.DataFrame(yearly_records)
    monthly_df = pd.DataFrame(monthly_records)
    equity_df = pd.DataFrame(equity_curve_records)

    return yearly_df, monthly_df, equity_df


def format_walk_forward_summary_table(
    results: list[WalkForwardStrategyResult],
) -> pd.DataFrame:
    """
    Format walk-forward results into a clean, comprehensive DataFrame for export and reporting.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        strat = r.strategy
        tr = r.train_perf
        val = r.val_perf
        hld = r.holdout_perf

        rows.append({
            "strategy_id": strat.strategy_id,
            "strategy_name": strat.strategy_name,
            "combination_type": strat.combination_type,
            "discovery_score": strat.discovery_score,
            "train_resolved": tr.clean_resolved_1r,
            "train_win_rate_pct": tr.clean_win_rate_1r_pct,
            "train_exp_1r": tr.expectancy_clean_1r,
            "train_net_r": tr.net_equity_r,
            "val_resolved": val.clean_resolved_1r,
            "val_win_rate_pct": val.clean_win_rate_1r_pct,
            "val_exp_1r": val.expectancy_clean_1r,
            "val_net_r": val.net_equity_r,
            "val_status": r.validation_status,
            "degradation_warning": r.validation_degradation_warning,
            "val_win_rate_drop_pct": r.win_rate_drop_val_pct,
            "holdout_resolved": hld.clean_resolved_1r if hld else 0,
            "holdout_win_rate_pct": hld.clean_win_rate_1r_pct if hld else float("nan"),
            "holdout_exp_1r": hld.expectancy_clean_1r if hld else float("nan"),
            "holdout_net_r": hld.net_equity_r if hld else float("nan"),
            "holdout_status": r.holdout_status,
            "overall_classification": r.overall_classification,
            "notes": r.notes,
        })
    return pd.DataFrame(rows)
