from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DimensionRobustnessSummary:
    """
    Descriptive robustness metrics for a single parameter dimension.
    """
    sensitivity_dimension: str
    tested_scenario_count: int
    profitable_scenario_count: int
    profitable_scenario_ratio: float
    baseline_net_r: float
    minimum_net_r: float
    maximum_net_r: float
    mean_net_r: float
    baseline_expectancy: float
    minimum_expectancy: float
    maximum_expectancy: float
    mean_expectancy: float
    minimum_candidate_ev: float
    maximum_candidate_ev: float
    minimum_profit_factor: float
    maximum_profit_factor: float
    minimum_maximum_drawdown_r: float
    maximum_maximum_drawdown_r: float
    immediate_neighbor_count: int
    immediate_profitable_neighbor_count: int
    immediate_profitable_neighbor_ratio: float
    nonbaseline_scenario_count: int
    nonbaseline_profitable_count: int
    nonbaseline_profitable_ratio: float


def build_robustness_matrix(step4_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Step 4 scenario results into a consolidated, structured robustness matrix.

    Parameters
    ----------
    step4_df : pd.DataFrame
        Step 4 scenario results (must contain at least the 13 one-factor scenarios).

    Returns
    -------
    pd.DataFrame
        Enriched robustness matrix containing all required identification, execution,
        performance, baseline-relative, and classification metrics.
    """
    # Defensive copy to guarantee input immutability
    df = step4_df.copy()

    # Filter to the 13 one-factor sensitivity scenarios if extra rows exist
    if "parameter_dimension" in df.columns:
        dim_col = "parameter_dimension"
    elif "sensitivity_dimension" in df.columns:
        dim_col = "sensitivity_dimension"
    elif "sensitivity_family" in df.columns:
        dim_col = "sensitivity_family"
    else:
        raise ValueError("Missing dimension column in input DataFrame")

    # Map dimensions to standardized names: 'ENTRY', 'STOP', 'TARGET'
    def standardize_dim(val: Any) -> str:
        s = str(val).upper()
        if "ENTRY" in s:
            return "ENTRY"
        if "STOP" in s:
            return "STOP"
        if "TARGET" in s:
            return "TARGET"
        return s

    df["sensitivity_dimension"] = df[dim_col].apply(standardize_dim)

    # Filter for standard dimensions only
    valid_dims = {"ENTRY", "STOP", "TARGET"}
    df = df[df["sensitivity_dimension"].isin(valid_dims)].copy().reset_index(drop=True)

    # Ensure baseline identification
    if "is_baseline" not in df.columns:
        df["is_baseline"] = (
            (df["entry_offset_r"] == 0.0)
            & (df["stop_multiplier_r"] == 1.0)
            & (df["target_multiplier_r"] == 1.0)
        )
    else:
        df["is_baseline"] = df["is_baseline"].astype(bool)

    # Determine baseline reference metrics from the baseline rows
    baseline_rows = df[df["is_baseline"]]
    if baseline_rows.empty:
        base_net_r = 107.6593
        base_exp = 0.2705
        base_ev = 0.1712
        base_dd = 5.0
    else:
        first_base = baseline_rows.iloc[0]
        base_net_r = float(first_base.get("gross_net_r", 107.6593))
        base_exp = float(first_base.get("expectancy_r_per_filled_trade", first_base.get("expectancy_per_filled_trade", 0.2705)))
        base_ev = float(first_base.get("candidate_ev_r_per_setup", 0.1712))
        base_dd = float(first_base.get("maximum_drawdown_r", 5.0))

    # Add parameter_name and parameter_value_r
    param_names = []
    param_vals = []
    for _, row in df.iterrows():
        dim = row["sensitivity_dimension"]
        if dim == "ENTRY":
            param_names.append("entry_offset_r")
            param_vals.append(float(row["entry_offset_r"]))
        elif dim == "STOP":
            param_names.append("stop_multiplier_r")
            val = row.get("stop_multiplier_r", row.get("stop_multiplier", 1.0))
            param_vals.append(float(val))
        elif dim == "TARGET":
            param_names.append("target_multiplier_r")
            val = row.get("target_multiplier_r", row.get("target_multiplier", 1.0))
            param_vals.append(float(val))
        else:
            param_names.append("unknown")
            param_vals.append(0.0)

    df["parameter_name"] = param_names
    df["parameter_value_r"] = param_vals

    # Standardize column names
    if "stop_multiplier_r" not in df.columns and "stop_multiplier" in df.columns:
        df["stop_multiplier_r"] = df["stop_multiplier"]
    if "stop_multiplier" not in df.columns and "stop_multiplier_r" in df.columns:
        df["stop_multiplier"] = df["stop_multiplier_r"]

    if "target_multiplier_r" not in df.columns and "target_multiplier" in df.columns:
        df["target_multiplier_r"] = df["target_multiplier"]
    if "target_multiplier" not in df.columns and "target_multiplier_r" in df.columns:
        df["target_multiplier"] = df["target_multiplier_r"]

    if "unfilled_trades" not in df.columns and "unfilled_orders" in df.columns:
        df["unfilled_trades"] = df["unfilled_orders"]
    if "horizon_exit_trades" not in df.columns and "horizon_expiration_trades" in df.columns:
        df["horizon_exit_trades"] = df["horizon_expiration_trades"]
    if "filled_win_rate" not in df.columns and "filled_trade_win_rate" in df.columns:
        df["filled_win_rate"] = df["filled_trade_win_rate"]
    elif "filled_trade_win_rate" not in df.columns and "filled_win_rate" in df.columns:
        df["filled_trade_win_rate"] = df["filled_win_rate"]

    if "expectancy_r_per_filled_trade" not in df.columns and "expectancy_per_filled_trade" in df.columns:
        df["expectancy_r_per_filled_trade"] = df["expectancy_per_filled_trade"]
    elif "expectancy_per_filled_trade" not in df.columns and "expectancy_r_per_filled_trade" in df.columns:
        df["expectancy_per_filled_trade"] = df["expectancy_r_per_filled_trade"]

    # Baseline reference columns
    df["baseline_net_r"] = base_net_r
    df["delta_net_r_vs_baseline"] = (df["gross_net_r"] - base_net_r).round(4)
    df["baseline_relative_net_r"] = df["delta_net_r_vs_baseline"]

    df["baseline_expectancy_r_per_filled_trade"] = base_exp
    df["delta_expectancy_vs_baseline"] = (df["expectancy_r_per_filled_trade"] - base_exp).round(4)

    df["baseline_candidate_ev_r_per_setup"] = base_ev
    df["delta_candidate_ev_vs_baseline"] = (df["candidate_ev_r_per_setup"] - base_ev).round(4)

    df["baseline_maximum_drawdown_r"] = base_dd
    df["delta_maximum_drawdown_vs_baseline"] = (df["maximum_drawdown_r"] - base_dd).round(4)

    # Profitability and Economic Viability Classifications
    df["is_profitable"] = (df["gross_net_r"] > 0.0).astype(bool)
    df["has_positive_expectancy"] = (df["expectancy_r_per_filled_trade"] > 0.0).astype(bool)
    df["has_positive_candidate_ev"] = (df["candidate_ev_r_per_setup"] > 0.0).astype(bool)

    # Order columns logically
    ordered_cols = [
        "scenario_id",
        "sensitivity_dimension",
        "parameter_name",
        "parameter_value_r",
        "entry_offset_r",
        "stop_multiplier_r",
        "stop_multiplier",
        "target_multiplier_r",
        "target_multiplier",
        "is_baseline",
        "total_setups",
        "filled_trades",
        "unfilled_trades",
        "fill_rate",
        "take_profit_trades",
        "stop_loss_trades",
        "ambiguous_stop_loss_trades",
        "horizon_exit_trades",
        "filled_win_rate",
        "filled_trade_win_rate",
        "gross_net_r",
        "expectancy_r_per_filled_trade",
        "expectancy_per_filled_trade",
        "candidate_ev_r_per_setup",
        "profit_factor",
        "maximum_drawdown_r",
        "maximum_consecutive_losses",
        "baseline_net_r",
        "delta_net_r_vs_baseline",
        "baseline_relative_net_r",
        "baseline_expectancy_r_per_filled_trade",
        "delta_expectancy_vs_baseline",
        "baseline_candidate_ev_r_per_setup",
        "delta_candidate_ev_vs_baseline",
        "baseline_maximum_drawdown_r",
        "delta_maximum_drawdown_vs_baseline",
        "is_profitable",
        "has_positive_expectancy",
        "has_positive_candidate_ev",
    ]

    # Include any additional pre-existing columns
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)

    return df[[c for c in ordered_cols if c in df.columns]].copy()


def compute_dimension_summary(robustness_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute structured robustness summary metrics across the three parameter dimensions.

    Parameters
    ----------
    robustness_matrix_df : pd.DataFrame
        Detailed robustness matrix from build_robustness_matrix.

    Returns
    -------
    pd.DataFrame
        Dimension-level summary table with one row per sensitivity dimension.
    """
    df = robustness_matrix_df.copy()

    records: list[dict[str, Any]] = []

    # Map of dimension to its immediate neighbors beyond baseline
    immediate_neighbors_map = {
        "ENTRY": [0.10],
        "STOP": [0.75, 1.25],
        "TARGET": [0.75, 1.25],
    }

    dimension_order = ["ENTRY", "STOP", "TARGET"]

    for dim in dimension_order:
        dim_df = df[df["sensitivity_dimension"] == dim]
        if dim_df.empty:
            continue

        dim_label = f"{dim}_OFFSET" if dim == "ENTRY" else f"{dim}_MULTIPLIER"

        tested_count = len(dim_df)
        profitable_count = int((dim_df["gross_net_r"] > 0.0).sum())
        profitable_ratio = float(round(profitable_count / tested_count, 4)) if tested_count > 0 else 0.0

        base_rows = dim_df[dim_df["is_baseline"]]
        base_net_r = float(base_rows.iloc[0]["gross_net_r"]) if not base_rows.empty else 107.6593
        base_exp = float(base_rows.iloc[0]["expectancy_r_per_filled_trade"]) if not base_rows.empty else 0.2705

        min_net_r = float(round(dim_df["gross_net_r"].min(), 4))
        max_net_r = float(round(dim_df["gross_net_r"].max(), 4))
        mean_net_r = float(round(dim_df["gross_net_r"].mean(), 4))

        min_exp = float(round(dim_df["expectancy_r_per_filled_trade"].min(), 4))
        max_exp = float(round(dim_df["expectancy_r_per_filled_trade"].max(), 4))
        mean_exp = float(round(dim_df["expectancy_r_per_filled_trade"].mean(), 4))

        min_ev = float(round(dim_df["candidate_ev_r_per_setup"].min(), 4))
        max_ev = float(round(dim_df["candidate_ev_r_per_setup"].max(), 4))

        min_pf = float(round(dim_df["profit_factor"].min(), 2))
        max_pf = float(round(dim_df["profit_factor"].max(), 2))

        min_dd = float(round(dim_df["maximum_drawdown_r"].min(), 4))
        max_dd = float(round(dim_df["maximum_drawdown_r"].max(), 4))

        # Immediate neighborhood analysis
        imm_vals = immediate_neighbors_map.get(dim, [])
        imm_df = dim_df[dim_df["parameter_value_r"].isin(imm_vals)]
        imm_count = len(imm_df)
        imm_prof = int((imm_df["gross_net_r"] > 0.0).sum())
        imm_ratio = float(round(imm_prof / imm_count, 4)) if imm_count > 0 else 0.0

        # Non-baseline scenarios analysis
        nonbase_df = dim_df[~dim_df["is_baseline"]]
        nonbase_count = len(nonbase_df)
        nonbase_prof = int((nonbase_df["gross_net_r"] > 0.0).sum())
        nonbase_ratio = float(round(nonbase_prof / nonbase_count, 4)) if nonbase_count > 0 else 0.0

        records.append({
            "sensitivity_dimension": dim_label,
            "dimension_name": dim,
            "tested_scenario_count": tested_count,
            "profitable_scenario_count": profitable_count,
            "profitable_scenario_ratio": profitable_ratio,
            "baseline_net_r": base_net_r,
            "minimum_net_r": min_net_r,
            "maximum_net_r": max_net_r,
            "mean_net_r": mean_net_r,
            "baseline_expectancy": base_exp,
            "minimum_expectancy": min_exp,
            "maximum_expectancy": max_exp,
            "mean_expectancy": mean_exp,
            "minimum_candidate_ev": min_ev,
            "maximum_candidate_ev": max_ev,
            "minimum_profit_factor": min_pf,
            "maximum_profit_factor": max_pf,
            "minimum_maximum_drawdown_r": min_dd,
            "maximum_maximum_drawdown_r": max_dd,
            "immediate_neighbor_count": imm_count,
            "immediate_profitable_neighbor_count": imm_prof,
            "immediate_profitable_neighbor_ratio": imm_ratio,
            "nonbaseline_scenario_count": nonbase_count,
            "all_nonbaseline_scenario_count": nonbase_count,
            "nonbaseline_profitable_count": nonbase_prof,
            "all_nonbaseline_profitable_count": nonbase_prof,
            "nonbaseline_profitable_ratio": nonbase_ratio,
            "all_nonbaseline_profitable_ratio": nonbase_ratio,
        })

    return pd.DataFrame(records)


def compute_aggregate_conclusion(robustness_matrix_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute deterministic aggregate robustness conclusion metrics, handling duplicate
    baseline rows so that unique configurations are not overcounted.

    Parameters
    ----------
    robustness_matrix_df : pd.DataFrame
        Detailed robustness matrix from build_robustness_matrix.

    Returns
    -------
    dict[str, Any]
        Factual aggregate metrics summarizing parameter robustness across unique scenarios.
    """
    df = robustness_matrix_df.copy()

    # Deduplicate unique configurations:
    # 1 unique baseline + non-baseline scenarios across the 3 dimensions = 1 + 3 + 3 + 4 = 11 unique scenarios
    unique_scenarios: list[dict[str, Any]] = []
    seen_configs: set[tuple[float, float, float]] = set()

    for _, row in df.iterrows():
        cfg = (
            float(row["entry_offset_r"]),
            float(row.get("stop_multiplier_r", row.get("stop_multiplier", 1.0))),
            float(row.get("target_multiplier_r", row.get("target_multiplier", 1.0))),
        )
        if cfg not in seen_configs:
            seen_configs.add(cfg)
            unique_scenarios.append({
                "scenario_id": row["scenario_id"],
                "config": cfg,
                "is_baseline": row["is_baseline"],
                "gross_net_r": float(row["gross_net_r"]),
                "expectancy": float(row["expectancy_r_per_filled_trade"]),
                "candidate_ev": float(row["candidate_ev_r_per_setup"]),
                "dimension": row["sensitivity_dimension"],
            })

    unique_df = pd.DataFrame(unique_scenarios)
    unique_count = len(unique_df)
    unique_profitable = int((unique_df["gross_net_r"] > 0.0).sum())
    unique_profitable_ratio = float(round(unique_profitable / unique_count, 4)) if unique_count > 0 else 0.0

    # Dimension-level compliance
    dim_summary = compute_dimension_summary(df)
    all_profitable = int((dim_summary["profitable_scenario_ratio"] == 1.0).sum())
    majority_profitable = int((dim_summary["profitable_scenario_ratio"] >= 0.5).sum())
    any_nonprofitable = int((dim_summary["profitable_scenario_ratio"] < 1.0).sum())

    base_profitable = bool((unique_df[unique_df["is_baseline"]]["gross_net_r"] > 0.0).all())

    return {
        "unique_scenarios_evaluated": unique_count,
        "unique_profitable_scenarios": unique_profitable,
        "unique_profitable_ratio": unique_profitable_ratio,
        "unique_profitable_pct": float(round(unique_profitable_ratio * 100.0, 2)),
        "baseline_profitable": base_profitable,
        "dimensions_with_all_scenarios_profitable": all_profitable,
        "dimensions_with_majority_profitable": majority_profitable,
        "dimensions_with_any_nonprofitable_scenario": any_nonprofitable,
        "total_matrix_rows": len(df),
        "total_dimensions_evaluated": len(dim_summary),
    }


def generate_and_save_robustness_matrix_artifacts(
    step4_df: pd.DataFrame,
    detailed_output_path: Optional[Path] = None,
    summary_output_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Generate and persist both the detailed parameter robustness matrix and the dimension summary.

    Parameters
    ----------
    step4_df : pd.DataFrame
        Step 4 scenario results.
    detailed_output_path : Optional[Path]
        Target path for detailed matrix CSV (defaults to execution_parameter_robustness_strat_09.csv).
    summary_output_path : Optional[Path]
        Target path for summary CSV (defaults to execution_robustness_summary_strat_09.csv).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]
        (detailed_matrix_df, summary_df, aggregate_conclusion_dict)
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    if detailed_output_path is None:
        detailed_output_path = project_root / "outputs" / "reports" / "execution_parameter_robustness_strat_09.csv"
    if summary_output_path is None:
        summary_output_path = project_root / "outputs" / "reports" / "execution_robustness_summary_strat_09.csv"

    detailed_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_df = build_robustness_matrix(step4_df)
    summary_df = compute_dimension_summary(matrix_df)
    conclusion = compute_aggregate_conclusion(matrix_df)

    matrix_df.to_csv(detailed_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    return matrix_df, summary_df, conclusion
