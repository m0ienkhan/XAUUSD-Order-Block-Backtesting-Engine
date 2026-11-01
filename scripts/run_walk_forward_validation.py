#!/usr/bin/env python3
"""
Two-Stage Out-of-Sample Walk-Forward Strategy Validation Runner.

Executes a strict chronological zero-lookahead validation protocol:
- Training (Discovery): 2016-01-01 to 2022-12-31
- Validation (Filtering): 2023-01-01 to 2024-12-31
- Final Holdout (Verification): 2025-01-01 to 2026-08-31
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics.walk_forward_validation import (
    FrozenFeatureBoundaries,
    FrozenStrategyDefinition,
    MultipleTestingFunnel,
    WalkForwardStrategyResult,
    compute_training_feature_boundaries,
    apply_frozen_feature_boundaries,
    split_datasets_by_period,
    evaluate_strategy_on_period,
    discover_and_freeze_strategies,
    filter_validation_survivors,
    evaluate_holdout_and_classify,
    compute_walk_forward_breakdown_tables,
    format_walk_forward_summary_table,
)
from src.analytics.reports import format_table


def run_walk_forward_research(
    enriched_csv: Path,
    path_csv: Path,
    output_dir: Path,
    min_samples: int = 20,
    min_val_samples: int = 20,
    min_holdout_samples: int = 20,
    top_n: int = 15,
    degradation_warning_threshold: float = 15.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  TWO-STAGE OUT-OF-SAMPLE WALK-FORWARD STRATEGY VALIDATION")
    print("=" * 70)

    print(f"Loading enriched data from {enriched_csv}...")
    enriched_df = pd.read_csv(enriched_csv)
    print(f"Loading path outcomes from {path_csv}...")
    path_df = pd.read_csv(path_csv)

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
    p_df = path_df[[c for c in path_cols if c in path_df.columns]].copy()
    full_merged = pd.merge(enriched_df, p_df, on=match_keys, how="inner")
    print(f"Full Population: {len(full_merged)} Selected Order Blocks")

    # Step 1: Chronological Partitioning
    print("\nStep 1: Chronological Partitioning into Train, Validation, and Holdout...")
    train_df, val_df, holdout_df = split_datasets_by_period(full_merged)
    print(f"  - Training (2016-01-01 to 2022-12-31)    : {len(train_df)} Selected OBs")
    print(f"  - Validation (2023-01-01 to 2024-12-31)  : {len(val_df)} Selected OBs")
    print(f"  - Final Holdout (2025-01-01 to 2026-08-31): {len(holdout_df)} Selected OBs")

    # Step 2: Compute Frozen Boundaries on Train ONLY
    print("\nStep 2: Computing Frozen Feature Quantile Boundaries exclusively on Training data...")
    boundaries = compute_training_feature_boundaries(train_df)
    print(f"  - Distance from Day Close Tertiles : Near <= ${boundaries.dist_q33:.2f} | Medium <= ${boundaries.dist_q66:.2f} | Far > ${boundaries.dist_q66:.2f}")
    print(f"  - OB Range Quartiles               : Q1 <= ${boundaries.size_q25:.2f} | Q2 <= ${boundaries.size_q50:.2f} | Q3 <= ${boundaries.size_q75:.2f} | Q4 > ${boundaries.size_q75:.2f}")
    print(f"  - FVG-to-OB Ratio Tertiles         : Small <= {boundaries.fvg_ratio_q33:.2f} | Medium <= {boundaries.fvg_ratio_q66:.2f} | Large > {boundaries.fvg_ratio_q66:.2f}")

    # Step 3: Apply Frozen Boundaries to all three datasets
    print("\nStep 3: Applying frozen Training boundaries across all partitions (Zero Lookahead)...")
    train_ann = apply_frozen_feature_boundaries(train_df, boundaries)
    val_ann = apply_frozen_feature_boundaries(val_df, boundaries)
    holdout_ann = apply_frozen_feature_boundaries(holdout_df, boundaries)
    full_ann = apply_frozen_feature_boundaries(full_merged, boundaries)

    # Step 4: Stage 1 - Strategy Discovery on Training data
    print("\nStep 4: STAGE 1 - Discovering and ranking candidate hypotheses on Training data...")
    candidate_strategies, funnel = discover_and_freeze_strategies(
        train_ann,
        min_samples=min_samples,
        top_n=top_n,
    )
    print(f"  - Candidate pool frozen for validation : {len(candidate_strategies)} strategies")

    # Evaluate Training baseline performance for all candidate strategies
    train_perfs = {
        strat.strategy_id: evaluate_strategy_on_period(train_ann, strat, "Train_2016_2022", min_samples=min_samples)
        for strat in candidate_strategies
    }

    # Step 5: Stage 2 - Validation Evaluation & Survival Filtering
    print("\nStep 5: STAGE 2 - Evaluating frozen candidate strategies on 2023-2024 Validation data...")
    validation_survivors, val_perfs, val_statuses, degradation_flags = filter_validation_survivors(
        val_ann,
        candidate_strategies,
        train_perfs,
        min_validation_samples=min_val_samples,
        degradation_warning_threshold=degradation_warning_threshold,
    )
    funnel.validation_survivors = len(validation_survivors)
    funnel.validation_insufficient_sample = sum(1 for s in val_statuses.values() if s == "INSUFFICIENT_VALIDATION_SAMPLE")
    funnel.validation_failed = sum(1 for s in val_statuses.values() if s.startswith("FAILED_"))
    print(f"  - Validation Survivors (meeting all 4 criteria) : {len(validation_survivors)} / {len(candidate_strategies)}")

    # Step 6: Stage 3 & 4 - Final Strategy Freeze and Holdout Evaluation
    print("\nStep 6: STAGE 4 - One-time final out-of-sample evaluation on 2025-2026 Holdout partition...")
    results, holdout_perfs = evaluate_holdout_and_classify(
        holdout_ann,
        candidate_strategies,
        validation_survivors,
        train_perfs,
        val_perfs,
        val_statuses,
        degradation_flags,
        min_holdout_samples=min_holdout_samples,
        min_holdout_partial_samples=10,
    )

    # Update funnel holdout statistics
    for r in results:
        if r.holdout_status == "VERIFIED_OOS_EDGE":
            funnel.holdout_verified_oos_edge += 1
        elif r.holdout_status == "POSITIVE_BUT_INSUFFICIENT_OOS_SAMPLE":
            funnel.holdout_positive_insufficient_sample += 1
        elif r.holdout_status == "INSUFFICIENT_HOLDOUT_SAMPLE":
            funnel.holdout_insufficient_sample += 1
        elif r.holdout_status == "FAILED_IN_HOLDOUT":
            funnel.holdout_failed += 1

    # Step 7: Compute detailed breakdown tables
    print("\nStep 7: Computing yearly, monthly, and equity curve breakdowns across 2016-2026...")
    yearly_df, monthly_df, equity_df = compute_walk_forward_breakdown_tables(full_ann, candidate_strategies)

    # Step 8: Generate Comprehensive Walk-Forward Report
    summary_df = format_walk_forward_summary_table(results)

    report_lines: list[str] = [
        "######################################################################",
        "  TWO-STAGE OUT-OF-SAMPLE WALK-FORWARD STRATEGY VALIDATION REPORT",
        "######################################################################",
        f"  Total Historical Population  : {len(full_merged)} Selected OBs (2016-2026)",
        f"  Training Discovery Period    : 2016-01-01 to 2022-12-31 ({len(train_df)} Selected OBs)",
        f"  Validation Period            : 2023-01-01 to 2024-12-31 ({len(val_df)} Selected OBs)",
        f"  Final Holdout Period         : 2025-01-01 to 2026-08-31 ({len(holdout_df)} Selected OBs)",
        "######################################################################\n",
        "======================================================================",
        "  A. MULTIPLE TESTING HYPOTHESIS ATTRITION FUNNEL",
        "======================================================================",
        f"  1. Total Candidate Hypotheses Generated (Train)  : {funnel.total_hypotheses_generated}",
        f"  2. Passing Min Training Sample Size (N >= {min_samples})      : {funnel.passed_sample_size_filter}",
        f"  3. Passing Positive Expectancy Filter (E > 0)    : {funnel.passed_positive_expectancy_filter}",
        f"  4. Passing Wilson CI Lower Bound (CI_Low >= 50%) : {funnel.passed_wilson_ci_filter}",
        f"  5. Removed by Deduplication / Overlap Filter     : {funnel.removed_by_deduplication}",
        f"  6. Candidate Strategy Pool Frozen for Validation : {funnel.candidate_pool_frozen_for_validation}",
        f"  7. Stage 2 Validation Survivors (N >= {min_val_samples}, E > 0)    : {funnel.validation_survivors}",
        f"  8. Stage 2 Validation Insufficient Sample (< {min_val_samples})  : {funnel.validation_insufficient_sample}",
        f"  9. Stage 2 Validation Failed (E <= 0 / Ambiguous): {funnel.validation_failed}",
        f" 10. Stage 4 Final Holdout Verified OOS Edges      : {funnel.holdout_verified_oos_edge}",
        f" 11. Stage 4 Positive but Insufficient OOS Sample  : {funnel.holdout_positive_insufficient_sample}",
        f" 12. Stage 4 Holdout Insufficient Sample (< 10)    : {funnel.holdout_insufficient_sample}",
        f" 13. Stage 4 Failed in Holdout (E <= 0)            : {funnel.holdout_failed}",
        "======================================================================\n",
    ]

    # Section B: Walk-Forward Multi-Stage Performance Table
    display_summary_cols = [
        "strategy_id",
        "strategy_name",
        "train_resolved",
        "train_win_rate_pct",
        "train_exp_1r",
        "val_resolved",
        "val_win_rate_pct",
        "val_exp_1r",
        "val_status",
        "degradation_warning",
        "holdout_resolved",
        "holdout_win_rate_pct",
        "holdout_exp_1r",
        "overall_classification",
    ]
    report_lines.append(format_table(summary_df[display_summary_cols], "B. WALK-FORWARD MULTI-STAGE PERFORMANCE SUMMARY"))

    # Section C: Validation Survivors Final Verification Table
    survivors_df = summary_df[summary_df["val_status"] == "VALIDATION_SURVIVOR"].copy()
    if not survivors_df.empty:
        report_lines.append(format_table(survivors_df[display_summary_cols], "C. VALIDATION SURVIVORS OUT-OF-SAMPLE VERIFICATION"))
    else:
        report_lines.append("\n  [No strategies survived all validation criteria]\n")

    # Section D: Yearly Performance of Top Validation Survivor
    if not survivors_df.empty:
        top_strat_id = survivors_df.iloc[0]["strategy_id"]
        top_yearly = yearly_df[yearly_df["strategy_id"] == top_strat_id].copy()
        report_lines.append(format_table(top_yearly[[
            "strategy_id",
            "year",
            "resolved_trades",
            "target_first",
            "stop_first",
            "clean_win_rate_pct",
            "net_r",
        ]], f"D. YEARLY PERFORMANCE BREAKDOWN: {top_strat_id} ({survivors_df.iloc[0]['strategy_name']})"))

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Step 9: Export artifacts
    print(f"\nSaving walk-forward validation artifacts to {output_dir}...")
    summary_df.to_csv(output_dir / "walk_forward_strategy_summary.csv", index=False)
    yearly_df.to_csv(output_dir / "walk_forward_yearly_performance.csv", index=False)
    monthly_df.to_csv(output_dir / "walk_forward_monthly_performance.csv", index=False)
    equity_df.to_csv(output_dir / "walk_forward_equity_curves.csv", index=False)

    with open(output_dir / "walk_forward_strategy_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Walk-forward strategy validation complete!")


def main():
    parser = argparse.ArgumentParser(description="Run Two-Stage Out-of-Sample Walk-Forward Strategy Validation")
    parser.add_argument(
        "--enriched-csv",
        type=str,
        default="outputs/reports/full_enriched_order_blocks_2016_2026.csv",
        help="Path to full enriched order blocks CSV",
    )
    parser.add_argument(
        "--path-csv",
        type=str,
        default="outputs/reports/full_path_outcomes_2016_2026.csv",
        help="Path to full path outcomes CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports",
        help="Directory to save output reports and CSVs",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum training sample size threshold for discovery",
    )
    parser.add_argument(
        "--min-validation-samples",
        type=int,
        default=20,
        help="Minimum validation sample size threshold for survival",
    )
    parser.add_argument(
        "--min-holdout-samples",
        type=int,
        default=20,
        help="Minimum holdout sample size threshold for full OOS verification",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top N candidate strategies to freeze from Training",
    )
    parser.add_argument(
        "--degradation-warning-threshold",
        type=float,
        default=15.0,
        help="Win-rate drop threshold (in percentage points) to flag degradation warning",
    )

    args = parser.parse_args()
    enriched_csv = Path(args.enriched_csv).resolve()
    path_csv = Path(args.path_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_walk_forward_research(
        enriched_csv=enriched_csv,
        path_csv=path_csv,
        output_dir=output_dir,
        min_samples=args.min_samples,
        min_val_samples=args.min_validation_samples,
        min_holdout_samples=args.min_holdout_samples,
        top_n=args.top_n,
        degradation_warning_threshold=args.degradation_warning_threshold,
    )


if __name__ == "__main__":
    main()
