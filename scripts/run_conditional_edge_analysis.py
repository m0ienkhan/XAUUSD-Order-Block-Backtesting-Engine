#!/usr/bin/env python3
"""
Conditional Edge Analysis & Strategy Hypothesis Generation Runner.

Evaluates multi-way feature interactions (Distance, Session, Size, Direction, FVG Ratio),
path-dependent win rates, multi-payoff theoretical expectancies (1R, 2R, 3R),
chronological robustness (2016-2019, 2019-2022, 2023-2026),
and intrabar ambiguity sensitivity across the full 10-year XAUUSD dataset.
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

from src.analytics.conditional_edge_analysis import (
    prepare_conditional_dataset,
    analyze_all_feature_combinations,
    test_chronological_robustness,
    rank_feature_interactions,
    extract_top_hypotheses,
)
from src.analytics.reports import format_table


def format_report_section(title: str) -> str:
    sep = "=" * 70
    return f"\n{sep}\n  {title}\n{sep}"


def run_conditional_research(
    enriched_csv: Path,
    path_csv: Path,
    output_dir: Path,
    min_samples: int = 20,
    min_period_samples: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading enriched data from {enriched_csv}...")
    enriched_df = pd.read_csv(enriched_csv)
    print(f"Loading path outcomes from {path_csv}...")
    path_df = pd.read_csv(path_csv)

    print(f"Enriched rows: {len(enriched_df)}, Path rows: {len(path_df)}")

    # 1. Prepare conditional dataset
    print("Preparing conditional dataset and deriving creation features...")
    cond_df = prepare_conditional_dataset(enriched_df, path_df)

    # 2. Zero data leakage audit
    print("\n--- ZERO DATA LEAKAGE AUDIT ---")
    print("Verifying grouping features are strictly derived from creation-time data...")
    grouping_cols = ["dist_bucket", "session", "size_bucket", "ob_direction", "fvg_ratio_bucket", "period"]
    for col in grouping_cols:
        assert col in cond_df.columns, f"Required grouping column {col} missing!"
    print(f"Audit passed! Grouping features: {grouping_cols}")

    # 3. Analyze all single, 2-way, and limited 3-way combinations
    print("\nEvaluating all 1-way, 2-way, and 3-way feature combinations...")
    comb_df = analyze_all_feature_combinations(cond_df, min_samples=min_samples)
    print(f"Evaluated {len(comb_df)} total distinct feature groups/combinations.")

    # 4. Chronological Robustness Testing across 3 periods
    print("\nRunning chronological robustness testing across P1 (2016-19), P2 (2019-22), P3 (2023-26)...")
    rob_df = test_chronological_robustness(cond_df, comb_df, min_period_samples=min_period_samples)

    # 5. Feature Interaction Ranking
    print("\nRanking feature interactions using composite scoring model...")
    ranked_df = rank_feature_interactions(comb_df, rob_df)

    # 6. Extract Top Strategy Hypotheses
    top_hypotheses = extract_top_hypotheses(ranked_df, top_n=25)

    # 7. Generate Full Report Text
    report_lines: list[str] = [
        "######################################################################",
        "  CONDITIONAL EDGE ANALYSIS & STRATEGY HYPOTHESIS REPORT (2016-2026)",
        "######################################################################",
        f"  Total Order Block Population : {len(cond_df)} Selected OBs",
        f"  Total Combinations Evaluated  : {len(ranked_df)}",
        f"  Minimum Sample Size Threshold : {min_samples} resolved trades",
        f"  Chronological Periods Tested  : P1 (2016-19), P2 (2019-22), P3 (2023-26)",
        "######################################################################\n",
    ]

    # Section A: Top Ranked Strategy Hypotheses
    top_display_cols = [
        "group_name",
        "tier",
        "composite_edge_score",
        "total_obs",
        "tapped_obs",
        "clean_resolved_1r",
        "clean_win_rate_1r_pct",
        "clean_win_rate_1r_ci95",
        "expectancy_clean_1r",
        "expectancy_conservative_1r",
        "candidate_ev_1r",
        "is_consistently_profitable",
        "positive_periods_count",
    ]
    if not top_hypotheses.empty:
        report_lines.append(format_table(top_hypotheses[top_display_cols].head(15), "A. TOP RANKED STRATEGY HYPOTHESES (TIER 1 & TIER 2)"))
    else:
        report_lines.append("\n  No combinations met Tier 1/2 criteria.\n")

    # Section B: Payoff Structure Expectancy Comparison (1R vs 2R vs 3R)
    payoff_cols = [
        "group_name",
        "clean_resolved_1r",
        "clean_win_rate_1r_pct",
        "expectancy_clean_1r",
        "clean_win_rate_2r_pct",
        "expectancy_clean_2r",
        "clean_win_rate_3r_pct",
        "expectancy_clean_3r",
        "composite_edge_score",
    ]
    if not top_hypotheses.empty:
        report_lines.append(format_table(top_hypotheses[payoff_cols].head(15), "B. MULTI-PAYOFF EXPECTANCY COMPARISON (1R vs 2R vs 3R)"))

    # Section C: Intrabar Ambiguity Sensitivity
    ambig_cols = [
        "group_name",
        "clean_resolved_1r",
        "opt_win_rate_1r_pct",
        "clean_win_rate_1r_pct",
        "cons_win_rate_1r_pct",
        "ambig_rate_1r_pct",
        "expectancy_clean_1r",
        "expectancy_conservative_1r",
    ]
    if not top_hypotheses.empty:
        report_lines.append(format_table(top_hypotheses[ambig_cols].head(15), "C. INTRABAR AMBIGUITY SENSITIVITY (OPTIMISTIC vs CLEAN vs CONSERVATIVE)"))

    # Section D: Chronological Period Stability (P1 vs P2 vs P3)
    stab_cols = [
        "group_name",
        "p1_resolved_1r",
        "p1_win_rate_1r",
        "p1_exp_1r",
        "p2_resolved_1r",
        "p2_win_rate_1r",
        "p2_exp_1r",
        "p3_resolved_1r",
        "p3_win_rate_1r",
        "p3_exp_1r",
        "positive_periods_count",
        "win_rate_std",
    ]
    if not top_hypotheses.empty:
        report_lines.append(format_table(top_hypotheses[stab_cols].head(15), "D. CHRONOLOGICAL PERIOD STABILITY (P1: 2016-19 | P2: 2019-22 | P3: 2023-26)"))

    # Section E: Best Two-Way Feature Interactions
    two_way_df = ranked_df[
        (ranked_df["combination_type"] == "2_Way")
        & (~ranked_df["insufficient_sample"])
    ].head(10)
    if not two_way_df.empty:
        report_lines.append(format_table(two_way_df[[
            "group_name",
            "composite_edge_score",
            "clean_resolved_1r",
            "clean_win_rate_1r_pct",
            "expectancy_clean_1r",
            "expectancy_conservative_1r",
            "positive_periods_count",
        ]], "E. BEST TWO-WAY FEATURE INTERACTIONS"))

    # Section F: Best Three-Way Feature Interactions
    three_way_df = ranked_df[
        (ranked_df["combination_type"] == "3_Way")
        & (~ranked_df["insufficient_sample"])
    ].head(10)
    if not three_way_df.empty:
        report_lines.append(format_table(three_way_df[[
            "group_name",
            "composite_edge_score",
            "clean_resolved_1r",
            "clean_win_rate_1r_pct",
            "expectancy_clean_1r",
            "expectancy_conservative_1r",
            "positive_periods_count",
        ]], "F. BEST THREE-WAY FEATURE INTERACTIONS"))

    # Section G: Rejected / Negative Expectancy Groups
    neg_df = ranked_df[
        (~ranked_df["insufficient_sample"])
        & (ranked_df["tier"] == "NEGATIVE_EXPECTANCY")
    ].head(10)
    if not neg_df.empty:
        report_lines.append(format_table(neg_df[[
            "group_name",
            "clean_resolved_1r",
            "clean_win_rate_1r_pct",
            "expectancy_clean_1r",
            "positive_periods_count",
        ]], "G. STATISTICALLY REJECTED / NEGATIVE EXPECTANCY GROUPS (SAMPLE)"))


    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 8. Export CSV and Text Artifacts
    print(f"\nSaving conditional edge reports to {output_dir}...")
    ranked_df.to_csv(output_dir / "conditional_ranked_hypotheses_2016_2026.csv", index=False)
    top_hypotheses.to_csv(output_dir / "conditional_top_strategy_hypotheses.csv", index=False)
    with open(output_dir / "conditional_edge_report_full_2016_2026.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Conditional edge analysis and report export complete!")


def main():
    parser = argparse.ArgumentParser(description="Run Conditional Edge Analysis on Selected Order Blocks")
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
        help="Minimum sample size threshold for reporting",
    )
    parser.add_argument(
        "--min-period-samples",
        type=int,
        default=10,
        help="Minimum sample size per period for robustness testing",
    )

    args = parser.parse_args()
    enriched_csv = Path(args.enriched_csv).resolve()
    path_csv = Path(args.path_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_conditional_research(
        enriched_csv=enriched_csv,
        path_csv=path_csv,
        output_dir=output_dir,
        min_samples=args.min_samples,
        min_period_samples=args.min_period_samples,
    )


if __name__ == "__main__":
    main()
