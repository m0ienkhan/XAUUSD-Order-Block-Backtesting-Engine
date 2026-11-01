from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_market_data
from src.data.trading_day import assign_trading_day
from src.structure.inside_bar import annotate_inside_bar_structure
from src.structure.fvg import detect_fvg
from src.pipeline.daily_ob_pipeline import build_daily_order_block_datasets
from src.analytics.ob_enrichment import enrich_selected_order_blocks
from src.analytics.path_analysis import batch_scan_path_dependent_outcomes
from src.analytics.statistical_analysis import (
    CREATION_TIME_FEATURES,
    analyze_by_direction,
    analyze_by_ob_size,
    analyze_by_time_and_session,
    analyze_by_fvg_characteristics,
    analyze_by_distance_features,
    analyze_unconditional_targets,
    summarize_path_dependent_results,
    analyze_mfe_mae_relationship,
)
from src.analytics.reports import generate_full_research_report


def perform_data_integrity_checks(
    df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
) -> dict:
    """Audit and verify dataset integrity metrics."""
    total_trading_days = df["trading_day"].nunique()
    total_candidates = len(candidates_df)
    total_selected = len(selected_df)

    bullish_count = int((selected_df["ob_direction"] == "bullish").sum()) if not selected_df.empty else 0
    bearish_count = int((selected_df["ob_direction"] == "bearish").sum()) if not selected_df.empty else 0

    has_fvg_count = int(selected_df["has_associated_fvg"].sum()) if not selected_df.empty else 0
    fvg_pct = (has_fvg_count / total_selected * 100.0) if total_selected > 0 else 0.0

    # Tap rates by horizon
    tap_rates = {}
    for h_tag in ["next_trading_day", "3_trading_days", "5_trading_days", "full"]:
        col = "was_future_tapped" if h_tag == "next_trading_day" else f"was_tapped_{'h3d' if h_tag == '3_trading_days' else 'h5d' if h_tag == '5_trading_days' else 'full'}"
        if col in enriched_df.columns and total_selected > 0:
            tapped_n = int(enriched_df[col].sum())
            tap_rates[h_tag] = (tapped_n, f"{(tapped_n / total_selected * 100.0):.2f}%")

    # Missing value checks
    creation_nulls = {k: int(enriched_df[k].isna().sum()) for k in CREATION_TIME_FEATURES if k in enriched_df.columns}

    audit_summary = {
        "total_candles": len(df),
        "total_trading_days": total_trading_days,
        "total_candidates": total_candidates,
        "total_selected_obs": total_selected,
        "bullish_obs": bullish_count,
        "bearish_obs": bearish_count,
        "selected_with_fvg_pct": f"{fvg_pct:.2f}%",
        "tap_rates_by_horizon": tap_rates,
        "creation_features_null_counts": {k: v for k, v in creation_nulls.items() if v > 0},
    }

    print("\n" + "="*70)
    print("  DATA INTEGRITY & PIPELINE AUDIT")
    print("="*70)
    print(f"  Total Market Candles        : {len(df):,}")
    print(f"  Total Distinct Trading Days : {total_trading_days:,}")
    print(f"  Total OB Candidates Evaluated: {total_candidates:,}")
    print(f"  Total Selected Order Blocks : {total_selected:,}")
    print(f"    - Bullish Selected OBs    : {bullish_count:,} ({(bullish_count/max(1,total_selected)*100):.1f}%)")
    print(f"    - Bearish Selected OBs    : {bearish_count:,} ({(bearish_count/max(1,total_selected)*100):.1f}%)")
    print(f"  Selected OBs with Valid FVG : {has_fvg_count:,} ({fvg_pct:.1f}%)")
    print("\n  Future Tap Rates by Horizon:")
    for h_name, (cnt, pct) in tap_rates.items():
        print(f"    - {h_name:18s}: {cnt:,} / {total_selected:,} ({pct})")
    
    null_issues = {k: v for k, v in creation_nulls.items() if v > 0}
    if null_issues:
        print(f"\n  [WARN] Missing values in creation features: {null_issues}")
    else:
        print("\n  [PASS] Zero missing values in mandatory creation-time features.")
    print("="*70 + "\n")

    return audit_summary


def run_full_statistical_research(
    csv_path: str = "data/raw/XAUUSD_15min_2016_2026.csv",
    start_day: str | None = None,
    end_day: str | None = None,
    horizon: str = "next_trading_day",
    output_dir: str = "outputs/reports",
    min_samples: int = 10,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Loading market data from {csv_path}...")
    df, _ = load_market_data(csv_path, timeframe_minutes=15)
    df = assign_trading_day(df)
    df = annotate_inside_bar_structure(df)
    df = detect_fvg(df)

    actual_start = start_day if start_day else str(df["trading_day"].min())
    actual_end = end_day if end_day else str(df["trading_day"].max())

    print(f"[2/5] Running frozen Order Block pipeline ({actual_start} -> {actual_end})...")
    candidates_df, selected_df = build_daily_order_block_datasets(
        df=df,
        start_day=actual_start,
        end_day=actual_end,
    )
    print(f"      Total candidates: {len(candidates_df):,}, Selected OBs: {len(selected_df):,}")

    print(f"[3/5] Enriching selected Order Blocks with multi-horizon behavior...")
    enriched_df = enrich_selected_order_blocks(
        df=df,
        selected_df=selected_df,
        horizons=["next_trading_day", "3_trading_days", "5_trading_days", "full"],
    )

    # Perform Data Integrity Check
    integrity_audit = perform_data_integrity_checks(
        df=df,
        candidates_df=candidates_df,
        selected_df=selected_df,
        enriched_df=enriched_df,
    )

    print(f"[4/5] Executing candle-by-candle path-dependent scans ({horizon})...")
    path_df = batch_scan_path_dependent_outcomes(
        df=df,
        enriched_df=enriched_df,
        horizon=horizon,
        target_multiples=[1.0, 2.0, 3.0, 5.0],
        stop_r=1.0,
    )

    print(f"[5/5] Generating comprehensive statistical research report (min_samples={min_samples})...")
    direction_df = analyze_by_direction(enriched_df, min_samples=min_samples)
    size_df = analyze_by_ob_size(enriched_df, n_buckets=4, min_samples=min_samples)
    session_df, dow_df, hour_df = analyze_by_time_and_session(enriched_df, min_samples=min_samples)
    delay_df, fvg_ratio_df = analyze_by_fvg_characteristics(enriched_df, min_samples=min_samples)
    dist_df = analyze_by_distance_features(enriched_df, min_samples=min_samples)
    uncond_targets_df = analyze_unconditional_targets(enriched_df, target_multiples=[1.0, 2.0, 3.0, 5.0], min_samples=min_samples)
    path_targets_df = summarize_path_dependent_results(path_df, target_multiples=[1.0, 2.0, 3.0, 5.0], min_samples=min_samples)
    mfe_mae_summary = analyze_mfe_mae_relationship(enriched_df, min_samples=min_samples)

    report = generate_full_research_report(
        direction_df=direction_df,
        size_df=size_df,
        session_df=session_df,
        dow_df=dow_df,
        fvg_df=fvg_ratio_df,
        dist_df=dist_df,
        uncond_targets_df=uncond_targets_df,
        path_targets_df=path_targets_df,
        mfe_mae_summary=mfe_mae_summary,
    )

    print(report)

    # Save artifacts
    print(f"\nSaving research datasets and summary reports to {out_path.resolve()}...")
    selected_df.to_csv(out_path / "full_selected_order_blocks_2016_2026.csv", index=False)
    enriched_df.to_csv(out_path / "full_enriched_order_blocks_2016_2026.csv", index=False)
    path_df.to_csv(out_path / "full_path_outcomes_2016_2026.csv", index=False)
    
    direction_df.to_csv(out_path / "statistical_direction_analysis.csv", index=False)
    size_df.to_csv(out_path / "statistical_size_analysis.csv", index=False)
    session_df.to_csv(out_path / "statistical_session_analysis.csv", index=False)
    dow_df.to_csv(out_path / "statistical_dow_analysis.csv", index=False)
    fvg_ratio_df.to_csv(out_path / "statistical_fvg_analysis.csv", index=False)
    dist_df.to_csv(out_path / "statistical_distance_analysis.csv", index=False)
    uncond_targets_df.to_csv(out_path / "statistical_unconditional_targets.csv", index=False)
    path_targets_df.to_csv(out_path / "statistical_path_targets.csv", index=False)
    
    with open(out_path / "research_report_full_2016_2026.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Research analysis and artifact export complete!\n")
    return enriched_df, path_df, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete XAUUSD Order Block Statistical Research Pipeline")
    parser.add_argument("--csv", type=str, default="data/raw/XAUUSD_15min_2016_2026.csv", help="Path to raw market CSV")
    parser.add_argument("--start", type=str, default=None, help="Start trading day (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End trading day (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=str, default="next_trading_day", help="Target analysis horizon")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Directory to save output CSVs")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum sample size threshold")

    args = parser.parse_args()
    run_full_statistical_research(
        csv_path=args.csv,
        start_day=args.start,
        end_day=args.end,
        horizon=args.horizon,
        output_dir=args.output_dir,
        min_samples=args.min_samples,
    )

