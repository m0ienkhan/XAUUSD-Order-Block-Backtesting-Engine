#!/usr/bin/env python3
"""
Execution-Level Strategy Validation Runner for STRAT_09 (2016-2026).

Executes bar-by-bar chronological simulation across:
- TRACK A (PRIMARY): Conservative Realistic Execution
- TRACK B (SECONDARY): Research Parity Reconciliation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.trading_day import assign_trading_day
from src.analytics.walk_forward_validation import (
    compute_training_feature_boundaries,
    apply_frozen_feature_boundaries,
    split_datasets_by_period,
)
from src.execution.contract import load_strat_09_contract
from src.execution.simulator import ExecutionSimulator
from src.execution.reconciliation import perform_reconciliation_audit
from src.analytics.reports import format_table


def run_execution_validation(
    market_csv: Path,
    enriched_csv: Path,
    path_csv: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  EXECUTION-LEVEL STRATEGY VALIDATION (STRAT_09: 2016-2026)")
    print("=" * 70)

    # 1. Load Market Data
    print(f"Loading 15-minute market data from {market_csv}...")
    market_df = pd.read_csv(market_csv)
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
    market_df = market_df.set_index("timestamp").sort_index()
    market_df = assign_trading_day(market_df)
    print(f"  - Market Bars Loaded : {len(market_df):,} candles ({market_df.index[0]} to {market_df.index[-1]})")

    # 2. Load Enriched OBs and Path Outcomes
    print(f"Loading enriched Order Blocks from {enriched_csv}...")
    enriched_df = pd.read_csv(enriched_csv)
    path_df = pd.read_csv(path_csv)

    train_df, _, _ = split_datasets_by_period(enriched_df)
    boundaries = compute_training_feature_boundaries(train_df)
    annotated_df = apply_frozen_feature_boundaries(enriched_df, boundaries)

    # 3. Load Frozen Contract for STRAT_09
    contract = load_strat_09_contract(output_dir / "walk_forward_strategy_summary.csv")
    print(f"\nLoaded Contract: {contract.strategy_id} ({contract.strategy_name})")
    print(f"  - Filters: {contract.filter_dict}")
    print(f"  - Target: {contract.target_r} R | Stop: {contract.stop_r} R")

    # 4. Run Dual-Track Simulation
    print("\nExecuting Chronological 15-Minute Bar Simulation...")
    sim = ExecutionSimulator(market_df, annotated_df, contract)
    sim_a, sim_b = sim.run_dual_track_simulation()

    # 5. Perform Parity Audit & Reconciliation
    print("Performing Reconciliation Parity Audit against Research Path Scanner...")
    audit_res, recon_report_text = perform_reconciliation_audit(
        sim_track_a=sim_a,
        sim_track_b=sim_b,
        path_outcomes_df=path_df,
        enriched_df=annotated_df,
    )

    # 6. Format Consolidated Performance Table
    sum_a = sim_a.summary
    sum_b = sim_b.summary

    comparison_df = pd.DataFrame([
        {
            "Metric": "Total Strategy Setups (OBs)",
            "Track A (Primary Conservative)": f"{sum_a.total_setups}",
            "Track B (Research Reconciliation)": f"{sum_b.total_setups}",
        },
        {
            "Metric": "Unfilled Orders (Day T+1 Expiration)",
            "Track A (Primary Conservative)": f"{sum_a.unfilled_orders} ({100 - sum_a.fill_rate_pct:.2f}%)",
            "Track B (Research Reconciliation)": f"{sum_b.unfilled_orders} ({100 - sum_b.fill_rate_pct:.2f}%)",
        },
        {
            "Metric": "Filled Trades",
            "Track A (Primary Conservative)": f"{sum_a.filled_trades} ({sum_a.fill_rate_pct:.2f}%)",
            "Track B (Research Reconciliation)": f"{sum_b.filled_trades} ({sum_b.fill_rate_pct:.2f}%)",
        },
        {
            "Metric": "Take Profit Trades (+1.0 R)",
            "Track A (Primary Conservative)": f"{sum_a.take_profit_trades}",
            "Track B (Research Reconciliation)": f"{sum_b.take_profit_trades}",
        },
        {
            "Metric": "Stop Loss Trades (-1.0 R)",
            "Track A (Primary Conservative)": f"{sum_a.stop_loss_trades}",
            "Track B (Research Reconciliation)": f"{sum_b.stop_loss_trades}",
        },
        {
            "Metric": "Ambiguous Same-Candle SL/TP",
            "Track A (Primary Conservative)": f"{sum_a.ambiguous_trades} (Enforced -1.0 R Loss)",
            "Track B (Research Reconciliation)": f"{sum_b.ambiguous_trades} (Excluded from Clean PnL)",
        },
        {
            "Metric": "Horizon Expiration Trades",
            "Track A (Primary Conservative)": f"{sum_a.horizon_exit_trades} (Closed @ Market Fractional R)",
            "Track B (Research Reconciliation)": f"{sum_b.horizon_exit_trades} (Excluded from Clean PnL)",
        },
        {
            "Metric": "Clean Win Rate (TP / (TP + SL))",
            "Track A (Primary Conservative)": f"{sum_a.clean_win_rate_pct:.2f}% (N = {sum_a.clean_resolved_trades})",
            "Track B (Research Reconciliation)": f"{sum_b.clean_win_rate_pct:.2f}% (N = {sum_b.clean_resolved_trades})",
        },
        {
            "Metric": "Win Rate on All Filled Trades (R > 0)",
            "Track A (Primary Conservative)": f"{sum_a.win_rate_filled_pct:.2f}% (N = {sum_a.filled_trades})",
            "Track B (Research Reconciliation)": f"N/A (Ambiguous Excluded)",
        },
        {
            "Metric": "Total Realized Return (Net R)",
            "Track A (Primary Conservative)": f"{sum_a.total_realized_r:+.2f} R",
            "Track B (Research Reconciliation)": f"{sum_b.total_realized_r:+.2f} R",
        },
        {
            "Metric": "Average Realized R per Filled Trade",
            "Track A (Primary Conservative)": f"{sum_a.avg_trade_r:+.4f} R",
            "Track B (Research Reconciliation)": f"{sum_b.avg_trade_r:+.4f} R",
        },
        {
            "Metric": "Candidate Expected Value (EV / Setup)",
            "Track A (Primary Conservative)": f"{sum_a.candidate_ev_r:+.4f} R / Setup",
            "Track B (Research Reconciliation)": f"{sum_b.candidate_ev_r:+.4f} R / Setup",
        },
        {
            "Metric": "Profit Factor",
            "Track A (Primary Conservative)": f"{sum_a.profit_factor:.2f}",
            "Track B (Research Reconciliation)": f"{sum_b.profit_factor:.2f}",
        },
        {
            "Metric": "Maximum Consecutive Wins / Losses",
            "Track A (Primary Conservative)": f"{sum_a.max_consecutive_wins} Wins / {sum_a.max_consecutive_losses} Losses",
            "Track B (Research Reconciliation)": f"{sum_b.max_consecutive_wins} Wins / {sum_b.max_consecutive_losses} Losses",
        },
        {
            "Metric": "Maximum Drawdown in R",
            "Track A (Primary Conservative)": f"-{sum_a.max_drawdown_r:.2f} R",
            "Track B (Research Reconciliation)": f"-{sum_b.max_drawdown_r:.2f} R",
        },
    ])

    report_lines: list[str] = [
        "######################################################################",
        "  EXECUTION-LEVEL BACKTEST VALIDATION REPORT: STRAT_09 (2016-2026)",
        "######################################################################",
        f"  Strategy ID                  : {contract.strategy_id}",
        f"  Strategy Definition          : {contract.strategy_name}",
        f"  Execution Horizon            : Strict Next Trading Day (Day T+1)",
        f"  Execution Friction           : Zero Spread / Zero Slippage (Gross Benchmark)",
        f"  Total Historical OB Setups   : {sum_a.total_setups} Setups across 2,776 Trading Days",
        "######################################################################\n",
        format_table(comparison_df, "A. DUAL-TRACK PERFORMANCE COMPARISON: PRIMARY vs RESEARCH RECONCILIATION"),
        "\n" + recon_report_text,
        format_table(sim_a.yearly_df, "B. PRIMARY TRACK A YEARLY PERFORMANCE BREAKDOWN (2016-2026)"),
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 7. Export all artifacts
    print(f"\nSaving execution backtest artifacts to {output_dir}...")
    sim_a.trades_df.to_csv(output_dir / "execution_trades_strat_09.csv", index=False)
    sim_a.equity_curve_df.to_csv(output_dir / "execution_equity_curve_strat_09.csv", index=False)
    sim_a.yearly_df.to_csv(output_dir / "execution_yearly_strat_09.csv", index=False)
    sim_a.monthly_df.to_csv(output_dir / "execution_monthly_strat_09.csv", index=False)
    audit_res.reconciliation_details_df.to_csv(output_dir / "execution_reconciliation_strat_09.csv", index=False)
    comparison_df.to_csv(output_dir / "execution_summary_strat_09.csv", index=False)

    with open(output_dir / "execution_report_strat_09.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Execution simulation and artifact export complete!")


def main():
    parser = argparse.ArgumentParser(description="Run Execution-Level Strategy Validation for STRAT_09")
    parser.add_argument(
        "--market-csv",
        type=str,
        default="data/raw/XAUUSD_15min_2016_2026.csv",
        help="Path to 15-minute market data CSV",
    )
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
        help="Directory to save execution reports and CSVs",
    )

    args = parser.parse_args()
    market_csv = Path(args.market_csv).resolve()
    enriched_csv = Path(args.enriched_csv).resolve()
    path_csv = Path(args.path_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_execution_validation(
        market_csv=market_csv,
        enriched_csv=enriched_csv,
        path_csv=path_csv,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
