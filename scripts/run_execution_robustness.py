#!/usr/bin/env python3
"""
Execution Realism & Robustness Stress Testing Runner for STRAT_09 (2016-2026).

Executes:
- Part 1 & 2: Cost Sensitivity Grid (60 scenarios)
- Part 3: Break-Even Friction Analysis (R & USD terms)
- Part 4 & 5: Parameter Robustness Matrix around Baseline (80 scenarios)
- Part 6: Combined Master Stress Scenarios (Idealized, Moderate, High, Severe)
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

from src.data.trading_day import assign_trading_day
from src.analytics.walk_forward_validation import (
    compute_training_feature_boundaries,
    apply_frozen_feature_boundaries,
    split_datasets_by_period,
)
from src.execution.contract import load_strat_09_contract, ExecutionContract
from src.execution.simulator import ExecutionSimulator
from src.execution.stress_testing import (
    run_cost_sensitivity_grid,
    run_break_even_analysis,
    run_parameter_robustness_grid,
    run_combined_stress_scenarios,
)
from src.analytics.reports import format_table


def run_robustness_stress_testing(
    market_csv: Path,
    enriched_csv: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  EXECUTION REALISM & ROBUSTNESS STRESS TESTING: STRAT_09")
    print("=" * 70)

    # 1. Load Data & Contract
    print(f"Loading 15m market data from {market_csv}...")
    market_df = pd.read_csv(market_csv)
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
    market_df = market_df.set_index("timestamp").sort_index()
    market_df = assign_trading_day(market_df)

    print(f"Loading enriched Order Blocks from {enriched_csv}...")
    enriched_df = pd.read_csv(enriched_csv)

    train_df, _, _ = split_datasets_by_period(enriched_df)
    boundaries = compute_training_feature_boundaries(train_df)
    annotated_df = apply_frozen_feature_boundaries(enriched_df, boundaries)

    contract = load_strat_09_contract(output_dir / "walk_forward_strategy_summary.csv")
    print(f"\nGoverning Contract: {contract.strategy_id} ({contract.strategy_name})")

    # 2. Run Baseline Simulation (Primary Track A)
    print("\nRunning Baseline Track A Execution Simulation...")
    sim = ExecutionSimulator(market_df, annotated_df, contract)
    baseline_res = sim.run_simulation(track="TRACK_A_CONSERVATIVE")
    base_sum = baseline_res.summary

    print(f"  - Setups: {base_sum.total_setups} | Filled: {base_sum.filled_trades} | Net Realized: {base_sum.total_realized_r:+.2f} R")
    print(f"  - Win Rate on Filled: {base_sum.win_rate_filled_pct:.2f}% | Candidate EV: {base_sum.candidate_ev_r:+.4f} R / Setup")

    # 3. Part 2: Cost Sensitivity Grid (60 cells)
    print("\nPart 2: Executing 60-Cell Cost Sensitivity Grid (5 Spreads x 4 Slippages x 3 Commissions)...")
    cost_grid_df = run_cost_sensitivity_grid(
        baseline_result=baseline_res,
        spread_scenarios=(0.00, 0.05, 0.10, 0.20, 0.30),
        slippage_scenarios=(0.00, 0.02, 0.05, 0.10),
        commission_scenarios=(0.00, 0.02, 0.05),
    )

    # 4. Part 3: Break-Even Cost Analysis
    print("\nPart 3: Computing Break-Even Friction Bounds...")
    be_result = run_break_even_analysis(baseline_res)
    print(f"  - Mean OB Range USD           : ${be_result.mean_ob_range_usd:.2f}")
    print(f"  - Break-Even Cost / Filled Trd : {be_result.break_even_cost_per_filled_trade_r:+.4f} R (${be_result.break_even_cost_per_filled_trade_usd:.2f} USD)")
    print(f"  - Break-Even Cost / Setup      : {be_result.break_even_cost_per_setup_r:+.4f} R (${be_result.break_even_cost_per_setup_usd:.2f} USD)")

    be_df = pd.DataFrame([{
        "Metric": "Break-Even Cost per Filled Trade",
        "Value (R)": f"{be_result.break_even_cost_per_filled_trade_r:+.4f} R",
        "Value (USD)": f"${be_result.break_even_cost_per_filled_trade_usd:.2f}",
    }, {
        "Metric": "Break-Even Cost per Candidate Setup",
        "Value (R)": f"{be_result.break_even_cost_per_setup_r:+.4f} R / Setup",
        "Value (USD)": f"${be_result.break_even_cost_per_setup_usd:.2f} / Setup",
    }, {
        "Metric": "Max Sustainable Round-Trip Friction",
        "Value (R)": f"{be_result.break_even_cost_per_filled_trade_r:+.4f} R",
        "Value (USD)": f"${be_result.max_sustainable_roundtrip_friction_usd:.2f}",
    }, {
        "Metric": "Mean Order Block Range",
        "Value (R)": "1.0000 R",
        "Value (USD)": f"${be_result.mean_ob_range_usd:.2f}",
    }])

    # 5. Part 4 & 5: Parameter Sensitivity Grid (80 combinations)
    print("\nPart 4 & 5: Executing Parameter Robustness Grid (4 Entries x 4 SL x 5 TP = 80 variations)...")
    param_grid_df = run_parameter_robustness_grid(
        market_df=market_df,
        annotated_obs_df=annotated_df,
        contract=contract,
        entry_offsets=(0.00, 0.10, 0.20, 0.30),
        sl_multipliers=(0.75, 1.00, 1.25, 1.50),
        tp_multipliers=(0.75, 1.00, 1.25, 1.50, 2.00),
    )

    # 6. Part 6: Combined Master Stress Scenarios
    print("\nPart 6: Evaluating 4 Predefined Combined Master Stress Scenarios...")
    master_scenarios_df = run_combined_stress_scenarios(baseline_res)

    # 7. Executive Robustness Summary Table
    summary_records = [{
        "Dimension": "Baseline (Idealized Gross)",
        "Configuration": "Zero Costs | Entry 0.00 | SL 1.0R | TP 1.0R",
        "Filled Trades": base_sum.filled_trades,
        "Win Rate %": f"{base_sum.win_rate_filled_pct:.2f}%",
        "Net Realized R": f"{base_sum.total_realized_r:+.2f} R",
        "Expectancy / Trd": f"{base_sum.avg_trade_r:+.4f} R",
        "Candidate EV": f"{base_sum.candidate_ev_r:+.4f} R",
        "Profit Factor": f"{base_sum.profit_factor:.2f}",
        "Max Drawdown": f"-{base_sum.max_drawdown_r:.2f} R",
    }]

    for _, sc in master_scenarios_df.iterrows():
        if sc["scenario_id"] == "SCENARIO_1_IDEALIZED":
            continue
        summary_records.append({
            "Dimension": sc["scenario_name"],
            "Configuration": f"Spread ${sc['spread_usd']:.2f}, Slip ${sc['slippage_usd']:.2f}, Comm ${sc['commission_usd']:.2f}",
            "Filled Trades": sc["filled_trades"],
            "Win Rate %": f"{sc['win_rate_filled_pct']:.2f}%",
            "Net Realized R": f"{sc['net_realized_r']:+.2f} R",
            "Expectancy / Trd": f"{sc['net_expectancy_filled_r']:+.4f} R",
            "Candidate EV": f"{sc['candidate_ev_r']:+.4f} R",
            "Profit Factor": f"{sc['profit_factor']:.2f}",
            "Max Drawdown": f"-{sc['max_drawdown_r']:.2f} R",
        })

    summary_df = pd.DataFrame(summary_records)

    # 8. Generate Comprehensive Text Report
    report_lines: list[str] = [
        "######################################################################",
        "  EXECUTION REALISM & ROBUSTNESS STRESS TESTING REPORT: STRAT_09",
        "######################################################################",
        f"  Target Strategy              : {contract.strategy_id} ({contract.strategy_name})",
        f"  Total Historical OB Setups   : {base_sum.total_setups} Setups (2016-2026)",
        f"  Baseline Filled Trades       : {base_sum.filled_trades} Trades ({base_sum.fill_rate_pct:.2f}% Fill Rate)",
        f"  Baseline Gross Net Realized  : {base_sum.total_realized_r:+.2f} R (Win Rate: {base_sum.win_rate_filled_pct:.2f}%)",
        "######################################################################\n",
        format_table(summary_df, "A. EXECUTIVE STRESS-TESTING SUMMARY MATRIX"),
        "\n" + format_table(be_df, "B. BREAK-EVEN EXECUTION FRICTION ANALYSIS"),
        "\n" + format_table(master_scenarios_df[[
            "scenario_id",
            "scenario_name",
            "total_friction_usd",
            "net_realized_r",
            "net_expectancy_filled_r",
            "candidate_ev_r",
            "profit_factor",
            "max_drawdown_r",
            "break_even_margin_remaining_usd",
            "break_even_margin_remaining_pct",
        ]], "C. COMBINED MASTER EXECUTION STRESS SCENARIOS"),
        "\n" + format_table(cost_grid_df.head(15)[[
            "scenario_id",
            "scenario_name",
            "total_friction_usd",
            "total_cost_r",
            "net_realized_r",
            "net_expectancy_filled_r",
            "candidate_ev_r",
            "profit_factor",
            "max_drawdown_r",
        ]], "D. COST SENSITIVITY GRID SAMPLE (Top 15 of 60 Scenarios)"),
        "\n" + format_table(param_grid_df.head(16)[[
            "scenario_id",
            "is_baseline",
            "entry_offset_r",
            "stop_r",
            "target_r",
            "filled_trades",
            "win_rate_filled_pct",
            "net_realized_r",
            "expectancy_filled_r",
            "profit_factor",
            "max_drawdown_r",
        ]], "E. PARAMETER ROBUSTNESS MATRIX SAMPLE (Selected Scenarios around Baseline)"),
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 9. Export Artifacts
    print(f"\nSaving robustness and stress testing artifacts to {output_dir}...")
    cost_grid_df.to_csv(output_dir / "execution_cost_sensitivity_strat_09.csv", index=False)
    be_df.to_csv(output_dir / "execution_break_even_analysis_strat_09.csv", index=False)
    param_grid_df.to_csv(output_dir / "execution_parameter_robustness_strat_09.csv", index=False)
    master_scenarios_df.to_csv(output_dir / "execution_combined_stress_scenarios_strat_09.csv", index=False)
    summary_df.to_csv(output_dir / "execution_robustness_summary_strat_09.csv", index=False)

    with open(output_dir / "execution_robustness_report_strat_09.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Robustness stress testing and artifact export complete!")


def main():
    parser = argparse.ArgumentParser(description="Run Execution Realism & Robustness Stress Testing for STRAT_09")
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
        "--output-dir",
        type=str,
        default="outputs/reports",
        help="Directory to save robustness reports and CSVs",
    )

    args = parser.parse_args()
    market_csv = Path(args.market_csv).resolve()
    enriched_csv = Path(args.enriched_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_robustness_stress_testing(
        market_csv=market_csv,
        enriched_csv=enriched_csv,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
