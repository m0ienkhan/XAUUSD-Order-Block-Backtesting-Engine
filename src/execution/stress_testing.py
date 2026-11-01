from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

from src.execution.cost_model import ExecutionCostModel, TradeCostBreakdown, compute_trade_cost_breakdown


@dataclass(frozen=True)
class CostSensitivityScenario:
    """
    Specification for a deterministic transaction cost sensitivity scenario.

    Parameters
    ----------
    scenario_id : str
        Deterministic unique identifier for this cost configuration.
    spread_usd : float
        Total round-trip bid/ask spread cost in USD price movement terms.
    entry_slippage_usd : float
        Adverse price movement on entry fill in USD.
    exit_slippage_usd : float
        Adverse price movement on exit fill in USD.
    commission_usd : float
        Canonical total round-trip commission expressed in USD price terms per ounce.
    commission_r_fixed : float, default=0.0
        Fixed round-trip commission expressed directly in R units.
    """
    scenario_id: str
    spread_usd: float
    entry_slippage_usd: float
    exit_slippage_usd: float
    commission_usd: float
    commission_r_fixed: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenario_id or not isinstance(self.scenario_id, str):
            raise ValueError(f"scenario_id must be a non-empty string, got: {self.scenario_id}")

        for name, val in [
            ("spread_usd", self.spread_usd),
            ("entry_slippage_usd", self.entry_slippage_usd),
            ("exit_slippage_usd", self.exit_slippage_usd),
            ("commission_usd", self.commission_usd),
            ("commission_r_fixed", self.commission_r_fixed),
        ]:
            if math.isnan(val) or math.isinf(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got: {val}")

    @property
    def total_slippage_usd(self) -> float:
        """Total round-trip slippage (entry + exit)."""
        return self.entry_slippage_usd + self.exit_slippage_usd

    @property
    def total_friction_usd(self) -> float:
        """Total round-trip price and commission friction in USD terms."""
        return self.spread_usd + self.total_slippage_usd + self.commission_usd

    def to_cost_model(self) -> ExecutionCostModel:
        """Convert to ExecutionCostModel instance."""
        return ExecutionCostModel(
            spread_usd=self.spread_usd,
            entry_slippage_usd=self.entry_slippage_usd,
            exit_slippage_usd=self.exit_slippage_usd,
            commission_usd=self.commission_usd,
            commission_r_fixed=self.commission_r_fixed,
        )


def generate_cost_sensitivity_grid(
    spread_levels: Sequence[float] = (0.00, 0.05, 0.10, 0.20, 0.30),
    slippage_levels: Sequence[float] = (0.00, 0.02, 0.05, 0.10),
    commission_levels: Sequence[float] = (0.00, 0.02, 0.05),
    commission_r_fixed: float = 0.0,
) -> list[CostSensitivityScenario]:
    """
    Generate the Cartesian grid of deterministic cost sensitivity scenarios.

    Slippage is applied symmetrically:
        entry_slippage_usd = slippage_level
        exit_slippage_usd = slippage_level

    Default Grid Sizing:
        5 spread levels x 4 slippage levels x 3 commission levels = 60 scenarios.

    Scenario ID Convention:
        COST_S{spread*100:03.0f}_SL{slip*100:03.0f}_C{comm*100:03.0f}
        e.g. COST_S000_SL000_C000, COST_S005_SL002_C002, COST_S030_SL010_C005
    """
    scenarios: list[CostSensitivityScenario] = []

    for spread in spread_levels:
        for slip in slippage_levels:
            for comm in commission_levels:
                s_code = int(round(spread * 100))
                sl_code = int(round(slip * 100))
                c_code = int(round(comm * 100))
                s_id = f"COST_S{s_code:03d}_SL{sl_code:03d}_C{c_code:03d}"

                scenarios.append(
                    CostSensitivityScenario(
                        scenario_id=s_id,
                        spread_usd=float(spread),
                        entry_slippage_usd=float(slip),
                        exit_slippage_usd=float(slip),
                        commission_usd=float(comm),
                        commission_r_fixed=float(commission_r_fixed),
                    )
                )

    return scenarios


def evaluate_scenario_on_trades(
    scenario: CostSensitivityScenario,
    trades_df: pd.DataFrame,
    total_setups: int,
) -> dict[str, Any]:
    """
    Apply a deterministic cost sensitivity scenario to completed baseline execution trades.

    Parameters
    ----------
    scenario : CostSensitivityScenario
        The specific cost configuration to evaluate.
    trades_df : pd.DataFrame
        Completed baseline trade records. Must contain 'realized_r' and 'ob_range'.
        Does NOT mutate the input DataFrame.
    total_setups : int
        Total candidate Order Block setups (denominator for Candidate EV).

    Returns
    -------
    dict[str, Any]
        Aggregated scenario performance metrics.
    """
    if total_setups <= 0:
        raise ValueError(f"total_setups must be strictly positive, got: {total_setups}")

    filled_trades = len(trades_df)
    if filled_trades == 0:
        return {
            "scenario_id": scenario.scenario_id,
            "spread_usd": scenario.spread_usd,
            "entry_slippage_usd": scenario.entry_slippage_usd,
            "exit_slippage_usd": scenario.exit_slippage_usd,
            "commission_usd": scenario.commission_usd,
            "commission_r_fixed": scenario.commission_r_fixed,
            "total_setups": total_setups,
            "filled_trades": 0,
            "gross_net_r": 0.0,
            "gross_expectancy_per_filled_trade": 0.0,
            "total_spread_cost_r": 0.0,
            "total_slippage_cost_r": 0.0,
            "total_commission_cost_r": 0.0,
            "total_execution_cost_r": 0.0,
            "average_cost_r_per_filled_trade": 0.0,
            "net_r": 0.0,
            "net_expectancy_per_filled_trade": 0.0,
            "candidate_ev_net_r_per_setup": 0.0,
            "net_win_rate": 0.0,
            "net_profit_factor": 0.0,
            "maximum_drawdown_r": 0.0,
            "maximum_consecutive_losses": 0,
            "baseline_relative_net_r": 0.0,
        }

    cost_model = scenario.to_cost_model()

    # Chronological trade ordering preservation
    if "entry_timestamp" in trades_df.columns:
        ordered_df = trades_df.sort_values(by="entry_timestamp").reset_index(drop=True)
    else:
        ordered_df = trades_df.copy().reset_index(drop=True)

    gross_r_series = ordered_df["realized_r"].astype(float)
    ob_range_series = ordered_df["ob_range"].astype(float)

    # Validate ob_ranges
    if (ob_range_series <= 0.0).any():
        raise ValueError("All ob_range values in trades_df must be strictly positive.")

    # Vectorized / deterministic per-trade cost conversion
    spread_cost_r_series = scenario.spread_usd / ob_range_series
    slippage_cost_r_series = scenario.total_slippage_usd / ob_range_series
    commission_cost_r_series = (scenario.commission_usd / ob_range_series) + scenario.commission_r_fixed
    total_cost_r_series = spread_cost_r_series + slippage_cost_r_series + commission_cost_r_series
    net_r_series = gross_r_series - total_cost_r_series

    # Population & Gross Metrics
    gross_net_r = float(round(gross_r_series.sum(), 4))
    gross_expectancy = float(round(gross_net_r / filled_trades, 4))

    # Aggregate Cost Metrics
    tot_spread_cost = float(round(spread_cost_r_series.sum(), 4))
    tot_slippage_cost = float(round(slippage_cost_r_series.sum(), 4))
    tot_comm_cost = float(round(commission_cost_r_series.sum(), 4))
    tot_exec_cost = float(round(tot_spread_cost + tot_slippage_cost + tot_comm_cost, 4))
    avg_cost_per_trade = float(round(tot_exec_cost / filled_trades, 4))

    # Net Return & Expectancy Metrics
    net_r = float(round(gross_net_r - tot_exec_cost, 4))
    net_expectancy = float(round(net_r / filled_trades, 4))
    candidate_ev = float(round(net_r / total_setups, 4))

    # Net Risk & Win/Loss Metrics (recalculated strictly from net_realized_r)
    net_wins_mask = net_r_series > 0.0
    net_losses_mask = net_r_series < 0.0
    net_win_count = int(net_wins_mask.sum())
    net_win_rate = float(round(net_win_count / filled_trades, 4))

    net_gross_gains = float(net_r_series[net_wins_mask].sum())
    net_gross_losses = float(abs(net_r_series[net_losses_mask].sum()))
    if net_gross_losses > 0.0:
        net_profit_factor = float(round(net_gross_gains / net_gross_losses, 2))
    elif net_gross_gains > 0.0:
        net_profit_factor = float("inf")
    else:
        net_profit_factor = 0.0

    # Sequential Equity Curve, Max Drawdown & Consecutive Losses from net_r
    cum_r = 0.0
    peak_r = 0.0
    max_dd = 0.0
    cur_loss_streak = 0
    max_loss_streak = 0

    for r_val in net_r_series:
        if r_val < 0.0:
            cur_loss_streak += 1
            if cur_loss_streak > max_loss_streak:
                max_loss_streak = cur_loss_streak
        else:
            cur_loss_streak = 0

        cum_r += r_val
        if cum_r > peak_r:
            peak_r = cum_r
        dd = peak_r - cum_r
        if dd > max_dd:
            max_dd = dd

    max_dd_r = float(round(max_dd, 4))
    baseline_relative_net_r = float(round(net_r - gross_net_r, 4))

    return {
        "scenario_id": scenario.scenario_id,
        "spread_usd": scenario.spread_usd,
        "entry_slippage_usd": scenario.entry_slippage_usd,
        "exit_slippage_usd": scenario.exit_slippage_usd,
        "commission_usd": scenario.commission_usd,
        "commission_r_fixed": scenario.commission_r_fixed,
        "total_setups": total_setups,
        "filled_trades": filled_trades,
        "gross_net_r": gross_net_r,
        "gross_expectancy_per_filled_trade": gross_expectancy,
        "total_spread_cost_r": tot_spread_cost,
        "total_slippage_cost_r": tot_slippage_cost,
        "total_commission_cost_r": tot_comm_cost,
        "total_execution_cost_r": tot_exec_cost,
        "average_cost_r_per_filled_trade": avg_cost_per_trade,
        "net_r": net_r,
        "net_expectancy_per_filled_trade": net_expectancy,
        "candidate_ev_net_r_per_setup": candidate_ev,
        "net_win_rate": net_win_rate,
        "net_profit_factor": net_profit_factor,
        "maximum_drawdown_r": max_dd_r,
        "maximum_consecutive_losses": max_loss_streak,
        "baseline_relative_net_r": baseline_relative_net_r,
    }


def run_cost_sensitivity_grid(
    trades_df: pd.DataFrame,
    total_setups: int,
    scenarios: Optional[Sequence[CostSensitivityScenario]] = None,
) -> pd.DataFrame:
    """
    Execute the cost sensitivity grid across completed baseline execution trades.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Baseline execution trades DataFrame.
    total_setups : int
        Total candidate setups count.
    scenarios : Sequence[CostSensitivityScenario], optional
        List of scenarios to evaluate. Defaults to the 60-cell Cartesian grid.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row represents one cost sensitivity scenario.
    """
    if scenarios is None:
        scenarios = generate_cost_sensitivity_grid()

    records: list[dict[str, Any]] = []
    for sc in scenarios:
        row = evaluate_scenario_on_trades(sc, trades_df, total_setups)
        records.append(row)

    return pd.DataFrame(records)


def load_baseline_execution_population(
    trades_csv: Optional[Path] = None,
    summary_csv: Optional[Path] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Load verified baseline STRAT_09 trade population and setup count from existing artifacts.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_trades_csv = project_root / "outputs" / "reports" / "execution_trades_strat_09.csv"
    default_summary_csv = project_root / "outputs" / "reports" / "execution_summary_strat_09.csv"

    t_path = trades_csv if trades_csv is not None else default_trades_csv
    s_path = summary_csv if summary_csv is not None else default_summary_csv

    if not t_path.exists():
        raise FileNotFoundError(f"Baseline trades artifact not found at: {t_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Baseline summary artifact not found at: {s_path}")

    trades_df = pd.read_csv(t_path)
    summary_df = pd.read_csv(s_path)

    # Extract total setups count from summary CSV
    setup_row = summary_df[summary_df["Metric"].str.contains("Total Strategy Setups", case=False, na=False)]
    if not setup_row.empty:
        total_setups = int(setup_row.iloc[0]["Track A (Primary Conservative)"])
    else:
        total_setups = 629  # Canonical verified fallback

    return trades_df, total_setups


@dataclass(frozen=True)
class BreakEvenAnalysisResult:
    """
    Detailed empirical break-even execution friction metrics for a strategy.
    """
    gross_net_r: float
    total_setups: int
    filled_trades: int

    gross_expectancy_per_filled_trade: float
    break_even_average_cost_r_per_filled_trade: float

    gross_candidate_ev_r_per_setup: float
    break_even_average_cost_r_per_setup: float

    exact_break_even_constant_usd_per_oz: float

    reciprocal_ob_range_sum: float
    arithmetic_mean_ob_range: float
    harmonic_mean_ob_range: float

    approximate_break_even_usd_from_mean_range: Optional[float] = None


def compute_break_even_analysis(
    trades_df: pd.DataFrame,
    total_setups: int,
) -> BreakEvenAnalysisResult:
    """
    Compute exact empirical break-even execution friction in R and USD terms.

    Formula:
        C*_usd = GrossNetR / sum(1 / ob_range_i)
    """
    if total_setups <= 0:
        raise ValueError(f"total_setups must be strictly positive, got: {total_setups}")

    filled_trades = len(trades_df)
    if filled_trades == 0:
        return BreakEvenAnalysisResult(
            gross_net_r=0.0,
            total_setups=total_setups,
            filled_trades=0,
            gross_expectancy_per_filled_trade=0.0,
            break_even_average_cost_r_per_filled_trade=0.0,
            gross_candidate_ev_r_per_setup=0.0,
            break_even_average_cost_r_per_setup=0.0,
            exact_break_even_constant_usd_per_oz=0.0,
            reciprocal_ob_range_sum=0.0,
            arithmetic_mean_ob_range=0.0,
            harmonic_mean_ob_range=0.0,
            approximate_break_even_usd_from_mean_range=0.0,
        )

    gross_r_series = trades_df["realized_r"].astype(float)
    ob_range_series = trades_df["ob_range"].astype(float)

    if (ob_range_series <= 0.0).any():
        raise ValueError("All ob_range values in trades_df must be strictly positive.")

    gross_net_r = float(round(gross_r_series.sum(), 4))
    gross_expectancy = float(round(gross_net_r / filled_trades, 4))
    be_cost_per_trade_r = gross_expectancy
    be_cost_per_setup_r = float(round(gross_net_r / total_setups, 4))

    # Exact empirical reciprocal distribution calculation
    reciprocal_sum = float((1.0 / ob_range_series).sum())
    exact_be_usd = float(round(gross_net_r / reciprocal_sum, 4)) if reciprocal_sum > 0 else 0.0

    arithmetic_mean_range = float(round(ob_range_series.mean(), 4))
    harmonic_mean_range = float(round(filled_trades / reciprocal_sum, 4)) if reciprocal_sum > 0 else 0.0
    approx_be_usd = float(round(gross_expectancy * arithmetic_mean_range, 4))

    return BreakEvenAnalysisResult(
        gross_net_r=gross_net_r,
        total_setups=total_setups,
        filled_trades=filled_trades,
        gross_expectancy_per_filled_trade=gross_expectancy,
        break_even_average_cost_r_per_filled_trade=be_cost_per_trade_r,
        gross_candidate_ev_r_per_setup=be_cost_per_setup_r,
        break_even_average_cost_r_per_setup=be_cost_per_setup_r,
        exact_break_even_constant_usd_per_oz=exact_be_usd,
        reciprocal_ob_range_sum=float(round(reciprocal_sum, 4)),
        arithmetic_mean_ob_range=arithmetic_mean_range,
        harmonic_mean_ob_range=harmonic_mean_range,
        approximate_break_even_usd_from_mean_range=approx_be_usd,
    )


def run_break_even_analysis(
    trades_df: pd.DataFrame,
    total_setups: int,
) -> BreakEvenAnalysisResult:
    """Canonical function to execute Break-Even Cost Analysis."""
    return compute_break_even_analysis(trades_df, total_setups)


def generate_break_even_summary_df(result: BreakEvenAnalysisResult) -> pd.DataFrame:
    """
    Generate one-row DataFrame capturing exact empirical break-even statistics.
    """
    return pd.DataFrame([{
        "gross_net_r": result.gross_net_r,
        "total_setups": result.total_setups,
        "filled_trades": result.filled_trades,
        "gross_expectancy_per_filled_trade": result.gross_expectancy_per_filled_trade,
        "break_even_average_cost_r_per_filled_trade": result.break_even_average_cost_r_per_filled_trade,
        "gross_candidate_ev_r_per_setup": result.gross_candidate_ev_r_per_setup,
        "break_even_average_cost_r_per_setup": result.break_even_average_cost_r_per_setup,
        "exact_break_even_constant_usd_per_oz": result.exact_break_even_constant_usd_per_oz,
        "reciprocal_ob_range_sum": result.reciprocal_ob_range_sum,
        "arithmetic_mean_ob_range": result.arithmetic_mean_ob_range,
        "harmonic_mean_ob_range": result.harmonic_mean_ob_range,
        "approximate_break_even_usd_from_mean_range": result.approximate_break_even_usd_from_mean_range,
    }])


def compute_scenario_break_even_margins(
    cost_grid_df: pd.DataFrame,
    gross_net_r: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute remaining edge, cost utilization, and break-even headroom multiplier
    for each cost sensitivity scenario.
    """
    if cost_grid_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    base_gross_r = gross_net_r if gross_net_r is not None else float(cost_grid_df["gross_net_r"].iloc[0])

    for _, row in cost_grid_df.iterrows():
        s_id = str(row["scenario_id"])
        g_r = float(row.get("gross_net_r", base_gross_r))
        tot_cost = float(row["total_execution_cost_r"])
        net_r = float(row["net_r"])

        # Break-Even Analysis D: Remaining Edge and Utilization
        remaining_edge_r = net_r  # Zero is the boundary; can be negative
        remaining_edge_fraction = float(round(net_r / g_r, 4)) if g_r != 0 else 0.0
        cost_utilization_fraction = float(round(tot_cost / g_r, 4)) if g_r != 0 else 0.0

        # Break-Even Analysis E: Cost Multipliers to Break-Even
        if tot_cost == 0.0:
            m_break_even = float("inf")
            headroom_m = float("inf")
        else:
            m_break_even = float(round(g_r / tot_cost, 4))
            headroom_m = float(round(m_break_even - 1.0, 4))

        records.append({
            "scenario_id": s_id,
            "gross_net_r": g_r,
            "total_execution_cost_r": tot_cost,
            "net_r": net_r,
            "remaining_edge_r": remaining_edge_r,
            "remaining_edge_fraction": remaining_edge_fraction,
            "cost_utilization_fraction": cost_utilization_fraction,
            "break_even_cost_multiplier": m_break_even,
            "cost_headroom_multiplier": headroom_m,
        })

    return pd.DataFrame(records)


def run_combined_stress_scenarios(*args: Any, **kwargs: Any) -> Any:
    """Stub placeholder for Step 6 Combined Stress Scenarios."""
    raise NotImplementedError("Pending Step 6: Combined Stress Scenarios")
