from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

from src.execution.cost_model import ExecutionCostModel, TradeCostBreakdown
from src.execution.robustness import (
    ExecutionParameterScenario,
    ParameterizedExecutionSimulator,
    load_robustness_input_datasets,
)


@dataclass(frozen=True)
class CombinedStressScenario:
    """
    Immutable specification of a combined execution stress scenario coupling
    transaction friction with execution parameter variations.

    Parameters
    ----------
    scenario_id : str
        Deterministic unique identifier (e.g. 'SCENARIO_1_IDEALIZED').
    scenario_name : str
        Short analytical label.
    scenario_description : str
        Detailed description of the stress assumptions.
    spread_usd : float, default=0.0
        Round-trip bid/ask spread cost in USD.
    entry_slippage_usd : float, default=0.0
        Adverse slippage on entry in USD.
    exit_slippage_usd : float, default=0.0
        Adverse slippage on exit in USD.
    commission_usd : float, default=0.0
        Round-trip commission in USD.
    commission_r_fixed : float, default=0.0
        Fixed round-trip commission in R units.
    entry_offset_r : float, default=0.0
        Entry offset in baseline structural OB range units R0.
    stop_multiplier_r : float, default=1.0
        Stop distance multiplier relative to baseline OB range R0.
    target_multiplier_r : float, default=1.0
        Target distance multiplier relative to baseline OB range R0.
    """
    scenario_id: str
    scenario_name: str
    scenario_description: str
    spread_usd: float = 0.0
    entry_slippage_usd: float = 0.0
    exit_slippage_usd: float = 0.0
    commission_usd: float = 0.0
    commission_r_fixed: float = 0.0
    entry_offset_r: float = 0.0
    stop_multiplier_r: float = 1.0
    target_multiplier_r: float = 1.0

    def __post_init__(self) -> None:
        if not self.scenario_id or not isinstance(self.scenario_id, str):
            raise ValueError(f"scenario_id must be a non-empty string, got: {self.scenario_id}")

        for val_name, val in [
            ("spread_usd", self.spread_usd),
            ("entry_slippage_usd", self.entry_slippage_usd),
            ("exit_slippage_usd", self.exit_slippage_usd),
            ("commission_usd", self.commission_usd),
            ("commission_r_fixed", self.commission_r_fixed),
            ("entry_offset_r", self.entry_offset_r),
        ]:
            if math.isnan(val) or math.isinf(val) or val < 0.0:
                raise ValueError(f"{val_name} must be a non-negative finite float, got: {val}")

        for mult_name, mult in [
            ("stop_multiplier_r", self.stop_multiplier_r),
            ("target_multiplier_r", self.target_multiplier_r),
        ]:
            if math.isnan(mult) or math.isinf(mult) or mult <= 0.0:
                raise ValueError(f"{mult_name} must be a strictly positive finite float, got: {mult}")

    @property
    def cost_model(self) -> ExecutionCostModel:
        """Instantiate the verified Step 1 ExecutionCostModel for this scenario."""
        return ExecutionCostModel(
            spread_usd=self.spread_usd,
            entry_slippage_usd=self.entry_slippage_usd,
            exit_slippage_usd=self.exit_slippage_usd,
            commission_usd=self.commission_usd,
            commission_r_fixed=self.commission_r_fixed,
        )

    @property
    def parameter_scenario(self) -> ExecutionParameterScenario:
        """Instantiate the isolated Step 4 ExecutionParameterScenario for this scenario."""
        return ExecutionParameterScenario(
            scenario_id=self.scenario_id,
            entry_offset_r=self.entry_offset_r,
            stop_multiplier_r=self.stop_multiplier_r,
            target_multiplier_r=self.target_multiplier_r,
            is_baseline=(
                self.entry_offset_r == 0.0
                and self.stop_multiplier_r == 1.0
                and self.target_multiplier_r == 1.0
                and self.spread_usd == 0.0
                and self.entry_slippage_usd == 0.0
                and self.exit_slippage_usd == 0.0
                and self.commission_usd == 0.0
                and self.commission_r_fixed == 0.0
            ),
        )


def generate_predefined_combined_scenarios() -> list[CombinedStressScenario]:
    """
    Generate the four canonical predefined Combined Execution Stress Scenarios.
    """
    return [
        CombinedStressScenario(
            scenario_id="SCENARIO_1_IDEALIZED",
            scenario_name="IDEALIZED",
            scenario_description="Zero-cost baseline execution with nominal frozen geometry",
            spread_usd=0.00,
            entry_slippage_usd=0.00,
            exit_slippage_usd=0.00,
            commission_usd=0.00,
            commission_r_fixed=0.00,
            entry_offset_r=0.00,
            stop_multiplier_r=1.00,
            target_multiplier_r=1.00,
        ),
        CombinedStressScenario(
            scenario_id="SCENARIO_2_MODERATE_REALISTIC",
            scenario_name="MODERATE_REALISTIC",
            scenario_description="Moderate transaction friction with nominal frozen geometry",
            spread_usd=0.10,
            entry_slippage_usd=0.02,
            exit_slippage_usd=0.02,
            commission_usd=0.02,
            commission_r_fixed=0.00,
            entry_offset_r=0.00,
            stop_multiplier_r=1.00,
            target_multiplier_r=1.00,
        ),
        CombinedStressScenario(
            scenario_id="SCENARIO_3_HIGH_FRICTION",
            scenario_name="HIGH_FRICTION",
            scenario_description="Materially higher transaction friction with deeper entry requirement",
            spread_usd=0.20,
            entry_slippage_usd=0.05,
            exit_slippage_usd=0.05,
            commission_usd=0.05,
            commission_r_fixed=0.00,
            entry_offset_r=0.10,
            stop_multiplier_r=1.00,
            target_multiplier_r=1.00,
        ),
        CombinedStressScenario(
            scenario_id="SCENARIO_4_SEVERE_STRESS",
            scenario_name="SEVERE_STRESS",
            scenario_description="Combined adverse stress: high friction, deeper entry, wider stop and target",
            spread_usd=0.30,
            entry_slippage_usd=0.10,
            exit_slippage_usd=0.10,
            commission_usd=0.05,
            commission_r_fixed=0.00,
            entry_offset_r=0.20,
            stop_multiplier_r=1.25,
            target_multiplier_r=1.25,
        ),
    ]


class CombinedStressSimulator:
    """
    Chronological execution simulator that couples execution parameter geometry
    with deterministic Step 1 transaction cost accounting.
    """

    def __init__(
        self,
        market_df: pd.DataFrame,
        enriched_ob_df: pd.DataFrame,
    ) -> None:
        self.market_df = market_df.copy()
        self.enriched_ob_df = enriched_ob_df.copy()

        # Re-use isolated ParameterizedExecutionSimulator setup preparation
        self._param_simulator = ParameterizedExecutionSimulator(self.market_df, self.enriched_ob_df)
        self.total_setups_count = self._param_simulator.total_setups_count

    def simulate_scenario(self, scenario: CombinedStressScenario) -> dict[str, Any]:
        """
        Simulate a combined stress scenario candle-by-candle with post-trade friction attribution.

        Returns
        -------
        dict[str, Any]
            Complete scenario record matching all required output metrics.
        """
        cost_model = scenario.cost_model
        param_scenario = scenario.parameter_scenario

        filled_trades = 0
        unfilled_trades = 0
        tp_trades = 0
        sl_trades = 0
        ambiguous_trades = 0
        horizon_trades = 0

        gross_r_list: list[float] = []
        spread_cost_list: list[float] = []
        slippage_cost_list: list[float] = []
        commission_cost_list: list[float] = []
        total_cost_list: list[float] = []
        net_r_list: list[float] = []
        entry_ts_list: list[pd.Timestamp] = []

        norm_tp_r = float(scenario.target_multiplier_r / scenario.stop_multiplier_r)
        norm_sl_r = -1.0

        for setup in self._param_simulator._setup_data:
            if not setup["has_day_slice"]:
                unfilled_trades += 1
                continue

            c_low = setup["c_low"]
            c_high = setup["c_high"]
            direction = setup["direction"]
            highs = setup["highs"]
            lows = setup["lows"]
            closes = setup["closes"]
            timestamps = setup["timestamps"]
            n_bars = len(highs)

            _, entry_p, sl_p, tp_p, ob_range, actual_risk_usd = param_scenario.calculate_order_levels(
                direction, c_low, c_high
            )

            is_filled = False
            filled_ts: Optional[pd.Timestamp] = None
            fill_bar_idx = -1
            trade_closed = False
            trade_gross_r = 0.0

            for bar_idx in range(n_bars):
                b_high = highs[bar_idx]
                b_low = lows[bar_idx]
                b_close = closes[bar_idx]
                b_ts = timestamps[bar_idx]

                # Step 1: Check fill if pending
                if not is_filled:
                    if direction == "bearish":
                        fill_condition = (b_high >= entry_p) and (b_low <= c_high)
                    else:
                        fill_condition = (b_low <= entry_p) and (b_high >= c_low)

                    if fill_condition:
                        is_filled = True
                        filled_ts = b_ts
                        fill_bar_idx = bar_idx

                        # Same-candle SL / TP triggers on fill bar
                        if direction == "bearish":
                            hit_tp = b_low <= tp_p
                            hit_sl = b_high >= sl_p
                        else:
                            hit_tp = b_high >= tp_p
                            hit_sl = b_low <= sl_p

                        if hit_tp and hit_sl:
                            trade_closed = True
                            ambiguous_trades += 1
                            trade_gross_r = norm_sl_r
                            break
                        elif hit_tp:
                            trade_closed = True
                            tp_trades += 1
                            trade_gross_r = norm_tp_r
                            break
                        elif hit_sl:
                            trade_closed = True
                            sl_trades += 1
                            trade_gross_r = norm_sl_r
                            break

                # Step 2: Position is open from earlier bar
                elif not trade_closed:
                    if direction == "bearish":
                        hit_tp = b_low <= tp_p
                        hit_sl = b_high >= sl_p
                    else:
                        hit_tp = b_high >= tp_p
                        hit_sl = b_low <= sl_p

                    if hit_tp and hit_sl:
                        trade_closed = True
                        ambiguous_trades += 1
                        trade_gross_r = norm_sl_r
                        break
                    elif hit_tp:
                        trade_closed = True
                        tp_trades += 1
                        trade_gross_r = norm_tp_r
                        break
                    elif hit_sl:
                        trade_closed = True
                        sl_trades += 1
                        trade_gross_r = norm_sl_r
                        break
                    elif bar_idx == n_bars - 1:
                        trade_closed = True
                        horizon_trades += 1
                        if direction == "bearish":
                            trade_gross_r = float(round((entry_p - b_close) / actual_risk_usd, 4))
                        else:
                            trade_gross_r = float(round((b_close - entry_p) / actual_risk_usd, 4))
                        break

            if not is_filled:
                unfilled_trades += 1
            else:
                filled_trades += 1
                entry_ts_list.append(filled_ts)

                # Deterministic Step 1 cost accounting attribution
                breakdown: TradeCostBreakdown = cost_model.compute_trade_cost_breakdown(
                    gross_realized_r=trade_gross_r,
                    ob_range=ob_range,
                    direction=direction,
                )

                gross_r_list.append(breakdown.gross_realized_r)
                spread_cost_list.append(breakdown.spread_cost_r)
                slippage_cost_list.append(breakdown.slippage_cost_r)
                commission_cost_list.append(breakdown.commission_cost_r)
                total_cost_list.append(breakdown.total_cost_r)
                net_r_list.append(breakdown.net_realized_r)

        # Aggregation
        total_gross_r = float(round(sum(gross_r_list), 4)) if gross_r_list else 0.0
        total_spread_cost = float(round(sum(spread_cost_list), 4)) if spread_cost_list else 0.0
        total_slippage_cost = float(round(sum(slippage_cost_list), 4)) if slippage_cost_list else 0.0
        total_commission_cost = float(round(sum(commission_cost_list), 4)) if commission_cost_list else 0.0
        total_exec_cost = float(round(sum(total_cost_list), 4)) if total_cost_list else 0.0

        net_r = float(round(total_gross_r - total_exec_cost, 4))

        fill_rate = float(round(filled_trades / self.total_setups_count, 4)) if self.total_setups_count > 0 else 0.0
        gross_exp = float(round(total_gross_r / filled_trades, 4)) if filled_trades > 0 else 0.0
        net_exp = float(round(net_r / filled_trades, 4)) if filled_trades > 0 else 0.0
        candidate_ev = float(round(net_r / self.total_setups_count, 4)) if self.total_setups_count > 0 else 0.0

        # Net risk metrics (evaluated on chronological net realized returns)
        if entry_ts_list and net_r_list:
            sorted_pairs = sorted(zip(entry_ts_list, net_r_list), key=lambda x: x[0])
            sorted_net_r = [x[1] for x in sorted_pairs]
        else:
            sorted_net_r = net_r_list

        net_wins = [r for r in sorted_net_r if r > 0.0]
        net_losses = [r for r in sorted_net_r if r < 0.0]
        net_win_rate = float(round(len(net_wins) / filled_trades, 4)) if filled_trades > 0 else 0.0

        gross_gains = sum(net_wins)
        gross_loss_sum = abs(sum(net_losses))
        if gross_loss_sum > 0.0:
            net_profit_factor = float(round(gross_gains / gross_loss_sum, 2))
        elif gross_gains > 0.0:
            net_profit_factor = float("inf")
        else:
            net_profit_factor = 0.0

        cum_r = 0.0
        peak_r = 0.0
        max_dd = 0.0
        cur_loss_streak, max_loss_streak = 0, 0

        for r_val in sorted_net_r:
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

        # Break-even edge analysis (Step 3 methodology)
        remaining_edge_r = net_r
        if total_gross_r != 0.0:
            cost_utilization_fraction = float(round(total_exec_cost / total_gross_r, 4))
            break_even_margin_fraction = float(round(net_r / total_gross_r, 4))
        else:
            cost_utilization_fraction = 0.0
            break_even_margin_fraction = 0.0

        if total_exec_cost > 0.0:
            be_cost_multiplier = float(round(total_gross_r / total_exec_cost, 4))
            cost_headroom_multiplier = float(round(be_cost_multiplier - 1.0, 4))
        else:
            be_cost_multiplier = float("inf")
            cost_headroom_multiplier = float("inf")

        return {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.scenario_name,
            "scenario_description": scenario.scenario_description,
            "spread_usd": scenario.spread_usd,
            "entry_slippage_usd": scenario.entry_slippage_usd,
            "exit_slippage_usd": scenario.exit_slippage_usd,
            "commission_usd": scenario.commission_usd,
            "commission_r_fixed": scenario.commission_r_fixed,
            "entry_offset_r": scenario.entry_offset_r,
            "stop_multiplier_r": scenario.stop_multiplier_r,
            "target_multiplier_r": scenario.target_multiplier_r,
            "total_setups": self.total_setups_count,
            "filled_trades": filled_trades,
            "unfilled_trades": unfilled_trades,
            "fill_rate": fill_rate,
            "take_profit_trades": tp_trades,
            "stop_loss_trades": sl_trades,
            "ambiguous_stop_loss_trades": ambiguous_trades,
            "horizon_exit_trades": horizon_trades,
            "gross_net_r": total_gross_r,
            "gross_expectancy_r_per_filled_trade": gross_exp,
            "total_spread_cost_r": total_spread_cost,
            "total_slippage_cost_r": total_slippage_cost,
            "total_commission_cost_r": total_commission_cost,
            "total_execution_cost_r": total_exec_cost,
            "net_r": net_r,
            "net_expectancy_r_per_filled_trade": net_exp,
            "candidate_ev_net_r_per_setup": candidate_ev,
            "net_win_rate": net_win_rate,
            "net_profit_factor": net_profit_factor,
            "maximum_drawdown_r": max_dd_r,
            "maximum_consecutive_losses": max_loss_streak,
            "remaining_edge_r": remaining_edge_r,
            "cost_utilization_fraction": cost_utilization_fraction,
            "break_even_margin_fraction": break_even_margin_fraction,
            "break_even_cost_multiplier": be_cost_multiplier,
            "cost_headroom_multiplier": cost_headroom_multiplier,
        }


def run_combined_stress_scenarios(
    market_df: pd.DataFrame,
    enriched_ob_df: pd.DataFrame,
    scenarios: Optional[Sequence[CombinedStressScenario]] = None,
) -> pd.DataFrame:
    """
    Execute combined execution stress scenarios across the STRAT_09 setup population.

    Parameters
    ----------
    market_df : pd.DataFrame
        Continuous 15m OHLC market data.
    enriched_ob_df : pd.DataFrame
        Enriched Order Blocks with frozen feature boundaries.
    scenarios : Optional[Sequence[CombinedStressScenario]]
        Scenarios to simulate (defaults to the 4 canonical predefined stress scenarios).

    Returns
    -------
    pd.DataFrame
        Structured results DataFrame with one row per scenario.
    """
    if scenarios is None:
        scenarios = generate_predefined_combined_scenarios()

    simulator = CombinedStressSimulator(market_df, enriched_ob_df)

    records: list[dict[str, Any]] = []
    for sc in scenarios:
        row = simulator.simulate_scenario(sc)
        records.append(row)

    return pd.DataFrame(records)


def generate_and_save_combined_stress_report(
    market_df: pd.DataFrame,
    enriched_ob_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Execute the four combined stress scenarios and persist results to CSV artifact.

    Parameters
    ----------
    market_df : pd.DataFrame
        Continuous 15m market data.
    enriched_ob_df : pd.DataFrame
        Enriched Order Blocks.
    output_path : Optional[Path]
        Target CSV path (defaults to outputs/reports/execution_combined_stress_scenarios_strat_09.csv).

    Returns
    -------
    tuple[pd.DataFrame, Path]
        (results_dataframe, saved_file_path)
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_path = project_root / "outputs" / "reports" / "execution_combined_stress_scenarios_strat_09.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = run_combined_stress_scenarios(market_df, enriched_ob_df)
    df.to_csv(output_path, index=False)
    return df, output_path
