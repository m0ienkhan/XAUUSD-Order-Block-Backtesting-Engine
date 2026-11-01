from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

from src.analytics.metrics import OBDirection
from src.execution.cost_model import ExecutionCostModel
from src.execution.order import OrderSide, OrderStatus, TradeResolution, Trade


@dataclass(frozen=True)
class ExecutionParameterScenario:
    """
    Immutable specification for an execution parameter robustness scenario around STRAT_09.

    Parameters
    ----------
    scenario_id : str
        Deterministic unique identifier for the scenario (e.g. 'ENTRY_OFF_010', 'STOP_125').
    entry_offset_r : float, default=0.0
        Entry offset in units of original structural OB range R0.
        For bearish STRAT_09: entry_price = candidate_low + (entry_offset_r * ob_range).
        0.0 = candidate_low; positive offset = deeper entry (higher short limit price).
    stop_multiplier_r : float, default=1.0
        Stop distance multiplier in units of original OB range R0.
        actual_risk_usd = stop_multiplier_r * ob_range.
        stop_price = entry_price + (stop_multiplier_r * ob_range).
    target_multiplier_r : float, default=1.0
        Target distance multiplier in units of original OB range R0:
        target_price = entry_price - (target_multiplier_r * ob_range).
    is_baseline : bool, default=False
        Flag indicating if this configuration represents the canonical baseline.
    parameter_dimension : str, default='BASELINE'
        Sensitivity dimension: 'ENTRY', 'STOP', 'TARGET', or 'BASELINE'.
    cost_model : ExecutionCostModel
        Transaction cost model (defaults to zero-cost for Step 4 geometry sensitivity).
    """
    scenario_id: str
    entry_offset_r: float = 0.0
    stop_multiplier_r: float = 1.0
    target_multiplier_r: float = 1.0
    is_baseline: bool = False
    parameter_dimension: str = "BASELINE"
    cost_model: ExecutionCostModel = field(default_factory=ExecutionCostModel)
    stop_multiplier: Optional[float] = None
    target_multiplier: Optional[float] = None
    sensitivity_dimension: Optional[str] = None
    analysis_type: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not isinstance(self.scenario_id, str):
            raise ValueError(f"scenario_id must be a non-empty string, got: {self.scenario_id}")

        if self.stop_multiplier is not None:
            object.__setattr__(self, "stop_multiplier_r", float(self.stop_multiplier))
        else:
            object.__setattr__(self, "stop_multiplier", self.stop_multiplier_r)

        if self.target_multiplier is not None:
            object.__setattr__(self, "target_multiplier_r", float(self.target_multiplier))
        else:
            object.__setattr__(self, "target_multiplier", self.target_multiplier_r)

        if self.sensitivity_dimension is not None:
            object.__setattr__(self, "parameter_dimension", str(self.sensitivity_dimension))
        else:
            object.__setattr__(self, "sensitivity_dimension", self.parameter_dimension)

        if math.isnan(self.entry_offset_r) or math.isinf(self.entry_offset_r) or self.entry_offset_r < 0.0:
            raise ValueError(f"entry_offset_r must be a non-negative finite float, got: {self.entry_offset_r}")

        if math.isnan(self.stop_multiplier_r) or math.isinf(self.stop_multiplier_r) or self.stop_multiplier_r <= 0.0:
            raise ValueError(f"stop_multiplier_r must be a strictly positive finite float, got: {self.stop_multiplier_r}")

        if math.isnan(self.target_multiplier_r) or math.isinf(self.target_multiplier_r) or self.target_multiplier_r <= 0.0:
            raise ValueError(f"target_multiplier_r must be a strictly positive finite float, got: {self.target_multiplier_r}")

    @property
    def sensitivity_family(self) -> str:
        return self.parameter_dimension

    @property
    def actual_risk_multiplier(self) -> float:
        return self.stop_multiplier_r

    @property
    def actual_reward_risk_ratio(self) -> float:
        """Normalized reward-to-risk ratio: target_multiplier_r / stop_multiplier_r."""
        return float(self.target_multiplier_r / self.stop_multiplier_r)

    def calculate_order_levels(
        self,
        direction: OBDirection | str,
        candidate_low: float,
        candidate_high: float,
    ) -> tuple[OrderSide, float, float, float, float, float]:
        """
        Calculate entry price, SL price, TP price, original OB range R0, and actual risk in USD.

        Formulas (Bearish):
            ob_range = candidate_high - candidate_low
            actual_risk_usd = stop_multiplier_r * ob_range
            entry_price = candidate_low + (entry_offset_r * ob_range)
            stop_price = entry_price + (stop_multiplier_r * ob_range)
            target_price = entry_price - (target_multiplier_r * ob_range)

        Returns
        -------
        tuple of (side, entry_price, stop_loss_price, target_price, ob_range, actual_risk_usd)
        """
        ob_range = max(float(candidate_high - candidate_low), 1e-8)
        actual_risk_usd = max(float(self.stop_multiplier_r * ob_range), 1e-8)

        dir_clean = str(direction).lower()
        if dir_clean in ("bearish", "sell"):
            side = OrderSide.SELL
            entry_price = float(candidate_low + (self.entry_offset_r * ob_range))
            stop_loss_price = float(entry_price + (self.stop_multiplier_r * ob_range))
            target_price = float(entry_price - (self.target_multiplier_r * ob_range))
        elif dir_clean in ("bullish", "buy"):
            side = OrderSide.BUY
            entry_price = float(candidate_high - (self.entry_offset_r * ob_range))
            stop_loss_price = float(entry_price - (self.stop_multiplier_r * ob_range))
            target_price = float(entry_price + (self.target_multiplier_r * ob_range))
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return side, entry_price, stop_loss_price, target_price, ob_range, actual_risk_usd


# Alias for backward compatibility
ParameterStressScenario = ExecutionParameterScenario


def generate_one_factor_parameter_scenarios() -> list[ExecutionParameterScenario]:
    """
    Generate the 13 independent one-factor parameter sensitivity scenarios across 3 families.

    1. Entry Sensitivity (4 scenarios):
       Vary entry offset in [0.00, 0.10, 0.20, 0.30] holding SL=1.00, TP=1.00.
    2. Stop Sensitivity (4 scenarios):
       Vary stop multiplier in [0.75, 1.00, 1.25, 1.50] holding Entry=0.00, TP=1.00.
    3. Target Sensitivity (5 scenarios):
       Vary target multiplier in [0.75, 1.00, 1.25, 1.50, 2.00] holding Entry=0.00, SL=1.00.

    Total: 4 + 4 + 5 = 13 scenarios.
    """
    scenarios: list[ExecutionParameterScenario] = []

    # 1. ENTRY SENSITIVITY (Hold SL=1.00R, TP=1.00R)
    entry_offsets = [0.00, 0.10, 0.20, 0.30]
    for e_off in entry_offsets:
        is_base = (e_off == 0.00)
        e_code = int(round(e_off * 100))
        s_id = f"ENTRY_OFF_{e_code:03d}"
        scenarios.append(
            ExecutionParameterScenario(
                scenario_id=s_id,
                entry_offset_r=float(e_off),
                stop_multiplier_r=1.00,
                target_multiplier_r=1.00,
                is_baseline=is_base,
                parameter_dimension="ENTRY",
            )
        )

    # 2. STOP SENSITIVITY (Hold Entry=0.00R, TP=1.00R)
    stop_multipliers = [0.75, 1.00, 1.25, 1.50]
    for s_mult in stop_multipliers:
        is_base = (s_mult == 1.00)
        s_code = int(round(s_mult * 100))
        s_id = f"STOP_{s_code:03d}"
        scenarios.append(
            ExecutionParameterScenario(
                scenario_id=s_id,
                entry_offset_r=0.00,
                stop_multiplier_r=float(s_mult),
                target_multiplier_r=1.00,
                is_baseline=is_base,
                parameter_dimension="STOP",
            )
        )

    # 3. TARGET SENSITIVITY (Hold Entry=0.00R, SL=1.00R)
    target_multipliers = [0.75, 1.00, 1.25, 1.50, 2.00]
    for t_mult in target_multipliers:
        is_base = (t_mult == 1.00)
        t_code = int(round(t_mult * 100))
        s_id = f"TARGET_{t_code:03d}"
        scenarios.append(
            ExecutionParameterScenario(
                scenario_id=s_id,
                entry_offset_r=0.00,
                stop_multiplier_r=1.00,
                target_multiplier_r=float(t_mult),
                is_baseline=is_base,
                parameter_dimension="TARGET",
            )
        )

    return scenarios


def generate_local_robustness_grid(
    entry_offsets: Sequence[float] = (0.00, 0.10, 0.20, 0.30),
    stop_multipliers: Sequence[float] = (0.75, 1.00, 1.25, 1.50),
    target_multipliers: Sequence[float] = (0.75, 1.00, 1.25, 1.50, 2.00),
) -> list[ExecutionParameterScenario]:
    """
    Generate the 80-scenario local robustness grid around the frozen baseline (4 x 4 x 5).
    """
    scenarios: list[ExecutionParameterScenario] = []

    for e_off in entry_offsets:
        for s_mult in stop_multipliers:
            for t_mult in target_multipliers:
                is_base = (e_off == 0.00 and s_mult == 1.00 and t_mult == 1.00)
                e_code = int(round(e_off * 100))
                s_code = int(round(s_mult * 100))
                t_code = int(round(t_mult * 100))
                s_id = "BASELINE" if is_base else f"GRID_E{e_code:03d}_S{s_code:03d}_T{t_code:03d}"
                dim = "BASELINE" if is_base else "GRID"

                scenarios.append(
                    ExecutionParameterScenario(
                        scenario_id=s_id,
                        entry_offset_r=float(e_off),
                        stop_multiplier_r=float(s_mult),
                        target_multiplier_r=float(t_mult),
                        is_baseline=is_base,
                        parameter_dimension=dim,
                    )
                )

    return scenarios


# Alias for backward compatibility
generate_parameter_robustness_grid = generate_local_robustness_grid


def generate_unique_parameter_scenarios() -> list[ExecutionParameterScenario]:
    """Generate unique non-duplicate parameter scenarios."""
    return generate_one_factor_parameter_scenarios()


def generate_combined_parameter_scenarios() -> list[ExecutionParameterScenario]:
    """Generate combined one-factor and local grid scenarios."""
    return generate_one_factor_parameter_scenarios() + generate_local_robustness_grid()


class ParameterizedExecutionSimulator:
    """
    Isolated execution simulator that evaluates parameterized execution geometry
    strictly on the frozen STRAT_09 setup population without lookahead.
    """

    def __init__(
        self,
        market_df: pd.DataFrame,
        enriched_ob_df: pd.DataFrame,
    ) -> None:
        self.market_df = market_df.copy()
        self.enriched_ob_df = enriched_ob_df.copy()

        # Extract eligible STRAT_09 setups: ob_direction == 'bearish' and dist_bucket == 'Medium_Dist'
        mask = (self.enriched_ob_df["ob_direction"] == "bearish") & (
            self.enriched_ob_df["dist_bucket"] == "Medium_Dist"
        )
        self.eligible_obs_df = self.enriched_ob_df[mask].copy().reset_index(drop=True)
        self.total_setups_count = len(self.eligible_obs_df)

        # Pre-extract Day T+1 market slices for fast, reproducible chronological simulation
        self._setup_data: list[dict[str, Any]] = []
        self._prepare_setup_data()

    def _prepare_setup_data(self) -> None:
        """Pre-extract Day T+1 bar slices for all eligible setups."""
        for idx, row in self.eligible_obs_df.iterrows():
            ob_trading_day = row["trading_day"]
            c_low = float(row["candidate_low"])
            c_high = float(row["candidate_high"])
            c_ts = pd.Timestamp(row["candidate_timestamp"])
            c_pos = int(row["candidate_position"])
            direction = str(row["ob_direction"])

            # Find next trading day
            future_mask = self.market_df["trading_day"] > ob_trading_day
            if not future_mask.any():
                self._setup_data.append({
                    "has_day_slice": False,
                    "c_low": c_low,
                    "c_high": c_high,
                    "ob_trading_day": ob_trading_day,
                    "direction": direction,
                })
                continue

            future_df = self.market_df[future_mask]
            future_days = future_df["trading_day"].drop_duplicates().tolist()
            if not future_days:
                self._setup_data.append({
                    "has_day_slice": False,
                    "c_low": c_low,
                    "c_high": c_high,
                    "ob_trading_day": ob_trading_day,
                    "direction": direction,
                })
                continue

            target_day = future_days[0]
            day_slice = future_df[future_df["trading_day"] == target_day]
            if day_slice.empty:
                self._setup_data.append({
                    "has_day_slice": False,
                    "c_low": c_low,
                    "c_high": c_high,
                    "ob_trading_day": ob_trading_day,
                    "direction": direction,
                })
                continue

            self._setup_data.append({
                "has_day_slice": True,
                "c_low": c_low,
                "c_high": c_high,
                "c_ts": c_ts,
                "c_pos": c_pos,
                "ob_trading_day": ob_trading_day,
                "direction": direction,
                "highs": day_slice["high"].to_numpy(dtype=float),
                "lows": day_slice["low"].to_numpy(dtype=float),
                "closes": day_slice["close"].to_numpy(dtype=float),
                "timestamps": list(day_slice.index),
            })

    def simulate_scenario(
        self,
        scenario: ExecutionParameterScenario,
        baseline_net_r: float = 107.6593,
    ) -> dict[str, Any]:
        """
        Simulate a parameter stress scenario sequentially candle-by-candle.

        Returns
        -------
        dict[str, Any]
            Comprehensive performance and robustness metrics.
        """
        filled_trades = 0
        unfilled_orders = 0
        tp_trades = 0
        sl_trades = 0
        ambiguous_trades = 0
        horizon_trades = 0

        realized_r_list: list[float] = []
        entry_ts_list: list[pd.Timestamp] = []

        # Target R normalization: TP = target_multiplier / stop_multiplier in scenario actual risk units
        norm_tp_r = float(scenario.target_multiplier_r / scenario.stop_multiplier_r)
        norm_sl_r = -1.0

        for setup in self._setup_data:
            if not setup["has_day_slice"]:
                unfilled_orders += 1
                continue

            c_low = setup["c_low"]
            c_high = setup["c_high"]
            direction = setup["direction"]
            highs = setup["highs"]
            lows = setup["lows"]
            closes = setup["closes"]
            timestamps = setup["timestamps"]
            n_bars = len(highs)

            _, entry_p, sl_p, tp_p, ob_range, actual_risk_usd = scenario.calculate_order_levels(
                direction, c_low, c_high
            )

            is_filled = False
            filled_ts: Optional[pd.Timestamp] = None
            fill_bar_idx = -1
            trade_closed = False

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

                        # Check same-candle SL / TP triggers on fill bar
                        if direction == "bearish":
                            hit_tp = b_low <= tp_p
                            hit_sl = b_high >= sl_p
                        else:
                            hit_tp = b_high >= tp_p
                            hit_sl = b_low <= sl_p

                        if hit_tp and hit_sl:
                            # Conservative Ambiguity Policy: Same-candle SL + TP = Stop Loss First
                            trade_closed = True
                            ambiguous_trades += 1
                            realized_r_list.append(-1.0)
                            entry_ts_list.append(filled_ts)
                            break
                        elif hit_tp:
                            trade_closed = True
                            tp_trades += 1
                            realized_r_list.append(norm_tp_r)
                            entry_ts_list.append(filled_ts)
                            break
                        elif hit_sl:
                            trade_closed = True
                            sl_trades += 1
                            realized_r_list.append(norm_sl_r)
                            entry_ts_list.append(filled_ts)
                            break

                # Step 2: Position is open from previous bar
                elif not trade_closed:
                    if direction == "bearish":
                        hit_tp = b_low <= tp_p
                        hit_sl = b_high >= sl_p
                    else:
                        hit_tp = b_high >= tp_p
                        hit_sl = b_low <= sl_p

                    if hit_tp and hit_sl:
                        # Conservative Ambiguity Policy
                        trade_closed = True
                        ambiguous_trades += 1
                        realized_r_list.append(-1.0)
                        entry_ts_list.append(filled_ts)
                        break
                    elif hit_tp:
                        trade_closed = True
                        tp_trades += 1
                        realized_r_list.append(norm_tp_r)
                        entry_ts_list.append(filled_ts)
                        break
                    elif hit_sl:
                        trade_closed = True
                        sl_trades += 1
                        realized_r_list.append(norm_sl_r)
                        entry_ts_list.append(filled_ts)
                        break
                    elif bar_idx == n_bars - 1:
                        # Final bar of next trading day: Horizon Expiration
                        trade_closed = True
                        horizon_trades += 1
                        if direction == "bearish":
                            frac_r = float(round((entry_p - b_close) / actual_risk_usd, 4))
                        else:
                            frac_r = float(round((b_close - entry_p) / actual_risk_usd, 4))
                        realized_r_list.append(frac_r)
                        entry_ts_list.append(filled_ts)
                        break

            if not is_filled:
                unfilled_orders += 1
            else:
                filled_trades += 1

        # Calculate Performance Metrics
        total_realized_r = float(round(sum(realized_r_list), 4)) if realized_r_list else 0.0
        gross_net_r = total_realized_r

        fill_rate = float(round(filled_trades / self.total_setups_count, 4)) if self.total_setups_count > 0 else 0.0
        expectancy = float(round(gross_net_r / filled_trades, 4)) if filled_trades > 0 else 0.0
        candidate_ev = float(round(gross_net_r / self.total_setups_count, 4)) if self.total_setups_count > 0 else 0.0

        # Win/loss and equity curve
        wins = [r for r in realized_r_list if r > 0.0]
        losses = [r for r in realized_r_list if r < 0.0]
        win_count = len(wins)
        win_rate = float(round(win_count / filled_trades, 4)) if filled_trades > 0 else 0.0

        gross_gains = sum(wins)
        gross_losses = abs(sum(losses))
        if gross_losses > 0.0:
            profit_factor = float(round(gross_gains / gross_losses, 2))
        elif gross_gains > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        # Sort chronological trades by entry_ts for sequential equity curve
        if entry_ts_list:
            sorted_pairs = sorted(zip(entry_ts_list, realized_r_list), key=lambda x: x[0])
            sorted_r = [x[1] for x in sorted_pairs]
        else:
            sorted_r = realized_r_list

        cum_r = 0.0
        peak_r = 0.0
        max_dd = 0.0
        cur_loss_streak, max_loss_streak = 0, 0
        cur_win_streak, max_win_streak = 0, 0

        for r_val in sorted_r:
            if r_val > 0.0:
                cur_win_streak += 1
                cur_loss_streak = 0
                if cur_win_streak > max_win_streak:
                    max_win_streak = cur_win_streak
            elif r_val < 0.0:
                cur_loss_streak += 1
                cur_win_streak = 0
                if cur_loss_streak > max_loss_streak:
                    max_loss_streak = cur_loss_streak

            cum_r += r_val
            if cum_r > peak_r:
                peak_r = cum_r
            dd = peak_r - cum_r
            if dd > max_dd:
                max_dd = dd

        max_dd_r = float(round(max_dd, 4))

        # Baseline-relative metrics
        baseline_relative_net_r = float(round(gross_net_r - baseline_net_r, 4))
        rel_pct = float(round((baseline_relative_net_r / baseline_net_r) * 100.0, 2)) if baseline_net_r != 0.0 else 0.0

        return {
            "scenario_id": scenario.scenario_id,
            "parameter_dimension": scenario.parameter_dimension,
            "sensitivity_dimension": scenario.parameter_dimension,
            "sensitivity_family": scenario.parameter_dimension,
            "is_baseline": scenario.is_baseline,
            "entry_offset_r": scenario.entry_offset_r,
            "stop_multiplier_r": scenario.stop_multiplier_r,
            "stop_multiplier": scenario.stop_multiplier_r,
            "target_multiplier_r": scenario.target_multiplier_r,
            "target_multiplier": scenario.target_multiplier_r,
            "actual_reward_risk_ratio": scenario.actual_reward_risk_ratio,
            "total_setups": self.total_setups_count,
            "filled_trades": filled_trades,
            "unfilled_trades": unfilled_orders,
            "unfilled_orders": unfilled_orders,
            "expired_unfilled": unfilled_orders,
            "fill_rate": fill_rate,
            "take_profit_trades": tp_trades,
            "stop_loss_trades": sl_trades,
            "ambiguous_stop_loss_trades": ambiguous_trades,
            "ambiguous_stopped_out": ambiguous_trades,
            "horizon_exit_trades": horizon_trades,
            "horizon_expiration_trades": horizon_trades,
            "filled_win_rate": win_rate,
            "filled_trade_win_rate": win_rate,
            "win_rate": win_rate,
            "win_rate_filled": win_rate,
            "gross_net_r": gross_net_r,
            "net_r": gross_net_r,
            "expectancy_r_per_filled_trade": expectancy,
            "expectancy_per_filled_trade": expectancy,
            "net_expectancy_per_filled_trade": expectancy,
            "candidate_ev_r_per_setup": candidate_ev,
            "profit_factor": profit_factor,
            "maximum_drawdown_r": max_dd_r,
            "maximum_consecutive_losses": max_loss_streak,
            "maximum_consecutive_wins": max_win_streak,
            "baseline_gross_net_r": baseline_net_r,
            "baseline_net_r": baseline_net_r,
            "baseline_relative_net_r": baseline_relative_net_r,
            "net_r_vs_baseline": baseline_relative_net_r,
            "net_r_difference_vs_baseline": baseline_relative_net_r,
            "baseline_relative_percent": rel_pct,
        }


def run_parameter_robustness_analysis(
    market_df: pd.DataFrame,
    enriched_ob_df: pd.DataFrame,
    scenarios: Optional[Sequence[ExecutionParameterScenario]] = None,
    baseline_net_r: float = 107.6593,
) -> pd.DataFrame:
    """
    Execute parameter robustness scenarios (defaulting to the 13 independent one-factor scenarios)
    against the verified STRAT_09 setup population.
    """
    if scenarios is None:
        scenarios = generate_one_factor_parameter_scenarios()

    simulator = ParameterizedExecutionSimulator(market_df, enriched_ob_df)

    records: list[dict[str, Any]] = []
    for sc in scenarios:
        row = simulator.simulate_scenario(sc, baseline_net_r=baseline_net_r)
        records.append(row)

    return pd.DataFrame(records)


# Alias for backward compatibility
run_parameter_robustness_grid = run_parameter_robustness_analysis
run_one_factor_parameter_sensitivity = run_parameter_robustness_analysis


def compute_robustness_diagnostics(
    robustness_df: pd.DataFrame,
    baseline_net_r: float = 107.6593,
) -> dict[str, Any]:
    """
    Compute descriptive robustness diagnostics across evaluated scenarios.
    """
    total = len(robustness_df)
    profitable = int((robustness_df["gross_net_r"] > 0.0).sum())
    pct_profitable = float(round((profitable / total) * 100.0, 2)) if total > 0 else 0.0
    positive_ev = int((robustness_df["candidate_ev_r_per_setup"] > 0.0).sum())

    return {
        "total_scenarios": total,
        "profitable_scenarios_count": profitable,
        "profitable_scenarios_pct": pct_profitable,
        "positive_candidate_ev_count": positive_ev,
        "median_net_r": float(round(robustness_df["gross_net_r"].median(), 4)),
        "mean_net_r": float(round(robustness_df["gross_net_r"].mean(), 4)),
        "min_net_r": float(round(robustness_df["gross_net_r"].min(), 4)),
        "max_net_r": float(round(robustness_df["gross_net_r"].max(), 4)),
        "baseline_net_r": baseline_net_r,
    }


def generate_and_save_parameter_robustness_report(
    market_df: pd.DataFrame,
    enriched_ob_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate the parameter sensitivity DataFrame (13 scenarios) and save to CSV.
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_path = project_root / "outputs" / "reports" / "execution_parameter_robustness_strat_09.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = run_parameter_robustness_analysis(market_df, enriched_ob_df)
    df.to_csv(output_path, index=False)
    return df, output_path


# Backward compatibility alias
save_parameter_robustness_report = generate_and_save_parameter_robustness_report


def load_robustness_input_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw continuous market data and enriched OBs with frozen feature boundaries.
    """
    from src.data.trading_day import assign_trading_day
    from src.analytics.walk_forward_validation import (
        compute_training_feature_boundaries,
        apply_frozen_feature_boundaries,
        split_datasets_by_period,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    market_csv = project_root / "data" / "raw" / "XAUUSD_15min_2016_2026.csv"
    enriched_csv = project_root / "outputs" / "reports" / "full_enriched_order_blocks_2016_2026.csv"

    if not market_csv.exists():
        raise FileNotFoundError(f"Market CSV not found at: {market_csv}")
    if not enriched_csv.exists():
        raise FileNotFoundError(f"Enriched OB CSV not found at: {enriched_csv}")

    market_df = pd.read_csv(market_csv)
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
    market_df = assign_trading_day(market_df.set_index("timestamp").sort_index())

    enriched_df = pd.read_csv(enriched_csv)
    train_df, _, _ = split_datasets_by_period(enriched_df)
    boundaries = compute_training_feature_boundaries(train_df)
    annotated_df = apply_frozen_feature_boundaries(enriched_df, boundaries)

    return market_df, annotated_df
