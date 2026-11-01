from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

from src.analytics.metrics import OBDirection
from src.execution.order import OrderStatus, Trade, TradeResolution
from src.execution.contract import ExecutionContract
from src.execution.performance import ExecutionPerformanceSummary, compute_execution_performance


@dataclass
class SimulationResult:
    """
    Complete output container for an execution track simulation run.
    """
    track: str
    contract: ExecutionContract
    summary: ExecutionPerformanceSummary
    trades: list[Trade]
    trades_df: pd.DataFrame
    equity_curve_df: pd.DataFrame
    yearly_df: pd.DataFrame
    monthly_df: pd.DataFrame
    orders_df: pd.DataFrame


class ExecutionSimulator:
    """
    Event-driven chronological 15-minute bar execution engine.
    """

    def __init__(
        self,
        market_df: pd.DataFrame,
        enriched_obs_df: pd.DataFrame,
        contract: ExecutionContract,
    ):
        """
        Parameters
        ----------
        market_df : pd.DataFrame
            Full 15m OHLC time series with DatetimeIndex and 'trading_day' column.
        enriched_obs_df : pd.DataFrame
            Selected and annotated Order Block candidate records.
        contract : ExecutionContract
            Execution contract specifying entry, SL, TP, and policies.
        """
        if not isinstance(market_df.index, pd.DatetimeIndex):
            raise TypeError("market_df index must be a DatetimeIndex.")
        if "trading_day" not in market_df.columns:
            raise KeyError("market_df must contain 'trading_day' column.")

        self.market_df = market_df.copy()
        self.enriched_obs_df = enriched_obs_df.copy()
        self.contract = contract

    def run_simulation(
        self,
        track: str = "TRACK_A_CONSERVATIVE",
    ) -> SimulationResult:
        """
        Execute the bar-by-bar simulation for the specified track.

        Parameters
        ----------
        track : str
            "TRACK_A_CONSERVATIVE" (Primary: same-candle ambiguity -> -1R, horizon exit -> fractional R)
            "TRACK_B_RESEARCH" (Secondary: identical to path_analysis.py, ambiguous/unresolved excluded)
        """
        # Filter eligible Order Blocks for this contract
        eligible_obs = []
        for _, row in self.enriched_obs_df.iterrows():
            if self.contract.is_ob_eligible(row):
                eligible_obs.append(row)

        total_setups_count = len(eligible_obs)
        if total_setups_count == 0:
            empty_summary, eq_df, y_df, m_df = compute_execution_performance(
                [], 0, self.contract.strategy_id, self.contract.strategy_name, track
            )
            return SimulationResult(
                track=track,
                contract=self.contract,
                summary=empty_summary,
                trades=[],
                trades_df=pd.DataFrame(),
                equity_curve_df=eq_df,
                yearly_df=y_df,
                monthly_df=m_df,
                orders_df=pd.DataFrame(),
            )

        executed_trades: list[Trade] = []
        order_records: list[dict[str, Any]] = []

        # Process each eligible setup
        for ob_idx, ob_row in enumerate(eligible_obs):
            ob_trading_day = ob_row["trading_day"]
            direction: OBDirection = ob_row["ob_direction"]
            c_low = float(ob_row["candidate_low"])
            c_high = float(ob_row["candidate_high"])
            c_ts = pd.Timestamp(ob_row["candidate_timestamp"])
            c_pos = int(ob_row["candidate_position"])
            cand_id = (ob_trading_day, c_pos, c_ts, direction)

            side, entry_p, sl_p, tp_p, ob_range = self.contract.calculate_order_levels(
                direction, c_low, c_high
            )

            # Resolve the single next trading day slice
            future_mask = self.market_df["trading_day"] > ob_trading_day
            if not future_mask.any():
                order_records.append({
                    "order_id": f"ORD_{ob_idx+1:04d}",
                    "ob_trading_day": ob_trading_day,
                    "candidate_timestamp": c_ts,
                    "status": OrderStatus.EXPIRED_UNFILLED.value,
                    "reason": "NO_FUTURE_MARKET_DATA",
                })
                continue

            future_df = self.market_df[future_mask]
            future_days = future_df["trading_day"].drop_duplicates().tolist()
            if not future_days:
                order_records.append({
                    "order_id": f"ORD_{ob_idx+1:04d}",
                    "ob_trading_day": ob_trading_day,
                    "candidate_timestamp": c_ts,
                    "status": OrderStatus.EXPIRED_UNFILLED.value,
                    "reason": "NO_NEXT_TRADING_DAY",
                })
                continue

            target_day = future_days[0]
            day_slice = future_df[future_df["trading_day"] == target_day]
            if day_slice.empty:
                order_records.append({
                    "order_id": f"ORD_{ob_idx+1:04d}",
                    "ob_trading_day": ob_trading_day,
                    "candidate_timestamp": c_ts,
                    "status": OrderStatus.EXPIRED_UNFILLED.value,
                    "reason": "EMPTY_NEXT_DAY_SLICE",
                })
                continue

            # Sequential bar-by-bar execution over Day T+1
            is_filled = False
            filled_ts: Optional[pd.Timestamp] = None
            fill_bar_idx = -1
            trade_closed = False

            order_id = f"ORD_{ob_idx+1:04d}"
            trade_id = f"TRD_{ob_idx+1:04d}"

            for bar_idx in range(len(day_slice)):
                bar = day_slice.iloc[bar_idx]
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])
                bar_ts = day_slice.index[bar_idx]

                # Step 1: Check fill if not yet filled
                if not is_filled:
                    if direction == "bearish":
                        fill_condition = (bar_high >= entry_p) and (bar_low <= c_high)
                    else:  # bullish
                        fill_condition = (bar_low <= entry_p) and (bar_high >= c_low)

                    if fill_condition:
                        is_filled = True
                        filled_ts = pd.Timestamp(bar_ts)
                        fill_bar_idx = bar_idx

                        # Check if entry bar also triggers SL / TP immediately
                        if direction == "bearish":
                            hit_tp = bar_low <= tp_p
                            hit_sl = bar_high >= sl_p
                        else:  # bullish
                            hit_tp = bar_high >= tp_p
                            hit_sl = bar_low <= sl_p

                        if hit_tp and hit_sl:
                            trade_closed = True
                            if track == "TRACK_A_CONSERVATIVE":
                                trade = Trade(
                                    trade_id=trade_id,
                                    order_id=order_id,
                                    strategy_id=self.contract.strategy_id,
                                    strategy_name=self.contract.strategy_name,
                                    candidate_identity=cand_id,
                                    ob_trading_day=ob_trading_day,
                                    candidate_timestamp=c_ts,
                                    side=side,
                                    entry_price=entry_p,
                                    entry_timestamp=filled_ts,
                                    exit_price=sl_p,
                                    exit_timestamp=filled_ts,
                                    stop_loss_price=sl_p,
                                    take_profit_price=tp_p,
                                    ob_range=ob_range,
                                    candles_held=0,
                                    resolution=TradeResolution.AMBIGUOUS_STOPPED_OUT,
                                    realized_r=-1.0,
                                    is_win=False,
                                    is_loss=True,
                                    is_ambiguous=True,
                                    is_horizon_exit=False,
                                    track=track,
                                    notes="Entry bar triggered both SL and TP (Conservative Loss).",
                                )
                                executed_trades.append(trade)
                            else:  # TRACK_B_RESEARCH
                                trade = Trade(
                                    trade_id=trade_id,
                                    order_id=order_id,
                                    strategy_id=self.contract.strategy_id,
                                    strategy_name=self.contract.strategy_name,
                                    candidate_identity=cand_id,
                                    ob_trading_day=ob_trading_day,
                                    candidate_timestamp=c_ts,
                                    side=side,
                                    entry_price=entry_p,
                                    entry_timestamp=filled_ts,
                                    exit_price=sl_p,
                                    exit_timestamp=filled_ts,
                                    stop_loss_price=sl_p,
                                    take_profit_price=tp_p,
                                    ob_range=ob_range,
                                    candles_held=0,
                                    resolution=TradeResolution.AMBIGUOUS_STOPPED_OUT,
                                    realized_r=float("nan"),
                                    is_win=False,
                                    is_loss=False,
                                    is_ambiguous=True,
                                    is_horizon_exit=False,
                                    track=track,
                                    notes="ambiguous_same_candle",
                                )
                                executed_trades.append(trade)
                            break

                        elif hit_tp:
                            trade_closed = True
                            trade = Trade(
                                trade_id=trade_id,
                                order_id=order_id,
                                strategy_id=self.contract.strategy_id,
                                strategy_name=self.contract.strategy_name,
                                candidate_identity=cand_id,
                                ob_trading_day=ob_trading_day,
                                candidate_timestamp=c_ts,
                                side=side,
                                entry_price=entry_p,
                                entry_timestamp=filled_ts,
                                exit_price=tp_p,
                                exit_timestamp=filled_ts,
                                stop_loss_price=sl_p,
                                take_profit_price=tp_p,
                                ob_range=ob_range,
                                candles_held=0,
                                resolution=TradeResolution.TAKE_PROFIT,
                                realized_r=1.0,
                                is_win=True,
                                is_loss=False,
                                is_ambiguous=False,
                                is_horizon_exit=False,
                                track=track,
                                notes="Target reached on entry bar.",
                            )
                            executed_trades.append(trade)
                            break

                        elif hit_sl:
                            trade_closed = True
                            trade = Trade(
                                trade_id=trade_id,
                                order_id=order_id,
                                strategy_id=self.contract.strategy_id,
                                strategy_name=self.contract.strategy_name,
                                candidate_identity=cand_id,
                                ob_trading_day=ob_trading_day,
                                candidate_timestamp=c_ts,
                                side=side,
                                entry_price=entry_p,
                                entry_timestamp=filled_ts,
                                exit_price=sl_p,
                                exit_timestamp=filled_ts,
                                stop_loss_price=sl_p,
                                take_profit_price=tp_p,
                                ob_range=ob_range,
                                candles_held=0,
                                resolution=TradeResolution.STOP_LOSS,
                                realized_r=-1.0,
                                is_win=False,
                                is_loss=True,
                                is_ambiguous=False,
                                is_horizon_exit=False,
                                track=track,
                                notes="Stop Loss hit on entry bar.",
                            )
                            executed_trades.append(trade)
                            break

                # Step 2: If position is open from previous bar, check subsequent bar
                elif not trade_closed:
                    candles_held = bar_idx - fill_bar_idx

                    if direction == "bearish":
                        hit_tp = bar_low <= tp_p
                        hit_sl = bar_high >= sl_p
                    else:  # bullish
                        hit_tp = bar_high >= tp_p
                        hit_sl = bar_low <= sl_p

                    if hit_tp and hit_sl:
                        trade_closed = True
                        if track == "TRACK_A_CONSERVATIVE":
                            trade = Trade(
                                trade_id=trade_id,
                                order_id=order_id,
                                strategy_id=self.contract.strategy_id,
                                strategy_name=self.contract.strategy_name,
                                candidate_identity=cand_id,
                                ob_trading_day=ob_trading_day,
                                candidate_timestamp=c_ts,
                                side=side,
                                entry_price=entry_p,
                                entry_timestamp=filled_ts,
                                exit_price=sl_p,
                                exit_timestamp=pd.Timestamp(bar_ts),
                                stop_loss_price=sl_p,
                                take_profit_price=tp_p,
                                ob_range=ob_range,
                                candles_held=candles_held,
                                resolution=TradeResolution.AMBIGUOUS_STOPPED_OUT,
                                realized_r=-1.0,
                                is_win=False,
                                is_loss=True,
                                is_ambiguous=True,
                                is_horizon_exit=False,
                                track=track,
                                notes="Post-fill bar triggered both SL and TP (Conservative Loss).",
                            )
                            executed_trades.append(trade)
                        else:  # TRACK_B_RESEARCH
                            trade = Trade(
                                trade_id=trade_id,
                                order_id=order_id,
                                strategy_id=self.contract.strategy_id,
                                strategy_name=self.contract.strategy_name,
                                candidate_identity=cand_id,
                                ob_trading_day=ob_trading_day,
                                candidate_timestamp=c_ts,
                                side=side,
                                entry_price=entry_p,
                                entry_timestamp=filled_ts,
                                exit_price=sl_p,
                                exit_timestamp=pd.Timestamp(bar_ts),
                                stop_loss_price=sl_p,
                                take_profit_price=tp_p,
                                ob_range=ob_range,
                                candles_held=candles_held,
                                resolution=TradeResolution.AMBIGUOUS_STOPPED_OUT,
                                realized_r=float("nan"),
                                is_win=False,
                                is_loss=False,
                                is_ambiguous=True,
                                is_horizon_exit=False,
                                track=track,
                                notes="ambiguous_same_candle",
                            )
                            executed_trades.append(trade)
                        break

                    elif hit_tp:
                        trade_closed = True
                        trade = Trade(
                            trade_id=trade_id,
                            order_id=order_id,
                            strategy_id=self.contract.strategy_id,
                            strategy_name=self.contract.strategy_name,
                            candidate_identity=cand_id,
                            ob_trading_day=ob_trading_day,
                            candidate_timestamp=c_ts,
                            side=side,
                            entry_price=entry_p,
                            entry_timestamp=filled_ts,
                            exit_price=tp_p,
                            exit_timestamp=pd.Timestamp(bar_ts),
                            stop_loss_price=sl_p,
                            take_profit_price=tp_p,
                            ob_range=ob_range,
                            candles_held=candles_held,
                            resolution=TradeResolution.TAKE_PROFIT,
                            realized_r=1.0,
                            is_win=True,
                            is_loss=False,
                            is_ambiguous=False,
                            is_horizon_exit=False,
                            track=track,
                            notes="Target hit on post-fill bar.",
                        )
                        executed_trades.append(trade)
                        break

                    elif hit_sl:
                        trade_closed = True
                        trade = Trade(
                            trade_id=trade_id,
                            order_id=order_id,
                            strategy_id=self.contract.strategy_id,
                            strategy_name=self.contract.strategy_name,
                            candidate_identity=cand_id,
                            ob_trading_day=ob_trading_day,
                            candidate_timestamp=c_ts,
                            side=side,
                            entry_price=entry_p,
                            entry_timestamp=filled_ts,
                            exit_price=sl_p,
                            exit_timestamp=pd.Timestamp(bar_ts),
                            stop_loss_price=sl_p,
                            take_profit_price=tp_p,
                            ob_range=ob_range,
                            candles_held=candles_held,
                            resolution=TradeResolution.STOP_LOSS,
                            realized_r=-1.0,
                            is_win=False,
                            is_loss=True,
                            is_ambiguous=False,
                            is_horizon_exit=False,
                            track=track,
                            notes="Stop Loss hit on post-fill bar.",
                        )
                        executed_trades.append(trade)
                        break

                    elif bar_idx == len(day_slice) - 1:
                        # Final bar of next trading day: Horizon Expiration
                        trade_closed = True
                        if track == "TRACK_A_CONSERVATIVE":
                            # Close at final bar close with fractional R
                            if direction == "bearish":
                                frac_r = float(round((entry_p - bar_close) / ob_range, 4))
                            else:
                                frac_r = float(round((bar_close - entry_p) / ob_range, 4))

                            trade = Trade(
                                trade_id=trade_id,
                                order_id=order_id,
                                strategy_id=self.contract.strategy_id,
                                strategy_name=self.contract.strategy_name,
                                candidate_identity=cand_id,
                                ob_trading_day=ob_trading_day,
                                candidate_timestamp=c_ts,
                                side=side,
                                entry_price=entry_p,
                                entry_timestamp=filled_ts,
                                exit_price=bar_close,
                                exit_timestamp=pd.Timestamp(bar_ts),
                                stop_loss_price=sl_p,
                                take_profit_price=tp_p,
                                ob_range=ob_range,
                                candles_held=candles_held,
                                resolution=TradeResolution.HORIZON_EXPIRATION,
                                realized_r=frac_r,
                                is_win=bool(frac_r > 0),
                                is_loss=bool(frac_r < 0),
                                is_ambiguous=False,
                                is_horizon_exit=True,
                                track=track,
                                notes=f"Horizon expiration market close @ {bar_close:.3f} (fractional R = {frac_r:+.4f} R).",
                            )
                            executed_trades.append(trade)
                        else:  # TRACK_B_RESEARCH
                            trade = Trade(
                                trade_id=trade_id,
                                order_id=order_id,
                                strategy_id=self.contract.strategy_id,
                                strategy_name=self.contract.strategy_name,
                                candidate_identity=cand_id,
                                ob_trading_day=ob_trading_day,
                                candidate_timestamp=c_ts,
                                side=side,
                                entry_price=entry_p,
                                entry_timestamp=filled_ts,
                                exit_price=bar_close,
                                exit_timestamp=pd.Timestamp(bar_ts),
                                stop_loss_price=sl_p,
                                take_profit_price=tp_p,
                                ob_range=ob_range,
                                candles_held=candles_held,
                                resolution=TradeResolution.HORIZON_EXPIRATION,
                                realized_r=float("nan"),
                                is_win=False,
                                is_loss=False,
                                is_ambiguous=False,
                                is_horizon_exit=True,
                                track=track,
                                notes="no_resolution_within_horizon",
                            )
                            executed_trades.append(trade)
                        break

            # If order was never filled during Day T+1
            if not is_filled:
                order_records.append({
                    "order_id": order_id,
                    "ob_trading_day": ob_trading_day,
                    "candidate_timestamp": c_ts,
                    "status": OrderStatus.EXPIRED_UNFILLED.value,
                    "reason": "PRICE_DID_NOT_TOUCH_ENTRY",
                })
            else:
                order_records.append({
                    "order_id": order_id,
                    "ob_trading_day": ob_trading_day,
                    "candidate_timestamp": c_ts,
                    "status": OrderStatus.FILLED.value,
                    "filled_timestamp": filled_ts,
                    "filled_price": entry_p,
                })

        # Compute summary and breakdowns
        summary, eq_df, y_df, m_df = compute_execution_performance(
            trades=executed_trades,
            total_setups_count=total_setups_count,
            strategy_id=self.contract.strategy_id,
            strategy_name=self.contract.strategy_name,
            track=track,
        )

        trades_df = pd.DataFrame([t.__dict__ for t in executed_trades]) if executed_trades else pd.DataFrame()
        orders_df = pd.DataFrame(order_records) if order_records else pd.DataFrame()

        return SimulationResult(
            track=track,
            contract=self.contract,
            summary=summary,
            trades=executed_trades,
            trades_df=trades_df,
            equity_curve_df=eq_df,
            yearly_df=y_df,
            monthly_df=m_df,
            orders_df=orders_df,
        )

    def run_dual_track_simulation(self) -> tuple[SimulationResult, SimulationResult]:
        """
        Run both Track A (Conservative Execution) and Track B (Research Reconciliation)
        over the exact same historical data and Order Block population.
        """
        track_a_result = self.run_simulation(track="TRACK_A_CONSERVATIVE")
        track_b_result = self.run_simulation(track="TRACK_B_RESEARCH")
        return track_a_result, track_b_result
