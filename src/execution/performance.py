from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import numpy as np

from src.execution.order import Trade, TradeResolution


@dataclass
class ExecutionPerformanceSummary:
    """
    Comprehensive performance metrics for an executed strategy backtest track.
    """
    strategy_id: str
    strategy_name: str
    track: str
    total_setups: int
    unfilled_orders: int
    filled_trades: int
    fill_rate_pct: float
    take_profit_trades: int
    stop_loss_trades: int
    ambiguous_trades: int
    horizon_exit_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_filled_pct: float     # Denominator: filled_trades
    win_rate_total_setups_pct: float # Denominator: total_setups
    total_realized_r: float
    avg_trade_r: float
    candidate_ev_r: float          # total_realized_r / total_setups
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown_r: float
    clean_resolved_trades: int     # For Track B reconciliation (TP + SL)
    clean_win_rate_pct: float      # For Track B reconciliation (TP / (TP + SL))


def compute_execution_performance(
    trades: list[Trade],
    total_setups_count: int,
    strategy_id: str,
    strategy_name: str,
    track: str,
) -> tuple[ExecutionPerformanceSummary, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute execution performance metrics, cumulative equity curve, yearly performance,
    and monthly performance tables.

    Returns
    -------
    tuple of (summary, equity_curve_df, yearly_df, monthly_df)
    """
    if not trades:
        empty_summary = ExecutionPerformanceSummary(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            track=track,
            total_setups=total_setups_count,
            unfilled_orders=total_setups_count,
            filled_trades=0,
            fill_rate_pct=0.0,
            take_profit_trades=0,
            stop_loss_trades=0,
            ambiguous_trades=0,
            horizon_exit_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_filled_pct=0.0,
            win_rate_total_setups_pct=0.0,
            total_realized_r=0.0,
            avg_trade_r=0.0,
            candidate_ev_r=0.0,
            profit_factor=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            max_drawdown_r=0.0,
            clean_resolved_trades=0,
            clean_win_rate_pct=0.0,
        )
        return empty_summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Convert trades to DataFrame
    records: list[dict[str, Any]] = []
    for t in trades:
        records.append({
            "trade_id": t.trade_id,
            "order_id": t.order_id,
            "strategy_id": t.strategy_id,
            "strategy_name": t.strategy_name,
            "ob_trading_day": t.ob_trading_day,
            "candidate_timestamp": t.candidate_timestamp,
            "side": t.side.value,
            "entry_price": t.entry_price,
            "entry_timestamp": t.entry_timestamp,
            "exit_price": t.exit_price,
            "exit_timestamp": t.exit_timestamp,
            "ob_range": t.ob_range,
            "candles_held": t.candles_held,
            "resolution": t.resolution.value if hasattr(t.resolution, "value") else str(t.resolution),
            "realized_r": t.realized_r,
            "is_win": t.is_win,
            "is_loss": t.is_loss,
            "is_ambiguous": t.is_ambiguous,
            "is_horizon_exit": t.is_horizon_exit,
            "track": t.track,
        })
    trades_df = pd.DataFrame(records).sort_values(by="entry_timestamp").reset_index(drop=True)

    # Calculate metrics
    filled_count = len(trades_df)
    unfilled_count = total_setups_count - filled_count
    fill_rate = float(round((filled_count / total_setups_count) * 100.0, 2)) if total_setups_count > 0 else 0.0

    tp_count = int((trades_df["resolution"] == TradeResolution.TAKE_PROFIT.value).sum())
    sl_count = int((trades_df["resolution"] == TradeResolution.STOP_LOSS.value).sum())
    ambig_count = int((trades_df["resolution"] == TradeResolution.AMBIGUOUS_STOPPED_OUT.value).sum())
    horizon_count = int((trades_df["resolution"] == TradeResolution.HORIZON_EXPIRATION.value).sum())

    # Wins / Losses in Track A (fractional R > 0 counts as win, < 0 as loss)
    win_mask = trades_df["realized_r"] > 0.0
    loss_mask = trades_df["realized_r"] < 0.0
    win_count = int(win_mask.sum())
    loss_count = int(loss_mask.sum())

    win_rate_filled = float(round((win_count / filled_count) * 100.0, 2)) if filled_count > 0 else 0.0
    win_rate_total = float(round((win_count / total_setups_count) * 100.0, 2)) if total_setups_count > 0 else 0.0

    # Clean metrics for Track B reconciliation
    clean_res = tp_count + sl_count
    clean_wr = float(round((tp_count / clean_res) * 100.0, 2)) if clean_res > 0 else 0.0

    valid_r = trades_df["realized_r"].dropna()
    total_r = float(round(valid_r.sum(), 4))
    avg_r = float(round(valid_r.mean(), 4)) if len(valid_r) > 0 else 0.0
    cand_ev = float(round(total_r / total_setups_count, 4)) if total_setups_count > 0 else 0.0

    gross_gains = float(valid_r[valid_r > 0].sum())
    gross_losses = float(abs(valid_r[valid_r < 0].sum()))
    profit_factor = float(round(gross_gains / gross_losses, 2)) if gross_losses > 0 else float("inf")

    # Sequential Equity Curve & Drawdown
    cum_r = 0.0
    peak_r = 0.0
    max_dd = 0.0
    cur_w_streak, max_w_streak = 0, 0
    cur_l_streak, max_l_streak = 0, 0
    equity_records: list[dict[str, Any]] = []

    for idx, row in trades_df.iterrows():
        r = row["realized_r"]
        if pd.isna(r):
            continue

        if r > 0:
            cur_w_streak += 1
            cur_l_streak = 0
            if cur_w_streak > max_w_streak:
                max_w_streak = cur_w_streak
        elif r < 0:
            cur_l_streak += 1
            cur_w_streak = 0
            if cur_l_streak > max_l_streak:
                max_l_streak = cur_l_streak

        cum_r += r
        if cum_r > peak_r:
            peak_r = cum_r
        dd = peak_r - cum_r
        if dd > max_dd:
            max_dd = dd

        equity_records.append({
            "trade_num": idx + 1,
            "entry_timestamp": row["entry_timestamp"],
            "exit_timestamp": row["exit_timestamp"],
            "resolution": row["resolution"],
            "return_r": round(r, 4),
            "cumulative_r": round(cum_r, 4),
            "drawdown_r": round(dd, 4),
        })

    equity_df = pd.DataFrame(equity_records)
    max_dd_r = float(round(max_dd, 4))

    # Yearly Breakdown
    entry_ts = pd.to_datetime(trades_df["entry_timestamp"], utc=True)
    trades_df["year"] = entry_ts.dt.year
    trades_df["year_month"] = entry_ts.dt.strftime("%Y-%m")

    yearly_records: list[dict[str, Any]] = []
    for yr, y_sub in trades_df.groupby("year"):
        y_valid = y_sub["realized_r"].dropna()
        y_wins = int((y_sub["realized_r"] > 0).sum())
        y_losses = int((y_sub["realized_r"] < 0).sum())
        y_trades = len(y_valid)
        y_wr = float(round((y_wins / y_trades) * 100.0, 2)) if y_trades > 0 else 0.0
        y_net_r = float(round(y_valid.sum(), 4))
        yearly_records.append({
            "strategy_id": strategy_id,
            "year": yr,
            "filled_trades": y_trades,
            "wins": y_wins,
            "losses": y_losses,
            "win_rate_pct": y_wr,
            "net_realized_r": y_net_r,
        })
    yearly_df = pd.DataFrame(yearly_records)

    # Monthly Breakdown
    monthly_records: list[dict[str, Any]] = []
    for ym, m_sub in trades_df.groupby("year_month"):
        m_valid = m_sub["realized_r"].dropna()
        m_wins = int((m_sub["realized_r"] > 0).sum())
        m_losses = int((m_sub["realized_r"] < 0).sum())
        m_trades = len(m_valid)
        m_wr = float(round((m_wins / m_trades) * 100.0, 2)) if m_trades > 0 else 0.0
        m_net_r = float(round(m_valid.sum(), 4))
        monthly_records.append({
            "strategy_id": strategy_id,
            "year_month": ym,
            "filled_trades": m_trades,
            "wins": m_wins,
            "losses": m_losses,
            "win_rate_pct": m_wr,
            "net_realized_r": m_net_r,
        })
    monthly_df = pd.DataFrame(monthly_records)

    summary = ExecutionPerformanceSummary(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        track=track,
        total_setups=total_setups_count,
        unfilled_orders=unfilled_count,
        filled_trades=filled_count,
        fill_rate_pct=fill_rate,
        take_profit_trades=tp_count,
        stop_loss_trades=sl_count,
        ambiguous_trades=ambig_count,
        horizon_exit_trades=horizon_count,
        winning_trades=win_count,
        losing_trades=loss_count,
        win_rate_filled_pct=win_rate_filled,
        win_rate_total_setups_pct=win_rate_total,
        total_realized_r=total_r,
        avg_trade_r=avg_r,
        candidate_ev_r=cand_ev,
        profit_factor=profit_factor,
        max_consecutive_wins=max_w_streak,
        max_consecutive_losses=max_l_streak,
        max_drawdown_r=max_dd_r,
        clean_resolved_trades=clean_res,
        clean_win_rate_pct=clean_wr,
    )

    return summary, equity_df, yearly_df, monthly_df
