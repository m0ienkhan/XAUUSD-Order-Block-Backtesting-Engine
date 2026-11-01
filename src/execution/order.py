from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED_UNFILLED = "EXPIRED_UNFILLED"


class TradeResolution(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    AMBIGUOUS_STOPPED_OUT = "AMBIGUOUS_STOPPED_OUT"  # Same 15m candle SL/TP resolved as SL in Track A
    HORIZON_EXPIRATION = "HORIZON_EXPIRATION"        # Position closed at final candle of Day T+1 with fractional R
    OPEN = "OPEN"


@dataclass
class Order:
    """
    Representation of an order placed by the execution engine.
    """
    order_id: str
    strategy_id: str
    candidate_identity: tuple[Any, int, Any, str]  # (trading_day, candidate_position, candidate_timestamp, direction)
    ob_trading_day: object
    candidate_timestamp: pd.Timestamp
    side: OrderSide
    order_type: OrderType
    limit_price: float
    stop_loss_price: float
    take_profit_price: float
    ob_range: float
    created_timestamp: pd.Timestamp
    valid_from_timestamp: pd.Timestamp
    expiration_timestamp: Optional[pd.Timestamp]
    status: OrderStatus = OrderStatus.PENDING
    filled_timestamp: Optional[pd.Timestamp] = None
    filled_price: Optional[float] = None


@dataclass
class Position:
    """
    Representation of an active open position in the market.
    """
    position_id: str
    order_id: str
    strategy_id: str
    candidate_identity: tuple[Any, int, Any, str]
    ob_trading_day: object
    side: OrderSide
    entry_price: float
    entry_timestamp: pd.Timestamp
    stop_loss_price: float
    take_profit_price: float
    ob_range: float
    current_candles_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0


@dataclass
class Trade:
    """
    Immutable representation of a completed, closed trade with full realized R attribution.
    """
    trade_id: str
    order_id: str
    strategy_id: str
    strategy_name: str
    candidate_identity: tuple[Any, int, Any, str]
    ob_trading_day: object
    candidate_timestamp: pd.Timestamp
    side: OrderSide
    entry_price: float
    entry_timestamp: pd.Timestamp
    exit_price: float
    exit_timestamp: pd.Timestamp
    stop_loss_price: float
    take_profit_price: float
    ob_range: float
    candles_held: int
    resolution: TradeResolution
    realized_r: float
    is_win: bool
    is_loss: bool
    is_ambiguous: bool
    is_horizon_exit: bool
    track: str  # "TRACK_A_CONSERVATIVE" or "TRACK_B_RESEARCH"
    notes: str = ""
