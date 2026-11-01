from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass(frozen=True)
class TradeCostBreakdown:
    """
    Immutable structured result capturing trade-level deterministic execution friction
    and net realized return attribution.
    """
    gross_realized_r: float
    spread_cost_r: float
    slippage_cost_r: float
    commission_cost_r: float
    total_cost_r: float
    net_realized_r: float
    ob_range: float
    effective_entry_price: Optional[float] = None
    effective_exit_price: Optional[float] = None

    def __post_init__(self) -> None:
        # Enforce exact identity: net_realized_r must equal gross_realized_r - total_cost_r
        expected_net = self.gross_realized_r - self.total_cost_r
        if not math.isclose(self.net_realized_r, expected_net, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Net R inconsistency: net_realized_r ({self.net_realized_r}) != "
                f"gross_realized_r ({self.gross_realized_r}) - total_cost_r ({self.total_cost_r})"
            )


@dataclass(frozen=True)
class ExecutionCostModel:
    """
    Deterministic transaction cost model for post-execution trade attribution.

    Parameters
    ----------
    spread_usd : float, default=0.0
        Total round-trip bid/ask spread cost in USD price movement terms.
        spread_cost_r = spread_usd / ob_range.
    entry_slippage_usd : float, default=0.0
        Adverse price movement on entry fill in USD.
    exit_slippage_usd : float, default=0.0
        Adverse price movement on exit fill in USD.
        slippage_cost_r = (entry_slippage_usd + exit_slippage_usd) / ob_range.
    commission_usd : float, default=0.0
        Total round-trip commission expressed as a USD-equivalent price cost.
        commission_cost_r = (commission_usd / ob_range) + commission_r_fixed.
    commission_r_fixed : float, default=0.0
        Fixed round-trip commission cost expressed directly in R units.
    """
    spread_usd: float = 0.0
    entry_slippage_usd: float = 0.0
    exit_slippage_usd: float = 0.0
    commission_usd: float = 0.0
    commission_r_fixed: float = 0.0
    commission_usd_per_oz: Optional[float] = None

    def __post_init__(self) -> None:
        # Handle backward-compatible alias commission_usd_per_oz
        if self.commission_usd_per_oz is not None and self.commission_usd == 0.0:
            object.__setattr__(self, "commission_usd", float(self.commission_usd_per_oz))
        elif self.commission_usd_per_oz is None:
            object.__setattr__(self, "commission_usd_per_oz", float(self.commission_usd))

        # Validate spread
        if math.isnan(self.spread_usd) or math.isinf(self.spread_usd) or self.spread_usd < 0.0:
            raise ValueError(f"spread_usd must be a non-negative finite float, got: {self.spread_usd}")

        # Validate entry slippage
        if math.isnan(self.entry_slippage_usd) or math.isinf(self.entry_slippage_usd) or self.entry_slippage_usd < 0.0:
            raise ValueError(f"entry_slippage_usd must be a non-negative finite float, got: {self.entry_slippage_usd}")

        # Validate exit slippage
        if math.isnan(self.exit_slippage_usd) or math.isinf(self.exit_slippage_usd) or self.exit_slippage_usd < 0.0:
            raise ValueError(f"exit_slippage_usd must be a non-negative finite float, got: {self.exit_slippage_usd}")

        # Validate commission USD
        if math.isnan(self.commission_usd) or math.isinf(self.commission_usd) or self.commission_usd < 0.0:
            raise ValueError(f"commission_usd must be a non-negative finite float, got: {self.commission_usd}")

        # Validate fixed commission R
        if math.isnan(self.commission_r_fixed) or math.isinf(self.commission_r_fixed) or self.commission_r_fixed < 0.0:
            raise ValueError(f"commission_r_fixed must be a non-negative finite float, got: {self.commission_r_fixed}")

    @property
    def total_roundtrip_price_friction_usd(self) -> float:
        """Total round-trip USD price friction (spread + entry slippage + exit slippage)."""
        return self.spread_usd + self.entry_slippage_usd + self.exit_slippage_usd

    @property
    def total_friction_usd(self) -> float:
        """Total round-trip price and commission friction in USD terms."""
        return self.total_roundtrip_price_friction_usd + self.commission_usd

    def calculate_effective_entry_price(
        self,
        direction: str,
        nominal_entry_price: float,
    ) -> float:
        """
        Compute informational/audit effective entry price with directional half-spread and slippage.
        - Bearish (Short / Sell): Effective entry is lower (worse fill for seller).
        - Bullish (Long / Buy): Effective entry is higher (worse fill for buyer).
        """
        if math.isnan(nominal_entry_price) or math.isinf(nominal_entry_price):
            raise ValueError(f"nominal_entry_price must be a finite float, got: {nominal_entry_price}")

        half_spread = self.spread_usd / 2.0
        dir_clean = str(direction).lower()
        if dir_clean in ("bearish", "sell"):
            return float(round(nominal_entry_price - half_spread - self.entry_slippage_usd, 6))
        elif dir_clean in ("bullish", "buy"):
            return float(round(nominal_entry_price + half_spread + self.entry_slippage_usd, 6))
        else:
            raise ValueError(f"Unsupported direction for effective price calculation: {direction}")

    def calculate_effective_exit_price(
        self,
        direction: str,
        nominal_exit_price: float,
    ) -> float:
        """
        Compute informational/audit effective exit price with directional half-spread and slippage.
        - Bearish (Short Exit / Buy to Cover): Effective exit is higher (worse cover price).
        - Bullish (Long Exit / Sell to Close): Effective exit is lower (worse sell price).
        """
        if math.isnan(nominal_exit_price) or math.isinf(nominal_exit_price):
            raise ValueError(f"nominal_exit_price must be a finite float, got: {nominal_exit_price}")

        half_spread = self.spread_usd / 2.0
        dir_clean = str(direction).lower()
        if dir_clean in ("bearish", "sell"):
            return float(round(nominal_exit_price + half_spread + self.exit_slippage_usd, 6))
        elif dir_clean in ("bullish", "buy"):
            return float(round(nominal_exit_price - half_spread - self.exit_slippage_usd, 6))
        else:
            raise ValueError(f"Unsupported direction for effective price calculation: {direction}")

    def compute_trade_cost_breakdown(
        self,
        gross_realized_r: float,
        ob_range: float,
        direction: Optional[str] = None,
        nominal_entry_price: Optional[float] = None,
        nominal_exit_price: Optional[float] = None,
    ) -> TradeCostBreakdown:
        """
        Calculate deterministic execution cost breakdown and net realized R for a trade.

        Parameters
        ----------
        gross_realized_r : float
            Gross realized return produced by nominal trade path execution.
        ob_range : float
            Strategy Order Block range in USD (must be strictly positive and finite).
        direction : str, optional
            'bearish' or 'bullish' for computing informational effective prices.
        nominal_entry_price : float, optional
            Nominal execution entry price (informational only).
        nominal_exit_price : float, optional
            Nominal execution exit price (informational only).

        Returns
        -------
        TradeCostBreakdown
            Immutable result container with component cost breakdown and net realized R.
        """
        if math.isnan(ob_range) or math.isinf(ob_range) or ob_range <= 0.0:
            raise ValueError(f"ob_range must be a strictly positive finite float, got: {ob_range}")

        if math.isnan(gross_realized_r) or math.isinf(gross_realized_r):
            raise ValueError(f"gross_realized_r must be a finite float, got: {gross_realized_r}")

        # Deterministic cost conversion to R-units
        spread_cost_r = self.spread_usd / ob_range
        slippage_cost_r = (self.entry_slippage_usd + self.exit_slippage_usd) / ob_range
        commission_cost_r = (self.commission_usd / ob_range) + self.commission_r_fixed
        total_cost_r = spread_cost_r + slippage_cost_r + commission_cost_r
        net_realized_r = gross_realized_r - total_cost_r

        # Informational effective prices
        effective_entry: Optional[float] = None
        effective_exit: Optional[float] = None
        if direction is not None and nominal_entry_price is not None:
            effective_entry = self.calculate_effective_entry_price(direction, nominal_entry_price)
        if direction is not None and nominal_exit_price is not None:
            effective_exit = self.calculate_effective_exit_price(direction, nominal_exit_price)

        return TradeCostBreakdown(
            gross_realized_r=float(gross_realized_r),
            spread_cost_r=float(spread_cost_r),
            slippage_cost_r=float(slippage_cost_r),
            commission_cost_r=float(commission_cost_r),
            total_cost_r=float(total_cost_r),
            net_realized_r=float(net_realized_r),
            ob_range=float(ob_range),
            effective_entry_price=effective_entry,
            effective_exit_price=effective_exit,
        )

    def compute_trade_cost_attribution(
        self,
        direction: str,
        nominal_entry_price: float,
        nominal_exit_price: float,
        ob_range: float,
    ) -> dict[str, float]:
        """
        Helper method providing dictionary-based trade cost attribution for compatibility.
        """
        safe_range = max(float(ob_range), 1e-8)
        dir_clean = str(direction).lower()
        if dir_clean in ("bearish", "sell"):
            gross_r = (nominal_entry_price - nominal_exit_price) / safe_range
        else:
            gross_r = (nominal_exit_price - nominal_entry_price) / safe_range

        b = self.compute_trade_cost_breakdown(
            gross_realized_r=gross_r,
            ob_range=safe_range,
            direction=direction,
            nominal_entry_price=nominal_entry_price,
            nominal_exit_price=nominal_exit_price,
        )

        return {
            "effective_entry_price": float(round(b.effective_entry_price, 4)) if b.effective_entry_price is not None else 0.0,
            "effective_exit_price": float(round(b.effective_exit_price, 4)) if b.effective_exit_price is not None else 0.0,
            "gross_realized_r": float(round(b.gross_realized_r, 4)),
            "spread_cost_r": float(round(b.spread_cost_r, 4)),
            "slippage_cost_r": float(round(b.slippage_cost_r, 4)),
            "commission_cost_r": float(round(b.commission_cost_r, 4)),
            "total_cost_r": float(round(b.total_cost_r, 4)),
            "net_realized_r": float(round(b.net_realized_r, 4)),
        }


def compute_trade_cost_breakdown(
    cost_model: ExecutionCostModel,
    gross_realized_r: float,
    ob_range: float,
    direction: Optional[str] = None,
    nominal_entry_price: Optional[float] = None,
    nominal_exit_price: Optional[float] = None,
) -> TradeCostBreakdown:
    """
    Public utility function to compute deterministic trade cost breakdown.
    """
    return cost_model.compute_trade_cost_breakdown(
        gross_realized_r=gross_realized_r,
        ob_range=ob_range,
        direction=direction,
        nominal_entry_price=nominal_entry_price,
        nominal_exit_price=nominal_exit_price,
    )
