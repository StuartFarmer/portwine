"""
Light‑weight portfolio & order bookkeeping for live execution.

Pure‑Python and dependency‑free so it can be unit‑tested without API keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class Position:
    ticker: str
    qty: int                  # positive = long, negative = short
    avg_px: float             # volume‑weighted average entry price


@dataclass
class Order:
    ticker: str
    qty: int                  # +buy / ‑sell; 0 forbidden
    order_type: str           # "market" for v1
    submitted_at: pd.Timestamp


@dataclass
class Portfolio:
    """
    Keeps an internal view of positions & cash, updated by polling a Broker
    (or directly by unit tests using `apply_fills`).
    """

    cash: float = 0.0
    _pos: Dict[str, Position] = field(default_factory=dict)

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def positions(self) -> List[Position]:
        return list(self._pos.values())

    def get_position(self, ticker: str) -> Position | None:
        return self._pos.get(ticker)

    def total_value(self, prices: Dict[str, float]) -> float:
        """
        Portfolio market value = Σ(value of positions) + cash.
        Missing prices -> position valued at 0.
        """
        pos_val = sum(p.qty * prices.get(p.ticker, 0.0) for p in self._pos.values())
        return pos_val + self.cash

    # --------------------------------------------------------------------- #
    # Mutation helpers
    # --------------------------------------------------------------------- #
    def apply_fills(self, fills: Dict[str, tuple[int, float]]) -> None:
        """
        Update portfolio after broker fills.

        Parameters
        ----------
        fills : dict
            {ticker -> (qty_filled, fill_price)}
        """
        for tkr, (qty, price) in fills.items():
            if qty == 0:
                continue

            prev = self._pos.get(tkr)
            if prev is None:
                # new position ↦ avg_px = fill price
                self._pos[tkr] = Position(tkr, qty, price)
            else:
                new_qty = prev.qty + qty
                if new_qty == 0:
                    # Flat -> remove
                    del self._pos[tkr]
                else:
                    # VWAP update
                    prev_cost = prev.avg_px * prev.qty
                    txn_cost = price * qty
                    avg_px = (prev_cost + txn_cost) / new_qty
                    self._pos[tkr] = Position(tkr, new_qty, avg_px)

            # cash ‑= buy, += sell
            self.cash -= qty * price
