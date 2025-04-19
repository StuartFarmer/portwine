"""
Concrete broker adaptor for Alpaca REST.

Requires `pip install alpaca‑trade‑api`.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "broker_alpaca requires `alpaca‑trade‑api` ‑ install via "
        "`pip install alpaca‑trade‑api`"
    ) from _e

from .broker_base import Broker
from .portfolio import Order, Position


class AlpacaBroker(Broker):
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        paper: bool = True,
    ):
        base_url = (
            base_url
            or ("https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets")
        )
        self._api = tradeapi.REST(
            api_key or os.getenv("APCA_API_KEY_ID"),
            api_secret or os.getenv("APCA_API_SECRET_KEY"),
            base_url,
            api_version="v2",
        )

    # ------------------------------------------------------------------ #
    # Broker interface
    # ------------------------------------------------------------------ #
    def get_positions(self) -> List[Position]:
        alpaca_pos = self._api.list_positions()
        return [
            Position(
                p.symbol,
                int(float(p.qty)),
                float(p.avg_entry_price),
            )
            for p in alpaca_pos
        ]

    def get_cash(self) -> float:
        acct = self._api.get_account()
        return float(acct.cash)

    def submit_orders(self, orders: List[Order]) -> None:
        for od in orders:
            side = "buy" if od.qty > 0 else "sell"
            self._api.submit_order(
                symbol=od.ticker,
                qty=abs(od.qty),
                side=side,
                type="market",
                time_in_force="day",
            )
            # no explicit handling of order ids yet
