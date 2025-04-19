from __future__ import annotations
from datetime import time
from typing import Dict, List

import pandas as pd
from portwine.strategies.base import StrategyBase
from .broker_base import Broker
from .portfolio import Order
from portwine.loaders.base import MarketDataLoader


class ExecutionEngine:
    # ---------- unchanged ctor ----------------------------------------- #
    def __init__(
        self,
        strategy: StrategyBase,
        data_loader: MarketDataLoader,   # <— any loader, live or historical
        broker: Broker,
        rebalance_time: time = time(16, 0),
        cash_buffer: float = 0.005,
    ):
        self.strategy = strategy
        self.data_loader = data_loader
        self.broker = broker
        self.rebalance_time = rebalance_time
        self.cash_buffer = cash_buffer

    # ---------- main loop --------------------------------------------- #
    def step(self, current_dt: pd.Timestamp):
        if current_dt.time() != self.rebalance_time:
            return []

        # 1) fetch bar(s) ------------------------------------------------ #
        bar_data = self.data_loader.next(self.strategy.tickers, current_dt)
        if not bar_data:
            raise RuntimeError("No data returned from loader")

        # 2) derive weights --------------------------------------------- #
        target_w = self._normalize_weights(
            self.strategy.step(current_dt, bar_data)
        )

        # 3) positions & cash ------------------------------------------- #
        positions = {p.ticker: p for p in self.broker.get_positions()}
        prices = {t: bd["close"] for t, bd in bar_data.items() if bd is not None}
        cash = self.broker.get_cash()
        port_val = sum(p.qty * prices.get(p.ticker, 0.0) for p in positions.values()) + cash

        # 4) desired qty ------------------------------------------------- #
        desired_qty = {
            t: int((w * (1 - self.cash_buffer) * port_val) / prices[t])
            for t, w in target_w.items()
            if t in prices
        }

        # 5) diff → orders ---------------------------------------------- #
        orders = self._build_orders(desired_qty, positions, current_dt)
        if orders:
            self.broker.submit_orders(orders)
        return orders

    # helper methods unchanged …
    # ------------------------------------------------------------------ #
    # Helpers (add at bottom of class)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
        """
        Rescale so |weights| sum to 1.  If all zeros, return zeros.
        """
        total = sum(abs(v) for v in w.values())
        if total == 0:
            return {t: 0.0 for t in w}
        return {t: v / total for t, v in w.items()}

    @staticmethod
    def _build_orders(
        desired_qty: Dict[str, int],
        current_pos: Dict[str, "Position"],
        ts: pd.Timestamp,
    ) -> List["Order"]:
        """
        Diff desired vs current and create signed market orders.
        """
        orders: List["Order"] = []
        for tkr, tgt in desired_qty.items():
            cur_qty = current_pos.get(tkr).qty if tkr in current_pos else 0
            diff = tgt - cur_qty
            if diff != 0:
                orders.append(Order(tkr, diff, "market", ts))
        return orders
