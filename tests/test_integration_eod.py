# tests/test_execution/test_integration_eod.py
import unittest
import pandas as pd

from portwine.execution.base import ExecutionEngine
from portwine.execution.broker_base import Broker
from portwine.execution.data_live_base import LiveMarketDataLoader
from portwine.execution.portfolio import Order
from portwine.strategies.base import StrategyBase


# --------------------------------------------------------------------------- #
# Simple mocks for an end‑to‑end dry‑run
# --------------------------------------------------------------------------- #
class StratEqual(StrategyBase):
    def step(self, ts, bar_data):
        w = 1.0 / len(self.tickers)
        return {t: w for t in self.tickers}


class DataFixed(LiveMarketDataLoader):
    def __init__(self, price):
        super().__init__()
        self.price = price

    def next(self, tickers, timestamp=None):
        return {k: {"close": self.price, "open": self.price * 0.99} for k in tickers}


class BrokerMemory(Broker):
    """In‑memory broker that fills at fixed $100."""

    def __init__(self, cash):
        self._cash = cash
        self._pos = {}
        self.orders: list[Order] = []

    # --- Broker interface -------------------------------------------------- #
    def get_positions(self):
        return [
            type("Pos", (), dict(ticker=k, qty=v, avg_px=100.0))
            for k, v in self._pos.items()
        ]

    def get_cash(self):
        return self._cash

    def submit_orders(self, orders):
        self.orders.extend(orders)
        for od in orders:
            self._pos[od.ticker] = self._pos.get(od.ticker, 0) + od.qty
            self._cash -= od.qty * 100.0


# --------------------------------------------------------------------------- #
# Integration test
# --------------------------------------------------------------------------- #
class TestIntegrationEOD(unittest.TestCase):
    def test_full_cycle(self):
        strat = StratEqual(["AAA", "BBB"])
        data = DataFixed(price=100)
        broker = BrokerMemory(cash=100_000)

        eng = ExecutionEngine(strat, data, broker, cash_buffer=0.0)
        eng.step(pd.Timestamp("2025-04-18 16:00"))

        self.assertEqual(len(broker.orders), 2)
        self.assertEqual(sum(abs(o.qty) for o in broker.orders), 1000)  # 500 + 500


if __name__ == "__main__":
    unittest.main()
