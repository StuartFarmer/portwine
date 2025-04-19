# tests/test_execution/test_weight_diff.py
import unittest
import pandas as pd

from portwine.execution.base import ExecutionEngine
from portwine.execution.portfolio import Position
from portwine.execution.broker_base import Broker
from portwine.execution.data_live_base import LiveMarketDataLoader
from portwine.strategies.base import StrategyBase


# --------------------------------------------------------------------------- #
# Minimal mocks
# --------------------------------------------------------------------------- #
class MockStrategy(StrategyBase):
    def __init__(self, tickers, fixed_w):
        super().__init__(tickers)
        self._w = fixed_w

    def step(self, current_date, daily_data):
        return self._w


class MockData(LiveMarketDataLoader):
    def __init__(self, prices):
        super().__init__()
        self.prices = prices
        
    def next(self, tickers, ts=None):
        return {t: {"close": self.prices[t]} for t in tickers}




class MockBroker(Broker):
    def __init__(self, pos, cash):
        self._pos = pos
        self._cash = cash
        self.submitted = []

    # --- Broker interface -------------------------------------------------- #
    def get_positions(self):
        return list(self._pos.values())

    def get_cash(self):
        return self._cash

    def submit_orders(self, orders):
        self.submitted.extend(orders)


# --------------------------------------------------------------------------- #
# Unit‑tests
# --------------------------------------------------------------------------- #
class TestWeightDiff(unittest.TestCase):
    """
    Target = 50 % AAPL + 50 % MSFT, currently flat, price = $100.
    Cash = 10 000 → need 50 shares of each.
    """

    def test_diff_and_orders(self):
        tickers = ["AAPL", "MSFT"]
        prices = {"AAPL": 100, "MSFT": 100}
        strat = MockStrategy(tickers, {"AAPL": 0.5, "MSFT": 0.5})
        data = MockData(prices)
        broker = MockBroker(pos={}, cash=10_000)

        eng = ExecutionEngine(strat, data, broker, cash_buffer=0.0)

        orders = eng.step(pd.Timestamp("2025-04-18 16:00"))

        self.assertEqual(len(orders), 2)
        qtys = {o.ticker: o.qty for o in orders}
        self.assertEqual(qtys, {"AAPL": 50, "MSFT": 50})


if __name__ == "__main__":
    unittest.main()
