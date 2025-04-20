import unittest
from datetime import timezone

from portwine.execution import ExecutionBase
from portwine.strategies.base import StrategyBase
from portwine.execution.brokers.base import Order


def make_exec_base(tickers):
    # only strategy.tickers matters for this helper method
    return ExecutionBase(
        strategy=StrategyBase(tickers),
        market_data_loader=None,
        broker=None,
        alternative_data_loader=None,
        timezone=timezone.utc
    )

class TestTargetPositionsToOrders(unittest.TestCase):
    def setUp(self):
        # exec_base not dependent on tickers for this method
        self.exec_base = make_exec_base([])

    def test_initial_buy_single(self):
        target = {'AAPL': 1000, 'MSFT': 0}
        current = {'AAPL': 0, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertIsInstance(order, Order)
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 1000.0)
        self.assertEqual(order.side, 'buy')

    def test_initial_buy_multiple(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 0, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 2)
        expected = [('AAPL', 1000.0, 'buy'), ('MSFT', 25.0, 'buy')]
        for exp, order in zip(expected, orders):
            ticker, qty, side = exp
            self.assertIsInstance(order, Order)
            self.assertEqual(order.ticker, ticker)
            self.assertEqual(order.quantity, qty)
            self.assertEqual(order.side, side)

    def test_add_on_buy_single(self):
        target = {'AAPL': 1000, 'MSFT': 0}
        current = {'AAPL': 900, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertIsInstance(order, Order)
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 100.0)
        self.assertEqual(order.side, 'buy')

    def test_add_on_buy_multiple(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 900, 'MSFT': 10}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 2)
        expected = [('AAPL', 100.0, 'buy'), ('MSFT', 15.0, 'buy')]
        for exp, order in zip(expected, orders):
            ticker, qty, side = exp
            self.assertIsInstance(order, Order)
            self.assertEqual(order.ticker, ticker)
            self.assertEqual(order.quantity, qty)
            self.assertEqual(order.side, side)

    def test_reduce_single(self):
        target = {'AAPL': 400, 'MSFT': 0}
        current = {'AAPL': 1000, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertIsInstance(order, Order)
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 600.0)
        self.assertEqual(order.side, 'sell')

    def test_reduce_multiple(self):
        target = {'AAPL': 400, 'MSFT': 10}
        current = {'AAPL': 900, 'MSFT': 50}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 2)
        expected = [('AAPL', 500.0, 'sell'), ('MSFT', 40.0, 'sell')]
        for exp, order in zip(expected, orders):
            ticker, qty, side = exp
            self.assertIsInstance(order, Order)
            self.assertEqual(order.ticker, ticker)
            self.assertEqual(order.quantity, qty)
            self.assertEqual(order.side, side)

    def test_mixed_add_and_reduce(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 500, 'MSFT': 50}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 2)
        expected = [('AAPL', 500.0, 'buy'), ('MSFT', 25.0, 'sell')]
        for exp, order in zip(expected, orders):
            ticker, qty, side = exp
            self.assertIsInstance(order, Order)
            self.assertEqual(order.ticker, ticker)
            self.assertEqual(order.quantity, qty)
            self.assertEqual(order.side, side)

    def test_initial_buy_and_add_on(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 0, 'MSFT': 10}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 2)
        expected = [('AAPL', 1000.0, 'buy'), ('MSFT', 15.0, 'buy')]
        for exp, order in zip(expected, orders):
            ticker, qty, side = exp
            self.assertIsInstance(order, Order)
            self.assertEqual(order.ticker, ticker)
            self.assertEqual(order.quantity, qty)
            self.assertEqual(order.side, side)

    def test_initial_buy_and_sell(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 0, 'MSFT': 50}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(len(orders), 2)
        expected = [('AAPL', 1000.0, 'buy'), ('MSFT', 25.0, 'sell')]
        for exp, order in zip(expected, orders):
            ticker, qty, side = exp
            self.assertIsInstance(order, Order)
            self.assertEqual(order.ticker, ticker)
            self.assertEqual(order.quantity, qty)
            self.assertEqual(order.side, side)

if __name__ == '__main__':
    unittest.main() 