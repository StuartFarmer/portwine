import unittest
from datetime import timezone

from portwine.execution.base import ExecutionBase
from portwine.strategies.base import StrategyBase


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
        self.assertEqual(orders, [['AAPL', '1000', 'buy']])

    def test_initial_buy_multiple(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 0, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(
            orders,
            [['AAPL', '1000', 'buy'], ['MSFT', '25', 'buy']]
        )

    def test_add_on_buy_single(self):
        target = {'AAPL': 1000, 'MSFT': 0}
        current = {'AAPL': 900, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(orders, [['AAPL', '100', 'buy']])

    def test_add_on_buy_multiple(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 900, 'MSFT': 10}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(
            orders,
            [['AAPL', '100', 'buy'], ['MSFT', '15', 'buy']]
        )

    def test_reduce_single(self):
        target = {'AAPL': 400, 'MSFT': 0}
        current = {'AAPL': 1000, 'MSFT': 0}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(orders, [['AAPL', '600', 'sell']])

    def test_reduce_multiple(self):
        target = {'AAPL': 400, 'MSFT': 10}
        current = {'AAPL': 900, 'MSFT': 50}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(
            orders,
            [['AAPL', '500', 'sell'], ['MSFT', '40', 'sell']]
        )

    def test_mixed_add_and_reduce(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 500, 'MSFT': 50}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(
            orders,
            [['AAPL', '500', 'buy'], ['MSFT', '25', 'sell']]
        )

    def test_initial_buy_and_add_on(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 0, 'MSFT': 10}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(
            orders,
            [['AAPL', '1000', 'buy'], ['MSFT', '15', 'buy']]
        )

    def test_initial_buy_and_sell(self):
        target = {'AAPL': 1000, 'MSFT': 25}
        current = {'AAPL': 0, 'MSFT': 50}
        orders = self.exec_base._target_positions_to_orders(target, current)
        self.assertEqual(
            orders,
            [['AAPL', '1000', 'buy'], ['MSFT', '25', 'sell']]
        )

if __name__ == '__main__':
    unittest.main() 