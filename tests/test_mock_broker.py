import unittest
from datetime import datetime
from portwine.brokers.base import Position, OrderNotFoundError, Order
from portwine.brokers.mock import MockBroker


class TestMockBroker(unittest.TestCase):
    def setUp(self):
        # Fresh broker before each test
        self.broker = MockBroker()

    def test_initial_account_default_equity(self):
        account = self.broker.get_account()
        self.assertEqual(account.equity, 1_000_000.0)

    def test_initial_account_custom_equity(self):
        custom = MockBroker(initial_equity=50_000.0)
        account = custom.get_account()
        self.assertEqual(account.equity, 50_000.0)

    def test_initial_positions_and_orders_empty(self):
        self.assertEqual(self.broker.get_positions(), {})
        self.assertEqual(self.broker.get_orders(), [])

    def test_get_position_not_held(self):
        pos = self.broker.get_position("AAPL")
        self.assertIsInstance(pos, Position)
        self.assertEqual(pos.ticker, "AAPL")
        self.assertEqual(pos.quantity, 0.0)

    def test_submit_buy_updates_position_and_returns_order(self):
        order = self.broker.submit_order("AAPL", 5)
        # Order attributes
        self.assertIsInstance(order, Order)
        self.assertEqual(order.ticker, "AAPL")
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.quantity, 5.0)
        self.assertEqual(order.order_type, "market")
        self.assertEqual(order.status, "filled")
        self.assertEqual(order.time_in_force, "day")
        self.assertEqual(order.remaining_quantity, 0.0)
        self.assertIsInstance(order.average_price, float)
        self.assertIsInstance(order.created_at, int)
        self.assertEqual(order.created_at, order.last_updated_at)
        # Position updated
        pos = self.broker.get_position("AAPL")
        self.assertEqual(pos.quantity, 5.0)
        # Orders list and lookup
        orders = self.broker.get_orders()
        self.assertIn(order, orders)
        fetched = self.broker.get_order(order.order_id)
        self.assertEqual(fetched, order)

    def test_submit_sell_without_prior_position(self):
        # Selling without a prior position yields a negative quantity
        order = self.broker.submit_order("TSLA", -10)
        self.assertEqual(order.side, "sell")
        pos = self.broker.get_position("TSLA")
        self.assertEqual(pos.quantity, -10.0)

    def test_buy_then_partial_sell(self):
        # Buy first
        buy = self.broker.submit_order("MSFT", 10)
        self.assertEqual(self.broker.get_position("MSFT").quantity, 10.0)
        # Then sell part
        sell = self.broker.submit_order("MSFT", -4)
        self.assertEqual(self.broker.get_position("MSFT").quantity, 6.0)
        self.assertEqual(sell.side, "sell")
        # Position still exists
        self.assertIn("MSFT", self.broker.get_positions())

    def test_positions_removal_when_zero(self):
        # Buy and then sell exactly to zero
        self.broker.submit_order("GOOG", 7)
        self.broker.submit_order("GOOG", -7)
        self.assertEqual(self.broker.get_position("GOOG").quantity, 0.0)
        self.assertNotIn("GOOG", self.broker.get_positions())

    def test_multiple_orders_unique_ids(self):
        o1 = self.broker.submit_order("X", 1)
        o2 = self.broker.submit_order("Y", 2)
        o3 = self.broker.submit_order("Z", 3)
        self.assertNotEqual(o1.order_id, o2.order_id)
        self.assertNotEqual(o2.order_id, o3.order_id)
        self.assertEqual([o.order_id for o in self.broker.get_orders()], [o1.order_id, o2.order_id, o3.order_id])

    def test_get_order_not_found_raises(self):
        with self.assertRaises(OrderNotFoundError):
            self.broker.get_order("nonexistent")

    def test_order_timestamps_are_equal_and_datetime(self):
        order = self.broker.submit_order("NFLX", 2)
        self.assertIsInstance(order.created_at, int)
        self.assertIsInstance(order.last_updated_at, int)
        self.assertEqual(order.created_at, order.last_updated_at)


if __name__ == "__main__":
    unittest.main()
