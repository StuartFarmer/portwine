import unittest
import time
from datetime import datetime
from typing import Dict, List

from portwine.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderNotFoundError,
    OrderExecutionError,
)


class MockBroker(BrokerBase):
    """In-memory mock broker conforming to the new BrokerBase API."""

    def __init__(self, initial_equity: float = 100000.0, fill_price: float = 100.0):
        self._equity = initial_equity
        self._fill_price = fill_price
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._next_order_id = 1

    def get_account(self) -> Account:
        ts = int(time.time())
        return Account(equity=self._equity, last_updated_at=ts)

    def get_positions(self) -> Dict[str, Position]:
        # Return a shallow copy so tests can't mutate internals
        return self._positions.copy()

    def get_position(self, ticker: str) -> Position:
        if ticker in self._positions:
            return self._positions[ticker]
        # default zero‐qty position
        return Position(ticker=ticker, quantity=0.0, last_updated_at=int(time.time()))

    def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order {order_id} not found")
        return self._orders[order_id]

    def get_orders(self) -> List[Order]:
        return list(self._orders.values())

    def submit_order(self, symbol: str, quantity: float) -> Order:
        oid = str(self._next_order_id)
        self._next_order_id += 1

        now_s = int(time.time())
        now_ms = int(time.time() * 1000)

        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)

        order = Order(
            order_id=oid,
            ticker=symbol,
            side=side,
            quantity=qty,
            order_type="market",
            status="filled",
            time_in_force="day",
            average_price=self._fill_price,
            remaining_quantity=0.0,
            created_at=now_ms,
            last_updated_at=now_s,
        )
        self._orders[oid] = order

        # update position
        prev_qty = self._positions.get(symbol, Position(symbol, 0.0, now_s)).quantity
        new_qty = prev_qty + quantity
        self._positions[symbol] = Position(ticker=symbol, quantity=new_qty, last_updated_at=now_s)

        return order

    def market_is_open(self, timestamp: datetime) -> bool:
        # Simplest stub: markets open Monday–Friday 9:30–16:00 UTC
        # For testing, return True if weekday < 5
        return timestamp.weekday() < 5


class TestMockBroker(unittest.TestCase):
    def setUp(self):
        self.broker = MockBroker(initial_equity=50_000.0, fill_price=123.45)

    def test_get_account(self):
        acct = self.broker.get_account()
        self.assertIsInstance(acct, Account)
        self.assertEqual(acct.equity, 50_000.0)
        self.assertIsInstance(acct.last_updated_at, int)
        # last_updated_at should be very recent
        self.assertAlmostEqual(acct.last_updated_at, int(time.time()), delta=2)

    def test_positions_initially_empty(self):
        self.assertEqual(self.broker.get_positions(), {})
        pos = self.broker.get_position("FOO")
        self.assertIsInstance(pos, Position)
        self.assertEqual(pos.ticker, "FOO")
        self.assertEqual(pos.quantity, 0.0)
        self.assertIsInstance(pos.last_updated_at, int)

    def test_submit_order_and_position_update(self):
        order = self.broker.submit_order("BAR", 10)
        # Order fields
        self.assertIsInstance(order, Order)
        self.assertEqual(order.order_id, "1")
        self.assertEqual(order.ticker, "BAR")
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.quantity, 10.0)
        self.assertEqual(order.order_type, "market")
        self.assertEqual(order.status, "filled")
        self.assertEqual(order.time_in_force, "day")
        self.assertEqual(order.average_price, 123.45)
        self.assertEqual(order.remaining_quantity, 0.0)
        self.assertIsInstance(order.created_at, int)
        self.assertIsInstance(order.last_updated_at, int)

        # Position was updated
        pos = self.broker.get_position("BAR")
        self.assertEqual(pos.quantity, 10.0)
        self.assertEqual(pos.last_updated_at, order.last_updated_at)

    def test_multiple_orders_and_get_orders(self):
        o1 = self.broker.submit_order("X", 1)
        o2 = self.broker.submit_order("X", -2)
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 2)
        self.assertIn(o1, orders)
        self.assertIn(o2, orders)

    def test_get_order_by_id(self):
        o = self.broker.submit_order("Z", 3)
        fetched = self.broker.get_order(o.order_id)
        self.assertEqual(fetched, o)

    def test_get_order_not_found_raises(self):
        with self.assertRaises(OrderNotFoundError):
            self.broker.get_order("999")

    def test_market_is_open_weekday(self):
        # Monday
        monday = datetime(2025, 4, 21, 12, 0, 0)
        self.assertTrue(self.broker.market_is_open(monday))
        # Sunday
        sunday = datetime(2025, 4, 20, 12, 0, 0)
        self.assertFalse(self.broker.market_is_open(sunday))


if __name__ == "__main__":
    unittest.main()
