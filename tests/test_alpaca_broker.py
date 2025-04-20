import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from portwine.brokers.alpaca import AlpacaBroker, _parse_datetime
from portwine.brokers.base import OrderExecutionError, OrderNotFoundError, Position, Order


class DummyResponse:
    def __init__(self, ok=True, status_code=200, json_data=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


class TestAlpacaBroker(unittest.TestCase):
    def setUp(self):
        # Create broker and override its HTTP session with a MagicMock
        self.broker = AlpacaBroker("key", "secret", base_url="https://api.example.com")
        self.session = MagicMock()
        self.broker._session = self.session

    def test_get_account_success(self):
        # Mock a successful account fetch
        resp = DummyResponse(ok=True, status_code=200, json_data={"equity": "15000"})
        self.session.get.return_value = resp
        account = self.broker.get_account()
        self.assertEqual(account.equity, 15000.0)

    def test_get_account_failure(self):
        # Mock a failed account fetch
        resp = DummyResponse(ok=False, status_code=500, json_data={"error": "fail"}, text="fail")
        self.session.get.return_value = resp
        with self.assertRaises(OrderExecutionError):
            self.broker.get_account()

    def test_get_positions_success(self):
        # Mock positions list
        data = [
            {"symbol": "AAPL", "qty": "10"},
            {"symbol": "TSLA", "qty": "5.5"}
        ]
        self.session.get.return_value = DummyResponse(ok=True, status_code=200, json_data=data)
        positions = self.broker.get_positions()
        self.assertIsInstance(positions, dict)
        self.assertEqual(len(positions), 2)
        self.assertTrue(all(isinstance(p, Position) for p in positions.values()))
        self.assertEqual(positions["AAPL"].quantity, 10.0)
        self.assertEqual(positions["TSLA"].quantity, 5.5)

    def test_get_positions_failure(self):
        # Mock error on positions fetch
        resp = DummyResponse(ok=False, status_code=500, text="error")
        self.session.get.return_value = resp
        with self.assertRaises(OrderExecutionError):
            self.broker.get_positions()

    def test_get_position_exists(self):
        # Mock single position
        resp = DummyResponse(ok=True, status_code=200, json_data={"symbol": "AAPL", "qty": "7"})
        self.session.get.return_value = resp
        pos = self.broker.get_position("AAPL")
        self.assertEqual(pos.ticker, "AAPL")
        self.assertEqual(pos.quantity, 7.0)

    def test_get_position_not_found(self):
        # Mock 404 for position
        resp = DummyResponse(ok=False, status_code=404)
        self.session.get.return_value = resp
        pos = self.broker.get_position("GOOG")
        self.assertEqual(pos.ticker, "GOOG")
        self.assertEqual(pos.quantity, 0.0)

    def test_get_position_error(self):
        # Mock other error
        resp = DummyResponse(ok=False, status_code=500, text="server error")
        self.session.get.return_value = resp
        with self.assertRaises(OrderExecutionError):
            self.broker.get_position("MSFT")

    def test_get_order_success(self):
        # Mock a filled order response
        created_at = "2021-04-14T09:30:00Z"
        updated_at = "2021-04-14T10:00:00Z"
        data = {
            "id": "123",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "3",
            "type": "market",
            "status": "filled",
            "time_in_force": "day",
            "filled_avg_price": "100.5",
            "filled_qty": "3",
            "created_at": created_at,
            "updated_at": updated_at
        }
        self.session.get.return_value = DummyResponse(ok=True, status_code=200, json_data=data)
        order = self.broker.get_order("123")
        self.assertIsInstance(order, Order)
        self.assertEqual(order.order_id, "123")
        self.assertEqual(order.ticker, "AAPL")
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.quantity, 3.0)
        self.assertEqual(order.order_type, "market")
        self.assertEqual(order.status, "filled")
        self.assertEqual(order.time_in_force, "day")
        self.assertAlmostEqual(order.average_price, 100.5)
        self.assertEqual(order.remaining_quantity, 0.0)
        self.assertIsInstance(order.created_at, datetime)
        self.assertIsInstance(order.last_updated_at, datetime)
        # Check correct parsing to UTC
        dt_created = _parse_datetime(created_at)
        self.assertEqual(order.created_at, dt_created)
        dt_updated = _parse_datetime(updated_at)
        self.assertEqual(order.last_updated_at, dt_updated)

    def test_get_order_not_found(self):
        # Mock 404 for order fetch
        resp = DummyResponse(ok=False, status_code=404, text="not found")
        self.session.get.return_value = resp
        with self.assertRaises(OrderNotFoundError):
            self.broker.get_order("nonexistent")

    def test_get_order_error(self):
        # Mock other error
        resp = DummyResponse(ok=False, status_code=500, text="server error")
        self.session.get.return_value = resp
        with self.assertRaises(OrderExecutionError):
            self.broker.get_order("123")

    def test_get_orders_success(self):
        # Mock list of orders
        data_list = [
            {
                "id": "1", "symbol": "AAPL", "side": "buy", "qty": "2", "type": "market",
                "status": "filled", "time_in_force": "day", "filled_avg_price": "50", "filled_qty": "2",
                "created_at": "2021-01-01T10:00:00Z", "updated_at": "2021-01-01T10:05:00Z"
            },
            {
                "id": "2", "symbol": "TSLA", "side": "sell", "qty": "1", "type": "limit",
                "status": "new", "time_in_force": "gtc", "filled_avg_price": None, "filled_qty": "0",
                "created_at": "2021-02-01T11:00:00Z", "updated_at": "2021-02-01T11:00:00Z"
            }
        ]
        self.session.get.return_value = DummyResponse(ok=True, status_code=200, json_data=data_list)
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 2)
        self.assertTrue(all(isinstance(o, Order) for o in orders))
        self.assertEqual(orders[0].order_id, "1")
        self.assertEqual(orders[1].ticker, "TSLA")
        # Fallback average_price when None
        self.assertEqual(orders[1].average_price, 0.0)

    def test_get_orders_failure(self):
        # Mock fetch error for orders list
        resp = DummyResponse(ok=False, status_code=500)
        self.session.get.return_value = resp
        with self.assertRaises(OrderExecutionError):
            self.broker.get_orders()

    def test_submit_order_success(self):
        # Mock a successful order submission
        created_at = "2021-03-01T12:00:00Z"
        updated_at = "2021-03-01T12:05:00Z"
        data = {
            "id": "xyz", "symbol": "MSFT", "side": "sell", "qty": "4", "type": "market",
            "status": "new", "time_in_force": "day", "filled_avg_price": "0", "filled_qty": "0",
            "created_at": created_at, "updated_at": updated_at
        }
        self.session.post.return_value = DummyResponse(ok=True, status_code=200, json_data=data)
        order = self.broker.submit_order("MSFT", -4)
        self.assertIsInstance(order, Order)
        self.assertEqual(order.side, "sell")
        self.assertEqual(order.quantity, 4.0)
        self.assertEqual(order.order_id, "xyz")
        self.assertEqual(order.time_in_force, "day")
        self.assertIsInstance(order.created_at, datetime)
        self.assertEqual(order.created_at, _parse_datetime(created_at))

    def test_submit_order_failure(self):
        # Mock failed submission
        resp = DummyResponse(ok=False, status_code=400, text="bad request")
        self.session.post.return_value = resp
        with self.assertRaises(OrderExecutionError):
            self.broker.submit_order("AAPL", 5)

    def test_parse_datetime_none(self):
        # None input returns None
        self.assertIsNone(_parse_datetime(None))

    def test_parse_datetime_with_offset(self):
        # ISO string with timezone offset (no trailing Z)
        ts = "2021-04-14T09:30:00+02:00"
        dt = _parse_datetime(ts)
        self.assertIsInstance(dt, datetime)
        # Should preserve the offset
        self.assertEqual(dt.isoformat(), "2021-04-14T09:30:00+02:00")

    def test_get_account_timestamp_type(self):
        # last_updated_at should be an integer UNIX ms timestamp
        resp = DummyResponse(ok=True, status_code=200, json_data={"equity": "12345"})
        self.session.get.return_value = resp
        account = self.broker.get_account()
        self.assertIsInstance(account.last_updated_at, int)

    def test_get_positions_timestamp_type(self):
        # last_updated_at should be set on each Position
        data = [{"symbol": "GOOG", "qty": "2"}]
        self.session.get.return_value = DummyResponse(ok=True, status_code=200, json_data=data)
        positions = self.broker.get_positions()
        for pos in positions.values():
            self.assertIsInstance(pos.last_updated_at, int)

    def test_get_order_avg_price_none(self):
        # filled_avg_price None should fallback to 0.0
        created_at = "2021-05-01T08:00:00Z"
        updated_at = "2021-05-01T08:01:00Z"
        data = {
            "id": "abc", "symbol": "XYZ", "side": "buy", "qty": "5", "type": "market",
            "status": "filled", "time_in_force": "day", "filled_avg_price": None, "filled_qty": "0",
            "created_at": created_at, "updated_at": updated_at
        }
        self.session.get.return_value = DummyResponse(ok=True, status_code=200, json_data=data)
        order = self.broker.get_order("abc")
        self.assertEqual(order.average_price, 0.0)


if __name__ == "__main__":
    unittest.main() 