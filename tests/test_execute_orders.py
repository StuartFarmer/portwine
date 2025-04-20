import unittest
from datetime import timezone

from portwine.execution import ExecutionBase
from portwine.brokers.base import Order, OrderExecutionError
from portwine.strategies.base import StrategyBase


class DummyBroker:
    """A fake broker that returns updated Order objects based on inputs."""
    def __init__(self):
        self.calls = []

    def submit_order(self, symbol: str, quantity: float) -> Order:
        # Record the call
        self.calls.append((symbol, quantity))
        # Return an Order with filled status and dummy metadata
        return Order(
            order_id=f"id-{symbol}",
            ticker=symbol,
            side="buy" if quantity > 0 else "sell",
            quantity=quantity,
            order_type="market",
            status="filled",
            time_in_force="day",
            average_price=123.45,
            remaining_quantity=0.0,
            created_at=1610000000000,
            last_updated_at=1610000001000,
        )


class ErrorBroker:
    """A fake broker that always raises an OrderExecutionError."""
    def submit_order(self, symbol: str, quantity: float) -> Order:
        raise OrderExecutionError("Broker failed to execute order")


def make_exec_base_with_broker(broker):
    return ExecutionBase(
        strategy=StrategyBase([]),
        market_data_loader=None,
        broker=broker,
        alternative_data_loader=None,
        timezone=timezone.utc
    )


class TestExecuteOrders(unittest.TestCase):
    def test_execute_orders_success(self):
        broker = DummyBroker()
        exec_base = make_exec_base_with_broker(broker)
        # Create two dummy orders to execute
        orders = [
            Order(order_id="", ticker="AAPL", side="buy", quantity=10.0,
                  order_type="", status="", time_in_force="", average_price=0.0,
                  remaining_quantity=0.0, created_at=0, last_updated_at=0),
            Order(order_id="", ticker="MSFT", side="sell", quantity=5.0,
                  order_type="", status="", time_in_force="", average_price=0.0,
                  remaining_quantity=0.0, created_at=0, last_updated_at=0),
        ]
        executed = exec_base._execute_orders(orders)
        # Should return a list of updated Order objects
        self.assertEqual(len(executed), 2)
        # Broker should have been called with each symbol and quantity
        self.assertEqual(broker.calls, [("AAPL", 10.0), ("MSFT", -5.0)])

        # Validate returned Order fields
        for updated, original in zip(executed, orders):
            self.assertIsInstance(updated, Order)
            # Order ID should be populated by broker
            self.assertTrue(updated.order_id.startswith("id-"))
            # Ticker and quantity should match original
            self.assertEqual(updated.ticker, original.ticker)
            self.assertEqual(updated.quantity, original.quantity)
            # Status and metadata from DummyBroker
            self.assertEqual(updated.status, "filled")
            self.assertEqual(updated.average_price, 123.45)
            self.assertEqual(updated.remaining_quantity, 0.0)
            self.assertEqual(updated.time_in_force, "day")
            self.assertEqual(updated.created_at, 1610000000000)
            self.assertEqual(updated.last_updated_at, 1610000001000)

    def test_execute_orders_error_propagates(self):
        broker = ErrorBroker()
        exec_base = make_exec_base_with_broker(broker)
        orders = [
            Order(order_id="", ticker="AAPL", side="buy", quantity=10.0,
                  order_type="", status="", time_in_force="", average_price=0.0,
                  remaining_quantity=0.0, created_at=0, last_updated_at=0)
        ]
        with self.assertRaises(OrderExecutionError):
            exec_base._execute_orders(orders)


if __name__ == '__main__':
    unittest.main() 