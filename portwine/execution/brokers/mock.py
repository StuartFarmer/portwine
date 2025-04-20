import threading
from datetime import datetime
from typing import Dict, List

from portwine.execution.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderNotFoundError,
)


class MockBroker(BrokerBase):
    """
    A simple in‑memory mock broker for testing.
    Fills all market orders immediately at a fixed price and
    tracks positions and orders in dictionaries.
    """

    def __init__(self, initial_equity: float = 1_000_000.0, fill_price: float = 100.0):
        self._equity = initial_equity
        self._fill_price = fill_price
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._lock = threading.Lock()

    def get_account(self) -> Account:
        """
        Return a snapshot of the account. Equity remains constant
        in this mock.
        """
        return Account(equity=self._equity)

    def get_positions(self) -> Dict[str, Position]:
        """
        Return all current positions.
        """
        # Return copies to avoid external mutation
        return {
            symbol: Position(symbol=symbol, quantity=pos.quantity)
            for symbol, pos in self._positions.items()
        }

    def get_position(self, ticker: str) -> Position:
        """
        Return position for a single ticker (zero if not held).
        """
        pos = self._positions.get(ticker)
        if pos is None:
            return Position(symbol=ticker, quantity=0.0)
        # Return a copy
        return Position(symbol=pos.symbol, quantity=pos.quantity)

    def get_order(self, order_id: str) -> Order:
        """
        Retrieve a single order by ID; raise if not found.
        """
        try:
            return self._orders[order_id]
        except KeyError:
            raise OrderNotFoundError(f"Order {order_id} not found")

    def get_orders(self) -> List[Order]:
        """
        Return a list of all orders submitted so far.
        """
        return list(self._orders.values())

    def submit_order(self, symbol: str, quantity: float) -> Order:
        """
        Simulate a market order fill:
          - Immediately 'fills' at self._fill_price
          - Updates in‑memory positions
          - Records the order with status 'filled'
        """
        with self._lock:
            self._order_counter += 1
            oid = str(self._order_counter)

        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        now = datetime.now()

        # Update position
        prev_qty = self._positions.get(symbol, Position(symbol, 0.0)).quantity
        new_qty = prev_qty + quantity
        if new_qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(symbol=symbol, quantity=new_qty)

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
            created_at=now,
            last_updated_at=now,
        )

        self._orders[oid] = order
        return order
