import threading
import time
from datetime import datetime
from typing import Dict, List

from portwine.brokers.base import (
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
        Return a snapshot of the account with a unix‐timestamp last_updated_at (ms).
        """
        ts = int(time.time() * 1_000)
        return Account(
            equity=self._equity,
            last_updated_at=ts
        )

    def get_positions(self) -> Dict[str, Position]:
        """
        Return all current positions, preserving each position's last_updated_at.
        """
        return {
            symbol: Position(
                ticker=pos.ticker,
                quantity=pos.quantity,
                last_updated_at=pos.last_updated_at
            )
            for symbol, pos in self._positions.items()
        }

    def get_position(self, ticker: str) -> Position:
        """
        Return position for a single ticker (zero if not held),
        with last_updated_at from the stored position or now if none.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            ts = int(time.time() * 1_000)
            return Position(ticker=ticker, quantity=0.0, last_updated_at=ts)
        return Position(
            ticker=pos.ticker,
            quantity=pos.quantity,
            last_updated_at=pos.last_updated_at
        )

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
          - Updates in‑memory positions with a unix‐timestamp last_updated_at (ms)
          - Records the order with status 'filled' and last_updated_at as unix timestamp (ms)
        """
        with self._lock:
            self._order_counter += 1
            oid = str(self._order_counter)

        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        now_ts = int(time.time() * 1_000)

        # Update position
        prev = self._positions.get(symbol)
        prev_qty = prev.quantity if prev is not None else 0.0
        new_qty = prev_qty + quantity

        if new_qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                ticker=symbol,
                quantity=new_qty,
                last_updated_at=now_ts
            )

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
            created_at=now_ts,
            last_updated_at=now_ts
        )

        self._orders[oid] = order
        return order

    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Stub implementation of market hours:
        - Returns True if it's a weekday (Monday=0 … Friday=4)
          between 09:30 and 16:00 in the local system time.
        """
        # Weekday check
        if timestamp.weekday() >= 5:
            return False

        # Hour/minute check: between 9:30 and 16:00
        hm = timestamp.hour * 60 + timestamp.minute
        open_time = 9 * 60 + 30   # 9:30 = 570 minutes
        close_time = 16 * 60      # 16:00 = 960 minutes
        return open_time <= hm < close_time
