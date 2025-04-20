import requests
from datetime import datetime
from typing import Dict, List

from portwine.execution.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderExecutionError,
    OrderNotFoundError,
    OrderCancelError,
)


def _parse_datetime(dt_str: str) -> datetime:
    """
    Parse an ISO‑8601 timestamp from Alpaca, handling the trailing 'Z'.
    """
    if dt_str is None:
        return None
    # Alpaca returns times like "2021-04-14T09:30:00Z"
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


class AlpacaBroker(BrokerBase):
    """
    Alpaca REST API implementation of BrokerBase using the `requests` library.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Args:
            api_key: Your Alpaca API key ID.
            api_secret: Your Alpaca secret key.
            base_url: Alpaca REST endpoint (paper or live).
        """
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json",
        })

    def get_account(self) -> Account:
        url = f"{self._base_url}/v2/account"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Account fetch failed: {resp.text}")
        data = resp.json()
        return Account(equity=float(data["equity"]))

    def get_positions(self) -> Dict[str, Position]:
        url = f"{self._base_url}/v2/positions"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Positions fetch failed: {resp.text}")
        positions = {}
        for p in resp.json():
            positions[p["symbol"]] = Position(
                symbol=p["symbol"],
                quantity=float(p["qty"])
            )
        return positions

    def get_position(self, ticker: str) -> Position:
        url = f"{self._base_url}/v2/positions/{ticker}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            return Position(symbol=ticker, quantity=0.0)
        if not resp.ok:
            raise OrderExecutionError(f"Position fetch failed: {resp.text}")
        p = resp.json()
        return Position(symbol=p["symbol"], quantity=float(p["qty"]))

    def get_order(self, order_id: str) -> Order:
        url = f"{self._base_url}/v2/orders/{order_id}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            raise OrderNotFoundError(f"Order {order_id} not found")
        if not resp.ok:
            raise OrderExecutionError(f"Order fetch failed: {resp.text}")
        o = resp.json()
        return Order(
            order_id=o["id"],
            ticker=o["symbol"],
            side=o["side"],
            quantity=float(o["qty"]),
            order_type=o["type"],
            status=o["status"],
            time_in_force=o["time_in_force"],
            average_price=float(o["filled_avg_price"] or 0.0),
            remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
            created_at=_parse_datetime(o["created_at"]),
            last_updated_at=_parse_datetime(o["updated_at"]),
        )

    def get_orders(self) -> List[Order]:
        url = f"{self._base_url}/v2/orders"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Orders fetch failed: {resp.text}")
        orders = []
        for o in resp.json():
            orders.append(Order(
                order_id=o["id"],
                ticker=o["symbol"],
                side=o["side"],
                quantity=float(o["qty"]),
                order_type=o["type"],
                status=o["status"],
                time_in_force=o["time_in_force"],
                average_price=float(o["filled_avg_price"] or 0.0),
                remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
                created_at=_parse_datetime(o["created_at"]),
                last_updated_at=_parse_datetime(o["updated_at"]),
            ))
        return orders

    def submit_order(self, symbol: str, quantity: float) -> Order:
        """
        Execute a market order on Alpaca.
        Positive quantity → buy, negative → sell.
        """
        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        url = f"{self._base_url}/v2/orders"
        resp = self._session.post(url, json=payload)
        if not resp.ok:
            raise OrderExecutionError(f"Order submission failed: {resp.text}")
        o = resp.json()
        return Order(
            order_id=o["id"],
            ticker=o["symbol"],
            side=o["side"],
            quantity=float(o["qty"]),
            order_type=o["type"],
            status=o["status"],
            time_in_force=o["time_in_force"],
            average_price=float(o["filled_avg_price"] or 0.0),
            remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
            created_at=_parse_datetime(o["created_at"]),
            last_updated_at=_parse_datetime(o["updated_at"]),
        )
