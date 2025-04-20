"""
Broker base class for trading interfaces.

This module provides the base class for all broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class Position:
    """Represents a trading position."""
    ticker: str
    quantity: float
    last_updated_at: int # UNIX timestamp in second for last time the data was updated

class TimeInForce(Enum):
    DAY = "day"
    GOOD_TILL_CANCELLED = "gtc"
    IMMEDIATE_OR_CANCEL = "ioc"
    FILL_OR_KILL = "fok"
    MARKET_ON_OPEN = "opg"
    MARKET_ON_CLOSE = "cls"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    
class OrderStatus(Enum):
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    PENDING = "pending"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

'''
    Order dataclass. All brokers must adhere to this standard. 
'''
@dataclass
class Order:
    order_id: str               # unique identifier for the order
    ticker: str                 # asset ticker
    side: str                   # buy or sell
    quantity: float             # amount / size of order
    order_type: str             # market, limit, etc
    status: str                 # submitted, rejected, filled, etc
    time_in_force: str          # gtc, fok, etc
    average_price: float        # average fill price of order
    remaining_quantity: float   # how much of the order still needs to be filled
    created_at: int        # when the order was created (UNIX timestamp milliseconds)
    last_updated_at: int   # when the data on this order was last updated with the broker

@dataclass
class Account:
    equity: float         # amount of money available to purchase securities (can include margin)
    last_updated_at: int
    # net_liquidation_value     liquid value, total_equity... all are names for the same thing, equity...
    # buying power includes margin, equity does not


class OrderExecutionError(Exception):
    """Raised when an order fails to execute."""
    pass


class OrderNotFoundError(Exception):
    """Raised when an order cannot be found."""
    pass


class OrderCancelError(Exception):
    """Raised when an order fails to cancel."""
    pass


class BrokerBase(ABC):
    """Base class for all broker implementations."""

    @abstractmethod
    def get_account(self) -> Account:
        """
        Get the current account information.
        
        Returns:
            Account object containing current account state
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbol to Position objects
        """
        pass

    @abstractmethod
    def get_position(self, ticker) -> Position:
        # Returns a position object for a given ticker
        # if there is no position, then an empty position is returned
        # with quantity 0
        pass

    @abstractmethod
    def get_order(self, order_id) -> Order:
        pass

    @abstractmethod
    def get_orders(self) -> List[Order]:
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        quantity: float,
    ) -> Order:
        """
        Execute a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            quantity: Order quantity
        
        Returns:
            Order object representing the executed order
        
        Raises:
            ValueError: If the order parameters are invalid
            OrderExecutionError: If the order fails to execute
        """
        pass

    @abstractmethod
    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the given datetime.
        """
        pass
