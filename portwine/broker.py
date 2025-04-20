"""
Broker base class for trading interfaces.

This module provides the base class for all broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    quantity: float
    side: OrderSide
    status: str
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = datetime.now()


@dataclass
class Account:
    """Represents an account state."""
    balance: float
    equity: float
    margin: float


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
    def execute_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
    ) -> Order:
        """
        Execute a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            quantity: Order quantity
            side: Order side (BUY or SELL)
        
        Returns:
            Order object representing the executed order
        
        Raises:
            ValueError: If the order parameters are invalid
            OrderExecutionError: If the order fails to execute
        """
        pass
