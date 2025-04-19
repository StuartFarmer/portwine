"""Base broker class for execution systems.

This module contains the base class for broker implementations, which handle
the interaction with trading platforms and exchanges.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timezone


class AccountInfo:
    """Container for account information."""
    
    def __init__(self, cash: float, portfolio_value: float, positions: Dict[str, Any]):
        """Initialize with account details.
        
        Args:
            cash: Available cash balance
            portfolio_value: Total portfolio value including cash and positions
            positions: Dictionary of positions by symbol
        """
        self.cash = cash
        self.portfolio_value = portfolio_value
        self.positions = positions


class Position:
    """Container for position information."""
    
    def __init__(
        self, 
        symbol: str, 
        qty: float, 
        market_value: float, 
        avg_entry_price: float,
        unrealized_pl: float
    ):
        """Initialize with position details.
        
        Args:
            symbol: The ticker symbol
            qty: Number of shares held
            market_value: Current value of the position
            avg_entry_price: Average cost basis
            unrealized_pl: Unrealized profit/loss
        """
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.avg_entry_price = avg_entry_price
        self.unrealized_pl = unrealized_pl


class Order:
    """Container for order information."""
    
    def __init__(
        self,
        symbol: str,
        qty: float,
        order_type: str,
        limit_price: Optional[float] = None,
        time_in_force: str = "day"
    ):
        """Initialize with order details.
        
        Args:
            symbol: The ticker symbol
            qty: Number of shares to buy/sell (negative for sell)
            order_type: One of: market, limit, stop, stop_limit
            limit_price: Price for limit orders
            time_in_force: One of: day, gtc, ioc, fok
        """
        self.symbol = symbol
        self.qty = qty
        self.order_type = order_type
        self.limit_price = limit_price
        self.time_in_force = time_in_force


class BrokerBase(ABC):
    """Base class for broker implementations.
    
    A broker is responsible for:
    - Checking if markets are open/closed
    - Getting account information (cash, positions)
    - Executing orders
    - Monitoring order status
    """
    
    def __init__(self):
        """Initialize the broker."""
        pass
    
    @abstractmethod
    def check_market_status(self) -> bool:
        """Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information.
        
        Returns:
            Dict: Dictionary containing account information:
                - cash: Available cash
                - portfolio_value: Total portfolio value
                - positions: Dictionary of positions by symbol
        """
        pass
    
    @abstractmethod
    def execute_order(
        self, 
        symbol: str, 
        qty: float, 
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """Execute an order.
        
        Args:
            symbol: The ticker symbol
            qty: Number of shares to buy/sell (negative for sell)
            order_type: One of: market, limit, stop, stop_limit
            limit_price: Price for limit orders
            time_in_force: One of: day, gtc, ioc, fok
            
        Returns:
            Dict: Order execution result
        """
        pass
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for a specific symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Dict or None: Position information or None if not held
        """
        account_info = self.get_account_info()
        positions = account_info.get("positions", {})
        return positions.get(symbol)
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights.
        
        Returns:
            Dict: Symbol -> weight mapping
        """
        account_info = self.get_account_info()
        portfolio_value = account_info.get("portfolio_value", 0)
        positions = account_info.get("positions", {})
        
        if portfolio_value <= 0:
            return {}
            
        weights = {}
        for symbol, position in positions.items():
            market_value = position.get("market_value", 0)
            weights[symbol] = market_value / portfolio_value
            
        return weights 