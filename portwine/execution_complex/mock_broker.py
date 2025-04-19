"""
Mock implementation of the BrokerBase class for testing.

This module provides a mock implementation of the broker interface
that can be used for testing purposes. It simulates account information,
market status, and order execution without connecting to a real broker.
"""

from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Union

from portwine.execution_complex.broker import BrokerBase, AccountInfo, Position, Order
from portwine.utils.market_calendar import MarketStatus


class MockBroker(BrokerBase):
    """
    Mock broker implementation for testing.
    
    This class provides a simulated broker environment for testing
    execution systems without connecting to a real broker.
    
    Attributes:
        cash: Current cash available.
        positions: Dictionary of current positions by symbol.
        orders: List of executed orders.
        failed_symbols: List of symbols that should fail on order execution.
        _is_market_open: Whether the market is currently open.
        _next_market_open: Next market open time.
        _next_market_close: Next market close time.
    """
    
    def __init__(self, initial_cash: float = 100000.0, 
                 initial_positions: Optional[Dict[str, Position]] = None,
                 failed_symbols: Optional[List[str]] = None,
                 timezone: str = "America/New_York"):
        """
        Initialize the mock broker.
        
        Args:
            initial_cash: Initial cash available.
            initial_positions: Dictionary of initial positions by symbol.
            failed_symbols: List of symbols that should fail on order execution.
            timezone: Timezone to use for market hours.
        """
        self.cash = initial_cash
        self.positions = initial_positions or {}
        self.orders = []
        self.failed_symbols = failed_symbols or []
        self.timezone = timezone
        
        # Default market status
        self._is_market_open = True
        now = datetime.now(pytz.timezone(self.timezone))
        self._next_market_open = now
        self._next_market_close = now + timedelta(hours=8)  # Default market day is 8 hours
        
    def set_market_status(self, is_open: bool, 
                          next_open: Optional[datetime] = None, 
                          next_close: Optional[datetime] = None):
        """
        Set the market status for testing.
        
        Args:
            is_open: Whether the market is currently open.
            next_open: Next market open time.
            next_close: Next market close time.
        """
        self._is_market_open = is_open
        
        now = datetime.now(pytz.timezone(self.timezone))
        self._next_market_open = next_open or now
        self._next_market_close = next_close or (now + timedelta(hours=8))
        
    def check_market_status(self) -> MarketStatus:
        """
        Check the current market status.
        
        Returns:
            MarketStatus object with current market status.
        """
        return MarketStatus(
            is_open=self._is_market_open, 
            next_open=self._next_market_open,
            next_close=self._next_market_close
        )
    
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.
        
        Returns:
            AccountInfo object with current account information.
        """
        # Calculate portfolio value
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            portfolio_value += position.current_price * position.quantity
            
        return AccountInfo(
            cash=self.cash, 
            portfolio_value=portfolio_value, 
            positions=self.positions
        )
    
    def execute_order(self, order: Order) -> bool:
        """
        Execute a simulated order.
        
        Args:
            order: Order to execute.
            
        Returns:
            True if the order was successfully executed, False otherwise.
        """
        # Check if this symbol should fail
        if order.symbol in self.failed_symbols:
            return False
        
        # Record the order
        self.orders.append(order)
        
        # Update positions and cash
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            
            # Update position
            if order.quantity > 0:  # Buy
                new_quantity = position.quantity + order.quantity
                total_cost = (position.quantity * position.cost_basis) + (order.quantity * order.price)
                new_cost_basis = total_cost / new_quantity if new_quantity > 0 else 0
                
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=new_quantity,
                    cost_basis=new_cost_basis,
                    current_price=order.price
                )
                
                # Update cash
                self.cash -= order.quantity * order.price
                
            else:  # Sell
                new_quantity = position.quantity + order.quantity  # order.quantity is negative for sells
                
                if new_quantity > 0:
                    # Partial sell
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=new_quantity,
                        cost_basis=position.cost_basis,  # Cost basis doesn't change on sells
                        current_price=order.price
                    )
                else:
                    # Complete sell
                    del self.positions[order.symbol]
                    
                # Update cash
                self.cash -= order.quantity * order.price  # order.quantity is negative, so this adds to cash
                
        else:
            # New position
            if order.quantity > 0:  # Can only create a position with a buy
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    cost_basis=order.price,
                    current_price=order.price
                )
                
                # Update cash
                self.cash -= order.quantity * order.price
            else:
                # Can't sell what we don't have
                return False
                
        return True
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to get position for.
            
        Returns:
            Position object if the position exists, None otherwise.
        """
        return self.positions.get(symbol)
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order
            
        Returns:
            Status string or None if not found
        """
        # In this mock implementation, all orders are immediately filled
        return "filled"
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns:
            True if successful, False otherwise
        """
        # No pending orders in this mock implementation
        return True
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
            True if successful, False otherwise
        """
        for symbol, position in list(self.positions.items()):
            qty = position.quantity
            if qty != 0:
                self.execute_order(Order(symbol=symbol, quantity=-qty))
        return True
    
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List of position objects
        """
        return list(self.positions.values())
    
    def get_cash(self) -> float:
        """
        Get available cash.
        
        Returns:
            Available cash amount
        """
        return self.cash
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        portfolio_value = self.cash
        for symbol, position in self.positions.items():
            portfolio_value += position.current_price * position.quantity
        return portfolio_value
    
    def reset(self, initial_cash: float = 100000.0, keep_positions: bool = False) -> None:
        """
        Reset the broker to initial state.
        
        Args:
            initial_cash: New initial cash balance
            keep_positions: Whether to keep existing positions
        """
        self.cash = initial_cash
        if not keep_positions:
            self.positions = {}
        self.orders = []
    
    def simulate_market_move(self, percent_change: float) -> None:
        """
        Simulate a market move by adjusting all position prices.
        
        Args:
            percent_change: Percentage change to apply to all positions
        """
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                old_price = position.current_price
                new_price = old_price * (1 + percent_change / 100.0)
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=position.quantity,
                    cost_basis=position.cost_basis,
                    current_price=new_price
                )
        
    def get_order_history(self) -> List[Order]:
        """
        Get history of all executed orders.
        
        Returns:
            List of order objects
        """
        return self.orders 