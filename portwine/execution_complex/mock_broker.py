"""
Mock broker implementation for testing.

This module provides a mock broker implementation that can be used for testing
strategies without connecting to a real broker.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from portwine.execution_complex.broker import BrokerBase

logger = logging.getLogger(__name__)


class MockBroker(BrokerBase):
    """
    Mock broker implementation for testing strategies.
    
    This class simulates a broker interface with in-memory state, allowing
    for easy testing of strategies without connecting to a real broker.
    """
    
    def __init__(
        self, 
        initial_cash: float = 100000.0, 
        initial_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        market_open: bool = True,
        fail_symbols: Optional[List[str]] = None
    ):
        """
        Initialize the mock broker.
        
        Args:
            initial_cash: Starting cash balance
            initial_positions: Initial positions dictionary (symbol -> position info)
            market_open: Whether the market should be considered open initially
            fail_symbols: List of symbols that should fail on order execution (for testing error handling)
        """
        super().__init__()
        self.cash = initial_cash
        self.positions = initial_positions or {}
        self.market_open = market_open
        self.fail_symbols = fail_symbols or []
        
        # Track order history
        self.orders = []
        
        logger.info(f"Initialized MockBroker with ${initial_cash:.2f} cash")
        if initial_positions:
            logger.info(f"Initial positions: {len(initial_positions)} assets")
        if fail_symbols:
            logger.info(f"Set up to fail orders for: {', '.join(fail_symbols)}")
    
    def check_market_status(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        return self.market_open
    
    def set_market_status(self, is_open: bool) -> None:
        """
        Set the market status for testing different scenarios.
        
        Args:
            is_open: Whether the market should be considered open
        """
        self.market_open = is_open
        logger.info(f"Market status set to {'open' if is_open else 'closed'}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information.
        
        Returns:
            Dictionary with account information
        """
        # Calculate portfolio value
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        portfolio_value = self.cash + position_value
        
        return {
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'positions': self.positions
        }
    
    def execute_order(
        self, 
        symbol: str, 
        qty: float, 
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False
    ) -> bool:
        """
        Execute a simulated order.
        
        Args:
            symbol: The ticker symbol
            qty: Quantity to buy/sell (positive for buy, negative for sell)
            order_type: Type of order (market, limit, stop, stop_limit)
            limit_price: Price for limit orders
            stop_price: Price for stop orders
            time_in_force: Time in force parameter (day, gtc, ioc, fok)
            extended_hours: Whether to allow trading during extended hours
            
        Returns:
            True if order execution succeeded, False otherwise
        """
        # Check if this symbol is set to fail
        if symbol in self.fail_symbols:
            logger.warning(f"Order for {symbol} failed (in fail_symbols list)")
            return False
        
        # Ensure we have a price for this symbol
        price = self._get_price(symbol)
        if price is None:
            logger.error(f"No price available for {symbol}")
            return False
        
        # For limit orders, check if the price is acceptable
        if order_type == "limit" and limit_price is not None:
            if (qty > 0 and price > limit_price) or (qty < 0 and price < limit_price):
                logger.warning(f"Limit price condition not met for {symbol}")
                return False
        
        order_value = abs(qty) * price
        
        # For buy orders, check if we have enough cash
        if qty > 0 and order_value > self.cash:
            logger.error(f"Insufficient cash for order: {symbol} {qty} @ ${price:.2f}")
            return False
        
        # For sell orders, check if we have the position
        if qty < 0:
            current_position = self.positions.get(symbol, {}).get('qty', 0)
            if abs(qty) > current_position:
                logger.error(f"Insufficient position for sell order: {symbol} {qty}")
                return False
        
        # Execute the order
        order = {
            'symbol': symbol,
            'qty': qty,
            'price': price,
            'order_type': order_type,
            'time_in_force': time_in_force,
            'timestamp': datetime.now(timezone.utc)
        }
        
        if limit_price is not None:
            order['limit_price'] = limit_price
            
        if stop_price is not None:
            order['stop_price'] = stop_price
            
        self.orders.append(order)
        
        # Update cash
        self.cash -= qty * price
        
        # Update position
        self._update_position(symbol, qty, price)
        
        logger.info(f"Executed order: {symbol} {qty} @ ${price:.2f}")
        return True
    
    def _update_position(self, symbol: str, qty: float, price: float) -> None:
        """
        Update a position after an order execution.
        
        Args:
            symbol: The ticker symbol
            qty: Quantity bought/sold
            price: Execution price
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'qty': 0,
                'market_value': 0,
                'avg_entry_price': price,
                'unrealized_pl': 0
            }
        
        position = self.positions[symbol]
        old_qty = position['qty']
        new_qty = old_qty + qty
        
        # Calculate new average price for buys
        if qty > 0:
            position['avg_entry_price'] = ((old_qty * position['avg_entry_price']) + (qty * price)) / new_qty
        
        position['qty'] = new_qty
        position['market_value'] = new_qty * price
        position['unrealized_pl'] = (price - position['avg_entry_price']) * new_qty
        
        # Remove position if quantity is zero
        if abs(new_qty) < 1e-6:
            del self.positions[symbol]
    
    def _get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Current price or None if not available
        """
        # First check if we have a position with this symbol, and use its price
        if symbol in self.positions:
            position = self.positions[symbol]
            if position['qty'] != 0:
                return position['market_value'] / position['qty']
        
        # Otherwise use a default price
        return 100.0  # Default price for testing
    
    def set_price(self, symbol: str, price: float) -> None:
        """
        Set the current price for a symbol.
        
        This is useful for testing market movements.
        
        Args:
            symbol: The ticker symbol
            price: New price
        """
        # Update market value and unrealized P&L for existing positions
        if symbol in self.positions:
            position = self.positions[symbol]
            old_price = position['market_value'] / position['qty'] if position['qty'] != 0 else 0
            position['market_value'] = position['qty'] * price
            position['unrealized_pl'] = (price - position['avg_entry_price']) * position['qty']
            
            logger.info(f"Updated price for {symbol}: ${old_price:.2f} -> ${price:.2f}")
        else:
            # Create a new position with zero quantity just to track the price
            self.positions[symbol] = {
                'symbol': symbol,
                'qty': 0,
                'market_value': 0,
                'avg_entry_price': price,
                'unrealized_pl': 0
            }
            logger.info(f"Set price for {symbol}: ${price:.2f}")
    
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
            qty = position['qty']
            if qty != 0:
                self.execute_order(symbol, -qty)
        return True
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions.
        
        Returns:
            List of position dictionaries
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
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        return self.cash + position_value
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position information for a specific symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Position information or None if not held
        """
        return self.positions.get(symbol)
    
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
        logger.info(f"Reset MockBroker with ${initial_cash:.2f} cash")
    
    def simulate_market_move(self, percent_change: float) -> None:
        """
        Simulate a market move by adjusting all position prices.
        
        Args:
            percent_change: Percentage change to apply to all positions
        """
        for symbol, position in self.positions.items():
            if position['qty'] != 0:
                old_price = position['market_value'] / position['qty']
                new_price = old_price * (1 + percent_change / 100.0)
                self.set_price(symbol, new_price)
        
        logger.info(f"Simulated market move of {percent_change:.2f}%")
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all executed orders.
        
        Returns:
            List of order dictionaries
        """
        return self.orders 