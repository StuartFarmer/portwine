"""
Mock broker implementation for testing.

This module provides a mock broker implementation that can be used for testing
strategies without connecting to a real broker.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from portwine.broker import BrokerBase, Order, Position, Account, OrderSide, OrderExecutionError


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
        self.logger = logging.getLogger(__name__)
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.market_open = market_open
        self.fail_symbols = fail_symbols or []
        
        # Initialize positions
        self._positions = {}
        if initial_positions:
            for symbol, pos_data in initial_positions.items():
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=pos_data.get('quantity', 0),
                    entry_price=pos_data.get('entry_price', 100.0),
                    current_price=pos_data.get('current_price', 100.0)
                )
        
        # Orders history
        self._orders = {}
        
        self.logger.info(f"Initialized MockBroker with ${initial_cash:.2f} cash")
        if initial_positions:
            self.logger.info(f"Initial positions: {len(initial_positions)} assets")
        if fail_symbols:
            self.logger.info(f"Set up to fail orders for: {', '.join(fail_symbols)}")
    
    def get_account(self) -> Account:
        """
        Get current account information.
        
        Returns:
            Account object with current information
        """
        # Calculate equity
        equity = self.cash
        for position in self._positions.values():
            equity += position.quantity * position.current_price
        
        return Account(
            balance=self.cash,
            equity=equity,
            margin=0.0  # No margin in the mock broker
        )
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbol to Position objects
        """
        return self._positions
    
    def execute_order(self, symbol: str, quantity: float, side: OrderSide) -> Order:
        """
        Execute a simulated order.
        
        Args:
            symbol: The ticker symbol
            quantity: Quantity to trade
            side: Order side (BUY or SELL)
            
        Returns:
            Order object representing the executed order
            
        Raises:
            OrderExecutionError: If order execution fails
        """
        # Check if market is open
        if not self.market_open:
            raise OrderExecutionError("Market is closed")
        
        # Check if this symbol is set to fail
        if symbol in self.fail_symbols:
            self.logger.warning(f"Order for {symbol} failed (in fail_symbols list)")
            raise OrderExecutionError(f"Order for {symbol} failed (in fail_symbols list)")
        
        # Get a price for this symbol (or use default)
        price = self._get_price(symbol)
        
        # Calculate order cost
        order_value = quantity * price
        
        # For buy orders, check if we have enough cash
        if side == OrderSide.BUY:
            if order_value > self.cash:
                raise OrderExecutionError(f"Insufficient cash for order: {symbol} {quantity} @ ${price:.2f}")
            
            # Update cash
            self.cash -= order_value
            
            # Update position
            if symbol in self._positions:
                position = self._positions[symbol]
                old_quantity = position.quantity
                old_entry_price = position.entry_price
                new_quantity = old_quantity + quantity
                
                # Calculate new average entry price
                new_entry_price = ((old_quantity * old_entry_price) + (quantity * price)) / new_quantity
                
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    entry_price=new_entry_price,
                    current_price=price
                )
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price
                )
        
        # For sell orders, check if we have the position
        else:  # side == OrderSide.SELL
            if symbol not in self._positions:
                raise OrderExecutionError(f"No position in {symbol} to sell")
            
            position = self._positions[symbol]
            if position.quantity < quantity:
                raise OrderExecutionError(f"Insufficient position for sell order: {symbol} {quantity}")
            
            # Update cash
            self.cash += order_value
            
            # Update position
            new_quantity = position.quantity - quantity
            if new_quantity > 0:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    entry_price=position.entry_price,
                    current_price=price
                )
            else:
                del self._positions[symbol]
        
        # Create order record
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            side=side,
            status="filled",
            filled_quantity=quantity,
            average_price=price,
            created_at=datetime.now()
        )
        
        # Save order
        self._orders[order_id] = order
        
        self.logger.info(f"Executed {'buy' if side == OrderSide.BUY else 'sell'} order for {quantity} shares of {symbol} at ${price:.2f}")
        return order
    
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
        self.logger.info(f"Market status set to {'open' if is_open else 'closed'}")
    
    def _get_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Current price
        """
        # First check if we have a position with this symbol
        if symbol in self._positions:
            return self._positions[symbol].current_price
        
        # Otherwise use a default price
        return 100.0
    
    def set_price(self, symbol: str, price: float) -> None:
        """
        Set the current price for a symbol.
        
        This is useful for testing market movements.
        
        Args:
            symbol: The ticker symbol
            price: New price
        """
        if symbol in self._positions:
            position = self._positions[symbol]
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=price
            )
            self.logger.info(f"Updated price for {symbol}: {position.current_price:.2f} -> {price:.2f}")
        else:
            # Create a dummy position with zero quantity
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                entry_price=price,
                current_price=price
            )
            self.logger.info(f"Set initial price for {symbol}: {price:.2f}")
    
    def reset(self, initial_cash: Optional[float] = None, keep_positions: bool = False) -> None:
        """
        Reset the broker state.
        
        Args:
            initial_cash: New initial cash amount, or None to use original
            keep_positions: Whether to keep current positions
        """
        self.cash = initial_cash if initial_cash is not None else self.initial_cash
        
        if not keep_positions:
            self._positions = {}
            
        self._orders = {}
        
        self.logger.info(f"Reset MockBroker with ${self.cash:.2f} cash, keep_positions={keep_positions}")
    
    def simulate_market_move(self, percent_change: float) -> None:
        """
        Simulate a market movement affecting all positions.
        
        Args:
            percent_change: Percentage change in prices (e.g., 0.05 for +5%)
        """
        for symbol, position in list(self._positions.items()):
            new_price = position.current_price * (1 + percent_change)
            self.set_price(symbol, new_price)
            
        self.logger.info(f"Simulated market move of {percent_change:.2%}")
    
    def get_order_history(self) -> Dict[str, Order]:
        """
        Get the history of orders.
        
        Returns:
            Dictionary of order objects
        """
        return self._orders 