"""
Mock execution_complex implementation for testing.

This module provides a mock implementation of the execution_complex interface
that can be used for testing without connecting to real brokers.
"""

from typing import Dict, List, Optional, Set

import pandas as pd

from portwine.execution_complex import AccountInfo, ExecutionBase, Order, Position
from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase


class MockExecution(ExecutionBase):
    """
    Mock implementation of execution_complex for testing.
    
    This implementation simulates an execution_complex environment without real trades.
    It tracks orders, positions, and portfolio value for testing purposes.
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_data_loader: MarketDataLoader,
        alternative_data_loader: Optional[MarketDataLoader] = None,
        initial_cash: float = 100000.0,
        initial_positions: Optional[Dict[str, Position]] = None,
        fail_symbols: Optional[Set[str]] = None,
    ):
        """
        Initialize the mock execution_complex.
        
        Parameters
        ----------
        strategy : StrategyBase
            Strategy implementation
        market_data_loader : MarketDataLoader
            Market data loader
        alternative_data_loader : MarketDataLoader, optional
            Alternative data loader
        initial_cash : float, default 100000.0
            Initial cash in the account
        initial_positions : Dict[str, Position], optional
            Initial positions in the account
        fail_symbols : Set[str], optional
            Set of symbols that should fail on order execution_complex
        """
        super().__init__(strategy, market_data_loader, alternative_data_loader)
        self.cash = initial_cash
        self.positions = initial_positions or {}
        self.executed_orders: List[Order] = []
        self.fail_symbols = fail_symbols or set()
        self.order_id_counter = 0
    
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.
        
        Returns
        -------
        AccountInfo
            Mock account information
        """
        # Calculate portfolio value based on current positions and cash
        portfolio_value = self.cash
        
        # Get current prices
        current_prices = {}
        for symbol in self.positions.keys():
            data = self.fetch_latest_data()
            if symbol in data and data[symbol] is not None:
                current_prices[symbol] = data[symbol]["close"]
        
        # Update position market values
        for symbol, position in self.positions.items():
            price = current_prices.get(symbol, position["avg_entry_price"])
            market_value = position["qty"] * price
            unrealized_pl = market_value - (position["qty"] * position["avg_entry_price"])
            
            # Update position values
            self.positions[symbol]["market_value"] = market_value
            self.positions[symbol]["unrealized_pl"] = unrealized_pl
            
            # Add to portfolio value
            portfolio_value += market_value
        
        return {
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "positions": self.positions,
        }
    
    def execute_order(self, order: Order) -> bool:
        """
        Execute a mock order.
        
        Parameters
        ----------
        order : Order
            Order to execute
            
        Returns
        -------
        bool
            True if order was executed, False otherwise
        """
        symbol = order["symbol"]
        qty = order["qty"]
        
        # Simulate failed orders
        if symbol in self.fail_symbols:
            return False
        
        # Get current price
        data = self.fetch_latest_data()
        if symbol not in data or data[symbol] is None:
            return False
        
        price = data[symbol]["close"]
        cost = price * qty
        
        # Check if we have enough cash for buys
        if qty > 0 and cost > self.cash:
            return False
        
        # For sells, check if we have enough shares
        if qty < 0 and (symbol not in self.positions or self.positions[symbol]["qty"] < abs(qty)):
            return False
        
        # Execute the trade
        self.order_id_counter += 1
        executed_order = order.copy()
        executed_order["execution_price"] = price
        executed_order["order_id"] = self.order_id_counter
        self.executed_orders.append(executed_order)
        
        # Update cash
        self.cash -= cost
        
        # Update position
        if symbol not in self.positions:
            # New position
            if qty > 0:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "market_value": cost,
                    "avg_entry_price": price,
                    "unrealized_pl": 0.0,
                }
        else:
            # Existing position
            current_position = self.positions[symbol]
            current_qty = current_position["qty"]
            current_value = current_position["avg_entry_price"] * current_qty
            
            new_qty = current_qty + qty
            
            if new_qty == 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Update position
                if qty > 0:
                    # Buy - update average price
                    new_value = current_value + cost
                    avg_price = new_value / new_qty
                else:
                    # Sell - keep average price the same
                    avg_price = current_position["avg_entry_price"]
                
                self.positions[symbol] = {
                    "symbol": symbol,
                    "qty": new_qty,
                    "market_value": new_qty * price,
                    "avg_entry_price": avg_price,
                    "unrealized_pl": (price - avg_price) * new_qty,
                }
        
        return True 