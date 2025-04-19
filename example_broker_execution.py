#!/usr/bin/env python
"""
Example script demonstrating the use of ExecutionBase with different broker implementations.

This example shows:
1. How to create broker implementations (MockBroker, AlpacaBroker)
2. How to use them directly with ExecutionBase
3. How to execute a simple trading strategy using this structure
"""

import argparse
import logging
import pandas as pd
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy
from portwine.execution_complex.base import ExecutionBase
from portwine.execution_complex.broker import BrokerBase, AccountInfo, Position, Order
from portwine.execution_complex.alpaca_broker import AlpacaBroker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("broker_example.log")
    ]
)
logger = logging.getLogger(__name__)


class MockBroker(BrokerBase):
    """
    Mock broker implementation for testing.
    
    This class simulates a broker interface with in-memory state.
    """
    
    def __init__(self, initial_cash: float = 100000.0, market_open: bool = True):
        """
        Initialize the mock broker.
        
        Args:
            initial_cash: Starting cash balance
            market_open: Whether the market should be considered open
        """
        super().__init__()
        self.cash = initial_cash
        self.positions = {}  # symbol -> position
        self.orders = []  # list of executed orders
        self.market_open = market_open
        self.prices = {}  # symbol -> price
        
        logger.info(f"Initialized MockBroker with ${initial_cash:.2f} cash")
    
    def check_market_status(self) -> bool:
        """Check if the market is currently open."""
        return self.market_open
    
    def set_market_status(self, is_open: bool):
        """Set the market status for testing."""
        self.market_open = is_open
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
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
        time_in_force: str = "day"
    ) -> bool:
        """
        Execute a simulated order.
        
        Args:
            symbol: The ticker symbol
            qty: Quantity to buy/sell (negative for sell)
            order_type: Order type (market, limit, etc.)
            limit_price: Limit price for limit orders
            time_in_force: Time in force parameter
            
        Returns:
            True if order execution succeeded, False otherwise
        """
        # Ensure we have a price for this symbol
        if symbol not in self.prices:
            logger.error(f"No price available for {symbol}")
            return False
        
        price = self.prices[symbol]
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
        self.orders.append(order)
        
        # Update cash
        self.cash -= qty * price
        
        # Update position
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
        
        logger.info(f"Executed order: {symbol} {qty} @ ${price:.2f}")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get status of an order by ID."""
        # In this mock implementation, all orders are immediately filled
        return "filled"
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        # No pending orders in this implementation
        return True
    
    def close_all_positions(self) -> bool:
        """Close all open positions."""
        for symbol, position in list(self.positions.items()):
            qty = position['qty']
            if qty != 0:
                self.execute_order(symbol, -qty)
        return True
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get list of current positions."""
        return list(self.positions.values())
    
    def get_cash(self) -> float:
        """Get available cash."""
        return self.cash
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        return self.cash + position_value
    
    def set_price(self, symbol: str, price: float):
        """Set the current price for a symbol (for testing)."""
        self.prices[symbol] = price
        
        # Update market value and unrealized P&L for existing positions
        if symbol in self.positions:
            position = self.positions[symbol]
            old_price = position['market_value'] / position['qty'] if position['qty'] != 0 else 0
            position['market_value'] = position['qty'] * price
            position['unrealized_pl'] = (price - position['avg_entry_price']) * position['qty']
            
            logger.info(f"Updated price for {symbol}: ${old_price:.2f} -> ${price:.2f}")


class MockDataLoader:
    """
    Mock data loader that returns fixed data.
    """
    
    def __init__(self, mock_data=None):
        """
        Initialize with optional mock data.
        
        Args:
            mock_data: Pre-defined mock data to use
        """
        self.mock_data = mock_data or {}
    
    def next(self, tickers, timestamp=None):
        """
        Get the next data point for the tickers.
        
        Args:
            tickers: List of tickers to get data for
            timestamp: Timestamp to get data for
            
        Returns:
            Dictionary of ticker data
        """
        return self.mock_data
    
    def set_mock_data(self, data):
        """Set the mock data to return."""
        self.mock_data = data


def create_mock_data(symbols):
    """Create mock price data for the given symbols."""
    mock_data = {}
    timestamp = pd.Timestamp.now(tz='UTC')
    
    for symbol in symbols:
        # Create a simple price based on the length of the symbol name
        # (just for demonstration)
        price = 100 + len(symbol) * 10
        
        mock_data[symbol] = {
            'timestamp': timestamp,
            'open': price * 0.99,
            'high': price * 1.01,
            'low': price * 0.98,
            'close': price,
            'volume': 1000
        }
    
    return mock_data


def run_mock_example():
    """Run example with MockBroker."""
    logger.info("=== Running example with MockBroker ===")
    
    # Create mock components
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL"]
    
    # Create a strategy
    strategy = SimpleMovingAverageStrategy(
        tickers=tickers,
        short_window=5,
        long_window=10,
        position_size=0.1
    )
    
    # Create a mock broker
    broker = MockBroker(initial_cash=100000.0)
    
    # Set mock prices in the broker
    for symbol in tickers:
        price = 100 + (ord(symbol[0]) - ord('A')) * 10  # Simple price based on first letter
        broker.set_price(symbol, price)
    
    # Create mock data loader
    mock_data = create_mock_data(tickers)
    data_loader = MockDataLoader(mock_data)
    
    # Create execution instance (using ExecutionBase directly)
    execution = ExecutionBase(
        strategy=strategy,
        market_data_loader=data_loader,
        broker=broker
    )
    
    # Run one step of the execution
    logger.info("Running execution step...")
    results = execution.step()
    
    # Log results
    logger.info("Execution results:")
    for symbol, success in results.items():
        logger.info(f"  {symbol}: {'Success' if success else 'Failed'}")
    
    # Log account state after execution
    account_info = broker.get_account_info()
    logger.info(f"Cash: ${account_info['cash']:.2f}")
    logger.info(f"Portfolio value: ${account_info['portfolio_value']:.2f}")
    logger.info("Positions:")
    for symbol, position in account_info['positions'].items():
        logger.info(f"  {symbol}: {position['qty']} shares, value: ${position['market_value']:.2f}")
    
    # Run with market closed to demonstrate that behavior
    broker.set_market_status(False)
    logger.info("Running execution step with market closed...")
    results = execution.step()
    logger.info(f"Results with market closed: {results}")
    
    # Return execution to market open
    broker.set_market_status(True)
    
    # Simulate price changes and run again
    logger.info("Simulating price changes and running again...")
    for symbol in tickers:
        # Increase prices by 5%
        new_price = broker.prices[symbol] * 1.05
        broker.set_price(symbol, new_price)
        
        # Update mock data
        mock_data[symbol]['close'] = new_price
        mock_data[symbol]['open'] = new_price * 0.99
        mock_data[symbol]['high'] = new_price * 1.01
        mock_data[symbol]['low'] = new_price * 0.98
    
    data_loader.set_mock_data(mock_data)
    
    # Run another step
    results = execution.step()
    
    # Log results after price changes
    logger.info("Execution results after price changes:")
    for symbol, success in results.items():
        logger.info(f"  {symbol}: {'Success' if success else 'Failed'}")
    
    # Log account state after execution
    account_info = broker.get_account_info()
    logger.info(f"Cash: ${account_info['cash']:.2f}")
    logger.info(f"Portfolio value: ${account_info['portfolio_value']:.2f}")
    logger.info("Positions:")
    for symbol, position in account_info['positions'].items():
        logger.info(f"  {symbol}: {position['qty']} shares, value: ${position['market_value']:.2f}")


def run_alpaca_example():
    """Run example with AlpacaBroker (if credentials are available)."""
    logger.info("=== Running example with AlpacaBroker ===")
    
    # Check for Alpaca credentials
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found in environment variables. Skipping AlpacaBroker example.")
        logger.warning("Set ALPACA_API_KEY and ALPACA_API_SECRET to run with Alpaca.")
        return
    
    try:
        # Create components for Alpaca example
        tickers = ["AAPL", "MSFT", "AMZN", "GOOGL"]
        
        # Create a strategy
        strategy = SimpleMovingAverageStrategy(
            tickers=tickers,
            short_window=5,
            long_window=10,
            position_size=0.1
        )
        
        # Create an Alpaca broker (using paper trading)
        broker = AlpacaBroker(
            api_key=api_key,
            api_secret=api_secret,
            paper_trading=True
        )
        
        # Create a data loader (using a simple mock for this example)
        # In a real scenario, you would use a proper market data loader
        data_loader = MockDataLoader(create_mock_data(tickers))
        
        # Create execution instance (using ExecutionBase directly with the Alpaca broker)
        execution = ExecutionBase(
            strategy=strategy,
            market_data_loader=data_loader,
            broker=broker
        )
        
        # Check if market is open
        is_open = broker.check_market_status()
        logger.info(f"Market is {'open' if is_open else 'closed'}")
        
        if is_open:
            # Run one step of the execution
            logger.info("Running execution step with Alpaca...")
            results = execution.step()
            
            # Log results
            logger.info("Execution results:")
            for symbol, success in results.items():
                logger.info(f"  {symbol}: {'Success' if success else 'Failed'}")
            
            # Get account info
            account_info = broker.get_account_info()
            logger.info(f"Cash: ${account_info.get('cash', 0):.2f}")
            logger.info(f"Portfolio value: ${account_info.get('portfolio_value', 0):.2f}")
        else:
            logger.info("Market is closed. Skipping execution step.")
        
    except Exception as e:
        logger.error(f"Error in Alpaca example: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run example broker execution")
    parser.add_argument("--broker", choices=["mock", "alpaca", "both"], default="mock",
                      help="Broker implementation to use for the example")
    
    args = parser.parse_args()
    
    try:
        if args.broker in ["mock", "both"]:
            run_mock_example()
            
        if args.broker in ["alpaca", "both"]:
            run_alpaca_example()
            
        logger.info("Example completed successfully")
    except Exception as e:
        logger.exception(f"Error running example: {e}")


if __name__ == "__main__":
    main() 