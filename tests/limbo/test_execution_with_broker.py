"""
Tests for the ExecutionBase class with different broker implementations.

This module verifies that the ExecutionBase class works correctly with
different broker implementations.
"""

import unittest

from portwine.execution import ExecutionBase
from portwine.execution.broker import BrokerBase


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, tickers=None):
        self.tickers = tickers or ["AAPL", "MSFT"]
        self.called_step = False
        self.signals = {"AAPL": 0.5, "MSFT": 0.5}
    
    def step(self, timestamp, data):
        """Process market data and generate signals."""
        self.called_step = True
        return self.signals
    
    def generate_signals(self):
        """Return the current signals."""
        return self.signals


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data=None):
        self.data = data or {}
        self.called_next = False
    
    def next(self, tickers, timestamp=None):
        """Return the next data point."""
        self.called_next = True
        
        # If no data provided, create some defaults
        if not self.data:
            self.data = {
                ticker: {
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 1000
                } for ticker in tickers
            }
            
        return self.data


class MockBroker(BrokerBase):
    """Mock broker for testing."""
    
    def __init__(self, market_open=True, cash=100000.0, positions=None):
        """Initialize with test data."""
        self.market_open = market_open
        self.cash = cash
        self.positions = positions or {}
        self.executed_orders = []
    
    def check_market_status(self):
        """Check if market is open."""
        return self.market_open
    
    def get_account_info(self):
        """Get account information."""
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        return {
            "cash": self.cash,
            "portfolio_value": self.cash + position_value,
            "positions": self.positions
        }
    
    def execute_order(self, symbol, qty, order_type="market", **kwargs):
        """Execute an order."""
        self.executed_orders.append({
            "symbol": symbol,
            "qty": qty,
            "order_type": order_type,
            **kwargs
        })
        return True
    
    def get_order_status(self, order_id):
        """Get order status."""
        return "filled"
    
    def cancel_all_orders(self):
        """Cancel all orders."""
        return True
    
    def close_all_positions(self):
        """Close all positions."""
        return True
    
    def get_positions(self):
        """Get all positions."""
        return list(self.positions.values())
    
    def get_cash(self):
        """Get available cash."""
        return self.cash
    
    def get_portfolio_value(self):
        """Get portfolio value."""
        position_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
        return self.cash + position_value


class TestExecutionWithBroker(unittest.TestCase):
    """Tests for the execution system with different broker implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MockStrategy(tickers=["AAPL", "MSFT"])
        self.data_loader = MockDataLoader()
        self.broker = MockBroker(market_open=True)
        
        # Create test positions
        self.broker.positions = {
            "AAPL": {
                "symbol": "AAPL",
                "qty": 10,
                "market_value": 1000.0,
                "avg_entry_price": 95.0,
                "unrealized_pl": 50.0
            }
        }
        
        # Create execution instance
        self.execution = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            broker=self.broker
        )
    
    def test_execution_with_market_open(self):
        """Test that execution works when market is open."""
        # Run execution step
        results = self.execution.step()
        
        # Verify that data loader and strategy were called
        self.assertTrue(self.data_loader.called_next)
        self.assertTrue(self.strategy.called_step)
        
        # Verify that orders were executed
        self.assertTrue(len(self.broker.executed_orders) > 0)
    
    def test_execution_with_market_closed(self):
        """Test that execution doesn't proceed when market is closed."""
        # Set market to closed
        self.broker.market_open = False
        
        # Run execution step
        results = self.execution.step()
        
        # Verify that no orders were executed
        self.assertEqual(len(self.broker.executed_orders), 0)
        
        # Verify that the result is an empty dict
        self.assertEqual(results, {})
    
    def test_execution_with_no_position_changes(self):
        """Test execution when no position changes are needed."""
        # Modify current positions to match target positions
        # If strategy returns 50/50 weights with $100k portfolio value,
        # target positions should be approximately 490 AAPL, 490 MSFT 
        # (with $100 price per share)
        portfolio_value = self.broker.get_portfolio_value()
        price = 100.0
        
        # First, set up mock data loader to return consistent prices
        self.data_loader.data = {
            "AAPL": {"close": price},
            "MSFT": {"close": price}
        }
        
        # Set up positions to exactly match what the strategy would recommend
        target_aapl_shares = int((portfolio_value * 0.5) / price)
        target_msft_shares = int((portfolio_value * 0.5) / price)
        
        self.broker.positions = {
            "AAPL": {
                "symbol": "AAPL",
                "qty": target_aapl_shares,
                "market_value": target_aapl_shares * price,
                "avg_entry_price": price,
                "unrealized_pl": 0.0
            },
            "MSFT": {
                "symbol": "MSFT",
                "qty": target_msft_shares,
                "market_value": target_msft_shares * price,
                "avg_entry_price": price,
                "unrealized_pl": 0.0
            }
        }
        
        # Run execution step
        results = self.execution.step()
        
        # Verify that no orders were executed
        self.assertEqual(len(self.broker.executed_orders), 0)
    
    def test_execution_with_different_broker(self):
        """Test that execution works with a different broker implementation."""
        # Create a custom broker implementation
        class CustomBroker(BrokerBase):
            def __init__(self):
                self.orders = []
                
            def check_market_status(self):
                return True
                
            def get_account_info(self):
                return {
                    "cash": 100000.0,
                    "portfolio_value": 100000.0,
                    "positions": {}
                }
                
            def execute_order(self, symbol, qty, **kwargs):
                self.orders.append((symbol, qty))
                return True
                
            def get_order_status(self, order_id):
                return "filled"
                
            def cancel_all_orders(self):
                return True
                
            def close_all_positions(self):
                return True
                
            def get_positions(self):
                return []
                
            def get_cash(self):
                return 100000.0
                
            def get_portfolio_value(self):
                return 100000.0
        
        # Create execution with the custom broker
        custom_broker = CustomBroker()
        execution = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            broker=custom_broker
        )
        
        # Run execution step
        results = execution.step()
        
        # Verify that orders were executed on the custom broker
        self.assertTrue(len(custom_broker.orders) > 0)
    
    def test_execution_error_handling(self):
        """Test that execution handles errors properly."""
        # Make broker raise an exception on execute_order
        def raise_error(*args, **kwargs):
            raise Exception("Test error")
            
        self.broker.execute_order = raise_error
        
        # Run execution step
        results = self.execution.step()
        
        # Verify that the result indicates failed execution
        self.assertTrue(all(not success for success in results.values()))


if __name__ == '__main__':
    unittest.main() 