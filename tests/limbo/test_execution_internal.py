"""
Unit tests for the internal helper methods of ExecutionBase.

These tests specifically focus on the internal helper methods
_get_current_positions, _calculate_target_positions, and _execute_orders.
"""

import unittest

from portwine.execution import ExecutionBase, OrderExecutionError
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader
from portwine.execution.broker import BrokerBase


class MockStrategy(StrategyBase):
    """Mock strategy for testing."""
    
    def __init__(self, tickers=None):
        """Initialize with optional tickers."""
        self.tickers = tickers or ['AAPL', 'MSFT', 'AMZN']
        self.step_called = False
    
    def step(self, current_date, daily_data):
        """Mock step method."""
        self.step_called = True
        return {ticker: 0.0 for ticker in self.tickers}


class MockDataLoader(MarketDataLoader):
    """Mock data loader for testing."""
    
    def __init__(self, data=None, should_fail=False):
        """Initialize with optional preloaded data and failure flag."""
        self.data = data or {}
        self.should_fail = should_fail
        self.next_called = False
    
    def next(self, tickers, timestamp=None):
        """Mock implementation of next method."""
        self.next_called = True
        if self.should_fail:
            raise Exception("Simulated data fetch error")
            
        result = {}
        for ticker in tickers:
            if ticker in self.data:
                result[ticker] = self.data[ticker]
            else:
                result[ticker] = {'close': 100.0}
                
        return result


class MockBroker(BrokerBase):
    """Mock broker for testing."""
    
    def __init__(self, account_info=None, market_open=True):
        """Initialize with optional account info and market status."""
        self.market_open = market_open
        self.account_info = account_info or {
            'cash': 10000.0,
            'portfolio_value': 12000.0,
            'positions': {}
        }
        self.check_market_status_called = False
        self.get_account_info_called = False
        self.execute_order_called = False
        self.execution_results = {}  # Symbol -> success
        self.execution_exceptions = {}  # Symbol -> exception
    
    def check_market_status(self):
        """Mock market status check."""
        self.check_market_status_called = True
        return self.market_open
    
    def get_account_info(self):
        """Mock account info getter."""
        self.get_account_info_called = True
        return self.account_info
    
    def execute_order(self, symbol, qty, order_type="market", **kwargs):
        """Mock order execution."""
        self.execute_order_called = True
        
        # Check if we should raise an exception for this symbol
        if symbol in self.execution_exceptions:
            raise self.execution_exceptions[symbol]
            
        # Return predefined result or True by default
        return self.execution_results.get(symbol, True)


class TestExecutionBaseInternal(unittest.TestCase):
    """Tests for internal helper methods of ExecutionBase."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MockStrategy()
        self.data_loader = MockDataLoader()
        self.broker = MockBroker()
        
        self.execution = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            broker=self.broker
        )
    
    #-------------------------------------------------------------------------
    # Tests for _get_current_positions
    #-------------------------------------------------------------------------
    
    def test_get_current_positions_with_positions(self):
        """Test getting current positions when positions exist."""
        # Setup broker with positions
        self.broker.account_info = {
            'cash': 10000.0,
            'portfolio_value': 25000.0,
            'positions': {
                'AAPL': {'qty': 10, 'market_value': 1500.0},
                'MSFT': {'qty': 20, 'market_value': 5000.0},
                'AMZN': {'qty': 5, 'market_value': 8500.0}
            }
        }
        
        positions, portfolio_value = self.execution._get_current_positions()
        
        # Verify positions
        self.assertEqual(len(positions), 3)
        self.assertEqual(positions['AAPL'], 10)
        self.assertEqual(positions['MSFT'], 20)
        self.assertEqual(positions['AMZN'], 5)
        
        # Verify portfolio value
        self.assertEqual(portfolio_value, 25000.0)
        
        # Verify broker method was called
        self.assertTrue(self.broker.get_account_info_called)
    
    def test_get_current_positions_empty(self):
        """Test getting current positions when no positions exist."""
        # Setup broker with no positions
        self.broker.account_info = {
            'cash': 10000.0,
            'portfolio_value': 10000.0,
            'positions': {}
        }
        
        positions, portfolio_value = self.execution._get_current_positions()
        
        # Verify empty positions
        self.assertEqual(positions, {})
        
        # Verify portfolio value
        self.assertEqual(portfolio_value, 10000.0)
    
    def test_get_current_positions_missing_qty(self):
        """Test getting positions when qty field is missing."""
        # Setup broker with a position missing qty
        self.broker.account_info = {
            'cash': 10000.0,
            'portfolio_value': 15000.0,
            'positions': {
                'AAPL': {'market_value': 1500.0},  # Missing qty
                'MSFT': {'qty': 20, 'market_value': 3500.0}
            }
        }
        
        positions, portfolio_value = self.execution._get_current_positions()
        
        # Verify positions (AAPL should default to 0)
        self.assertEqual(positions['AAPL'], 0)
        self.assertEqual(positions['MSFT'], 20)
    
    def test_get_current_positions_missing_positions_field(self):
        """Test handling when positions field is missing in account info."""
        # Setup broker with missing positions field
        self.broker.account_info = {
            'cash': 10000.0,
            'portfolio_value': 10000.0
            # Missing positions field
        }
        
        positions, portfolio_value = self.execution._get_current_positions()
        
        # Verify empty positions
        self.assertEqual(positions, {})
    
    def test_get_current_positions_missing_portfolio_value(self):
        """Test handling when portfolio_value is missing."""
        # Setup broker with missing portfolio_value
        self.broker.account_info = {
            'cash': 10000.0,
            'positions': {'AAPL': {'qty': 10}}
            # Missing portfolio_value
        }
        
        positions, portfolio_value = self.execution._get_current_positions()
        
        # Verify positions are correct
        self.assertEqual(positions['AAPL'], 10)
        
        # Verify portfolio value defaults to 0
        self.assertEqual(portfolio_value, 0)
    
    #-------------------------------------------------------------------------
    # Tests for _calculate_target_positions
    #-------------------------------------------------------------------------
    
    def test_calculate_target_positions_normal(self):
        """Test normal calculation of target positions."""
        target_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'AMZN': 0.2}
        portfolio_value = 10000.0
        prices = {'AAPL': 150.0, 'MSFT': 250.0, 'AMZN': 3200.0}
        
        target_positions = self.execution._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        
        # Expected shares = (weight * portfolio_value) / price
        self.assertEqual(len(target_positions), 3)
        self.assertAlmostEqual(target_positions['AAPL'], (0.4 * 10000) / 150.0, places=10)
        self.assertAlmostEqual(target_positions['MSFT'], (0.3 * 10000) / 250.0, places=10)
        self.assertAlmostEqual(target_positions['AMZN'], (0.2 * 10000) / 3200.0, places=10)
    
    def test_calculate_target_positions_zero_weight(self):
        """Test calculation when a symbol has zero weight."""
        target_weights = {'AAPL': 0.0, 'MSFT': 0.5}
        portfolio_value = 10000.0
        prices = {'AAPL': 150.0, 'MSFT': 250.0}
        
        target_positions = self.execution._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        
        # Zero weight should result in 0 shares
        self.assertAlmostEqual(target_positions['AAPL'], 0.0, places=10)
        self.assertAlmostEqual(target_positions['MSFT'], (0.5 * 10000) / 250.0, places=10)
    
    def test_calculate_target_positions_missing_price(self):
        """Test when a price is missing for a symbol."""
        target_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'AMZN': 0.2}
        portfolio_value = 10000.0
        prices = {'AAPL': 150.0, 'MSFT': 250.0}  # AMZN price missing
        
        target_positions = self.execution._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        
        # Should only have positions for symbols with prices
        self.assertEqual(len(target_positions), 2)
        self.assertIn('AAPL', target_positions)
        self.assertIn('MSFT', target_positions)
        self.assertNotIn('AMZN', target_positions)
    
    def test_calculate_target_positions_zero_price(self):
        """Test when a price is zero for a symbol."""
        target_weights = {'AAPL': 0.4, 'MSFT': 0.3}
        portfolio_value = 10000.0
        prices = {'AAPL': 150.0, 'MSFT': 0.0}  # Zero price for MSFT
        
        target_positions = self.execution._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        
        # Should only have positions for symbols with positive prices
        self.assertEqual(len(target_positions), 1)
        self.assertIn('AAPL', target_positions)
        self.assertNotIn('MSFT', target_positions)
    
    def test_calculate_target_positions_empty_weights(self):
        """Test with empty target weights."""
        target_weights = {}
        portfolio_value = 10000.0
        prices = {'AAPL': 150.0, 'MSFT': 250.0}
        
        target_positions = self.execution._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        
        # Should have empty positions
        self.assertEqual(target_positions, {})
    
    def test_calculate_target_positions_zero_portfolio_value(self):
        """Test with zero portfolio value."""
        target_weights = {'AAPL': 0.4, 'MSFT': 0.3}
        portfolio_value = 0.0
        prices = {'AAPL': 150.0, 'MSFT': 250.0}
        
        target_positions = self.execution._calculate_target_positions(
            target_weights, portfolio_value, prices
        )
        
        # All positions should be zero
        self.assertEqual(target_positions['AAPL'], 0.0)
        self.assertEqual(target_positions['MSFT'], 0.0)
    
    #-------------------------------------------------------------------------
    # Tests for _execute_orders
    #-------------------------------------------------------------------------
    
    def test_execute_orders_success(self):
        """Test successful execution of orders."""
        orders = [
            {'symbol': 'AAPL', 'qty': 10, 'order_type': 'market'},
            {'symbol': 'MSFT', 'qty': -5, 'order_type': 'limit'}
        ]
        
        # Both orders succeed by default
        results = self.execution._execute_orders(orders)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results['AAPL'])
        self.assertTrue(results['MSFT'])
        
        # Verify broker method was called
        self.assertTrue(self.broker.execute_order_called)
    
    def test_execute_orders_empty(self):
        """Test execution with empty orders list."""
        results = self.execution._execute_orders([])
        
        # Should return empty results
        self.assertEqual(results, {})
        
        # Broker method should not be called
        self.assertFalse(self.broker.execute_order_called)
    
    def test_execute_orders_failure(self):
        """Test when broker returns False for order execution."""
        orders = [
            {'symbol': 'AAPL', 'qty': 10},
            {'symbol': 'MSFT', 'qty': -5}
        ]
        
        # MSFT order should fail
        self.broker.execution_results = {'MSFT': False}
        
        results = self.execution._execute_orders(orders)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results['AAPL'])
        self.assertFalse(results['MSFT'])
    
    def test_execute_orders_exception(self):
        """Test when broker raises exception during order execution."""
        orders = [
            {'symbol': 'AAPL', 'qty': 10},
            {'symbol': 'MSFT', 'qty': -5}
        ]
        
        # MSFT order should raise exception
        self.broker.execution_exceptions = {
            'MSFT': OrderExecutionError("Failed to execute order")
        }
        
        results = self.execution._execute_orders(orders)
        
        # Check results - MSFT should be False due to exception
        self.assertEqual(len(results), 2)
        self.assertTrue(results['AAPL'])
        self.assertFalse(results['MSFT'])
    
    def test_execute_orders_different_order_types(self):
        """Test execution with different order types."""
        orders = [
            {'symbol': 'AAPL', 'qty': 10, 'order_type': 'market'},
            {'symbol': 'MSFT', 'qty': -5, 'order_type': 'limit'},
            {'symbol': 'AMZN', 'qty': 2, 'order_type': 'stop'}
        ]
        
        # Implement a mock to verify different order types are passed correctly
        original_execute = self.broker.execute_order
        order_types_seen = {}
        
        def mock_execute(symbol, qty, order_type="market", **kwargs):
            order_types_seen[symbol] = order_type
            return True
            
        self.broker.execute_order = mock_execute
        
        results = self.execution._execute_orders(orders)
        
        # Restore original method
        self.broker.execute_order = original_execute
        
        # Check order types
        self.assertEqual(order_types_seen['AAPL'], 'market')
        self.assertEqual(order_types_seen['MSFT'], 'limit')
        self.assertEqual(order_types_seen['AMZN'], 'stop')
    
    def test_execute_orders_missing_qty(self):
        """Test that orders missing quantity field raise an exception."""
        # Create test orders - one missing qty, one with qty
        orders = [
            {'symbol': 'AAPL'},  # Missing qty
            {'symbol': 'MSFT', 'qty': 5}
        ]
        
        # Execute orders - should raise OrderExecutionError
        with self.assertRaises(OrderExecutionError) as cm:
            self.execution._execute_orders(orders)
            
        # Verify the error message mentions the missing quantity
        self.assertIn('Missing quantity', str(cm.exception))
        self.assertIn('AAPL', str(cm.exception))


if __name__ == '__main__':
    unittest.main() 