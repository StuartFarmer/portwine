"""
Unit tests for the ExecutionBase class.

These tests verify that the ExecutionBase class correctly handles
initialization, data fetching, and price retrieval.
"""

import unittest
import pandas as pd
from datetime import timezone

from portwine.execution import ExecutionBase, DataFetchError
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
        self.next_args = None
    
    def next(self, tickers, timestamp=None):
        """Mock implementation of next method."""
        self.next_called = True
        self.next_args = (tickers, timestamp)
        
        if self.should_fail:
            raise Exception("Simulated data fetch error")
            
        result = {}
        for ticker in tickers:
            if ticker in self.data:
                result[ticker] = self.data[ticker]
            else:
                result[ticker] = {'close': 100.0}  # Default data
                
        return result


class MockBroker(BrokerBase):
    """Mock broker for testing."""
    
    def __init__(self, market_open=True):
        """Initialize with market open status."""
        self.market_open = market_open
        self.check_market_status_called = False
        self.get_account_info_called = False
        self.execute_order_called = False
        self.executed_orders = {}
    
    def check_market_status(self):
        """Mock market status check."""
        self.check_market_status_called = True
        return self.market_open
    
    def get_account_info(self):
        """Mock account info getter."""
        self.get_account_info_called = True
        return {
            'cash': 10000.0,
            'portfolio_value': 12000.0,
            'positions': {}
        }
    
    def execute_order(self, symbol, qty, order_type="market", **kwargs):
        """Mock order execution."""
        self.execute_order_called = True
        self.executed_orders[symbol] = {
            'qty': qty,
            'order_type': order_type,
            **kwargs
        }
        return True


class TestExecutionBase(unittest.TestCase):
    """Test suite for the ExecutionBase class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.strategy = MockStrategy()
        self.data_loader = MockDataLoader()
        self.broker = MockBroker()
        
        self.execution = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            broker=self.broker
        )
    
    def test_initialization(self):
        """Test that initialization correctly sets attributes."""
        self.assertEqual(self.execution.strategy, self.strategy)
        self.assertEqual(self.execution.market_data_loader, self.data_loader)
        self.assertEqual(self.execution.broker, self.broker)
        self.assertEqual(self.execution.tickers, self.strategy.tickers)
        self.assertEqual(self.execution.min_change_pct, 0.01)  # Default value
        self.assertEqual(self.execution.min_order_value, 1.0)  # Default value
    
    def test_initialization_with_optional_params(self):
        """Test initialization with optional parameters."""
        alt_data_loader = MockDataLoader()
        execution = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            broker=self.broker,
            alternative_data_loader=alt_data_loader,
            min_change_pct=0.02,
            min_order_value=10.0
        )
        
        self.assertEqual(execution.alternative_data_loader, alt_data_loader)
        self.assertEqual(execution.min_change_pct, 0.02)
        self.assertEqual(execution.min_order_value, 10.0)
    
    def test_initialization_missing_required_params(self):
        """Test initialization with missing required parameters."""
        # Test missing strategy
        with self.assertRaises(TypeError):
            ExecutionBase(
                market_data_loader=self.data_loader,
                broker=self.broker
            )
        
        # Test missing market_data_loader
        with self.assertRaises(TypeError):
            ExecutionBase(
                strategy=self.strategy,
                broker=self.broker
            )
        
        # Test missing broker
        with self.assertRaises(TypeError):
            ExecutionBase(
                strategy=self.strategy,
                market_data_loader=self.data_loader
            )
    
    def test_fetch_latest_data_success(self):
        """Test successful fetching of latest data."""
        timestamp = pd.Timestamp.now(tz=timezone.utc)
        test_data = {
            'AAPL': {'close': 150.0, 'open': 148.0},
            'MSFT': {'close': 250.0, 'open': 248.0},
            'AMZN': {'close': 3200.0, 'open': 3180.0}
        }
        
        self.data_loader.data = test_data
        result = self.execution.fetch_latest_data(timestamp)
        
        # Check that the data loader was called with correct parameters
        self.assertTrue(self.data_loader.next_called)
        self.assertEqual(self.data_loader.next_args[0], self.strategy.tickers)
        self.assertEqual(self.data_loader.next_args[1], timestamp)
        
        # Check that the result contains the correct data
        self.assertEqual(result, test_data)
    
    def test_fetch_latest_data_no_timestamp(self):
        """Test fetching latest data with no timestamp."""
        test_data = {
            'AAPL': {'close': 150.0},
            'MSFT': {'close': 250.0},
            'AMZN': {'close': 3200.0}
        }
        
        self.data_loader.data = test_data
        result = self.execution.fetch_latest_data()
        
        # Check that the data loader was called
        self.assertTrue(self.data_loader.next_called)
        
        # The timestamp should be None or automatically created
        if self.data_loader.next_args[1] is not None:
            self.assertIsInstance(self.data_loader.next_args[1], pd.Timestamp)
        
        # Check that the result contains the correct data
        self.assertEqual(result, test_data)
    
    def test_fetch_latest_data_with_alternative_loader(self):
        """Test fetching latest data with alternative data loader."""
        # Setup main data
        main_data = {
            'AAPL': {'close': 150.0, 'open': 148.0},
            'MSFT': {'close': 250.0, 'open': 248.0},
            'AMZN': {'close': 3200.0, 'open': 3180.0}
        }
        self.data_loader.data = main_data
        
        # Setup alternative data
        alt_data = {
            'AAPL': {'sentiment': 0.8},
            'MSFT': {'sentiment': 0.6},
            'AMZN': {'sentiment': 0.7}
        }
        
        # Create alt data loader and execution with it
        alt_loader = MockDataLoader(data=alt_data)
        execution = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=self.data_loader,
            broker=self.broker,
            alternative_data_loader=alt_loader
        )
        
        # Expected merged data
        expected_data = {
            'AAPL': {'close': 150.0, 'open': 148.0, 'sentiment': 0.8},
            'MSFT': {'close': 250.0, 'open': 248.0, 'sentiment': 0.6},
            'AMZN': {'close': 3200.0, 'open': 3180.0, 'sentiment': 0.7}
        }
        
        # Test fetching with both loaders
        result = execution.fetch_latest_data()
        
        # Check that both loaders were called
        self.assertTrue(self.data_loader.next_called)
        self.assertTrue(alt_loader.next_called)
        
        # Check that data was merged correctly
        self.assertEqual(result, expected_data)
    
    def test_fetch_latest_data_error(self):
        """Test error handling when fetching data fails."""
        self.data_loader.should_fail = True
        
        with self.assertRaises(DataFetchError):
            self.execution.fetch_latest_data()
    
    def test_get_current_prices_success(self):
        """Test successful retrieval of current prices."""
        test_data = {
            'AAPL': {'close': 150.0, 'open': 148.0},
            'MSFT': {'close': 250.0, 'open': 248.0},
            'AMZN': {'close': 3200.0, 'open': 3180.0}
        }
        
        expected_prices = {
            'AAPL': 150.0,
            'MSFT': 250.0,
            'AMZN': 3200.0
        }
        
        self.data_loader.data = test_data
        prices = self.execution.get_current_prices(self.strategy.tickers)
        
        # Check that the prices match expected values
        self.assertEqual(prices, expected_prices)
    
    def test_get_current_prices_missing_data(self):
        """Test price retrieval with missing data for some tickers."""
        test_data = {
            'AAPL': {'close': 150.0},
            # MSFT is missing close
            'MSFT': {'open': 248.0},
            'AMZN': {'close': 3200.0}
        }
        
        expected_prices = {
            'AAPL': 150.0,
            # MSFT should be missing
            'AMZN': 3200.0
        }
        
        self.data_loader.data = test_data
        prices = self.execution.get_current_prices(self.strategy.tickers)
        
        # Check that prices include only tickers with close data
        self.assertEqual(prices, expected_prices)
    
    def test_get_current_prices_empty_tickers(self):
        """Test price retrieval with empty ticker list."""
        prices = self.execution.get_current_prices([])
        
        # Should return empty dictionary
        self.assertEqual(prices, {})
    
    def test_get_current_prices_error_propagation(self):
        """Test that errors from fetch_latest_data propagate to get_current_prices."""
        self.data_loader.should_fail = True
        
        with self.assertRaises(DataFetchError):
            self.execution.get_current_prices(self.strategy.tickers)
    
    def test_step_market_open(self):
        """Test step execution when market is open."""
        timestamp = pd.Timestamp.now(tz=timezone.utc)
        test_data = {
            'AAPL': {'close': 150.0},
            'MSFT': {'close': 250.0},
            'AMZN': {'close': 3200.0}
        }
        self.data_loader.data = test_data
        
        # Mock broker to return positions
        self.broker.get_account_info = lambda: {
            'cash': 10000.0,
            'portfolio_value': 12000.0,
            'positions': {
                'AAPL': {'qty': 10},
                'MSFT': {'qty': 5}
            }
        }
        
        # Set up strategy to return non-zero allocations

        # Run step
        results = self.execution.step(timestamp)
        
        # Check that broker methods were called
        self.assertTrue(self.broker.check_market_status_called)
        self.assertTrue(self.broker.execute_order_called)
        
        # At least one order should have been executed
        self.assertGreater(len(results), 0)
    
    def test_step_market_closed(self):
        """Test step execution when market is closed."""
        self.broker.market_open = False
        timestamp = pd.Timestamp.now(tz=timezone.utc)
        
        results = self.execution.step(timestamp)
        
        # Check that market status was checked
        self.assertTrue(self.broker.check_market_status_called)
        
        # No orders should be executed when market is closed
        self.assertEqual(results, {})
        self.assertFalse(self.broker.execute_order_called)
    
    def test_step_with_error(self):
        """Test step execution when an error occurs."""
        # Make the data loader fail
        self.data_loader.should_fail = True
        timestamp = pd.Timestamp.now(tz=timezone.utc)
        
        # Should return empty dict on error, not raise
        results = self.execution.step(timestamp)
        self.assertEqual(results, {})


if __name__ == '__main__':
    unittest.main() 