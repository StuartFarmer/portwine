"""
Unit tests for the ExecutionBase class.

These tests verify that the ExecutionBase class correctly handles
execution timing, strategy calls, and trade execution.
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from portwine.execution.base import ExecutionBase
from portwine.execution.broker import BrokerBase, AccountInfo, Position, Order
from portwine.utils.market_calendar import MarketStatus


class SimpleTestStrategy:
    """A simple strategy for testing."""
    
    def __init__(self):
        self.step_called = False
        self.tickers = ['AAPL', 'MSFT', 'AMZN']
    
    def step(self, current_date, daily_data):
        """Simple strategy step that returns a fixed position change."""
        self.step_called = True
        return {'AAPL': 10, 'MSFT': -5, 'AMZN': 0}


class MockBroker(BrokerBase):
    """Mock implementation of BrokerBase for testing."""
    
    def __init__(self, is_market_open=True):
        self.is_market_open = is_market_open
        self.orders_executed = {}
        self.get_account_info_called = False
        self.check_market_status_called = False
    
    def check_market_status(self):
        """Check if the market is open."""
        self.check_market_status_called = True
        market_status = MarketStatus(
            is_open=self.is_market_open,
            next_open=None,
            next_close=None
        )
        return market_status

    def get_account_info(self):
        """Get the account information."""
        self.get_account_info_called = True

        positions = {
            'AAPL': Position(
                symbol='AAPL',
                qty=100,
                market_value=15000.0,
                avg_entry_price=150.0,
                unrealized_pl=0.0
            ),
            'MSFT': Position(
                symbol='MSFT',
                qty=50,
                market_value=10000.0,
                avg_entry_price=200.0,
                unrealized_pl=0.0
            )
        }

        account_info = AccountInfo(
            cash=100000.0,
            portfolio_value=125000.0,
            positions=positions
        )
        return account_info

    def execute_order(self, symbol, qty, order_type="market"):
        """Execute an order."""
        order = Order(
            symbol=symbol,
            qty=qty,
            order_type=order_type,
            status="filled"
        )
        self.orders_executed[symbol] = order
        return True


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self):
        self.next_called = False

    def next(self, tickers, timestamp):
        """Return mock data for the tickers."""
        self.next_called = True
        return {ticker: {'close': 100.0} for ticker in tickers}


class TestExecutionBase(unittest.TestCase):
    """Test the ExecutionBase class."""
    
    def setUp(self):
        """Set up the test."""
        self.strategy = SimpleTestStrategy()
        self.broker = MockBroker()
        self.data_loader = MockDataLoader()
        self.schedule = MagicMock()

        self.execution = ExecutionBase(
            strategy=self.strategy,
            broker=self.broker,
            schedule=self.schedule,
            market_data_loader=self.data_loader
        )
    
    def test_initialization(self):
        """Test that the execution system is initialized correctly."""
        self.assertEqual(self.execution.strategy, self.strategy)
        self.assertEqual(self.execution.broker, self.broker)
        self.assertEqual(self.execution.schedule, self.schedule)
        self.assertEqual(self.execution.market_data_loader, self.data_loader)
        self.assertEqual(self.execution.tickers, self.strategy.tickers)
        self.assertFalse(self.execution._running)
    
    def test_step_market_open(self):
        """Test that step executes the strategy and returns orders when market is open."""
        self.broker.is_market_open = True

        result = self.execution.step()

        # Check that the broker methods were called
        self.assertTrue(self.broker.check_market_status_called)
        self.assertTrue(self.broker.get_account_info_called)

        # Check that the strategy step was called
        self.assertTrue(self.strategy.step_called)

        # Check that orders were executed
        self.assertEqual(len(result), 2)  # AAPL and MSFT, AMZN has quantity 0
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        self.assertNotIn('AMZN', result)

        # Check the broker's orders
        self.assertEqual(len(self.broker.orders_executed), 2)
        self.assertIn('AAPL', self.broker.orders_executed)
        self.assertIn('MSFT', self.broker.orders_executed)

    def test_step_market_closed(self):
        """Test that step does not execute orders when the market is closed."""
        self.broker.is_market_open = False

        result = self.execution.step()
        
        # Check that the broker method was called
        self.assertTrue(self.broker.check_market_status_called)
        
        # Check that other methods were not called
        self.assertFalse(self.broker.get_account_info_called)
        self.assertFalse(self.strategy.step_called)
        
        # Check that no orders were executed
        self.assertEqual(len(result), 0)
        self.assertEqual(len(self.broker.orders_executed), 0)
    
    def test_run_once(self):
        """Test run_once method execution."""
        self.broker.is_market_open = True
        
        # Run a single step
        result = self.execution.run_once()
        
        # Check that it called the necessary methods
        self.assertTrue(self.broker.check_market_status_called)
        self.assertTrue(self.strategy.step_called)

        # Check that orders were executed
        self.assertEqual(len(result), 2)  # AAPL and MSFT
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)

    def test_fetch_latest_data(self):
        """Test that fetch_latest_data calls the data loader correctly."""
        timestamp = pd.Timestamp.now()
        
        data = self.execution.fetch_latest_data(timestamp)

        # Check that the data loader was called
        self.assertTrue(self.data_loader.next_called)
        
        # Check that the data was returned correctly
        self.assertEqual(len(data), len(self.strategy.tickers))
        for ticker in self.strategy.tickers:
            self.assertIn(ticker, data)
            self.assertEqual(data[ticker]['close'], 100.0)

    def test_get_current_prices(self):
        """Test that get_current_prices returns the correct prices."""
        prices = self.execution.get_current_prices(self.strategy.tickers)

        # Check that the prices were returned correctly
        self.assertEqual(len(prices), len(self.strategy.tickers))
        for ticker in self.strategy.tickers:
            self.assertIn(ticker, prices)
            self.assertEqual(prices[ticker], 100.0)

    def test_run_with_schedule(self):
        """Test that run uses the schedule correctly."""
        # Configure the schedule to yield a time and then raise StopIteration
        next_time = pd.Timestamp.now()
        self.schedule.__iter__.return_value = iter([next_time])

        # Run with the schedule
        self.execution.run()
        
        # Check that it called next on the schedule
        self.schedule.__iter__.assert_called_once()

        # Check that the broker and strategy were called
        self.assertTrue(self.broker.check_market_status_called)
        self.assertTrue(self.strategy.step_called)

    def test_stop(self):
        """Test that stop sets _running to False."""
        self.execution._running = True
        self.execution.stop()
        self.assertFalse(self.execution._running)


if __name__ == '__main__':
    unittest.main() 