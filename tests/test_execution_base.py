"""
Unit tests for the ExecutionBase class.

These tests verify that the ExecutionBase class correctly handles
execution timing, strategy calls, and trade execution.
"""

import datetime
import signal
import time
import unittest
from unittest.mock import MagicMock, patch, call

from portwine.execution_complex.execution_base import ExecutionBase
from portwine.execution_complex.broker import BrokerBase, AccountInfo, Order


class SimpleTestStrategy:
    """A simple strategy for testing."""
    
    def __init__(self):
        self.step_called = False
        self.tickers = ['AAPL', 'MSFT', 'AMZN']
    
    def step(self, account_info, market_status):
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
        market_status = MagicMock()
        market_status.is_open = self.is_market_open
        return market_status
    
    def get_account_info(self):
        """Get the account information."""
        self.get_account_info_called = True
        account_info = AccountInfo(
            cash=100000.0,
            equity=150000.0,
            buying_power=200000.0,
            positions={'AAPL': 100, 'MSFT': 50}
        )
        return account_info
    
    def execute_order(self, symbol, quantity):
        """Execute an order."""
        order = Order(
            id=f"order-{symbol}-{quantity}",
            symbol=symbol,
            quantity=quantity,
            status="filled",
            filled_quantity=quantity,
            filled_avg_price=100.0,
            side="buy" if quantity > 0 else "sell",
            created_at=datetime.datetime.now()
        )
        self.orders_executed[symbol] = order
        return order


class TestExecutionBase(unittest.TestCase):
    """Test the ExecutionBase class."""
    
    def setUp(self):
        """Set up the test."""
        self.strategy = SimpleTestStrategy()
        self.broker = MockBroker()
        self.execution = ExecutionBase(
            strategy=self.strategy,
            broker=self.broker,
            max_iterations=1
        )
    
    def test_initialization(self):
        """Test that the execution system is initialized correctly."""
        self.assertEqual(self.execution.strategy, self.strategy)
        self.assertEqual(self.execution.broker, self.broker)
        self.assertEqual(self.execution.max_iterations, 1)
        self.assertTrue(self.execution.running)
        self.assertEqual(self.execution._original_handlers, {})
    
    def test_signal_handler_setup(self):
        """Test that the signal handlers are set up correctly."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        
        self.execution._setup_signal_handlers()
        
        # Check that the original handlers are saved
        self.assertEqual(self.execution._original_handlers[signal.SIGINT], original_sigint)
        self.assertEqual(self.execution._original_handlers[signal.SIGTERM], original_sigterm)
        
        # Check that the new handlers are set
        self.assertEqual(signal.getsignal(signal.SIGINT), self.execution.signal_handler)
        self.assertEqual(signal.getsignal(signal.SIGTERM), self.execution.signal_handler)
        
        # Clean up
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
    
    def test_signal_handler_cleanup(self):
        """Test that the signal handlers are cleaned up correctly."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        
        self.execution._setup_signal_handlers()
        self.execution._cleanup_signal_handlers()
        
        # Check that the original handlers are restored
        self.assertEqual(signal.getsignal(signal.SIGINT), original_sigint)
        self.assertEqual(signal.getsignal(signal.SIGTERM), original_sigterm)
    
    def test_signal_handler(self):
        """Test that the signal handler sets running to False."""
        self.execution.running = True
        self.execution.signal_handler(signal.SIGINT, None)
        self.assertFalse(self.execution.running)
    
    def test_get_next_execution_time(self):
        """Test that _get_next_execution_time returns the next execution time."""
        next_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
        schedule_iterator = iter([next_time])
        
        result = self.execution._get_next_execution_time(schedule_iterator)
        
        self.assertEqual(result, next_time)
        self.assertTrue(self.execution.running)
    
    def test_get_next_execution_time_exhausted(self):
        """Test that _get_next_execution_time handles exhausted iterators."""
        schedule_iterator = iter([])
        
        result = self.execution._get_next_execution_time(schedule_iterator)
        
        self.assertIsNone(result)
        self.assertFalse(self.execution.running)
    
    @patch('time.sleep')
    def test_wait_until_execution_time(self, mock_sleep):
        """Test that _wait_until_execution_time waits until the execution time."""
        # Test waiting for a future time
        future_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
        self.execution._wait_until_execution_time(future_time)
        mock_sleep.assert_called_once()
        
        # Test not waiting for a past time
        mock_sleep.reset_mock()
        past_time = datetime.datetime.now() - datetime.timedelta(seconds=10)
        self.execution._wait_until_execution_time(past_time)
        mock_sleep.assert_not_called()
    
    def test_execute_iteration_market_open(self):
        """Test that _execute_iteration executes the strategy step and returns orders."""
        self.broker.is_market_open = True
        
        result = self.execution._execute_iteration()
        
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
    
    def test_execute_iteration_market_closed(self):
        """Test that _execute_iteration does not execute orders when the market is closed."""
        self.broker.is_market_open = False
        
        result = self.execution._execute_iteration()
        
        # Check that the broker method was called
        self.assertTrue(self.broker.check_market_status_called)
        
        # Check that other methods were not called
        self.assertFalse(self.broker.get_account_info_called)
        self.assertFalse(self.strategy.step_called)
        
        # Check that no orders were executed
        self.assertEqual(len(result), 0)
        self.assertEqual(len(self.broker.orders_executed), 0)
    
    @patch.object(ExecutionBase, '_get_next_execution_time')
    @patch.object(ExecutionBase, '_wait_until_execution_time')
    @patch.object(ExecutionBase, '_execute_iteration')
    def test_run_one_iteration_with_schedule(self, mock_execute, mock_wait, mock_get_time):
        """Test that run_one_iteration with a schedule calls the right methods."""
        next_time = datetime.datetime.now()
        mock_get_time.return_value = next_time
        mock_execute.return_value = {'AAPL': 'test_order'}
        
        schedule_iterator = MagicMock()
        result = self.execution.run_one_iteration(schedule_iterator)
        
        mock_get_time.assert_called_once_with(schedule_iterator)
        mock_wait.assert_called_once_with(next_time)
        mock_execute.assert_called_once()
        self.assertEqual(result, {'AAPL': 'test_order'})
    
    @patch.object(ExecutionBase, '_execute_iteration')
    def test_run_one_iteration_without_schedule(self, mock_execute):
        """Test that run_one_iteration without a schedule calls the right methods."""
        mock_execute.return_value = {'AAPL': 'test_order'}
        
        result = self.execution.run_one_iteration()
        
        mock_execute.assert_called_once()
        self.assertEqual(result, {'AAPL': 'test_order'})
    
    @patch.object(ExecutionBase, '_setup_signal_handlers')
    @patch.object(ExecutionBase, '_cleanup_signal_handlers')
    @patch.object(ExecutionBase, 'run_one_iteration')
    def test_run_with_max_iterations(self, mock_run_one, mock_cleanup, mock_setup):
        """Test that run with max_iterations stops after the right number of iterations."""
        self.execution.max_iterations = 3
        mock_run_one.return_value = {}
        
        self.execution.run()
        
        mock_setup.assert_called_once()
        self.assertEqual(mock_run_one.call_count, 3)
        mock_cleanup.assert_called_once()
    
    def test_stop(self):
        """Test that stop sets running to False."""
        self.execution.running = True
        self.execution.stop()
        self.assertFalse(self.execution.running)


if __name__ == '__main__':
    unittest.main() 