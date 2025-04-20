import unittest
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytz

from portwine.execution.base import ExecutionBase
from portwine.execution.broker import BrokerBase, AccountInfo, Position
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader
from portwine.utils.schedule_iterator import ScheduleIterator
from portwine.utils.market_calendar import MarketStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_execution_run')


class ShortIntervalScheduleIterator(ScheduleIterator):
    """
    A schedule iterator that yields times a few seconds into the future.
    This allows for testing the run loop without long waits.
    """
    def __init__(
        self, 
        interval_seconds: float = 2.0, 
        timezone: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None
    ):
        """
        Initialize the iterator.
        
        Parameters
        ----------
        interval_seconds : float, default 2.0
            Seconds between scheduled times
        timezone : Optional[str], default None
            Timezone name, or UTC if None
        start_date : Optional[pd.Timestamp], default None
            Start date, or now if None
        """
        super().__init__(timezone=timezone, start_date=start_date)
        self.interval_seconds = interval_seconds
    
    def __next__(self) -> pd.Timestamp:
        """Get the next scheduled time."""
        # Return a time that's interval_seconds in the future
        now = pd.Timestamp.now(tz=self.timezone)
        next_time = now + pd.Timedelta(seconds=self.interval_seconds)
        return next_time


class LimitedScheduleIterator(ShortIntervalScheduleIterator):
    """
    A schedule iterator that limits the number of iterations to avoid infinite loops.
    """
    def __init__(
        self, 
        max_iterations: int = 3,
        interval_seconds: float = 0.1, 
        timezone: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None
    ):
        """
        Initialize the iterator with a maximum iteration count.
        """
        super().__init__(interval_seconds, timezone, start_date)
        self.max_iterations = max_iterations
        self.iterations = 0
    
    def __next__(self) -> pd.Timestamp:
        """Get the next scheduled time, but limit the number of iterations."""
        if self.iterations >= self.max_iterations:
            raise StopIteration("Max iterations reached")
        
        self.iterations += 1
        return super().__next__()


class MockScheduleIterator(ScheduleIterator):
    """
    A simple mock schedule iterator for testing timezone handling.
    Returns a timestamp a few seconds in the future with the specified timezone.
    """
    def __init__(
        self, 
        timezone: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None
    ):
        """
        Initialize the iterator.
        
        Parameters
        ----------
        timezone : Optional[str], default None
            Timezone name, or UTC if None
        start_date : Optional[pd.Timestamp], default None
            Start date, or now if None
        """
        super().__init__(timezone=timezone, start_date=start_date)
    
    def __next__(self) -> pd.Timestamp:
        """Get the next scheduled time."""
        # Return a time that's 1 second in the future with the specified timezone
        now = pd.Timestamp.now(tz=self.timezone)
        next_time = now + pd.Timedelta(seconds=1.0)
        return next_time


class MockStrategy(StrategyBase):
    """Mock strategy for testing."""
    
    def __init__(self, tickers: List[str]):
        """Initialize with tickers."""
        super().__init__(tickers)
        self.step_called = 0
        self.target_weights = {ticker: 0.0 for ticker in tickers}
        self.last_time_zone = None
        self.raise_error = False
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Process daily data and return allocations."""
        self.step_called += 1
        logger.info(f"Strategy.step called ({self.step_called})")
        
        # Store the timezone of the current_date
        if current_date is not None and hasattr(current_date, 'tz'):
            self.last_time_zone = str(current_date.tz) if current_date.tz else None
        
        # Optionally raise an error for testing error handling
        if self.raise_error:
            raise ValueError("Test error from MockStrategy")
            
        return self.target_weights
    
    def generate_signals(self) -> Dict[str, float]:
        """Return current signals."""
        return self.target_weights


class MockDataLoader(MarketDataLoader):
    """Mock data loader for testing."""
    
    def __init__(self):
        """Initialize the mock data loader."""
        self.next_called = 0
    
    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        """Return next batch of data."""
        self.next_called += 1
        logger.info(f"DataLoader.next called ({self.next_called})")
        return {ticker: {'close': 100.0} for ticker in tickers}
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load historical data for a ticker."""
        return None


class MockBroker(BrokerBase):
    """Mock broker for testing."""
    
    def __init__(self, market_open: bool = True):
        """Initialize with default market status."""
        self.market_open = market_open
        self.check_market_status_called = 0
        self.get_account_info_called = 0
        self.execute_order_called = 0
        self.positions = {}  # Dictionary of positions by symbol
    
    def check_market_status(self) -> MarketStatus:
        """Check if market is open."""
        self.check_market_status_called += 1
        logger.info(f"Broker.check_market_status called ({self.check_market_status_called})")
        
        now = datetime.now(pytz.UTC)
        next_open = now + pd.Timedelta(hours=16) if not self.market_open else None
        next_close = now + pd.Timedelta(hours=8) if self.market_open else None
        
        return MarketStatus(
            is_open=self.market_open,
            next_open=next_open,
            next_close=next_close
        )
    
    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        self.get_account_info_called += 1
        logger.info(f"Broker.get_account_info called ({self.get_account_info_called})")
        
        # Convert dictionary positions to Position objects
        position_objects = {}
        for symbol, pos_data in self.positions.items():
            position_objects[symbol] = Position(
                symbol=symbol,
                qty=pos_data.get('qty', 0),
                market_value=pos_data.get('market_value', 0),
                avg_entry_price=pos_data.get('avg_entry_price', 0),
                unrealized_pl=pos_data.get('unrealized_pl', 0)
            )
            
        return AccountInfo(
            cash=10000.0,
            portfolio_value=10000.0 + sum(p.market_value for p in position_objects.values()),
            positions=position_objects
        )
    
    def execute_order(self, symbol: str, qty: float, order_type: str = "market") -> bool:
        """Execute an order."""
        self.execute_order_called += 1
        logger.info(f"Broker.execute_order called ({self.execute_order_called})")
        
        # Update positions based on the order
        if symbol not in self.positions:
            self.positions[symbol] = {
                'qty': qty,
                'market_value': qty * 100.0,  # Assume price of 100
                'avg_entry_price': 100.0,
                'unrealized_pl': 0.0
            }
        else:
            # Update existing position
            self.positions[symbol]['qty'] += qty
            self.positions[symbol]['market_value'] = self.positions[symbol]['qty'] * 100.0
            
            # Remove position if quantity becomes zero
            if self.positions[symbol]['qty'] == 0:
                del self.positions[symbol]
                
        return True
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status."""
        return "filled"
    
    def cancel_all_orders(self) -> bool:
        """Cancel all orders."""
        return True
    
    def close_all_positions(self) -> bool:
        """Close all positions."""
        self.positions = {}
        return True
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        return [{'symbol': k, **v} for k, v in self.positions.items()]
    
    def get_cash(self) -> float:
        """Get cash balance."""
        return 10000.0
    
    def get_portfolio_value(self) -> float:
        """Get portfolio value."""
        return 10000.0 + sum(p.get('market_value', 0) for p in self.positions.values())


class TestExecution(ExecutionBase):
    """Concrete implementation of ExecutionBase for testing."""
    
    def __init__(self, strategy, broker, schedule=None, market_data_loader=None, timezone=None):
        """Initialize with strategy, broker, and optional timezone."""
        self.market_data_loader = market_data_loader if market_data_loader else MockDataLoader()
        
        super().__init__(
            strategy=strategy, 
            broker=broker,
            schedule=schedule,
            market_data_loader=self.market_data_loader,
            timezone=timezone
        )


class TestExecutionRun(unittest.TestCase):
    """Tests for the ExecutionBase.run method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.strategy = MockStrategy(self.tickers)
        self.data_loader = MockDataLoader()
        self.broker = MockBroker()
        self.schedule = LimitedScheduleIterator(max_iterations=3, interval_seconds=0.1)
        
        # Create execution instance
        self.execution = TestExecution(
            strategy=self.strategy,
            broker=self.broker,
            schedule=self.schedule,
            market_data_loader=self.data_loader
        )
    
    def test_basic_execution(self):
        """Test basic execution with default parameters."""
        # Reset counters
        self.strategy.step_called = 0
        self.broker.check_market_status_called = 0
        
        # Run execution
        self.execution.run(max_iterations=3)
        
        # Check that the expected methods were called
        self.assertEqual(self.strategy.step_called, 3, "Strategy.step should be called 3 times")
        self.assertEqual(self.broker.check_market_status_called, 3, 
                      "Broker.check_market_status should be called 3 times")
    
    def test_market_closed(self):
        """Test behavior when market is closed."""
        # Set market as closed
        self.broker.market_open = False
        
        # Reset counters
        self.broker.check_market_status_called = 0
        self.strategy.step_called = 0
        
        # Run a single step
        self.execution.run_once()
            
        # Check that market status was checked but strategy was not called
        self.assertEqual(self.broker.check_market_status_called, 1, 
                      "Broker.check_market_status should be called once")
        self.assertEqual(self.strategy.step_called, 0, 
                      "Strategy.step should not be called when market is closed")
    
    def test_stop_method(self):
        """Test that the stop method works correctly."""
        # Run in a separate thread
        thread = threading.Thread(target=self.execution.run)
        thread.daemon = True
        thread.start()
        
        # Wait briefly to ensure execution has started
        time.sleep(0.5)
        
        # Stop the execution
        self.execution.stop()
        
        # Wait for thread to terminate
        thread.join(timeout=1.0)
        
        # Check that execution was stopped
        self.assertFalse(self.execution._running, "Execution should be stopped")
    
    def test_timezone_handling(self):
        """Test that timezone is correctly passed to strategy."""
        # Set a custom timezone
        timezone = "America/New_York"
        
        # Create execution with the timezone
        execution = TestExecution(
            strategy=self.strategy,
            broker=self.broker,
            market_data_loader=self.data_loader,
            timezone=timezone
        )
        
        # Reset strategy counter
        self.strategy.step_called = 0
        self.strategy.last_time_zone = None
        
        # Run one step
        execution.run_once()
        
        # Check that the timezone was passed correctly to the strategy
        self.assertEqual(self.strategy.step_called, 1, "Strategy.step should be called once")
        self.assertIsNotNone(self.strategy.last_time_zone, "Strategy should receive timezone-aware timestamp")
        self.assertIn("America/New_York", self.strategy.last_time_zone, 
                   "Strategy should receive timestamp with New York timezone")
    
    def test_generate_orders(self):
        """Test that orders are generated correctly based on target weights and positions."""
        # Add some positions to the broker
        self.broker.positions = {
            'AAPL': {
                'qty': 10,
                'market_value': 1500.0,
                'avg_entry_price': 150.0,
                'unrealized_pl': 0.0
            },
            'MSFT': {
                'qty': 5,
                'market_value': 1200.0,
                'avg_entry_price': 240.0,
                'unrealized_pl': 0.0
            }
        }
        
        # Set some target weights that will require both buying and selling
        self.strategy.target_weights = {
            'AAPL': 0.2,  # Reduce position
            'MSFT': 0.3,  # Increase position
            'GOOGL': 0.1  # New position
        }
        
        # Reset counters
        self.strategy.step_called = 0
        self.broker.execute_order_called = 0
        
        # Run a single step
        self.execution.run_once()
        
        # Verify that orders were executed correctly
        self.assertEqual(self.strategy.step_called, 1, "Strategy.step should be called once")
        self.assertTrue(self.broker.execute_order_called > 0, "Orders should be executed")
        
        # Check positions after execution
        account_info = self.broker.get_account_info()
        self.assertIn('GOOGL', account_info.positions, "New position should be created")
        
        # Check quantity changes based on the target weights
        # Since we use a mock price of 100 for all symbols and the initial portfolio value is 12700.0,
        # the target quantities should be:
        # AAPL: 12700 * 0.2 / 100 = 25 (we already have 10, so buy 15 more)
        # MSFT: 12700 * 0.3 / 100 = 38 (we already have 5, so buy 33 more)
        # GOOGL: 12700 * 0.1 / 100 = 12 (we have 0, so buy 12)
        self.assertEqual(account_info.positions['AAPL'].qty, 25, "AAPL quantity should be updated to 25")
        self.assertEqual(account_info.positions['MSFT'].qty, 38, "MSFT quantity should be updated to 38")
        self.assertEqual(account_info.positions['GOOGL'].qty, 12, "GOOGL quantity should be 12")
        
        # Verify portfolio value is correct after trades
        expected_portfolio_value = 10000.0 + (25 * 100.0) + (38 * 100.0) + (12 * 100.0)
        self.assertEqual(account_info.portfolio_value, expected_portfolio_value, 
                      "Portfolio value should be updated correctly")

    def test_account_info_fields(self):
        """Test that account info fields are correctly accessed in step method."""
        # Set the strategy to return a simple weight
        self.strategy.target_weights = {'AAPL': 0.5}
        
        # Run a single step
        self.execution.run_once()
        
        # If we reach this point without exceptions, the test passes
        # The fix for account.equity vs account.portfolio_value is working
        self.assertTrue(True)

    def test_run_once(self):
        """Test that run_once works correctly."""
        # Reset counters
        self.strategy.step_called = 0
        self.broker.check_market_status_called = 0
        
        # Run a single step
        self.execution.run_once()
        
        # Check that the step completed successfully
        self.assertEqual(self.strategy.step_called, 1, "Strategy.step should be called once")
        self.assertEqual(self.broker.check_market_status_called, 1, 
                      "Broker.check_market_status should be called once")

    def test_market_closed_step(self):
        """Test behavior when market is closed using step."""
        # Set market as closed
        self.broker.market_open = False
        
        # Reset counters
        self.strategy.step_called = 0
        self.broker.check_market_status_called = 0
        
        # Run step
        orders = self.execution.step()
        
        # Check results
        self.assertEqual(len(orders), 0, "No orders should be executed when market is closed")
        self.assertEqual(self.strategy.step_called, 0, "Strategy.step should not be called when market is closed")
        self.assertEqual(self.broker.check_market_status_called, 1, 
                      "Broker.check_market_status should be called once")

    def test_custom_timezone(self):
        """Test that execution can use a custom timezone passed at initialization."""
        # Create execution with custom timezone
        timezone = "Europe/Paris"
        execution = TestExecution(
            strategy=self.strategy,
            broker=self.broker,
            timezone=timezone
        )
        
        # Reset strategy counter
        self.strategy.step_called = 0
        self.strategy.last_time_zone = None
        
        # Run one step
        execution.run_once()
        
        # Check that the custom timezone from execution was used
        self.assertEqual(self.strategy.step_called, 1, "Strategy.step should be called once")
        self.assertIsNotNone(self.strategy.last_time_zone, "Strategy should receive timezone-aware timestamp")
        self.assertIn("Europe/Paris", self.strategy.last_time_zone, 
                    "Strategy should receive timestamp with Paris timezone")


if __name__ == '__main__':
    unittest.main() 