"""
Tests for schedule iterators focusing on timezone handling without patching.

These tests use the MockScheduleIterator and custom calendars to test
timezone conversions and handling without the need for patching.
"""
import unittest
from datetime import datetime, time

import pandas as pd
import pytz

from portwine.utils.schedule_iterator import ScheduleIterator
from portwine.utils.custom_schedule_iterators import (
    FixedTimeScheduleIterator,
    IntradayScheduleIterator,
    CompositeScheduleIterator
)


class MockScheduleIterator(ScheduleIterator):
    """
    A simple mock schedule iterator for testing.
    Provides predefined timestamps for testing without requiring actual time to pass.
    """
    def __init__(
        self,
        timestamps=None,
        timezone=None,
        start_date=None
    ):
        """
        Initialize the mock iterator with predefined timestamps.
        
        Parameters
        ----------
        timestamps : list of pd.Timestamp, default None
            List of timestamps to return in sequence, or None to generate default timestamps
        timezone : str or pytz.timezone, default None
            Timezone name or timezone object, or UTC if None
        start_date : pd.Timestamp or datetime, default None
            Start date for generating timestamps if timestamps is None
        """
        super().__init__(timezone=timezone, start_date=start_date)
        
        if timestamps is not None:
            self.timestamps = timestamps
        else:
            # Generate a sequence of timestamps at hourly intervals
            start = start_date if start_date is not None else pd.Timestamp.now(tz=self.timezone)
            self.timestamps = [
                start + pd.Timedelta(hours=i) 
                for i in range(1, 10)  # Generate 9 timestamps
            ]
        
        self.index = 0
    
    def __next__(self):
        """Get the next timestamp from the predefined sequence."""
        if self.index >= len(self.timestamps):
            raise StopIteration("No more timestamps")
        
        timestamp = self.timestamps[self.index]
        
        # Convert to the desired timezone if needed
        if hasattr(timestamp, 'tz') and timestamp.tz is not None and timestamp.tz != self.timezone:
            timestamp = timestamp.tz_convert(self.timezone)
        elif not hasattr(timestamp, 'tz') or timestamp.tz is None:
            timestamp = pd.Timestamp(timestamp, tz=self.timezone)
        
        self.index += 1
        return timestamp


class TestTimezoneHandling(unittest.TestCase):
    """Test timezone handling in schedule iterators."""
    
    def test_base_iterator_timezone(self):
        """Test that ScheduleIterator handles timezone conversion correctly."""
        # Create timestamps in UTC
        timestamps = [
            pd.Timestamp('2023-05-01 12:00:00', tz='UTC'),
            pd.Timestamp('2023-05-02 12:00:00', tz='UTC')
        ]
        
        # Test various timezone conversions
        timezones = [
            'America/New_York',    # UTC-4 or UTC-5
            'Europe/London',       # UTC+0 or UTC+1
            'Asia/Tokyo',          # UTC+9
            'Australia/Sydney'     # UTC+10 or UTC+11
        ]
        
        for tz in timezones:
            with self.subTest(timezone=tz):
                # Create iterator with timezone conversion
                iterator = MockScheduleIterator(
                    timestamps=timestamps.copy(),
                    timezone=tz
                )
                
                # Get the timestamps and verify they're converted
                for i in range(len(timestamps)):
                    time = next(iterator)
                    self.assertEqual(time.tz.zone, tz)
                    
                    # Check conversion is correct
                    utc_time = timestamps[i].tz_convert(tz)
                    self.assertEqual(time, utc_time)
    
    def test_fixed_time_timezone(self):
        """Test that FixedTimeScheduleIterator handles timezones correctly."""
        # Create a fixed time iterator with specific timezone
        fixed_time = time(9, 30)  # 9:30 AM
        iterator = FixedTimeScheduleIterator(
            execution_time=fixed_time,
            timezone="America/New_York",
            days_of_week=[0, 1, 2, 3, 4]  # Monday-Friday
        )
        
        # Get the next few execution times
        execution_times = []
        for _ in range(3):
            execution_times.append(next(iterator))
        
        # Verify all times are at 9:30 AM New York time
        for exec_time in execution_times:
            self.assertEqual(exec_time.tz.zone, "America/New_York")
            self.assertEqual(exec_time.hour, 9)
            self.assertEqual(exec_time.minute, 30)
            
            # Verify it's a weekday (not Saturday or Sunday)
            self.assertIn(exec_time.dayofweek, [0, 1, 2, 3, 4])


class TestExecutionTimezoneHandling(unittest.TestCase):
    """Test timezone handling in the execution system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip imports if needed
        try:
            from portwine.execution_complex.base import ExecutionBase
            from portwine.execution_complex.broker import BrokerBase, AccountInfo
            from portwine.strategies.base import StrategyBase
            from portwine.loaders.base import MarketDataLoader
        except ImportError:
            self.skipTest("Required modules not available")
        
        # Import the classes we need for testing
        from portwine.execution_complex.base import ExecutionBase
        from portwine.execution_complex.broker import BrokerBase, AccountInfo
        from portwine.strategies.base import StrategyBase
        from portwine.loaders.base import MarketDataLoader
        
        # Create a mock strategy
        class MockStrategy(StrategyBase):
            def __init__(self, tickers):
                super().__init__(tickers)
                self.step_called = 0
                self.last_timestamp = None
                self.last_timezone = None
                
            def step(self, current_date, daily_data):
                self.step_called += 1
                self.last_timestamp = current_date
                self.last_timezone = str(current_date.tz) if hasattr(current_date, 'tz') and current_date.tz else None
                return {ticker: 0.0 for ticker in self.tickers}
        
        # Create a mock broker
        class MockBroker(BrokerBase):
            def check_market_status(self):
                return True
                
            def get_account_info(self):
                return {'cash': 10000.0, 'portfolio_value': 10000.0, 'positions': {}}
                
            def execute_order(self, symbol, qty, order_type="market"):
                return True
        
        # Create a mock data loader
        class MockDataLoader(MarketDataLoader):
            def __init__(self):
                self.next_called = 0
                
            def next(self, tickers, timestamp):
                self.next_called += 1
                return {ticker: {'close': 100.0} for ticker in tickers}
        
        # Store classes for use in tests
        self.ExecutionBase = ExecutionBase
        self.MockStrategy = MockStrategy
        self.MockBroker = MockBroker
        self.MockDataLoader = MockDataLoader
    
    def test_execution_timezone_passing(self):
        """Test that execution passes timezone correctly to the strategy."""
        # Skip if setup failed
        if not hasattr(self, 'ExecutionBase'):
            self.skipTest("Setup failed to load required classes")
        
        # Create a concrete execution class for testing
        class TestExecution(self.ExecutionBase):
            def __init__(self, strategy, broker, data_loader=None, timezone=None):
                self.data_loader = data_loader if data_loader else self.MockDataLoader()
                super().__init__(strategy=strategy, broker=broker, market_data_loader=self.data_loader)
                self.timezone = timezone
                
            def _execute_iteration(self):
                timestamp = pd.Timestamp.now(tz=self.timezone or 'UTC')
                return super()._execute_iteration()
        
        # Create components with different timezones
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        strategy = self.MockStrategy(tickers)
        broker = self.MockBroker()
        data_loader = self.MockDataLoader()
        
        # Test different timezone configurations
        test_cases = [
            ('UTC', None),                 # UTC timezone
            ('America/New_York', None),    # NY timezone
            ('Europe/London', None),       # London timezone
            ('Asia/Tokyo', None),          # Tokyo timezone
            (None, 'America/New_York'),    # No execution timezone, but scheduler timezone
            (None, None)                   # No timezone specified
        ]
        
        for exec_tz, sched_tz in test_cases:
            with self.subTest(execution_timezone=exec_tz, scheduler_timezone=sched_tz):
                # Create execution with specified timezone
                execution = TestExecution(
                    strategy=strategy,
                    broker=broker,
                    data_loader=data_loader,
                    timezone=exec_tz
                )
                
                # Create scheduler with specified timezone
                scheduler = MockScheduleIterator(timezone=sched_tz)
                
                # Reset counters
                strategy.step_called = 0
                strategy.last_timezone = None
                
                # Run one iteration
                execution.run_one_iteration(scheduler)
                
                # Check that strategy was called with correct timezone
                self.assertEqual(strategy.step_called, 1, "Strategy step should be called once")
                
                # The timezone should be from the execution if specified, otherwise from the scheduler
                expected_timezone = exec_tz or sched_tz or 'UTC'
                
                # Strip 'tzfile' or other prefixes that might be in the timezone string
                actual_timezone = strategy.last_timezone
                if actual_timezone and '/' in actual_timezone:
                    actual_timezone = actual_timezone.split('/')[-2] + '/' + actual_timezone.split('/')[-1]
                
                # Assert the timezone matches what we expect
                self.assertIn(str(expected_timezone), str(actual_timezone),
                            f"Expected timezone '{expected_timezone}' not found in '{actual_timezone}'")
                
                # Make sure the timestamp is timezone-aware
                self.assertIsNotNone(strategy.last_timestamp.tz,
                                   "Timestamp should be timezone-aware")

if __name__ == "__main__":
    unittest.main() 