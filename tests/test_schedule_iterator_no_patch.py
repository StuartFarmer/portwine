import unittest
from datetime import datetime, timedelta
import pandas as pd
import pandas_market_calendars as mcal
import pytz
from typing import Optional, Dict, Any, List

from portwine.utils.schedule_iterator import ScheduleIterator, DailyMarketScheduleIterator


# Create a test calendar class to avoid needing to patch
class TestCalendar:
    """Test calendar that returns predetermined market schedules"""
    
    def __init__(self, mock_schedule):
        """Initialize with a predefined schedule"""
        self.mock_schedule = mock_schedule
    
    def schedule(self, start_date, end_date):
        """Return the mock schedule"""
        return self.mock_schedule


# Modified DailyMarketScheduleIterator that accepts a calendar instance instead of fetching one
class TestDailyMarketScheduleIterator(DailyMarketScheduleIterator):
    """Test version of DailyMarketScheduleIterator that accepts a calendar instance"""
    
    def __init__(self,
                 test_calendar,
                 exchange: str = "NYSE",
                 minutes_before_close: int = 15,
                 timezone = None,
                 start_date = None):
        """
        Initialize with a test calendar instance
        
        Parameters
        ----------
        test_calendar : TestCalendar
            Calendar instance to use for testing
        """
        # Initialize the ScheduleIterator part first
        ScheduleIterator.__init__(self, timezone, start_date)
        
        # Set attributes directly instead of calling DailyMarketScheduleIterator.__init__
        self.exchange = exchange
        self.minutes_before_close = minutes_before_close
        self.calendar = test_calendar
        
        # Keep track of loaded market days
        self._loaded_market_schedule = None
        self._load_next_market_schedule()


class TestScheduleIterator(unittest.TestCase):
    """Base tests for the ScheduleIterator abstract base class"""

    def test_abstract_next(self):
        """Test that the base class raises NotImplementedError for __next__"""
        # Create a concrete subclass but don't override __next__
        class ConcreteScheduleIterator(ScheduleIterator):
            pass
            
        # This should raise a TypeError because __next__ is abstract
        with self.assertRaises(TypeError):
            iterator = ConcreteScheduleIterator()


class TestDailyMarketScheduleIteratorTests(unittest.TestCase):
    """Tests for the DailyMarketScheduleIterator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.timezone = pytz.timezone("America/New_York")
        self.exchange = "NYSE"
        self.minutes_before_close = 15
        
        # Create market schedule with proper open/close times
        market_days = pd.DatetimeIndex([
            # Week 1
            pd.Timestamp("2023-01-02 09:30", tz="America/New_York"),  # Monday
            pd.Timestamp("2023-01-03 09:30", tz="America/New_York"),  # Tuesday
            pd.Timestamp("2023-01-04 09:30", tz="America/New_York"),  # Wednesday
            pd.Timestamp("2023-01-05 09:30", tz="America/New_York"),  # Thursday
            pd.Timestamp("2023-01-06 09:30", tz="America/New_York"),  # Friday
            # Week 2
            # No Monday (Jan 9) - Holiday
            pd.Timestamp("2023-01-10 09:30", tz="America/New_York"),  # Tuesday
            pd.Timestamp("2023-01-11 09:30", tz="America/New_York"),  # Wednesday
            pd.Timestamp("2023-01-12 09:30", tz="America/New_York"),  # Thursday
            pd.Timestamp("2023-01-13 09:30", tz="America/New_York"),  # Friday
        ])
        
        self.market_schedule = pd.DataFrame({
            'market_open': [
                pd.Timestamp(f"2023-01-0{i} 09:30", tz="America/New_York") for i in range(2, 7)
            ] + [
                pd.Timestamp(f"2023-01-1{i} 09:30", tz="America/New_York") for i in range(0, 4)
            ],
            'market_close': [
                pd.Timestamp(f"2023-01-0{i} 16:00", tz="America/New_York") for i in range(2, 7)
            ] + [
                pd.Timestamp(f"2023-01-1{i} 16:00", tz="America/New_York") for i in range(0, 4)
            ]
        }, index=market_days)
        
        # Create test calendar with our mock schedule
        self.test_calendar = TestCalendar(self.market_schedule)

    def test_initialization(self):
        """Test initialization of the iterator"""
        iterator = TestDailyMarketScheduleIterator(
            test_calendar=self.test_calendar,
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone
        )
        
        self.assertEqual(iterator.exchange, self.exchange)
        self.assertEqual(iterator.minutes_before_close, self.minutes_before_close)
        self.assertEqual(iterator.timezone, self.timezone)
        self.assertEqual(iterator.calendar, self.test_calendar)

    def test_next_returns_correct_time(self):
        """Test that __next__ returns the correct time (15 minutes before close)"""
        # Start on Jan 1, 2023 (Sunday)
        start_date = pd.Timestamp("2023-01-01 12:00", tz="America/New_York")
        
        iterator = TestDailyMarketScheduleIterator(
            test_calendar=self.test_calendar,
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # First call should return Jan 2, 15:45 (Monday)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-02 15:45", tz="America/New_York")
        self.assertEqual(next_time, expected_time)
        
        # Next call should return Jan 3, 15:45 (Tuesday)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-03 15:45", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    def test_skips_weekend(self):
        """Test that the iterator correctly skips weekends"""
        # Start on Jan 6, 2023 (Friday) after market close
        start_date = pd.Timestamp("2023-01-06 16:30", tz="America/New_York")
        
        iterator = TestDailyMarketScheduleIterator(
            test_calendar=self.test_calendar,
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # Should skip to the next trading day (Tuesday, as Monday is a holiday in our mock data)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-10 15:45", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    def test_skips_holidays(self):
        """Test that the iterator correctly skips holidays"""
        # Start on Jan 9, 2023 (Monday, holiday in our mock data)
        start_date = pd.Timestamp("2023-01-09 12:00", tz="America/New_York")
        
        iterator = TestDailyMarketScheduleIterator(
            test_calendar=self.test_calendar,
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # Should skip to the next trading day (Tuesday)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-10 15:45", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    def test_custom_minutes_before_close(self):
        """Test using a custom minutes_before_close value"""
        # Start on Jan 1, 2023 (Sunday)
        start_date = pd.Timestamp("2023-01-01 12:00", tz="America/New_York")
        
        # Use 30 minutes before close
        iterator = TestDailyMarketScheduleIterator(
            test_calendar=self.test_calendar,
            exchange=self.exchange,
            minutes_before_close=30,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # Should return 30 minutes before close for the next trading day (Monday)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-02 15:30", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    def test_different_timezone(self):
        """Test using a different timezone"""
        # Start on Jan 1, 2023 (Sunday)
        start_date = pd.Timestamp("2023-01-01 12:00", tz="America/New_York")
        
        # Use Pacific timezone
        pacific_tz = pytz.timezone("America/Los_Angeles")
        iterator = TestDailyMarketScheduleIterator(
            test_calendar=self.test_calendar,
            exchange=self.exchange,
            minutes_before_close=15,
            timezone=pacific_tz,
            start_date=start_date
        )
        
        # Should return time in Pacific timezone for the next trading day (Monday)
        next_time = next(iterator)
        # 15:45 ET = 12:45 PT
        expected_time = pd.Timestamp("2023-01-02 12:45", tz="America/Los_Angeles")
        self.assertEqual(next_time, expected_time)


# Additional test to demonstrate how to test the actual implementation
class TestActualDailyMarketScheduleIterator(unittest.TestCase):
    """Tests for the actual DailyMarketScheduleIterator class using a real calendar"""
    
    def test_with_real_calendar(self):
        """Test with a real calendar for one iteration"""
        # This test uses the actual implementation and a real calendar
        # We only verify basic functionality to avoid external dependencies in tests
        
        # Use a specific date in the past to make test deterministic
        start_date = pd.Timestamp("2022-01-03 09:00", tz="America/New_York")  # Monday
        
        iterator = DailyMarketScheduleIterator(
            exchange="NYSE",
            minutes_before_close=15,
            timezone="America/New_York",
            start_date=start_date
        )
        
        # Just verify we get a timestamp back, not its exact value
        # since that would depend on the external calendar
        next_time = next(iterator)
        self.assertIsInstance(next_time, pd.Timestamp)
        self.assertEqual(next_time.hour, 15)
        self.assertEqual(next_time.minute, 45)


class MockScheduleIterator(ScheduleIterator):
    """
    A simple mock schedule iterator for testing.
    Provides predefined timestamps for testing without requiring actual time to pass.
    """
    def __init__(
        self,
        timestamps: Optional[List[pd.Timestamp]] = None,
        timezone: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None
    ):
        """
        Initialize the mock iterator with predefined timestamps.
        
        Parameters
        ----------
        timestamps : Optional[List[pd.Timestamp]], default None
            List of timestamps to return in sequence, or None to generate default timestamps
        timezone : Optional[str], default None
            Timezone name, or UTC if None
        start_date : Optional[pd.Timestamp], default None
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
    
    def __next__(self) -> pd.Timestamp:
        """Get the next timestamp from the predefined sequence."""
        if self.index >= len(self.timestamps):
            raise StopIteration("No more timestamps")
        
        timestamp = self.timestamps[self.index]
        
        # Convert to the desired timezone if needed
        if timestamp.tz != self.timezone:
            timestamp = timestamp.tz_convert(self.timezone)
        
        self.index += 1
        return timestamp


class TestMockScheduleIterator(unittest.TestCase):
    """Tests for the MockScheduleIterator class."""
    
    def test_default_timestamps(self):
        """Test that the iterator generates default timestamps correctly."""
        # Create iterator with a fixed start date
        start_date = pd.Timestamp('2023-05-01 10:00:00', tz='UTC')
        iterator = MockScheduleIterator(start_date=start_date)
        
        # Get the first timestamp and verify it's 1 hour after start
        first_time = next(iterator)
        self.assertEqual(first_time, pd.Timestamp('2023-05-01 11:00:00', tz='UTC'))
        
        # Get the second timestamp and verify it's 2 hours after start
        second_time = next(iterator)
        self.assertEqual(second_time, pd.Timestamp('2023-05-01 12:00:00', tz='UTC'))
    
    def test_custom_timestamps(self):
        """Test that the iterator returns custom timestamps correctly."""
        # Create custom timestamps
        timestamps = [
            pd.Timestamp('2023-05-01 12:00:00', tz='UTC'),
            pd.Timestamp('2023-05-02 12:00:00', tz='UTC'),
            pd.Timestamp('2023-05-03 12:00:00', tz='UTC')
        ]
        
        # Create iterator with custom timestamps
        iterator = MockScheduleIterator(timestamps=timestamps)
        
        # Check that all timestamps are returned in order
        for expected in timestamps:
            actual = next(iterator)
            self.assertEqual(actual, expected)
        
        # Check that StopIteration is raised when exhausted
        with self.assertRaises(StopIteration):
            next(iterator)
    
    def test_timezone_conversion(self):
        """Test that timestamps are converted to the requested timezone."""
        # Create a timestamp in UTC
        timestamps = [pd.Timestamp('2023-05-01 12:00:00', tz='UTC')]
        
        # Create iterator with timezone conversion
        iterator = MockScheduleIterator(
            timestamps=timestamps,
            timezone='America/New_York'
        )
        
        # Get the first timestamp and verify it's converted
        time = next(iterator)
        self.assertEqual(time.tz.zone, 'America/New_York')
        self.assertEqual(time.hour, 8)  # 12 UTC = 8 EDT (assuming EDT is in effect)


if __name__ == "__main__":
    unittest.main() 