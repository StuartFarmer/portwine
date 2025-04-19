import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import pandas_market_calendars as mcal
import pytz

from portwine.utils.schedule_iterator import ScheduleIterator, DailyMarketScheduleIterator


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


class TestDailyMarketScheduleIterator(unittest.TestCase):
    """Tests for the DailyMarketScheduleIterator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.timezone = pytz.timezone("America/New_York")
        self.exchange = "NYSE"
        self.minutes_before_close = 15
        
        # Create a mock calendar with predictable market hours
        self.mock_calendar = MagicMock()
        
        # Mock market days: Monday-Friday for two weeks, skipping one holiday
        self.market_days = pd.DatetimeIndex([
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
        
        # Mock market schedule with proper open/close times
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
        }, index=self.market_days)

    @patch("pandas_market_calendars.get_calendar")
    def test_initialization(self, mock_get_calendar):
        """Test initialization of the iterator"""
        mock_get_calendar.return_value = self.mock_calendar
        self.mock_calendar.schedule.return_value = self.market_schedule
        
        iterator = DailyMarketScheduleIterator(
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone
        )
        
        self.assertEqual(iterator.exchange, self.exchange)
        self.assertEqual(iterator.minutes_before_close, self.minutes_before_close)
        self.assertEqual(iterator.timezone, self.timezone)
        mock_get_calendar.assert_called_once_with(self.exchange)

    @patch("pandas_market_calendars.get_calendar")
    def test_next_returns_correct_time(self, mock_get_calendar):
        """Test that __next__ returns the correct time (15 minutes before close)"""
        mock_get_calendar.return_value = self.mock_calendar
        self.mock_calendar.schedule.return_value = self.market_schedule
        
        # Start on Jan 1, 2023 (Sunday)
        start_date = pd.Timestamp("2023-01-01 12:00", tz="America/New_York")
        
        iterator = DailyMarketScheduleIterator(
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

    @patch("pandas_market_calendars.get_calendar")
    def test_skips_weekend(self, mock_get_calendar):
        """Test that the iterator correctly skips weekends"""
        mock_get_calendar.return_value = self.mock_calendar
        self.mock_calendar.schedule.return_value = self.market_schedule
        
        # Start on Jan 6, 2023 (Friday) after market close
        start_date = pd.Timestamp("2023-01-06 16:30", tz="America/New_York")
        
        iterator = DailyMarketScheduleIterator(
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # Should skip to the next trading day (Tuesday, as Monday is a holiday in our mock data)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-10 15:45", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    @patch("pandas_market_calendars.get_calendar")
    def test_skips_holidays(self, mock_get_calendar):
        """Test that the iterator correctly skips holidays"""
        mock_get_calendar.return_value = self.mock_calendar
        self.mock_calendar.schedule.return_value = self.market_schedule
        
        # Start on Jan 9, 2023 (Monday, holiday in our mock data)
        start_date = pd.Timestamp("2023-01-09 12:00", tz="America/New_York")
        
        iterator = DailyMarketScheduleIterator(
            exchange=self.exchange,
            minutes_before_close=self.minutes_before_close,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # Should skip to the next trading day (Tuesday)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-10 15:45", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    @patch("pandas_market_calendars.get_calendar")
    def test_custom_minutes_before_close(self, mock_get_calendar):
        """Test using a custom minutes_before_close value"""
        mock_get_calendar.return_value = self.mock_calendar
        self.mock_calendar.schedule.return_value = self.market_schedule
        
        # Start on Jan 1, 2023 (Sunday)
        start_date = pd.Timestamp("2023-01-01 12:00", tz="America/New_York")
        
        # Use 30 minutes before close
        iterator = DailyMarketScheduleIterator(
            exchange=self.exchange,
            minutes_before_close=30,
            timezone=self.timezone,
            start_date=start_date
        )
        
        # Should return 30 minutes before close for the next trading day (Monday)
        next_time = next(iterator)
        expected_time = pd.Timestamp("2023-01-02 15:30", tz="America/New_York")
        self.assertEqual(next_time, expected_time)

    @patch("pandas_market_calendars.get_calendar")
    def test_different_timezone(self, mock_get_calendar):
        """Test using a different timezone"""
        mock_get_calendar.return_value = self.mock_calendar
        self.mock_calendar.schedule.return_value = self.market_schedule
        
        # Start on Jan 1, 2023 (Sunday)
        start_date = pd.Timestamp("2023-01-01 12:00", tz="America/New_York")
        
        # Use Pacific timezone
        pacific_tz = pytz.timezone("America/Los_Angeles")
        iterator = DailyMarketScheduleIterator(
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


if __name__ == "__main__":
    unittest.main() 