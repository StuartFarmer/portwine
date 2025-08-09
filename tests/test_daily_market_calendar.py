import pytest
import datetime
import numpy as np
from unittest.mock import Mock, patch
from portwine.backtester.core import DailyMarketCalendar, validate_dates


class TestDailyMarketCalendar:
    
    def test_init(self):
        """Test calendar initialization"""
        calendar = DailyMarketCalendar('NYSE')
        assert calendar.calendar is not None
        assert hasattr(calendar.calendar, 'schedule')
    
    def test_init_different_calendar(self):
        """Test calendar initialization with different calendar"""
        calendar = DailyMarketCalendar('NASDAQ')
        assert calendar.calendar is not None
    
    def test_schedule(self):
        """Test schedule method"""
        calendar = DailyMarketCalendar('NYSE')
        result = calendar.schedule('2024-01-01', '2024-01-05')
        assert result is not None
        assert 'market_open' in result.columns
    
    def test_validate_dates_valid(self):
        """Test valid date validation"""
        calendar = DailyMarketCalendar('NYSE')
        assert validate_dates('2024-01-01', '2024-01-05') is True
    
    def test_validate_dates_with_none_end_date(self):
        """Test validation with None end date"""
        calendar = DailyMarketCalendar('NYSE')
        assert validate_dates('2024-01-01', None) is True
    
    def test_validate_dates_invalid_order(self):
        """Test invalid date order"""
        calendar = DailyMarketCalendar('NYSE')
        with pytest.raises(AssertionError, match="End date must be after start date"):
            validate_dates('2024-01-05', '2024-01-01')
    
    def test_validate_dates_same_date(self):
        """Test validation with same start and end date"""
        calendar = DailyMarketCalendar('NYSE')
        with pytest.raises(AssertionError, match="End date must be after start date"):
            validate_dates('2024-01-01', '2024-01-01')
    
    def test_validate_dates_invalid_start_date_type(self):
        """Test validation with invalid start date type"""
        calendar = DailyMarketCalendar('NYSE')
        with pytest.raises(AssertionError, match="Start date is required in string format"):
            validate_dates(123, '2024-01-05')
    
    def test_get_datetime_index(self):
        """Test datetime index generation"""
        calendar = DailyMarketCalendar('NYSE')
        result = calendar.get_datetime_index('2024-01-01', '2024-01-05')
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert result.dtype == 'datetime64[ns]'
    
    def test_get_datetime_index_with_none_end_date(self):
        """Test datetime index generation with None end date (uses today)"""
        calendar = DailyMarketCalendar('NYSE')
        result = calendar.get_datetime_index('2024-01-01', None)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0 