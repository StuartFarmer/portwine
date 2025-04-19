"""
Schedule Iterator Module

This module provides iterators for generating scheduled execution times
for trading strategies. Each iterator yields datetime objects representing
the next time the strategy should be executed.
"""

import abc
from datetime import datetime, timedelta
from typing import Optional, Iterator, Union

import pandas as pd
import pandas_market_calendars as mcal
import pytz


class ScheduleIterator(abc.ABC, Iterator):
    """
    Abstract base class for schedule iterators.
    
    A schedule iterator is an iterator that yields datetime objects
    representing the next time a strategy should be executed.
    """
    
    def __init__(self, 
                 timezone: Optional[Union[str, pytz.timezone]] = None,
                 start_date: Optional[Union[str, datetime, pd.Timestamp]] = None):
        """
        Initialize the schedule iterator.
        
        Parameters
        ----------
        timezone : Optional[Union[str, pytz.timezone]], default None
            The timezone to use for the schedule. If None, UTC is used.
        start_date : Optional[Union[str, datetime, pd.Timestamp]], default None
            The start date from which to begin the schedule.
            If None, the current time is used.
        """
        # Set timezone
        if timezone is None:
            self.timezone = pytz.UTC
        elif isinstance(timezone, str):
            self.timezone = pytz.timezone(timezone)
        else:
            self.timezone = timezone
            
        # Set start date
        if start_date is None:
            self.current_time = pd.Timestamp.now(tz=self.timezone)
        elif isinstance(start_date, str):
            self.current_time = pd.Timestamp(start_date, tz=self.timezone)
        elif isinstance(start_date, datetime):
            if start_date.tzinfo is None:
                start_date = self.timezone.localize(start_date)
            self.current_time = pd.Timestamp(start_date)
        else:
            # Assume it's a Timestamp
            if start_date.tzinfo is None:
                start_date = self.timezone.localize(start_date)
            self.current_time = start_date
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    @abc.abstractmethod
    def __next__(self) -> pd.Timestamp:
        """
        Return the next execution time.
        
        Returns
        -------
        pd.Timestamp
            The next time the strategy should be executed.
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement __next__")


class DailyMarketScheduleIterator(ScheduleIterator):
    """
    Iterator that yields the next trading day's schedule time.
    
    This iterator uses pandas_market_calendars to determine the next
    trading day and returns a time that is a specified number of minutes
    before market close.
    """
    
    def __init__(self,
                 exchange: str = "NYSE",
                 minutes_before_close: int = 15,
                 timezone: Optional[Union[str, pytz.timezone]] = None,
                 start_date: Optional[Union[str, datetime, pd.Timestamp]] = None):
        """
        Initialize the daily market schedule iterator.
        
        Parameters
        ----------
        exchange : str, default "NYSE"
            The exchange to use for determining market hours.
        minutes_before_close : int, default 15
            Number of minutes before market close to schedule execution.
        timezone : Optional[Union[str, pytz.timezone]], default None
            The timezone to use for the schedule. If None, UTC is used.
        start_date : Optional[Union[str, datetime, pd.Timestamp]], default None
            The start date from which to begin the schedule.
            If None, the current time is used.
        """
        super().__init__(timezone, start_date)
        self.exchange = exchange
        self.minutes_before_close = minutes_before_close
        self.calendar = mcal.get_calendar(exchange)
        
        # Keep track of loaded market days
        self._loaded_market_schedule = None
        self._load_next_market_schedule()
        
    def _load_next_market_schedule(self):
        """
        Load market schedule for the next month starting from current_time.
        
        This fetches market open/close times for a period ahead,
        so we don't need to fetch for every call to __next__.
        """
        start_date = self.current_time.date()
        end_date = (self.current_time + pd.Timedelta(days=30)).date()
        
        self._loaded_market_schedule = self.calendar.schedule(
            start_date=start_date,
            end_date=end_date
        )
        
        # Ensure index has timezone info for comparison
        if not self._loaded_market_schedule.empty:
            # Check if index already has timezone
            if self._loaded_market_schedule.index[0].tzinfo is None:
                # Localize index to UTC (default for exchange calendars)
                self._loaded_market_schedule.index = self._loaded_market_schedule.index.tz_localize('UTC')
    
    def __next__(self) -> pd.Timestamp:
        """
        Return the next trading day's execution time.
        
        This will be `minutes_before_close` minutes before the market close
        on the next trading day after the current time.
        
        Returns
        -------
        pd.Timestamp
            The time to execute the strategy, minutes_before_close minutes before
            market close on the next trading day.
        """
        # If we've exhausted our loaded schedule, load more
        if (self._loaded_market_schedule is None or 
            self._loaded_market_schedule.empty or
            self.current_time.date() > self._loaded_market_schedule.index[-1].date()):
            self._load_next_market_schedule()
        
        # Ensure current_time has timezone for comparison
        if self.current_time.tzinfo is None:
            self.current_time = pd.Timestamp(self.current_time, tz='UTC')
            
        # Find the next market day - convert to UTC for comparison if needed
        current_time_utc = self.current_time.tz_convert('UTC') if self.current_time.tzinfo is not None else self.current_time
        
        # Find next market days by filtering the schedule
        next_market_days = self._loaded_market_schedule[
            self._loaded_market_schedule.index.tz_convert('UTC') > current_time_utc
        ]
        
        # If no market days found, load schedule farther in future
        if next_market_days.empty:
            self.current_time = self.current_time + pd.Timedelta(days=30)
            self._load_next_market_schedule()
            return next(self)
        
        # Get the next market day
        next_market_day = next_market_days.iloc[0]
        
        # Calculate execution time (minutes_before_close minutes before market close)
        market_close = next_market_day["market_close"]
        execution_time = market_close - pd.Timedelta(minutes=self.minutes_before_close)
        
        # Update current time
        self.current_time = execution_time
        
        # Convert to target timezone if needed
        if execution_time.tzinfo != self.timezone:
            execution_time = execution_time.astimezone(self.timezone)
        
        return execution_time 