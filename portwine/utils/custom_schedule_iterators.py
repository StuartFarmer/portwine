"""
Custom Schedule Iterators

This module provides additional implementations of the ScheduleIterator abstract class
for different scheduling needs beyond the basic DailyMarketScheduleIterator.
"""

import abc
from datetime import datetime, time
from typing import Optional, Union, List, Tuple

import pandas as pd
import pytz
import pandas_market_calendars as mcal

from portwine.utils.schedule_iterator import ScheduleIterator


class FixedTimeScheduleIterator(ScheduleIterator):
    """
    Iterator that yields fixed times on specified days.
    
    This iterator schedules execution at the same time every day,
    optionally filtered to specific days of the week.
    """
    
    def __init__(
        self,
        execution_time: Union[str, time, Tuple[int, int]],
        days_of_week: Optional[List[int]] = None,
        timezone: Optional[Union[str, pytz.timezone]] = None,
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None
    ):
        """
        Initialize with fixed execution time and days.
        
        Parameters
        ----------
        execution_time : Union[str, time, Tuple[int, int]]
            The time of day to execute, in any of these formats:
            - String "HH:MM" (24-hour format)
            - datetime.time object
            - Tuple of (hour, minute)
        days_of_week : Optional[List[int]], default None
            List of days to include (0=Monday, 6=Sunday), or None for all days
        timezone : Optional[Union[str, pytz.timezone]], default None
            Timezone to use for calculations
        start_date : Optional[Union[str, datetime, pd.Timestamp]], default None
            Start date from which to begin scheduling
        """
        super().__init__(timezone, start_date)
        
        # Parse execution time
        if isinstance(execution_time, str):
            hour, minute = map(int, execution_time.split(':'))
            self.hour = hour
            self.minute = minute
        elif isinstance(execution_time, time):
            self.hour = execution_time.hour
            self.minute = execution_time.minute
        elif isinstance(execution_time, tuple) and len(execution_time) == 2:
            self.hour, self.minute = execution_time
        else:
            raise ValueError(
                "execution_time must be a string ('HH:MM'), time object, or tuple (hour, minute)"
            )
        
        # Set days of week (0=Monday, 6=Sunday to match pd.Timestamp.dayofweek)
        self.days_of_week = days_of_week
    
    def __next__(self) -> pd.Timestamp:
        """
        Return the next execution time.
        
        Returns
        -------
        pd.Timestamp
            The next scheduled execution time
        """
        # Start from current time
        current = self.current_time
        
        # Find the next execution time
        while True:
            # Set time to execution time
            next_time = current.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
            
            # If next_time is in the past, move to the next day
            if next_time <= current:
                next_time = next_time + pd.Timedelta(days=1)
            
            # Check if the day of week is allowed
            if self.days_of_week is not None:
                day_of_week = next_time.dayofweek  # 0=Monday, 6=Sunday
                if day_of_week not in self.days_of_week:
                    # Move to the next day and try again
                    current = next_time
                    continue
            
            # We found a valid time
            break
        
        # Update current time
        self.current_time = next_time
        
        return next_time


class IntradayScheduleIterator(ScheduleIterator):
    """
    Iterator that yields times at regular intervals during market hours.
    
    This iterator schedules execution at regular intervals during market hours,
    for example every 15 minutes while the market is open.
    """
    
    def __init__(
        self,
        interval_minutes: int = 15,
        exchange: str = "NYSE",
        timezone: Optional[Union[str, pytz.timezone]] = None,
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None
    ):
        """
        Initialize with interval and exchange.
        
        Parameters
        ----------
        interval_minutes : int, default 15
            Interval between executions in minutes
        exchange : str, default "NYSE"
            Exchange to use for market hours
        timezone : Optional[Union[str, pytz.timezone]], default None
            Timezone to use for calculations
        start_date : Optional[Union[str, datetime, pd.Timestamp]], default None
            Start date from which to begin scheduling
        """
        super().__init__(timezone, start_date)
        self.interval_minutes = interval_minutes
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)
        
        # Load market schedule for the upcoming period
        self._loaded_market_schedule = None
        self._load_next_market_schedule()
        
        # Keep track of the next intraday time
        self._next_intraday_time = None
    
    def _load_next_market_schedule(self):
        """Load market schedule for the next month."""
        start_date = self.current_time.date()
        end_date = (self.current_time + pd.Timedelta(days=30)).date()
        
        self._loaded_market_schedule = self.calendar.schedule(
            start_date=start_date,
            end_date=end_date
        )
        
        # Ensure index has timezone info
        if not self._loaded_market_schedule.empty:
            if self._loaded_market_schedule.index[0].tzinfo is None:
                self._loaded_market_schedule.index = self._loaded_market_schedule.index.tz_localize('UTC')
    
    def _get_next_market_day(self) -> Optional[pd.Series]:
        """Get the next trading day after the current time."""
        if (self._loaded_market_schedule is None or 
            self._loaded_market_schedule.empty or
            self.current_time.date() > self._loaded_market_schedule.index[-1].date()):
            self._load_next_market_schedule()
        
        # Ensure current_time has timezone for comparison
        current_time_utc = self.current_time.tz_convert('UTC') if self.current_time.tzinfo is not None else self.current_time
        
        # Find next market days
        next_market_days = self._loaded_market_schedule[
            self._loaded_market_schedule.index.tz_convert('UTC') > current_time_utc
        ]
        
        if next_market_days.empty:
            # Load schedule for a later period
            self.current_time = self.current_time + pd.Timedelta(days=30)
            self._load_next_market_schedule()
            return self._get_next_market_day()
        
        return next_market_days.iloc[0]
    
    def _generate_intraday_times(self, market_open: pd.Timestamp, market_close: pd.Timestamp) -> List[pd.Timestamp]:
        """Generate intraday execution times between market_open and market_close."""
        # Total minutes in market day
        total_minutes = int((market_close - market_open).total_seconds() / 60)
        
        # Number of intervals
        num_intervals = total_minutes // self.interval_minutes
        
        # Generate times
        times = []
        for i in range(num_intervals + 1):  # +1 to include market close
            interval_time = market_open + pd.Timedelta(minutes=i * self.interval_minutes)
            if interval_time <= market_close:
                times.append(interval_time)
        
        return times
    
    def __next__(self) -> pd.Timestamp:
        """
        Return the next execution time.
        
        Returns
        -------
        pd.Timestamp
            The next scheduled execution time
        """
        # If we have intraday times and the next one is in the future, return it
        if self._next_intraday_time is not None and len(self._next_intraday_time) > 0:
            next_time = self._next_intraday_time[0]
            
            # If the next time is in the future, use it
            if next_time > self.current_time:
                # Update the list and current time
                self._next_intraday_time = self._next_intraday_time[1:]
                self.current_time = next_time
                return next_time
        
        # Get the next market day
        next_market_day = self._get_next_market_day()
        
        # Get market open and close times
        market_open = next_market_day["market_open"]
        market_close = next_market_day["market_close"]
        
        # Generate intraday times
        intraday_times = self._generate_intraday_times(market_open, market_close)
        
        # Find the first time that's after current_time
        future_times = [t for t in intraday_times if t > self.current_time]
        
        if not future_times:
            # No future times today, move to next day
            self.current_time = market_close + pd.Timedelta(minutes=1)
            return next(self)
        
        # Save remaining times for next calls
        next_time = future_times[0]
        self._next_intraday_time = future_times[1:]
        
        # Update current time
        self.current_time = next_time
        
        # Convert to target timezone if needed
        if next_time.tzinfo != self.timezone:
            next_time = next_time.astimezone(self.timezone)
        
        return next_time


class CompositeScheduleIterator(ScheduleIterator):
    """
    Iterator that combines multiple schedule iterators.
    
    This iterator takes a list of schedule iterators and yields the next
    earliest time from any of them, allowing for complex schedules combining
    different scheduling patterns.
    """
    
    def __init__(
        self,
        iterators: List[ScheduleIterator],
        timezone: Optional[Union[str, pytz.timezone]] = None
    ):
        """
        Initialize with a list of schedule iterators.
        
        Parameters
        ----------
        iterators : List[ScheduleIterator]
            List of schedule iterators to combine
        timezone : Optional[Union[str, pytz.timezone]], default None
            Timezone to use for calculations
        """
        # Use the first iterator's time as the start time if available
        start_time = None
        if iterators:
            # Take earliest current_time from iterators
            start_time = min(iterator.current_time for iterator in iterators)
        
        super().__init__(timezone, start_time)
        
        if not iterators:
            raise ValueError("Must provide at least one iterator")
        
        self.iterators = iterators
        
        # Initialize next times
        self.next_times = [next(iterator) for iterator in iterators]
    
    def __next__(self) -> pd.Timestamp:
        """
        Return the next earliest execution time from any iterator.
        
        Returns
        -------
        pd.Timestamp
            The next scheduled execution time
        """
        if not self.next_times:
            raise StopIteration("No more scheduled times")
        
        # Find the earliest time
        earliest_idx = min(range(len(self.next_times)), key=lambda i: self.next_times[i])
        earliest_time = self.next_times[earliest_idx]
        
        # Get the next time from that iterator
        try:
            self.next_times[earliest_idx] = next(self.iterators[earliest_idx])
        except StopIteration:
            # If an iterator is exhausted, remove it
            del self.iterators[earliest_idx]
            del self.next_times[earliest_idx]
        
        # Update current time
        self.current_time = earliest_time
        
        return earliest_time 