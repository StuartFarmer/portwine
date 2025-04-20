"""
Daily Executor Module

This module provides a flexible executor that can run trading strategies
on a configurable schedule, including intraday, market open/close, 
and custom time-based schedules.
"""

import json
import time
import logging
import importlib
import schedule
import pytz
import re
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

import pandas as pd
import pandas_market_calendars as mcal
from portwine.utils.schedule_iterator import ScheduleIterator, DailyMarketScheduleIterator

logger = logging.getLogger(__name__)

class DailyExecutor:
    """
    Executor for running trading strategies on flexible schedules.
    
    This class handles the orchestration of:
    - Loading configuration
    - Initializing strategy, execution, and data components
    - Running strategies on configurable schedules (daily, intraday, market-based)
    - Proper shutdown and resource cleanup
    
    Scheduling options include:
    - Fixed time (e.g., "15:45")
    - Market event based (e.g., "market_open", "market_close")
    - Offset from market events (e.g., "market_open+30m", "market_close-15m")
    - Intraday intervals (e.g., "interval:30m" for every 30 minutes during market hours)
    """
    
    # Market event patterns
    MARKET_EVENT_PATTERN = re.compile(r'^(market_open|market_close)(?:([+-])(\d+)([mh]))?$')
    INTERVAL_PATTERN = re.compile(r'^interval:(\d+)([mh])$')
    
    def __init__(self, config: Dict[Any, Any]):
        """
        Initialize the executor with configuration.
        
        Args:
            config: Dictionary containing configuration for strategy, 
                   execution, data loading, and scheduling.
        """
        self.config = config
        self.strategy = None
        self.executor = None
        self.data_loader = None
        self.initialized = False
        
        # Extract execution schedule settings
        self.schedule_config = config.get("schedule", {})
        self.run_time = self.schedule_config.get("run_time", "15:45")  # Default to 15:45 ET
        self.time_zone = self.schedule_config.get("time_zone", "US/Eastern")
        self.days = self.schedule_config.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        
        # Exchange calendar for market timing
        self.exchange = self.schedule_config.get("exchange", "NYSE")
        self.calendar = mcal.get_calendar(self.exchange)
        
        # Intraday settings
        self.intraday_schedule = self.schedule_config.get("intraday", None)
        self.market_hours_only = self.schedule_config.get("market_hours_only", True)
        
        # Job registry
        self._scheduled_jobs = []
    
    @classmethod
    def from_config_file(cls, config_file_path: str) -> 'DailyExecutor':
        """
        Create a DailyExecutor from a configuration file.
        
        Args:
            config_file_path: Path to the JSON configuration file.
            
        Returns:
            DailyExecutor instance configured with the file contents.
        """
        with open(config_file_path, 'r') as file:
            config = json.load(file)
        
        return cls(config)
    
    def _import_class(self, class_path: str) -> Any:
        """
        Dynamically import a class from its string path.
        
        Args:
            class_path: String in format "module.submodule.ClassName"
            
        Returns:
            The imported class object
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def initialize(self) -> None:
        """
        Initialize all components based on configuration.
        
        This includes:
        - Data loader
        - Strategy
        - Execution system
        """
        if self.initialized:
            logger.warning("Components already initialized")
            return
            
        logger.info("Initializing components...")
        
        # Initialize data loader
        if "data_loader" in self.config:
            data_loader_config = self.config["data_loader"]
            data_loader_class = self._import_class(data_loader_config["class"])
            data_loader_params = data_loader_config.get("params", {})
            self.data_loader = data_loader_class(**data_loader_params)
            logger.info(f"Initialized data loader: {data_loader_class.__name__}")
        
        # Initialize strategy
        if "strategy" in self.config:
            strategy_config = self.config["strategy"]
            strategy_class = self._import_class(strategy_config["class"])
            strategy_params = strategy_config.get("params", {})
            
            # If tickers are specified separately, add them to params
            if "tickers" in strategy_config:
                strategy_params["tickers"] = strategy_config["tickers"]
                
            # Create the strategy, passing the data loader if needed
            if self.data_loader is not None and "data_loader" not in strategy_params:
                strategy_params["data_loader"] = self.data_loader
                
            self.strategy = strategy_class(**strategy_params)
            logger.info(f"Initialized strategy: {strategy_class.__name__}")
        
        # Initialize execution system
        if "execution" in self.config:
            execution_config = self.config["execution"]
            execution_class = self._import_class(execution_config["class"])
            execution_params = execution_config.get("params", {})
            
            # Add strategy if not already provided
            if self.strategy is not None and "strategy" not in execution_params:
                execution_params["strategy"] = self.strategy
                
            self.executor = execution_class(**execution_params)
            logger.info(f"Initialized execution system: {execution_class.__name__}")
        
        self.initialized = True
        logger.info("All components initialized successfully")
    
    def run_once(self) -> None:
        """
        Run the strategy execution once.
        
        This is useful for manual execution or testing.
        """
        if not self.initialized:
            logger.error("Components not initialized. Call initialize() first.")
            return
            
        # Check if we should only run during market hours
        if self.market_hours_only and not self._is_market_open():
            logger.info("Market is closed. Skipping execution as market_hours_only is enabled.")
            return
            
        logger.info("Running strategy execution...")
        try:
            # Update market data
            if self.data_loader:
                logger.info("Updating market data...")
                if hasattr(self.data_loader, 'update_data'):
                    self.data_loader.update_data()
            
            # Generate signals
            logger.info("Generating trading signals...")
            if self.strategy:
                if hasattr(self.strategy, 'generate_signals'):
                    signals = self.strategy.generate_signals()
                    logger.info(f"Generated {len(signals)} signals")
                else:
                    logger.warning("Strategy does not have generate_signals method")
            
            # Execute trades
            logger.info("Executing trades...")
            if self.executor:
                if hasattr(self.executor, 'execute'):
                    self.executor.execute()
                else:
                    # Try step method as an alternative
                    if hasattr(self.executor, 'step'):
                        self.executor.step()
                    else:
                        logger.warning("Executor does not have execute or step method")
                
            logger.info("Strategy execution completed successfully")
            
        except Exception as e:
            logger.exception(f"Error during strategy execution: {e}")
    
    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        now = datetime.now(pytz.timezone(self.time_zone))
        today_str = now.strftime('%Y-%m-%d')
        
        # Get schedule for today
        schedule_df = self.calendar.schedule(start_date=today_str, end_date=today_str)
        
        if schedule_df.empty:
            return False  # Not a trading day
        
        # Get market open and close times
        market_open = schedule_df.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule_df.iloc[0]['market_close'].to_pydatetime()
        
        # Check if current time is between market open and close
        return market_open <= now <= market_close
    
    def _get_market_times(self, date: Optional[datetime] = None) -> Dict[str, datetime]:
        """
        Get market open and close times for a specific date.
        
        Args:
            date: The date to check (default: today)
            
        Returns:
            Dict containing market_open and market_close times, or None if not a trading day
        """
        if date is None:
            date = datetime.now(pytz.timezone(self.time_zone))
            
        date_str = date.strftime('%Y-%m-%d')
        
        # Get schedule for the date
        schedule_df = self.calendar.schedule(start_date=date_str, end_date=date_str)
        
        if schedule_df.empty:
            return None  # Not a trading day
        
        # Get market open and close times
        market_open = schedule_df.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule_df.iloc[0]['market_close'].to_pydatetime()
        
        return {
            'market_open': market_open,
            'market_close': market_close
        }
    
    def _parse_time_expression(self, time_expr: str, date: Optional[datetime] = None) -> Optional[datetime]:
        """
        Parse a time expression into a specific datetime.
        
        Supports:
        - Fixed time ("HH:MM")
        - Market events ("market_open", "market_close")
        - Offsets from market events ("market_open+30m", "market_close-15m")
        
        Args:
            time_expr: Time expression string
            date: Base date to use (default: today)
            
        Returns:
            datetime object if valid time expression, None otherwise
        """
        if date is None:
            date = datetime.now(pytz.timezone(self.time_zone))
            
        # Check for fixed time format (HH:MM)
        if re.match(r'^\d{1,2}:\d{2}$', time_expr):
            hours, minutes = map(int, time_expr.split(':'))
            return date.replace(hour=hours, minute=minutes, second=0, microsecond=0)
        
        # Check for market event with optional offset
        market_match = self.MARKET_EVENT_PATTERN.match(time_expr)
        if market_match:
            event, sign, offset_val, offset_unit = market_match.groups()
            
            # Get market times for the day
            market_times = self._get_market_times(date)
            if not market_times:
                logger.warning(f"Not a trading day: {date.strftime('%Y-%m-%d')}")
                return None
            
            # Get base event time
            base_time = market_times[event]
            
            # Apply offset if provided
            if offset_val:
                offset_val = int(offset_val)
                if sign == '-':
                    offset_val = -offset_val
                
                if offset_unit == 'm':
                    base_time = base_time + timedelta(minutes=offset_val)
                elif offset_unit == 'h':
                    base_time = base_time + timedelta(hours=offset_val)
            
            return base_time
        
        logger.warning(f"Invalid time expression: {time_expr}")
        return None
    
    def _setup_intraday_schedule(self) -> None:
        """
        Set up intraday scheduling based on configuration.
        """
        if not self.intraday_schedule:
            return
            
        # Handle interval-based scheduling
        interval_match = self.INTERVAL_PATTERN.match(self.intraday_schedule)
        if interval_match:
            interval_val, interval_unit = interval_match.groups()
            interval_val = int(interval_val)
            
            # Calculate interval in minutes
            if interval_unit == 'h':
                interval_minutes = interval_val * 60
            else:
                interval_minutes = interval_val
            
            # For intraday intervals, we need to dynamically schedule
            # at the start of each day and clear at the end
            for day in self.days:
                # Schedule a job to set up the day's intraday schedule
                getattr(schedule.every(), day.lower()).at("00:01").do(
                    self._setup_intraday_for_day, interval_minutes=interval_minutes
                )
            
            logger.info(f"Set up intraday interval scheduling: every {interval_val}{interval_unit} on {', '.join(self.days)}")
    
    def _setup_intraday_for_day(self, interval_minutes: int) -> None:
        """
        Set up intraday schedule for a specific day.
        
        Args:
            interval_minutes: Interval in minutes between executions
        """
        # Clear any existing intraday jobs
        self._clear_intraday_jobs()
        
        # Get market hours for today
        today = datetime.now(pytz.timezone(self.time_zone))
        market_times = self._get_market_times(today)
        
        if not market_times:
            logger.info(f"Today {today.strftime('%Y-%m-%d')} is not a trading day. No intraday schedule needed.")
            return
        
        # Start from market open
        current_time = market_times['market_open']
        end_time = market_times['market_close']
        
        # Schedule jobs at intervals until market close
        while current_time < end_time:
            # Convert to local time string for schedule library
            time_str = current_time.astimezone(pytz.timezone(self.time_zone)).strftime('%H:%M')
            
            # Only schedule future jobs
            now = datetime.now(pytz.timezone(self.time_zone))
            if current_time > now:
                job = schedule.every().day.at(time_str).do(self.run_once)
                self._scheduled_jobs.append(job)
                logger.debug(f"Scheduled intraday execution at {time_str}")
            
            # Move to next interval
            current_time += timedelta(minutes=interval_minutes)
    
    def _clear_intraday_jobs(self) -> None:
        """Clear all scheduled intraday jobs."""
        for job in self._scheduled_jobs:
            schedule.cancel_job(job)
        
        self._scheduled_jobs = []
        logger.debug("Cleared intraday scheduled jobs")
    
    def _create_schedule_iterator(self) -> ScheduleIterator:
        """
        Create an appropriate schedule iterator based on configuration.
        
        Returns
        -------
        ScheduleIterator
            The schedule iterator that will yield execution times
        """
        # Default configuration
        timezone = self.time_zone
        
        # Check if the run_time is a market event-based time
        if self.MARKET_EVENT_PATTERN.match(self.run_time):
            # For market-based times, use a DailyMarketScheduleIterator
            market_match = self.MARKET_EVENT_PATTERN.match(self.run_time)
            event_type = market_match.group(1)
            op = market_match.group(2)
            amount = int(market_match.group(3)) if market_match.group(3) else 0
            unit = market_match.group(4) if market_match.group(4) else 'm'
            
            # Calculate minutes from close
            if event_type == "market_close":
                if op == "-":
                    minutes_before_close = amount * (60 if unit == 'h' else 1)
                elif op == "+":
                    # After market close (negative minutes before close)
                    minutes_before_close = -amount * (60 if unit == 'h' else 1)
                else:
                    # Exactly at market close
                    minutes_before_close = 0
            else:  # market_open
                # Convert to minutes before close
                # First calculate the offset from market open
                offset_minutes = amount * (60 if unit == 'h' else 1)
                if op == "-":
                    offset_minutes = -offset_minutes
                
                # We'll use a custom implementation later, but for now
                # we'll estimate the trading day as 6.5 hours (390 minutes)
                minutes_before_close = 390 - offset_minutes
            
            return DailyMarketScheduleIterator(
                exchange=self.exchange,
                minutes_before_close=minutes_before_close,
                timezone=timezone
            )
        else:
            # For fixed times, we'll use a DailyMarketScheduleIterator with manual calculation
            # Parse the fixed time
            try:
                hour, minute = map(int, self.run_time.split(':'))
                
                # Get market hours for calculation
                now = datetime.now(pytz.timezone(timezone))
                market_times = self._get_market_times(now)
                
                if market_times:
                    market_close = market_times['market_close']
                    
                    # Calculate time difference from market close
                    fixed_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # Convert both to minutes since midnight for easy comparison
                    fixed_minutes = hour * 60 + minute
                    close_minutes = market_close.hour * 60 + market_close.minute
                    
                    # Calculate minutes before close
                    minutes_before_close = close_minutes - fixed_minutes
                    
                    # Handle case where time is after market close
                    if minutes_before_close < 0:
                        minutes_before_close = 0  # Execute at market close
                    
                    return DailyMarketScheduleIterator(
                        exchange=self.exchange,
                        minutes_before_close=minutes_before_close,
                        timezone=timezone
                    )
                else:
                    # No market times available, use a default
                    logger.warning("Could not determine market hours. Using default 15 minutes before close.")
                    return DailyMarketScheduleIterator(
                        exchange=self.exchange,
                        minutes_before_close=15,
                        timezone=timezone
                    )
                
            except (ValueError, TypeError):
                logger.error(f"Invalid time format: {self.run_time}. Using default 15 minutes before close.")
                return DailyMarketScheduleIterator(
                    exchange=self.exchange,
                    minutes_before_close=15,
                    timezone=timezone
                )
    
    def run_scheduled(self) -> None:
        """
        Run the strategy on a schedule based on the configuration.
        
        This will block indefinitely until interrupted.
        """
        if not self.initialized:
            logger.error("Components not initialized. Call initialize() first.")
            return
        
        # Create appropriate schedule iterator
        schedule_iterator = self._create_schedule_iterator()
        
        logger.info(f"Scheduled execution on {', '.join(self.days)}")
        
        # Get the first execution time
        next_run_time = next(schedule_iterator)
        
        # Log when the next run will occur
        self._log_next_run_time(next_run_time)
        
        # Run the scheduler indefinitely
        try:
            while True:
                # Get the current time
                now = pd.Timestamp.now(tz=schedule_iterator.timezone)
                
                # Check if it's time to run
                if now >= next_run_time:
                    # Check if today is in the configured days
                    current_day = now.day_name()
                    if current_day in self.days:
                        logger.info(f"Executing scheduled run at {now}")
                        self.run_once()
                    else:
                        logger.info(f"Skipping execution on {current_day} (not in configured days)")
                    
                    # Get the next execution time
                    next_run_time = next(schedule_iterator)
                    self._log_next_run_time(next_run_time)
                
                # Sleep to avoid busy waiting
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
    
    def _log_next_run_time(self, next_run_time: pd.Timestamp) -> None:
        """
        Calculate and log the time until the next scheduled run.
        
        Parameters
        ----------
        next_run_time : pd.Timestamp
            The next scheduled run time
        """
        now = pd.Timestamp.now(tz=next_run_time.tzinfo)
        time_diff = next_run_time - now
        
        # Convert to hours and minutes
        seconds = time_diff.total_seconds()
        hours, remainder = divmod(seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        logger.info(f"Next run scheduled in {int(hours)} hours and {int(minutes)} minutes at {next_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    def shutdown(self) -> None:
        """
        Clean shutdown of all components.
        """
        logger.info("Shutting down components...")
        
        # Shutdown execution system
        if self.executor:
            try:
                if hasattr(self.executor, 'shutdown'):
                    self.executor.shutdown()
                logger.info("Execution system shut down")
            except Exception as e:
                logger.exception(f"Error shutting down execution system: {e}")
        
        # Shutdown strategy
        if self.strategy:
            try:
                if hasattr(self.strategy, 'shutdown'):
                    self.strategy.shutdown()
                logger.info("Strategy shut down")
            except Exception as e:
                logger.exception(f"Error shutting down strategy: {e}")
                
        # Shutdown data loader
        if self.data_loader:
            try:
                if hasattr(self.data_loader, 'shutdown'):
                    self.data_loader.shutdown()
                logger.info("Data loader shut down")
            except Exception as e:
                logger.exception(f"Error shutting down data loader: {e}")
        
        self.initialized = False
        logger.info("All components shut down successfully") 