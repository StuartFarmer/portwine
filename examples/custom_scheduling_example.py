#!/usr/bin/env python3
"""
Example of using custom schedule iterators with ExecutionBase

This script demonstrates how to use the additional schedule iterators from
the custom_schedule_iterators module for complex scheduling patterns.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import time

import pandas as pd
import pytz

# Adjust path to import from parent directory if running directly
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy
from portwine.execution_complex.mock_broker import MockBroker
from portwine.loaders.alpaca import AlpacaMarketDataLoader
from portwine.utils.custom_schedule_iterators import (
    FixedTimeScheduleIterator,
    IntradayScheduleIterator,
    CompositeScheduleIterator
)
from portwine.utils.schedule_iterator import DailyMarketScheduleIterator
from portwine.execution_complex.base import ExecutionBase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("custom_scheduling_example")


def create_fixed_time_scheduler():
    """Create a fixed time scheduler (e.g., 10:00 AM on weekdays)."""
    logger.info("Creating fixed time scheduler: 10:00 AM on weekdays")
    
    # Business days (Monday=0 to Friday=4)
    business_days = [0, 1, 2, 3, 4]
    
    # Create scheduler
    scheduler = FixedTimeScheduleIterator(
        execution_time="10:00",
        days_of_week=business_days,
        timezone="America/New_York"
    )
    
    return scheduler


def create_intraday_scheduler():
    """Create an intraday scheduler (e.g., every 30 minutes during market hours)."""
    logger.info("Creating intraday scheduler: every 30 minutes during market hours")
    
    # Create scheduler
    scheduler = IntradayScheduleIterator(
        interval_minutes=30,
        exchange="NYSE",
        timezone="America/New_York"
    )
    
    return scheduler


def create_composite_scheduler():
    """
    Create a composite scheduler that combines multiple scheduling patterns.
    
    For example, execute:
    - At market open
    - At noon
    - 15 minutes before market close
    """
    logger.info("Creating composite scheduler with multiple execution times")
    
    # Create market open scheduler
    market_open_scheduler = DailyMarketScheduleIterator(
        exchange="NYSE",
        minutes_before_close=390,  # Typical 6.5 hour trading day = 390 minutes
        timezone="America/New_York"
    )
    
    # Create noon scheduler
    noon_scheduler = FixedTimeScheduleIterator(
        execution_time="12:00",
        days_of_week=[0, 1, 2, 3, 4],  # Weekdays
        timezone="America/New_York"
    )
    
    # Create market close scheduler
    market_close_scheduler = DailyMarketScheduleIterator(
        exchange="NYSE",
        minutes_before_close=15,
        timezone="America/New_York"
    )
    
    # Combine schedulers
    scheduler = CompositeScheduleIterator(
        iterators=[market_open_scheduler, noon_scheduler, market_close_scheduler],
        timezone="America/New_York"
    )
    
    return scheduler


def create_test_execution():
    """Create a test execution system with mock components."""
    logger.info("Creating test execution system")
    
    # Create mock components
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    strategy = SimpleMovingAverageStrategy(
        tickers=tickers,
        short_window=20,
        long_window=50,
        position_size=0.1
    )
    broker = MockBroker(market_open=True)
    data_loader = AlpacaMarketDataLoader(
        api_key="test_key",
        api_secret="test_secret",
        paper_trading=True
    )
    
    # Create concrete execution class
    class TestExecution(ExecutionBase):
        pass
    
    # Create and return execution system
    return TestExecution(strategy, data_loader, broker)


def print_upcoming_schedule(scheduler, count=5):
    """Print the next few scheduled execution times."""
    logger.info(f"Upcoming {count} execution times:")
    
    # Create a copy of the scheduler to avoid advancing the original
    times = []
    for i in range(count):
        time = next(scheduler)
        times.append(time)
        logger.info(f"  {i+1}. {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    return times


def run_demo(scheduler_type, max_iterations=3):
    """Run a demonstration with the specified scheduler type."""
    # Create execution system
    execution = create_test_execution()
    
    # Create scheduler based on type
    if scheduler_type == "fixed":
        scheduler = create_fixed_time_scheduler()
    elif scheduler_type == "intraday":
        scheduler = create_intraday_scheduler()
    elif scheduler_type == "composite":
        scheduler = create_composite_scheduler()
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Print upcoming schedule
    print_upcoming_schedule(scheduler, count=5)
    
    # Create a new scheduler for execution
    if scheduler_type == "fixed":
        run_scheduler = create_fixed_time_scheduler()
    elif scheduler_type == "intraday":
        run_scheduler = create_intraday_scheduler()
    elif scheduler_type == "composite":
        run_scheduler = create_composite_scheduler()
    
    # For demo purposes, create a special short interval scheduler
    class ShortDemoScheduler(FixedTimeScheduleIterator):
        """A scheduler that generates times a few seconds apart for demo purposes."""
        
        def __next__(self):
            """Return a time a few seconds in the future."""
            next_time = pd.Timestamp.now(tz=self.timezone) + pd.Timedelta(seconds=2)
            self.current_time = next_time
            return next_time
    
    logger.info("Using short demo scheduler for demonstration purposes")
    demo_scheduler = ShortDemoScheduler(
        execution_time="00:00",  # Not used in the override
        timezone="America/New_York"
    )
    
    # Run with the demo scheduler
    logger.info(f"Running execution with {max_iterations} iterations")
    execution.run(demo_scheduler, max_iterations=max_iterations)
    
    logger.info("Demonstration completed")


def main():
    """Run the example."""
    parser = argparse.ArgumentParser(description="Demonstrate custom scheduling")
    parser.add_argument(
        "--type", 
        type=str, 
        default="fixed",
        choices=["fixed", "intraday", "composite"],
        help="Type of scheduler to demonstrate"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=3,
        help="Number of iterations to run"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting custom scheduling example with {args.type} scheduler")
        run_demo(args.type, args.iterations)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error: {e}")


if __name__ == "__main__":
    main() 