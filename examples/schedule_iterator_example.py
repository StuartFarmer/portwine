#!/usr/bin/env python3
"""
Example script demonstrating how to use the ScheduleIterator module.

This script shows how to create a DailyMarketScheduleIterator to schedule
a daily trading strategy execution at a specified time before market close.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz

# Add the parent directory to the path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from portwine.utils.schedule_iterator import DailyMarketScheduleIterator


def print_next_execution_times(iterator, count=10):
    """Print the next n execution times from the iterator."""
    print(f"Next {count} execution times:")
    for i in range(count):
        next_time = next(iterator)
        print(f"  {i + 1}. {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")


def main():
    """Run example demonstrations of schedule iterators."""
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("schedule_iterator_example")
    
    # Example 1: Basic usage with default parameters
    logger.info("Example 1: Basic usage with default parameters")
    
    # Create iterator with default parameters (NYSE, 15 minutes before close, UTC)
    iterator = DailyMarketScheduleIterator()
    
    # Print the next 10 execution times
    print_next_execution_times(iterator)
    print()
    
    # Example 2: Custom parameters
    logger.info("Example 2: Custom parameters")
    
    # Create iterator with custom parameters
    iterator = DailyMarketScheduleIterator(
        exchange="NASDAQ",
        minutes_before_close=30,
        timezone="America/New_York",
        start_date=pd.Timestamp("2023-05-01")  # Start from a specific date
    )
    
    # Print the next 10 execution times
    print_next_execution_times(iterator)
    print()
    
    # Example 3: Different timezone
    logger.info("Example 3: West coast timezone")
    
    # Create iterator with West coast timezone
    iterator = DailyMarketScheduleIterator(
        exchange="NYSE",
        minutes_before_close=15,
        timezone="America/Los_Angeles"
    )
    
    # Print the next 10 execution times
    print_next_execution_times(iterator)
    print()
    
    # Example 4: Use with a while loop for continuous execution
    logger.info("Example 4: Use in a while loop (simulation)")
    
    # Create iterator
    iterator = DailyMarketScheduleIterator(
        timezone="America/New_York"
    )
    
    # Simulate running for 3 days
    now = pd.Timestamp.now(tz="America/New_York")
    end_time = now + pd.Timedelta(days=3)
    
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("Schedule:")
    
    # Get the first execution time
    next_time = next(iterator)
    count = 1
    
    # Simulate time passing
    while next_time < end_time and count <= 5:
        print(f"  {count}. {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # In a real application, you would wait until next_time, then:
        # 1. Execute your strategy
        # 2. Get the next execution time
        
        # Get the next execution time
        next_time = next(iterator)
        count += 1


if __name__ == "__main__":
    main() 