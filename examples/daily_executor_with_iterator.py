#!/usr/bin/env python3
"""
Example script demonstrating the updated DailyExecutor with schedule iterators.

This script shows how to configure and run a trading strategy using
the updated DailyExecutor class with the new schedule iterator feature.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd
import pytz

# Add the parent directory to the path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from portwine.utils.daily_executor import DailyExecutor


def setup_logging(config):
    """Set up logging based on configuration."""
    logging_config = config.get("logging", {})
    level = logging_config.get("level", "INFO")
    log_file = logging_config.get("file")
    log_format = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=handlers
    )


def create_mock_config(run_time):
    """Create a sample configuration for demonstration."""
    return {
        "strategy": {
            "class": "portwine.strategies.simple_moving_average.SimpleMovingAverageStrategy",
            "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
            "params": {
                "short_window": 20,
                "long_window": 50,
                "position_size": 0.1
            }
        },
        "execution": {
            "class": "portwine.execution_complex.MockExecution",
            "params": {
                "initial_cash": 100000.0,
                "fail_symbols": []
            }
        },
        "data_loader": {
            "class": "portwine.loaders.alpaca.AlpacaMarketDataLoader",
            "params": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "cache_dir": "./data/alpaca_cache",
                "paper_trading": True
            }
        },
        "schedule": {
            "run_time": run_time,
            "time_zone": "US/Eastern",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "exchange": "NYSE",
            "market_hours_only": True
        },
        "logging": {
            "level": "INFO",
            "file": "iterator_example.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


def main():
    """Run the daily executor example."""
    parser = argparse.ArgumentParser(description="Run a trading strategy using DailyExecutor with schedule iterators")
    parser.add_argument(
        "--run-time",
        type=str,
        default="15:45",
        help="Time to run (e.g., '15:45', 'market_close-15m', 'market_open+30m')"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run the strategy once instead of scheduling"
    )
    parser.add_argument(
        "--sim-days",
        type=int, 
        default=0,
        help="Simulate running for specified number of days instead of blocking indefinitely"
    )
    args = parser.parse_args()

    try:
        # Create config with specified run time
        config = create_mock_config(args.run_time)
        
        # Set up logging
        setup_logging(config)
        logger = logging.getLogger("daily_executor_example")
        
        # Create executor from config
        logger.info(f"Creating executor with run time: {args.run_time}")
        executor = DailyExecutor(config)
        
        # Initialize all components
        logger.info("Initializing executor components")
        executor.initialize()
        
        if args.run_once:
            logger.info("Running strategy once...")
            executor.run_once()
        elif args.sim_days > 0:
            logger.info(f"Simulating scheduled execution for {args.sim_days} days...")
            
            # Create schedule iterator
            schedule_iterator = executor._create_schedule_iterator()
            
            # Get current time
            now = pd.Timestamp.now(tz=schedule_iterator.timezone)
            
            # Calculate end time
            end_time = now + pd.Timedelta(days=args.sim_days)
            
            logger.info(f"Simulation period: {now} to {end_time}")
            
            # Get first execution time
            next_time = next(schedule_iterator)
            logger.info(f"First execution time: {next_time}")
            
            # Simulate time passing
            while next_time < end_time:
                # In a real scenario, we'd wait until this time
                logger.info(f"Would execute at: {next_time}")
                
                # Get the next execution time
                next_time = next(schedule_iterator)
        else:
            logger.info("Starting scheduled execution...")
            executor.run_scheduled()
    
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'executor' in locals():
            executor.shutdown()


if __name__ == "__main__":
    main() 