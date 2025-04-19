#!/usr/bin/env python3
"""
Example of using ExecutionBase with ScheduleIterator

This script demonstrates how to use the ExecutionBase.run method with a ScheduleIterator
to schedule trading at specific times, replacing the functionality of daily_executor.py.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import pandas as pd

from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy
from portwine.execution_complex.mock_broker import MockBroker
from portwine.execution_complex.alpaca_broker import AlpacaBroker
from portwine.loaders.alpaca import AlpacaMarketDataLoader
from portwine.utils.schedule_iterator import DailyMarketScheduleIterator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading.log")
    ]
)
logger = logging.getLogger("portwine.execution_example")


def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise


def create_components_from_config(config):
    """Create strategy, broker, and data loader from config."""
    # Create data loader
    data_loader_config = config.get("data_loader", {})
    data_loader_class = data_loader_config.get("class", "portwine.loaders.alpaca.AlpacaMarketDataLoader")
    data_loader_params = data_loader_config.get("params", {})
    
    # Handle API keys from environment variables
    if "api_key" in data_loader_params and data_loader_params["api_key"].startswith("${"):
        env_var = data_loader_params["api_key"].strip("${}")
        data_loader_params["api_key"] = os.environ.get(env_var, "")
    
    if "api_secret" in data_loader_params and data_loader_params["api_secret"].startswith("${"):
        env_var = data_loader_params["api_secret"].strip("${}")
        data_loader_params["api_secret"] = os.environ.get(env_var, "")
    
    # Create the data loader
    if data_loader_class == "portwine.loaders.alpaca.AlpacaMarketDataLoader":
        data_loader = AlpacaMarketDataLoader(**data_loader_params)
    else:
        raise ValueError(f"Unsupported data loader class: {data_loader_class}")
    
    # Create strategy
    strategy_config = config.get("strategy", {})
    strategy_class = strategy_config.get("class", "portwine.strategies.simple_moving_average.SimpleMovingAverageStrategy")
    strategy_params = strategy_config.get("params", {})
    tickers = strategy_config.get("tickers", ["AAPL", "MSFT", "GOOGL"])
    
    # Create the strategy
    if strategy_class == "portwine.strategies.simple_moving_average.SimpleMovingAverageStrategy":
        strategy = SimpleMovingAverageStrategy(tickers=tickers, **strategy_params)
    else:
        raise ValueError(f"Unsupported strategy class: {strategy_class}")
    
    # Create broker
    broker_config = config.get("execution", {})
    broker_class = broker_config.get("class", "portwine.execution_complex.mock_broker.MockBroker")
    broker_params = broker_config.get("params", {})
    
    # Handle API keys from environment variables
    if "api_key" in broker_params and broker_params["api_key"].startswith("${"):
        env_var = broker_params["api_key"].strip("${}")
        broker_params["api_key"] = os.environ.get(env_var, "")
    
    if "api_secret" in broker_params and broker_params["api_secret"].startswith("${"):
        env_var = broker_params["api_secret"].strip("${}")
        broker_params["api_secret"] = os.environ.get(env_var, "")
    
    # Create the broker
    if broker_class == "portwine.execution_complex.MockBroker":
        broker = MockBroker(**broker_params)
    elif broker_class == "portwine.execution_complex.AlpacaBroker":
        broker = AlpacaBroker(**broker_params)
    else:
        raise ValueError(f"Unsupported broker class: {broker_class}")
    
    return strategy, broker, data_loader


def create_schedule_from_config(config):
    """Create a schedule iterator from config."""
    schedule_config = config.get("schedule", {})
    
    # Get schedule parameters
    run_time = schedule_config.get("run_time", "15:45")
    timezone = schedule_config.get("time_zone", "America/New_York")
    exchange = schedule_config.get("exchange", "NYSE")
    
    # Create schedule iterator
    if "market_close" in run_time:
        # Parse time offset from market close
        if "-" in run_time:
            parts = run_time.split("-")
            minutes_before_close = int(parts[1].replace("m", ""))
        else:
            minutes_before_close = 15  # Default
            
        # Create market schedule iterator
        return DailyMarketScheduleIterator(
            exchange=exchange,
            minutes_before_close=minutes_before_close,
            timezone=timezone
        )
    else:
        # For fixed time schedules, convert to minutes before close
        # This is a simplification - in a full implementation, you might
        # want to support fixed times directly with a different iterator
        try:
            hour, minute = map(int, run_time.split(":"))
            # Assuming market closes at 16:00, calculate minutes before close
            minutes_before_close = (16 - hour) * 60 - minute
            
            # Create market schedule iterator
            return DailyMarketScheduleIterator(
                exchange=exchange,
                minutes_before_close=minutes_before_close,
                timezone=timezone
            )
        except ValueError:
            logger.error(f"Invalid run_time format: {run_time}, using default 15 minutes before close")
            return DailyMarketScheduleIterator(
                exchange=exchange,
                minutes_before_close=15,
                timezone=timezone
            )


def create_execution_system(strategy, broker, data_loader):
    """Create an execution system with strategy, broker, and data loader."""
    # Import dynamically to avoid circular imports
    from portwine.execution_complex.base import ExecutionBase
    
    # Create a concrete implementation of ExecutionBase
    class ConcreteExecution(ExecutionBase):
        pass
    
    # Create and return the execution system
    return ConcreteExecution(strategy, data_loader, broker)


def main():
    """Run the example."""
    parser = argparse.ArgumentParser(description="Run trading execution with schedule")
    parser.add_argument("--config", type=str, default="example_config.json",
                        help="Path to configuration file")
    parser.add_argument("--run-once", action="store_true",
                        help="Run once instead of scheduling")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Maximum number of iterations to run")
    
    args = parser.parse_args()
    
    try:
        # Resolve the config path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
        
        # Load the configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        
        # Create components
        strategy, broker, data_loader = create_components_from_config(config)
        
        # Create execution system
        execution = create_execution_system(strategy, broker, data_loader)
        
        # Run once or scheduled
        if args.run_once:
            logger.info("Running once...")
            execution.step()
        else:
            # Create schedule
            schedule = create_schedule_from_config(config)
            
            # Print next few scheduled times
            logger.info("Upcoming scheduled execution times:")
            for i in range(3):
                next_time = next(schedule)
                logger.info(f"  {i+1}. {next_time}")
                
            # Reset the schedule
            schedule = create_schedule_from_config(config)
            
            # Run with schedule
            logger.info("Starting scheduled execution...")
            execution.run(schedule, max_iterations=args.iterations)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting...")
    except Exception as e:
        logger.exception(f"Error: {e}")
    

if __name__ == "__main__":
    main() 