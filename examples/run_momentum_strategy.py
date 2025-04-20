#!/usr/bin/env python3
"""
Example script demonstrating how to use the daily executor to run a momentum strategy.
"""
import os
import logging
import json
from portwine.utils.daily_executor import DailyExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MomentumStrategy")

if __name__ == "__main__":
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load from config file
    config_path = os.path.join(current_dir, "example_config.json")
    executor = DailyExecutor.from_config_file(config_path)
    
    # Or set up components with a configuration dictionary
    """
    # Example of creating a config dictionary manually
    config = {
        "strategy": {
            "class": "portwine.examples.execution_example.SimpleMomentumStrategy",
            "tickers": ["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            "params": {
                "lookback_days": 10,
                "top_n": 2
            }
        },
        "execution": {
            "class": "portwine.execution_alpaca.AlpacaExecution",
            "params": {
                "paper": True,
                "api_key": os.environ.get("ALPACA_API_KEY"),
                "api_secret": os.environ.get("ALPACA_API_SECRET")
            }
        },
        "data_loader": {
            "class": "portwine.loaders.alpaca.AlpacaMarketDataLoader",
            "params": {
                "cache_dir": "./data/alpaca_cache",
                "api_key": os.environ.get("ALPACA_API_KEY"),
                "api_secret": os.environ.get("ALPACA_API_SECRET")
            }
        },
        "schedule": {
            "run_time": "15:45",
            "time_zone": "America/New_York",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
    }
    
    # Create the executor with the config dictionary
    # executor = DailyExecutor(config)
    """
    
    # Initialize components
    executor.initialize()
    
    # Run once immediately without scheduling
    # executor.run_once()
    
    # Or run scheduled at the specified time
    executor.run_scheduled() 