#!/usr/bin/env python3
"""
Example script demonstrating how to use the daily executor to run a momentum strategy.
"""
import os
import logging
from portwine.utils.daily_executor import DailyExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MomentumStrategy")

if __name__ == "__main__":
    # Load from config file
    executor = DailyExecutor.from_config_file("example_config.json")
    
    # Or set up components explicitly
    # from portwine.examples.execution_example import SimpleMomentumStrategy
    # from portwine.execution_alpaca import AlpacaExecution
    # from portwine.loaders.alpaca import AlpacaMarketDataLoader
    #
    # executor = DailyExecutor(
    #     strategy_class="portwine.examples.execution_example.SimpleMomentumStrategy",
    #     strategy_params={"lookback_days": 10, "top_n": 2},
    #     tickers=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
    #     execution_class="portwine.execution_alpaca.AlpacaExecution",
    #     data_loader_class="portwine.loaders.alpaca.AlpacaMarketDataLoader",
    #     data_loader_params={"cache_dir": "./data/alpaca_cache"},
    #     paper_trading=True,
    #     run_time="15:45",
    #     timezone="America/New_York"
    # )
    
    # Run once immediately without scheduling
    # executor.run_once()
    
    # Or run scheduled at the specified time
    executor.run_scheduled() 