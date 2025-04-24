#!/usr/bin/env python3
"""
Example script to run SimpleMovingAverageStrategy with AlpacaBroker and AlpacaMarketDataLoader.
"""

import os
from datetime import datetime
from portwine.scheduler import daily_schedule

from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy
from portwine.brokers.alpaca import AlpacaBroker
from portwine.loaders.alpaca import AlpacaMarketDataLoader
from portwine.execution import ExecutionBase


def main():
    # Set up tickers and strategy parameters
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    short_window = 20
    long_window = 50
    position_size = 0.1

    # Initialize the strategy
    strategy = SimpleMovingAverageStrategy(
        tickers=tickers,
        short_window=short_window,
        long_window=long_window,
        position_size=position_size
    )

    # Initialize market data loader
    market_data_loader = AlpacaMarketDataLoader(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
        start_date="2020-01-01",
        cache_dir="cache",
        paper_trading=True
    )

    # Initialize the broker
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    broker = AlpacaBroker(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
        base_url=base_url
    )

    # Create execution engine
    executor = ExecutionBase(strategy, market_data_loader, broker)

    # Schedule execution at market close for today on NYSE
    print("Scheduling execution at market close for today...")
    schedule = daily_schedule(after_open_minutes=0, before_close_minutes=0, interval_seconds=60, calendar_name="NYSE")
    executor.run(schedule)
    print("Scheduled execution finished.")


if __name__ == "__main__":
    main()
