#!/usr/bin/env python3
"""
Example script to run SimpleMovingAverageStrategy with AlpacaBroker and AlpacaMarketDataLoader, testing executor warmup.
"""

import os
from datetime import datetime
from portwine.scheduler import daily_schedule
import pandas as pd
import pandas_market_calendars as mcal

from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy
from portwine.brokers.alpaca import AlpacaBroker
from portwine.loaders.alpaca import AlpacaMarketDataLoader
from portwine.execution import ExecutionBase
from portwine.loaders.polygon import PolygonMarketDataLoader

def main():
    # Set timezone for the entire script
    timezone = "America/New_York"

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

    # Initialize market data loader (Polygon) with script timezone
    loader = PolygonMarketDataLoader(
        api_key=os.getenv("POLYGON_API_KEY"),
        data_dir="data",
        timezone=timezone
    )

    # Initialize the broker
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    broker = AlpacaBroker(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
        base_url=base_url
    )

    # Create execution engine
    executor = ExecutionBase(strategy, loader, broker)

    # --- MAIN SCHEDULE ---
    start_lookup = (pd.Timestamp.now(tz=timezone) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    schedule = daily_schedule(
        after_open_minutes=0,
        before_close_minutes=0,
        calendar_name="NYSE",
        interval_seconds=60*60
    )

    # Run with warmup_start_date to test executor warmup
    executor.run(schedule, warmup_start_date=start_lookup)
    print("Scheduled execution finished.")

if __name__ == "__main__":
    main() 