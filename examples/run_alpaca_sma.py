#!/usr/bin/env python3
"""
Example script to run SimpleMovingAverageStrategy with AlpacaBroker and AlpacaMarketDataLoader.
"""

import os
from datetime import datetime
from portwine.scheduler import daily_schedule
import pandas as pd
import pandas_market_calendars as mcal
import itertools

from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy
from portwine.brokers.alpaca import AlpacaBroker
from portwine.loaders.alpaca import AlpacaMarketDataLoader
from portwine.execution import ExecutionBase
from portwine.loaders.polygon import PolygonMarketDataLoader


def warmup_strategy(
    strategy,
    loader,
    tickers,
    calendar_name: str,
    warmup_days: int,
    buffer_factor: float = 2.5,
    timezone: str = "America/New_York"
):
    calendar = mcal.get_calendar(calendar_name)
    today = pd.Timestamp.now(tz=timezone).normalize()
    lookback_days = int(warmup_days * buffer_factor)
    start_lookup = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    schedule_df = calendar.schedule(start_date=start_lookup, end_date=today.strftime("%Y-%m-%d"))
    trading_days = schedule_df.index.tz_localize(None)

    if len(trading_days) < warmup_days + 1:
        raise ValueError("Still not enough trading days after large lookback.")

    warmup_start_date = trading_days[-(warmup_days+1)].strftime("%Y-%m-%d")
    yesterday = trading_days[-2].strftime("%Y-%m-%d")

    # Fetch and cache historical data for each ticker for the warm-up period
    for ticker in tickers:
        print(f"Fetching historical data for {ticker} from {warmup_start_date} to {yesterday}...")
        loader.fetch_historical_data(ticker, warmup_start_date, yesterday)

    warmup_schedule = daily_schedule(
        after_open_minutes=0,
        before_close_minutes=0,
        calendar_name=calendar_name,
        start_date=warmup_start_date,
        end_date=yesterday
    )

    for ts in warmup_schedule:
        dt = pd.to_datetime(ts, unit='ms').tz_localize("UTC").tz_convert(timezone).normalize()
        dt = dt.tz_localize(None)
        daily_data = loader.next(tickers, dt)
        strategy.step(dt, daily_data)
        print(f"Warmup step at {dt}: {strategy.current_signals}")


def warmup_strategy_with_iterator(
    strategy,
    loader,
    tickers,
    calendar_name: str,
    start_date: str,
    after_open_minutes: int = 0,
    before_close_minutes: int = 0,
    interval_seconds: int = None,
    timezone: str = "America/New_York"
):
    import pandas as pd
    now = pd.Timestamp.now(tz=timezone)
    schedule = daily_schedule(
        after_open_minutes=after_open_minutes,
        before_close_minutes=before_close_minutes,
        interval_seconds=interval_seconds,
        calendar_name=calendar_name,
        start_date=start_date
    )
    steps = 0
    try:
        ts = next(schedule)
        dt = pd.to_datetime(ts, unit='ms').tz_localize("UTC").tz_convert(timezone).normalize()
        while dt < now:
            dt_naive = dt.tz_localize(None)
            daily_data = loader.next(tickers, dt_naive)
            strategy.step(dt_naive, daily_data)
            print(f"Warmup step at {dt_naive}: {strategy.current_signals}")
            steps += 1
            if steps % 100 == 0:
                print(f"Warm-up progress: {steps} steps...")
            ts = next(schedule)
            dt = pd.to_datetime(ts, unit='ms').tz_localize("UTC").tz_convert(timezone).normalize()
    except StopIteration:
        print(f"Warm-up complete after {steps} steps (schedule exhausted).")
        return schedule
    print(f"Warm-up complete after {steps} steps (reached now).")
    return schedule


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

    # Initialize market data loader (Polygon)
    market_data_loader = PolygonMarketDataLoader(
        api_key=os.getenv("POLYGON_API_KEY"),
        data_dir="data"
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

    # --- WARM-UP PERIOD using schedule iterator ---
    import pandas as pd
    start_lookup = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    schedule = warmup_strategy_with_iterator(
        strategy,
        market_data_loader,
        tickers,
        calendar_name="NYSE",
        start_date=start_lookup,
        after_open_minutes=0,
        before_close_minutes=0,
        interval_seconds=60,
        timezone="America/New_York"
    )
    # --- END WARM-UP ---

    executor.run(schedule)
    print("Scheduled execution finished.")


if __name__ == "__main__":
    main()
