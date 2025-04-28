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


def prepend(item, iterator):
    yield item
    yield from iterator


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
    import pandas_market_calendars as mcal
    now = pd.Timestamp.now(tz=timezone)
    schedule = daily_schedule(
        after_open_minutes=after_open_minutes,
        before_close_minutes=before_close_minutes,
        interval_seconds=interval_seconds,
        calendar_name=calendar_name,
        start_date=start_date
    )
    calendar = mcal.get_calendar(calendar_name)
    steps = 0
    last_data = {t: None for t in tickers}
    try:
        ts = next(schedule)
        while True:
            # Convert ts to a timezone-aware datetime using the provided timezone
            dt_aware = pd.to_datetime(ts, unit='ms', utc=True).tz_convert(timezone)
            now = pd.Timestamp.now(tz=timezone)
            if dt_aware >= now:
                break  # Do not advance the generator, leave ts at the next interval
            # Check if valid trading datetime
            sched = calendar.schedule(start_date=dt_aware.strftime("%Y-%m-%d"), end_date=dt_aware.strftime("%Y-%m-%d"))
            if sched.empty:
                ts = next(schedule)
                continue
            open_time = sched.iloc[0]["market_open"].tz_convert(timezone)
            close_time = sched.iloc[0]["market_close"].tz_convert(timezone)
            if not (open_time <= dt_aware <= close_time):
                ts = next(schedule)
                continue
            # Fetch data with ffill
            daily_data = loader.next(tickers, dt_aware, ffill=True)
            # Forward-fill missing values
            for t in tickers:
                if daily_data[t] is None and last_data[t] is not None:
                    daily_data[t] = last_data[t]
                elif daily_data[t] is not None:
                    last_data[t] = daily_data[t]
            strategy.step(dt_aware, daily_data)
            print(f"Warmup step at {dt_aware}: {strategy.current_signals}")
            steps += 1
            if steps % 100 == 0:
                print(f"Warm-up progress: {steps} steps...")
            ts = next(schedule)
    except StopIteration:
        print(f"Warm-up complete after {steps} steps (schedule exhausted).")
        return schedule
    print(f"Warm-up complete after {steps} steps (reached now).")
    return prepend(ts, schedule)


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

    # --- WARM-UP PERIOD using schedule iterator ---
    import pandas as pd
    start_lookup = (pd.Timestamp.now(tz=timezone) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    schedule = warmup_strategy_with_iterator(
        strategy,
        loader,
        tickers,
        calendar_name="NYSE",
        start_date=start_lookup,
        after_open_minutes=0,
        before_close_minutes=0,
        interval_seconds=120,
        timezone=timezone
    )
    # --- END WARM-UP ---

    # Print the next timestamp in the schedule after warmup
    try:
        next_ts = next(schedule)
        print(f"Next timestamp in schedule after warmup: {pd.to_datetime(next_ts, unit='ms', utc=True).tz_convert(timezone)}")
        # Print the next 5 timestamps after warmup
        print("Next 5 timestamps after warmup:")
        print(pd.to_datetime(next_ts, unit='ms', utc=True).tz_convert(timezone))
        for _ in range(4):
            ts = next(schedule)
            print(pd.to_datetime(ts, unit='ms', utc=True).tz_convert(timezone))
    except StopIteration:
        print("Schedule exhausted after warmup.")
        return

    # Rewind the generator by one since we advanced it for printing
    schedule = prepend(next_ts, schedule)

    executor.run(schedule)
    print("Scheduled execution finished.")


if __name__ == "__main__":
    main()
