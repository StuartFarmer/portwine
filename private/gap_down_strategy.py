#!/usr/bin/env python3
"""
Example script demonstrating a warmup run for the VolumeConstrainedGapDownStrategy using PolygonMarketDataLoader.
Runs daily 1 minute after open and 5 minutes before close on NYSE, with a one-week warmup period.
"""

import os
from datetime import datetime, time
import pandas as pd
from portwine.scheduler import daily_schedule
from portwine.execution import ExecutionBase
from portwine.loaders.broker import BrokerDataLoader
from portwine.brokers.alpaca import AlpacaBroker
from portwine.loaders.polygon import PolygonMarketDataLoader
from portwine.strategies.base import StrategyBase
import pandas_market_calendars as mcal

# Optional: set your API credentials directly, or rely on environment variables
API_KEY = os.getenv("POLYGON_API_KEY") or "444iQQIqlLbXm4sxdnoclHBfEFpvppIP"
API_SECRET = os.getenv("ALPACA_API_SECRET") or "zQ9WQXe4IcVh4VYYhO1ZkpdHG6mfiUUg6caEmGy6"
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or "PK7K7VQMS2290SIMDLK2"
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


class VolumeConstrainedGapDownStrategy(StrategyBase):
    """
    On each bar:
      - Morning session (before 12:00 pm): for each ticker with a negative overnight gap
        (yesterday's close → today's open),
          • If ACCOUNT:PORTFOLIO_VALUE is available, compute capacity = volume_pct * (prev_volume * prev_close)
            and set weight = capacity / portfolio_value.
          • If ACCOUNT:PORTFOLIO_VALUE is missing, fall back to equal-weight sizing among the gap-down tickers.
      - Afternoon session (on or after 12:00 pm): exit all positions.
    """
    def __init__(self, tickers: list[str], volume_pct: float = 0.01):
        super().__init__(tickers)
        self.volume_pct = volume_pct
        self.prev_close = {t: None for t in tickers}
        self.prev_volume = {t: None for t in tickers}
        print(f"\nInitialized VolumeConstrainedGapDownStrategy with:")
        print(f"  Tickers: {tickers}")
        print(f"  Volume percentage: {volume_pct:.2%}")

    def step(self, current_date: pd.Timestamp, bar_data: dict[str, pd.Series]) -> dict[str, float]:
        print(f"\n{'='*80}")
        print(f"Strategy step at {current_date}")
        print(f"Received bar data for {len(bar_data)} symbols")
        
        signals: dict[str, float] = {}
        acct = bar_data.get("ACCOUNT:PORTFOLIO_VALUE")
        port_value = acct["close"] if acct is not None else None
        print(f"Portfolio value from account data: {port_value if port_value is not None else 'Not available'}")

        if current_date.time() < time(12, 0):
            print(f'\nMorning session at {current_date}')
            losers: list[str] = []
            for t in self.tickers:
                bd = bar_data.get(t)
                prev_c = self.prev_close[t]
                prev_v = self.prev_volume[t]
                
                print(f"\nAnalyzing {t}:")
                print(f"  Previous close: {prev_c}")
                print(f"  Previous volume: {prev_v}")
                if bd is not None:
                    print(f"  Current bar data:")
                    print(f"    Open: {bd.get('open')}")
                    print(f"    Close: {bd.get('close')}")
                    print(f"    Volume: {bd.get('volume')}")
                    print(f"    High: {bd.get('high')}")
                    print(f"    Low: {bd.get('low')}")
                else:
                    print("  No current bar data available")
                
                if bd is None or prev_c is None or prev_v is None:
                    print(f"  {t} returned no data or not enough historical info.")
                    continue

                gap = (bd["open"] - prev_c) / prev_c
                print(f"  Gap calculation:")
                print(f"    Today's open: {bd['open']}")
                print(f"    Previous close: {prev_c}")
                print(f"    Gap: {gap:.2%}")
                if gap < 0:
                    losers.append(t)
                    print(f"  Added to losers list (negative gap)")

            print(f"\nFound {len(losers)} stocks with negative gaps: {losers}")

            if not losers:
                print("No gap-downs: setting all positions to flat")
                for t in self.tickers:
                    signals[t] = 0.0
            else:
                if port_value is None:
                    print(f"No portfolio value available, using equal weights (1/{len(losers)})")
                    w = 1.0 / len(losers)
                    for t in self.tickers:
                        signals[t] = w if t in losers else 0.0
                        print(f"  {t}: weight = {signals[t]:.4f}")
                else:
                    print(f"Portfolio value: {port_value:,.2f}")
                    for t in self.tickers:
                        if t in losers:
                            cap = self.volume_pct * (self.prev_volume[t] * self.prev_close[t])
                            signals[t] = cap / port_value
                            print(f"  {t} calculations:")
                            print(f"    Previous volume: {self.prev_volume[t]:,.0f}")
                            print(f"    Previous close: ${self.prev_close[t]:,.2f}")
                            print(f"    Volume percentage: {self.volume_pct:.2%}")
                            print(f"    Capacity: ${cap:,.2f}")
                            print(f"    Final weight: {signals[t]:.4f}")
                        else:
                            signals[t] = 0.0
                            print(f"  {t}: weight = 0.0 (not in losers list)")

        else:
            print(f'\nAfternoon session at {current_date}')
            for t in self.tickers:
                signals[t] = 0.0
                print(f"  {t}: weight = 0.0 (afternoon exit)")
            
            print("\nUpdating previous day's data for next day's gap calculation:")
            for t in self.tickers:
                bd = bar_data.get(t)
                if bd is not None:
                    self.prev_close[t] = bd["close"]
                    self.prev_volume[t] = bd["volume"]
                    print(f"  {t}:")
                    print(f"    Updated close: ${bd['close']:,.2f}")
                    print(f"    Updated volume: {bd['volume']:,.0f}")
                else:
                    print(f"  {t}: No data available for update")

        print(f"\nFinal signals for all tickers:")
        for t, s in signals.items():
            print(f"  {t}: {s:.4f}")
        print(f"{'='*80}\n")

        return signals


def fetch_and_validate_data(loader: PolygonMarketDataLoader, tickers: list[str], warmup_start_date: str) -> list[str]:
    """
    Fetches historical data for each ticker and returns a list of tickers that have data.
    Ensures we have enough historical data for gap calculations.
    """
    valid_tickers = []
    # Get data from one day before warmup start to ensure we have data for gap calculation
    end_date = pd.Timestamp.now(tz=loader.timezone)
    start_date = (pd.Timestamp(warmup_start_date, tz=loader.timezone) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\nFetching data from {start_date} to {end_date} to cover warmup period plus one day prior\n")
    
    for ticker in tickers:
        df = loader.fetch_historical_data(ticker, from_date=start_date)
        if df is not None and not df.empty and len(df) >= 2:  # Need at least 2 days of data
            valid_tickers.append(ticker)
            print(f"Ticker {ticker} has sufficient data ({len(df)} days).")
        else:
            print(f"Ticker {ticker} has insufficient data.")
    return valid_tickers


def main():
    timezone = "America/New_York"

    # Alpaca broker setup
    broker = AlpacaBroker(api_key=ALPACA_API_KEY, api_secret=API_SECRET, base_url=BASE_URL)

    # Polygon market data loader
    loader = PolygonMarketDataLoader(
        api_key=API_KEY,
        data_dir="data",
        timezone=timezone
    )

    # Alternative loader (optional) for broker account data
    broker_loader = BrokerDataLoader(broker=broker)

    # Create the strategy with desired tickers
    tickers = [
        "NCLTY", "RCRUY", "TWFG", "HSHCY",
        "SOBKY", "SMCAY", "NLOP", "DSFIY",
        "KDDIY", "FSUN", "SMNNY", "MITEY", 
        "SSMXY", "OTSKY", "CLPHY", "SVNDY",
        "FANUY", "PNGAY", "STBFY", "AVBP",
        "ABLLL"
    ]

    # Get calendar for trading days
    calendar = mcal.get_calendar("NYSE")
    now = pd.Timestamp.now(tz=timezone)
    
    # Get the previous trading day for warmup end
    schedule = calendar.schedule(
        start_date=(now - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d")
    )
    if schedule.empty:
        print("No trading days found in the last week. Exiting.")
        return
        
    # Get the last trading day before today
    last_trading_day = schedule.index[-2] if len(schedule) > 1 else schedule.index[0]
    warmup_end_date = last_trading_day.strftime("%Y-%m-%d")
    warmup_start_date = (pd.Timestamp(warmup_end_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"\nWarmup period will run from {warmup_start_date} to {warmup_end_date}")

    # Fetch and validate data for each ticker
    valid_tickers = fetch_and_validate_data(loader, tickers, warmup_start_date)
    if not valid_tickers:
        print("No valid tickers found. Exiting.")
        return

    strategy = VolumeConstrainedGapDownStrategy(tickers=valid_tickers)

    # Create the execution engine
    executor = ExecutionBase(
        strategy=strategy,
        market_data_loader=loader,
        broker=broker,
        alternative_data_loader=broker_loader
    )

    # Custom warmup loop

    AFTER_OPEN_MINUTES = 1
    BEFORE_CLOSE_MINUTES = 5

    print("\nRunning custom warmup period...")
    warmup_schedule = daily_schedule(
        after_open_minutes=AFTER_OPEN_MINUTES,
        before_close_minutes=BEFORE_CLOSE_MINUTES,
        calendar_name="NYSE",
        start_date=warmup_start_date,
        end_date=warmup_end_date
    )
    
    # Manually step through warmup period
    for event_time in warmup_schedule:
        print(f"\nProcessing warmup event at {event_time}")
        # Convert Unix timestamp to pandas Timestamp
        event_timestamp = pd.Timestamp(event_time, unit='ms', tz=timezone)
        # Fetch market data for this timestamp using next()
        bar_data = loader.next(valid_tickers, event_timestamp)
        
        # Run strategy step
        signals = strategy.step(event_timestamp, bar_data)
        print(f"Warmup signals at {event_timestamp}: {signals}")

    # Then run the live schedule
    print("\nStarting live trading...")
    live_schedule = daily_schedule(
        after_open_minutes=AFTER_OPEN_MINUTES,
        before_close_minutes=BEFORE_CLOSE_MINUTES,
        calendar_name="NYSE"
    )
    executor.run(live_schedule)

    print("Scheduled execution finished.")


if __name__ == "__main__":
    main()