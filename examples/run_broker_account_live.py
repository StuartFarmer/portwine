#!/usr/bin/env python3
"""
Example script demonstrating live trading with BrokerDataLoader.
Pulls account equity via BROKER:ACCOUNT and sizes a small position in AAPL based on 1% of equity.
"""
import os
from datetime import datetime
from portwine.scheduler import daily_schedule
from portwine.execution import ExecutionBase
from portwine.loaders.broker import BrokerDataLoader
from portwine.brokers.alpaca import AlpacaBroker
from portwine.loaders.alpaca import AlpacaMarketDataLoader
from portwine.strategies.base import StrategyBase


class BrokerAccountStrategy(StrategyBase):
    """Simple strategy that uses account equity to size a AAPL position."""
    def __init__(self):
        super().__init__(tickers=["BROKER:ACCOUNT", "AAPL"] )

    def step(self, current_date, data):
        # Fetch latest account equity
        acct = data.get("BROKER:ACCOUNT")
        equity = acct["equity"] if acct is not None else 0.0

        # Get latest AAPL close price
        aapl = data.get("AAPL")
        if aapl is None or equity <= 0:
            return {}
        price = aapl.get("close", 0.0)

        # Target 1% of equity in AAPL
        target_value = 0.01 * equity
        shares = target_value / price if price > 0 else 0.0
        return {"AAPL": shares}


def main():
    # Load credentials from environment
    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")
    BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Initialize Alpaca broker and market data
    broker = AlpacaBroker(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL)
    market_loader = AlpacaMarketDataLoader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        start_date=datetime.utcnow().date().isoformat(),
        cache_dir="cache",
        paper_trading=True
    )

    # Wrap broker as alternative data loader
    broker_loader = BrokerDataLoader(broker=broker)

    # Create strategy and executor
    strategy = BrokerAccountStrategy()
    executor = ExecutionBase(
        strategy=strategy,
        market_data_loader=market_loader,
        broker=broker,
        alternative_data_loader=broker_loader
    )

    # Schedule to run 1 minute after market open every day on NYSE
    print("Scheduling daily run at 1 minute after open")
    schedule = daily_schedule(
        after_open_minutes=1,
        calendar_name="NYSE"
    )
    executor.run(schedule)


if __name__ == "__main__":
    main() 