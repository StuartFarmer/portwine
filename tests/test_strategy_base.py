import unittest
import pandas as pd
from datetime import datetime
from portwine.strategies.base import StrategyBase
from portwine.backtester.core import Backtester
from portwine.data.interface import DataInterface
from tests.helpers import MockDataStore
from unittest.mock import Mock

class MockDailyMarketCalendar:
    """Test-specific DailyMarketCalendar that mimics data-driven behavior"""
    def __init__(self, calendar_name):
        self.calendar_name = calendar_name
        # For testing, we'll use all calendar days to match original behavior
        
    def schedule(self, start_date, end_date):
        """Return all calendar days to match original data-driven behavior"""
        days = pd.date_range(start_date, end_date, freq="D")
        # Set market close to match the data timestamps (00:00:00)
        closes = [pd.Timestamp(d.date()) for d in days]
        return pd.DataFrame({"market_close": closes}, index=days)
    
    def get_datetime_index(self, start_date, end_date):
        """Return datetime index for the given date range"""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2020-01-10'
        
        # Return all calendar days
        days = pd.date_range(start_date, end_date, freq="D")
        return days.to_numpy()

# Fake market data loader for integration testing
class FakeLoader:
    def __init__(self):
        # 5 days of dummy data
        self.dates = pd.date_range('2025-01-01', '2025-01-05', freq='D')
        self.dfs = {}
        for t in ['X', 'Y']:
            # create a DataFrame with constant prices
            self.dfs[t] = pd.DataFrame({
                'open':   1.0,
                'high':   1.0,
                'low':    1.0,
                'close':  1.0,
                'volume': 100
            }, index=self.dates)
    
    def fetch_data(self, tickers):
        dfs = {}
        for t in tickers:
            # create a DataFrame with constant prices
            dfs[t] = pd.DataFrame({
                'open':   1.0,
                'high':   1.0,
                'low':    1.0,
                'close':  1.0,
                'volume': 100
            }, index=self.dates)
        return dfs

class MockDataInterface(DataInterface):
    """DataInterface backed by MockDataStore for testing"""
    def __init__(self, mock_data=None):
        store = MockDataStore()
        if mock_data:
            store.load_bulk(mock_data)
        super().__init__(store)
        self.current_timestamp = None

    def set_current_timestamp(self, dt):
        self.current_timestamp = dt

    def __getitem__(self, ticker):
        return super().__getitem__(ticker)

    def exists(self, ticker, start_date, end_date):
        return self.data_loader.exists(ticker, start_date, end_date)

class TestStrategy(StrategyBase):
    """Concrete strategy for testing."""
    def step(self, current_date, daily_data):
        # Equal weight strategy
        valid_tickers = [t for t in daily_data.keys() if daily_data.get(t) is not None]
        n = len(valid_tickers)
        weight = 1.0 / n if n > 0 else 0.0
        return {ticker: weight for ticker in valid_tickers}

class TestStrategyBase(unittest.TestCase):
    def test_dedup_tickers(self):
        # duplicates should be removed, preserving order
        s = TestStrategy(['A', 'B', 'A', 'C', 'B'])
        # Should return a list with unique tickers (order not guaranteed)
        self.assertIsInstance(s.tickers, list)
        self.assertCountEqual(s.tickers, ['A', 'B', 'C'])

class TestBacktesterIntegration(unittest.TestCase):
    def test_backtest_runs_and_respects_dedup(self):
        loader = FakeLoader()
        data_interface = MockDataInterface(loader.dfs)
        calendar = MockDailyMarketCalendar("NYSE")
        bt = Backtester(
            data=data_interface,
            calendar=calendar
        )
        # Initialize strategy with duplicate tickers
        s = TestStrategy(['X', 'X', 'Y'])
        # After init, duplicates must be removed
        self.assertIsInstance(s.tickers, list)
        self.assertCountEqual(s.tickers, ['X', 'Y'])
        # Run backtest; should not error
        res = bt.run_backtest(s)
        # Should return a dict including 'strategy_returns'
        self.assertIsInstance(res, dict)
        self.assertIn('strategy_returns', res)
        # Verify the returns series has entries for the 5 data days (1st day may be NaN if pct_change)
        sr = res['strategy_returns']
        self.assertGreaterEqual(len(sr), 4)  # at least 4 valid return entries

if __name__ == "__main__":
    unittest.main()
