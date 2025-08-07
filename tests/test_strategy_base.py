import unittest
import pandas as pd
from datetime import datetime
from portwine.strategies.base import StrategyBase
from portwine.backtester.core import NewBacktester
from portwine.data.interface import DataInterface
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
    """Mock DataInterface for testing"""
    def __init__(self, mock_data=None):
        self.mock_data = mock_data or {}
        self.current_timestamp = None
        
        # Create a proper mock loader instead of using Mock()
        class MockLoader:
            def __init__(self, mock_data):
                self.mock_data = mock_data
                
            def next(self, tickers, ts):
                """Mock next method that returns data for the given timestamp."""
                result = {}
                for ticker in tickers:
                    if ticker in self.mock_data:
                        data = self.mock_data[ticker]
                        # Find the data at or before the given timestamp
                        if not data.empty:
                            # Find the closest date at or before ts
                            mask = data.index <= ts
                            if mask.any():
                                latest_idx = data.index[mask].max()
                                row = data.loc[latest_idx]
                                result[ticker] = {
                                    'open': float(row['open']),
                                    'high': float(row['high']),
                                    'low': float(row['low']),
                                    'close': float(row['close']),
                                    'volume': float(row['volume'])
                                }
                            else:
                                result[ticker] = None
                        else:
                            result[ticker] = None
                    else:
                        result[ticker] = None
                return result
                
            def fetch_data(self, tickers):
                """Mock fetch_data method."""
                return {t: self.mock_data.get(t) for t in tickers if t in self.mock_data}
        
        self.data_loader = MockLoader(self.mock_data)
        
    def set_current_timestamp(self, dt):
        self.current_timestamp = dt
        
    def __getitem__(self, ticker):
        if ticker in self.mock_data:
            data = self.mock_data[ticker]
            if self.current_timestamp is not None:
                # Return data for the current timestamp
                dt_python = pd.Timestamp(self.current_timestamp)
                if hasattr(data, 'index'):
                    try:
                        idx = data.index.get_loc(dt_python)
                        return {
                            'close': data['close'].iloc[idx],
                            'open': data['open'].iloc[idx],
                            'high': data['high'].iloc[idx],
                            'low': data['low'].iloc[idx],
                            'volume': data['volume'].iloc[idx]
                        }
                    except KeyError:
                        return None
                else:
                    return data
            return data
        return None
        
    def exists(self, ticker, start_date, end_date):
        return ticker in self.mock_data

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
        bt = NewBacktester(
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
