import unittest
from datetime import time
import pandas as pd
import numpy as np
from unittest.mock import Mock

from portwine.backtester.core import Backtester
from portwine.strategies.base import StrategyBase
from portwine.data.interface import RestrictedDataInterface

class MockRestrictedDataInterface(RestrictedDataInterface):
    def __init__(self, mock_data=None):
        # Create a mock data loader
        self.mock_data_loader = Mock()
        self.mock_data = mock_data or {}
        self.set_timestamp_calls = []
        self.get_calls = []
        self.current_timestamp = None
        
        # Initialize parent with mock loaders
        super().__init__({None: self.mock_data_loader})
        
        # Add data_loader attribute for compatibility
        self.data_loader = self.mock_data_loader
        
        # Configure the mock data loader to return proper data
        def mock_next(tickers, timestamp):
            result = {}
            for ticker in tickers:
                if ticker in self.mock_data:
                    data = self.mock_data[ticker]
                    if self.current_timestamp is not None:
                        dt_python = pd.Timestamp(self.current_timestamp)
                        # Use the exact timestamps that match the data
                        dates = pd.to_datetime([
                            '2025-04-14 09:30', '2025-04-14 16:00',
                            '2025-04-15 09:30', '2025-04-15 16:00',
                        ])
                        try:
                            idx = dates.get_loc(dt_python)
                            result[ticker] = {
                                'close': float(data['close'][idx]),
                                'open': float(data['open'][idx]),
                                'high': float(data['high'][idx]),
                                'low': float(data['low'][idx]),
                                'volume': float(data['volume'][idx])
                            }
                        except (KeyError, IndexError):
                            # Fallback to first value if index not found
                            result[ticker] = {
                                'close': float(data['close'][0]),
                                'open': float(data['open'][0]),
                                'high': float(data['high'][0]),
                                'low': float(data['low'][0]),
                                'volume': float(data['volume'][0])
                            }
                    else:
                        # Fallback to first value if no timestamp set
                        result[ticker] = {
                            'close': float(data['close'][0]),
                            'open': float(data['open'][0]),
                            'high': float(data['high'][0]),
                            'low': float(data['low'][0]),
                            'volume': float(data['volume'][0])
                        }
                else:
                    result[ticker] = {
                        'close': 100.0,
                        'open': 100.0,
                        'high': 105.0,
                        'low': 95.0,
                        'volume': 1000000
                    }
            return result
        
        self.mock_data_loader.next = mock_next

    def set_current_timestamp(self, dt):
        self.set_timestamp_calls.append(dt)
        self.current_timestamp = dt
        super().set_current_timestamp(dt)

    def set_restricted_tickers(self, tickers, prefix=None):
        self.restricted_tickers = tickers
        super().set_restricted_tickers(tickers, prefix)

    def __getitem__(self, ticker):
        self.get_calls.append(ticker)
        if ticker in self.mock_data:
            data = self.mock_data[ticker]
            if self.current_timestamp is not None:
                dt_python = pd.Timestamp(self.current_timestamp)
                # Use the exact timestamps that match the data
                dates = pd.to_datetime([
                    '2025-04-14 09:30', '2025-04-14 16:00',
                    '2025-04-15 09:30', '2025-04-15 16:00',
                ])
                try:
                    idx = dates.get_loc(dt_python)
                    return {
                        'close': data['close'][idx],
                        'open': data['open'][idx],
                        'high': data['high'][idx],
                        'low': data['low'][idx],
                        'volume': data['volume'][idx]
                    }
                except KeyError:
                    return None
            return data
        return None

    def exists(self, ticker, start_date, end_date):
        return ticker in self.mock_data

class MockDailyMarketCalendar:
    """Test-specific DailyMarketCalendar for intraday tests"""
    def __init__(self, calendar_name):
        self.calendar_name = calendar_name
        
    def get_datetime_index(self, start_date, end_date):
        """Return datetime index for the given date range"""
        if start_date is None:
            start_date = '2025-04-14'
        if end_date is None:
            end_date = '2025-04-15'
        
        # Return intraday timestamps
        all_dates = pd.to_datetime([
            '2025-04-14 09:30', '2025-04-14 16:00',
            '2025-04-15 09:30', '2025-04-15 16:00',
        ])
        
        # For the date filtering test, return only the exact date requested
        if start_date == end_date:
            start_dt = pd.Timestamp(start_date)
            filtered_dates = all_dates[all_dates == start_dt]
        else:
            # For other tests, return all dates in the range
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            # Use inclusive range - for end_date, include the entire day
            if end_dt.hour == 0 and end_dt.minute == 0:
                # If end_date has no time component, include the entire day
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            filtered_dates = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]
        
        return filtered_dates.to_numpy()


class OvernightIntradayStrategy(StrategyBase):
    """
    Goes long only on the 16:00 bar; flat at all other times.
    """
    def __init__(self, tickers):
        super().__init__(tickers)

    def step(self, current_date, bar_data):
        if current_date.time() == time(16, 0):
            return {t: 1.0 for t in self.tickers}
        return {t: 0.0 for t in self.tickers}


class IntradayOvernightStrategy(StrategyBase):
    """
    Goes long only on the 09:30 bar; flat at all other times.
    """
    def __init__(self, tickers):
        super().__init__(tickers)

    def step(self, current_date, bar_data):
        if current_date.time() == time(9, 30):
            return {t: 1.0 for t in self.tickers}
        return {t: 0.0 for t in self.tickers}


class MockIntradayLoader:
    """
    Synthetic intraday OHLCV for ticker 'TEST':
      2025-04-14 09:30, 16:00
      2025-04-15 09:30, 16:00
    """
    def __init__(self):
        dates = pd.to_datetime([
            '2025-04-14 09:30', '2025-04-14 16:00',
            '2025-04-15 09:30', '2025-04-15 16:00',
        ])
        df = pd.DataFrame({
            'open':   [1, 1, 1, 1],
            'high':   [1, 1, 1, 1],
            'low':    [1, 1, 1, 1],
            'close':  [1, 1, 1, 1],
            'volume': [100, 100, 100, 100],
        }, index=dates)
        self.data = {'TEST': df}

    def fetch_data(self, tickers):
        return {t: self.data[t] for t in tickers}


class CustomIntradayLoader:
    """
    Synthetic intraday OHLCV for ticker 'TEST' with varying close prices:
      2025‑04‑14 09:30 -> 100
      2025‑04‑14 16:00 -> 104
      2025‑04‑15 09:30 -> 108
      2025‑04‑15 16:00 -> 102
    """
    def __init__(self):
        dates = pd.to_datetime([
            '2025-04-14 09:30', '2025-04-14 16:00',
            '2025-04-15 09:30', '2025-04-15 16:00',
        ])
        self.df = pd.DataFrame({
            'open':   [100,   104,   108,   102],
            'high':   [101,   105,   109,   103],
            'low':    [ 99,   103,   107,   101],
            'close':  [100,   104,   108,   102],
            'volume': [1000,  1000,  1000,  1000],
        }, index=dates)

    def fetch_data(self, tickers):
        return {t: self.df for t in tickers}


class TestIntradayBacktester(unittest.TestCase):
    def setUp(self):
        # Create mock data interface
        self.data_interface = MockRestrictedDataInterface()
        
        # Convert MockIntradayLoader data to the new format
        loader = MockIntradayLoader()
        for ticker, data in loader.data.items():
            self.data_interface.mock_data[ticker] = {
                'close': data['close'].values,
                'open': data['open'].values,
                'high': data['high'].values,
                'low': data['low'].values,
                'volume': data['volume'].values
            }
        
        self.bt = Backtester(self.data_interface, calendar=MockDailyMarketCalendar("NYSE"))

    def test_overnight_intraday_signals_raw(self):
        strat = OvernightIntradayStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat, 
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        sig = res['signals_df']

        # 09:30 -> flat
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 09:30'), 'TEST'], 0.0)
        # 16:00 -> long
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 16:00'), 'TEST'], 1.0)
        # Next day same pattern
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 09:30'), 'TEST'], 0.0)
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 16:00'), 'TEST'], 1.0)

    def test_intraday_overnight_signals_raw(self):
        strat = IntradayOvernightStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat, 
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        sig = res['signals_df']

        # 09:30 -> long
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 09:30'), 'TEST'], 1.0)
        # 16:00 -> flat
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 16:00'), 'TEST'], 0.0)
        # Next day same pattern
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 09:30'), 'TEST'], 1.0)
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 16:00'), 'TEST'], 0.0)

    def test_signals_shifted(self):
        strat = OvernightIntradayStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # default shift_signals=True (handled by Backtester)
        res = self.bt.run_backtest(
            strat,
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        sig = res['signals_df']

        # First bar (2025-04-14 09:30) => raw signal = 0
        self.assertEqual(sig.iloc[0]['TEST'], 0.0)
        # Raw signal at 2025-04-14 16:00 = 1 (no shifting in signals_df)
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 16:00'), 'TEST'], 1.0)

    def test_start_end_date_filtering(self):
        strat = OvernightIntradayStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat,
            start_date='2025-04-14 16:00',
            end_date='2025-04-14 16:00',
            benchmark=equal_weight_benchmark
        )
        sig = res['signals_df']

        # The calendar should return only the filtered date
        self.assertListEqual(
            list(sig.index),
            [pd.Timestamp('2025-04-14 16:00')]
        )
        self.assertEqual(sig.iloc[0]['TEST'], 1.0)

    def test_union_ts_merges_and_sorts(self):
        strat = OvernightIntradayStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat, 
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        sig = res['signals_df']

        expected = pd.to_datetime([
            '2025-04-14 09:30',
            '2025-04-14 16:00',
            '2025-04-15 09:30',
            '2025-04-15 16:00'
        ])
        pd.testing.assert_index_equal(sig.index, expected, check_names=False)


class TestIntradayReturnCalculations(unittest.TestCase):
    def setUp(self):
        # Create mock data interface
        self.data_interface = MockRestrictedDataInterface()
        
        # Convert CustomIntradayLoader data to the new format
        loader = CustomIntradayLoader()
        for ticker in ['TEST']:  # CustomIntradayLoader returns same data for all tickers
            self.data_interface.mock_data[ticker] = {
                'close': loader.df['close'].values,
                'open': loader.df['open'].values,
                'high': loader.df['high'].values,
                'low': loader.df['low'].values,
                'volume': loader.df['volume'].values
            }
        
        # Create a custom backtester that uses the mock data interface directly
        class CustomBacktester(Backtester):
            def __init__(self, data_interface, calendar):
                self.data = data_interface
                self.restricted_data = data_interface  # Use the mock interface directly
                self.calendar = calendar
        
        self.bt = CustomBacktester(self.data_interface, calendar=MockDailyMarketCalendar("NYSE"))

        # precompute the four percent returns:
        # first bar: 09:30 -> no prior bar -> pct_change = NaN -> filled to 0
        # 16:00: (104/100 -1) = 0.04
        # next 09:30: (108/104 -1) ≈ 0.038461538
        # next 16:00: (102/108 -1) ≈ -0.055555556
        self.expected_ret = {
            '2025-04-14 09:30': 0.0,
            '2025-04-14 16:00':  0.04,
            '2025-04-15 09:30':  (108/104) - 1,
            '2025-04-15 16:00':  (102/108) - 1,
        }

    def test_tickers_returns(self):
        """tickers_returns matches the true overnight/intraday pct_change."""
        strat = OvernightIntradayStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat, 
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        ret_df = res['tickers_returns']

        for ts_str, exp in self.expected_ret.items():
            ts = pd.Timestamp(ts_str)
            self.assertAlmostEqual(
                ret_df.loc[ts, 'TEST'],
                exp,
                places=8,
                msg=f"ret_df at {ts} should be {exp}"
            )

    def test_strategy_returns_overnight_intraday(self):
        """
        OvernightIntradayStrategy only captures intraday bars (16:00),
        so its strategy_returns at 16:00 should equal the intraday pct_change,
        and zero at the opens.
        """
        strat = OvernightIntradayStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat, 
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        sr = res['strategy_returns']



        # With signal shifting: signals from previous day are used for returns
        # 09:30 bars => no previous signal => 0
        self.assertEqual(sr[pd.Timestamp('2025-04-14 09:30')], 0.0)
        # 09:30 bars => previous signal from 16:00 => intraday pct_change
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-15 09:30')],
            self.expected_ret['2025-04-15 09:30'],
            places=8
        )

        # 16:00 bars => previous signal from 09:30 (0.0) => 0.0
        self.assertEqual(sr[pd.Timestamp('2025-04-14 16:00')], 0.0)
        # 16:00 bars => previous signal from 09:30 (0.0) => 0.0
        self.assertEqual(sr[pd.Timestamp('2025-04-15 16:00')], 0.0)

    def test_strategy_returns_intraday_overnight(self):
        """
        IntradayOvernightStrategy only captures overnight bars (09:30),
        so its strategy_returns at 09:30 should equal the overnight pct_change,
        and zero at the closes.
        """
        strat = IntradayOvernightStrategy(['TEST'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat,
            start_date='2025-04-14',
            end_date='2025-04-15',
            benchmark=equal_weight_benchmark
        )
        
        sr = res['strategy_returns']
        
        # With signal shifting: signals from previous day are used for returns
        # 09:30 bars => no previous signal => 0
        self.assertEqual(sr[pd.Timestamp('2025-04-14 09:30')], 0.0)
        # 09:30 bars => previous signal from 16:00 (0.0) => 0.0
        self.assertEqual(sr[pd.Timestamp('2025-04-15 09:30')], 0.0)

        # 16:00 bars => previous signal from 09:30 (1.0) => overnight pct_change
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-14 16:00')],
            self.expected_ret['2025-04-14 16:00'],
            places=8
        )
        # 16:00 bars => previous signal from 09:30 (1.0) => overnight pct_change
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-15 16:00')],
            self.expected_ret['2025-04-15 16:00'],
            places=8
        )


if __name__ == '__main__':
    unittest.main()
