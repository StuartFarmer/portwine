import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from portwine.backtester.core import NewBacktester, _split_tickers
from portwine.strategies.base import StrategyBase
from portwine.data.interface import DataInterface, MultiDataInterface, RestrictedDataInterface
from portwine.loaders.base import MarketDataLoader
from portwine.backtester.benchmarks import InvalidBenchmarkError
from portwine.backtester.core import BenchmarkTypes
from unittest.mock import Mock
import pandas as pd


class MockMarketDataLoader(MarketDataLoader):
    """Simple mock market data loader"""
    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}
        
    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        return self.mock_data.get(ticker)


class MockAlternativeDataLoader(MarketDataLoader):
    """Simple mock alternative data loader"""
    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}
        
    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        # For alternative data, we need to map the symbol back to the full ticker name
        # The mock_data has keys like 'FRED:GDP', but we're called with 'GDP'
        for full_ticker in self.mock_data.keys():
            if ':' in full_ticker:
                prefix, symbol = full_ticker.split(':', 1)
                if symbol == ticker:
                    return self.mock_data[full_ticker]
        return None


class MockDailyMarketCalendar:
    """Test-specific DailyMarketCalendar that mimics data-driven behavior"""
    def __init__(self, calendar_name):
        self.calendar_name = calendar_name
        
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


class MockBusinessDayCalendar:
    """Test-specific calendar that matches business day frequency"""
    def __init__(self, calendar_name):
        self.calendar_name = calendar_name
        
    def schedule(self, start_date, end_date):
        """Return business days to match the test data"""
        days = pd.date_range(start_date, end_date, freq="B")
        # Set market close to match the data timestamps (00:00:00)
        closes = [pd.Timestamp(d.date()) for d in days]
        return pd.DataFrame({"market_close": closes}, index=days)
    
    def get_datetime_index(self, start_date, end_date):
        """Return datetime index for the given date range"""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2020-01-14'
        
        # Return business days
        days = pd.date_range(start_date, end_date, freq="B")
        return days.to_numpy()


class MixedDataStrategy(StrategyBase):
    """Strategy that uses both regular and alternative data"""

    def __init__(self, regular_tickers, alt_tickers):
        combined = list(regular_tickers) + list(alt_tickers)
        super().__init__(combined)
        self.regular_tickers = regular_tickers
        self.alt_tickers = alt_tickers
        self.step_calls = []

    def step(self, current_date, daily_data):
        # Record invocation
        self.step_calls.append((current_date, daily_data))
        # Determine which tickers have data
        tickers_with_data = [t for t in self.tickers if daily_data.get(t) is not None]
        n = len(tickers_with_data)
        if n == 0:
            return {}
        # Equal-weight allocation among tickers with data
        return {t: (1.0 / n if t in tickers_with_data else 0.0) for t in self.tickers}


class TestStrategy(StrategyBase):
    """Strategy for testing date filtering with alternative data"""

    def __init__(self, regular_tickers, alt_tickers):
        combined = list(regular_tickers) + list(alt_tickers)
        super().__init__(combined)
        self.step_calls = []
        self.dates_seen = []

    def step(self, current_date, daily_data):
        self.step_calls.append((current_date, daily_data))
        self.dates_seen.append(current_date)
        # Allocate equally only to regular tickers
        regular = [t for t in self.tickers if ":" not in t]
        if not regular:
            return {}
        weight = 1.0 / len(regular)
        return {t: (weight if t in regular else 0.0) for t in self.tickers}


class TestBacktesterWithAltData(unittest.TestCase):
    """Test cases for Backtester with alternative data support"""

    def setUp(self):
        # Date range for testing
        self.dates = pd.date_range(start='2020-01-01', end='2020-01-10')
        self.regular_tickers = ['AAPL', 'MSFT']
        self.alt_tickers = ['FRED:GDP', 'FRED:FEDFUNDS', 'BARCHARTINDEX:ADDA']

        # Generate mock data
        self.regular_data = self._create_price_data(self.regular_tickers)
        self.alt_data = self._create_alt_data(self.alt_tickers)

        # Create mock loaders
        self.market_loader = MockMarketDataLoader(self.regular_data)
        self.alt_loader = MockAlternativeDataLoader(self.alt_data)
        
        # Create data interfaces
        self.data_interface = MultiDataInterface({
            None: self.market_loader,
            'FRED': self.alt_loader,
            'BARCHARTINDEX': self.alt_loader
        })
        
        # Create market-only data interface
        self.market_only_data_interface = MultiDataInterface({
            None: self.market_loader
        })
        
        self.calendar = MockDailyMarketCalendar("NYSE")
        
        self.backtester = NewBacktester(
            data=self.data_interface,
            calendar=self.calendar
        )
        self.market_only_backtester = NewBacktester(
            data=self.market_only_data_interface,
            calendar=self.calendar
        )

    def _create_price_data(self, tickers):
        data = {}
        for i, t in enumerate(tickers):
            base = 100.0 * (i + 1)
            prices = [base + j for j in range(len(self.dates))]
            data[t] = pd.DataFrame({
                'open': prices,
                'high': [p + 2 for p in prices],
                'low': [p - 2 for p in prices],
                'close': prices,
                'volume': [1_000_000] * len(self.dates)
            }, index=self.dates)
        return data

    def _create_alt_data(self, tickers):
        data = {}
        for i, t in enumerate(tickers):
            base = 10.0 * (i + 1)
            vals = [base + np.random.normal(0, 0.1) for _ in range(len(self.dates))]
            data[t] = pd.DataFrame({
                'open': vals,
                'high': vals,
                'low': vals,
                'close': vals,
                'volume': [0] * len(self.dates)
            }, index=self.dates)
        return data

    def test_initialization(self):
        self.assertIs(self.backtester.data, self.data_interface)
        self.assertIs(self.backtester.calendar, self.calendar)

    def test_ticker_parsing(self):
        from portwine.backtester.core import _split_tickers
        regular, alternative = _split_tickers(
            set(self.regular_tickers + self.alt_tickers)
        )
        self.assertEqual(set(regular), set(self.regular_tickers))
        self.assertEqual(set(alternative), set(self.alt_tickers))

    def test_mixed_data_strategy(self):
        strat = MixedDataStrategy(self.regular_tickers, self.alt_tickers)
        results = self.backtester.run_backtest(
            strategy=strat
        )
        self.assertIsNotNone(results)
        # Regular tickers appear in signals
        for t in self.regular_tickers:
            self.assertIn(t, results['signals_df'].columns)
        # Alt tickers do not appear
        for t in self.alt_tickers:
            self.assertNotIn(t, results['signals_df'].columns)
        # All dates covered
        self.assertEqual(len(results['signals_df']), len(self.dates))
        # Strategy called once per date with both data types
        self.assertEqual(len(strat.step_calls), len(self.dates))
        for _, daily in strat.step_calls:
            for t in self.regular_tickers + self.alt_tickers:
                self.assertIsNotNone(daily.get(t))

    def test_alt_data_benchmark_fails(self):
        strat = MixedDataStrategy(self.regular_tickers, [])
        with self.assertRaises(InvalidBenchmarkError):
            self.backtester.run_backtest(
                strategy=strat,
                benchmark=self.alt_tickers[0]
            )

    def test_without_alt_loader(self):
        # Create a strategy with only regular tickers for the market-only backtester
        strat = MixedDataStrategy(self.regular_tickers, [])
        results = self.market_only_backtester.run_backtest(
            strategy=strat
        )
        self.assertIsNotNone(results)
        # Ensure no alt data passed to strategy
        for _, daily in strat.step_calls:
            # The strategy should only have access to regular tickers
            for ticker in strat.tickers:
                self.assertIsNotNone(daily.get(ticker))
            # Alternative tickers should not be accessible since they're not in the strategy's universe
            for alt in self.alt_tickers:
                self.assertIsNone(daily.get(alt))


class TestAltDataDateFiltering(unittest.TestCase):
    """Test proper date filtering when alternative data is present"""

    def setUp(self):
        # Market dates: business days
        self.market_dates = pd.date_range(start='2020-01-01', end='2020-01-14', freq='B')
        # Alternative dates: monthly and weekly series
        self.alt_monthly = pd.date_range(start='2019-12-15', end='2020-02-15', freq='MS')
        self.alt_weekly = pd.date_range(start='2019-12-15', end='2020-02-15', freq='W')

        # Build mock datasets
        self.market_data = {
            'AAPL': self._make_price(self.market_dates, 100),
            'MSFT': self._make_price(self.market_dates, 200),
            'SPY':  self._make_price(self.market_dates, 300)
        }
        self.alt_data = {
            'ECON:GDP':    self._make_alt(self.alt_monthly, 1000),
            'ECON:CPI':    self._make_alt(self.alt_monthly, 200),
            'ECON:RATES':  self._make_alt(self.alt_weekly,   3)
        }

        self.market_loader = MockMarketDataLoader(self.market_data)
        self.alt_loader = MockAlternativeDataLoader(self.alt_data)
        
        # Create data interface
        self.data_interface = MultiDataInterface({
            None: self.market_loader,
            'ECON': self.alt_loader
        })
        self.calendar = MockBusinessDayCalendar("NYSE")
        
        self.backtester = NewBacktester(
            data=self.data_interface,
            calendar=self.calendar
        )

    def _make_price(self, dates, base):
        vals = [base + i for i in range(len(dates))]
        return pd.DataFrame({
            'open':  vals,
            'high':  [v + 2 for v in vals],
            'low':   [v - 2 for v in vals],
            'close': vals,
            'volume':[1_000_000] * len(dates)
        }, index=dates)

    def _make_alt(self, dates, base):
        vals = [base + (i * 0.1) for i in range(len(dates))]
        return pd.DataFrame({
            'open':  vals,
            'high':  vals,
            'low':   vals,
            'close': vals,
            'volume':[0] * len(dates)
        }, index=dates)

    def test_dates_from_market_data_only(self):
        strat = TestStrategy(['AAPL', 'MSFT'], list(self.alt_data.keys()))
        results = self.backtester.run_backtest(
            strategy=strat
        )
        # Only market dates should appear
        idx = results['signals_df'].index
        self.assertEqual(len(idx), len(self.market_dates))
        pd.testing.assert_index_equal(idx, self.market_dates, check_names=False)
        # Strategy called exactly once per market date
        self.assertEqual(len(strat.step_calls), len(self.market_dates))
        self.assertEqual(strat.dates_seen, list(self.market_dates))

    def test_alternative_data_does_not_affect_trading_calendar(self):
        strat = TestStrategy(['AAPL', 'MSFT'], ['ECON:GDP'])
        results = self.backtester.run_backtest(
            strategy=strat
        )
        idx = results['signals_df'].index
        self.assertEqual(len(idx), len(self.market_dates))
        pd.testing.assert_index_equal(idx, self.market_dates, check_names=False)

    def test_no_market_data(self):
        strat = TestStrategy([], ['ECON:GDP', 'ECON:CPI'])
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(strategy=strat)

    def test_with_date_filtering(self):
        strat = TestStrategy(['AAPL'], ['ECON:GDP'])
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-06',
            end_date='2020-01-10'
        )
        # Should only include the specified date range
        expected_dates = pd.date_range('2020-01-06', '2020-01-10', freq='B')
        self.assertEqual(len(results['signals_df']), len(expected_dates))


class TestOutputsExcludeAltTickers(unittest.TestCase):
    """Test that alternative tickers are excluded from output DataFrames"""

    def setUp(self):
        # Build 5 days of simple price data for 'A'
        self.dates = pd.date_range('2020-01-01', periods=5)
        self.data = {
            'A': pd.DataFrame({
                'open': [100] * 5,
                'high': [105] * 5,
                'low': [95] * 5,
                'close': [102] * 5,
                'volume': [1000000] * 5
            }, index=self.dates),
            'ALT:B': pd.DataFrame({
                'open': [10] * 5,
                'high': [10] * 5,
                'low': [10] * 5,
                'close': [10] * 5,
                'volume': [0] * 5
            }, index=self.dates)
        }
        
        self.market_loader = MockMarketDataLoader({'A': self.data['A']})
        self.alt_loader = MockAlternativeDataLoader({'ALT:B': self.data['ALT:B']})
        self.data_interface = MultiDataInterface({
            None: self.market_loader,
            'ALT': self.alt_loader
        })
        self.calendar = MockDailyMarketCalendar("NYSE")
        self.backtester = NewBacktester(
            data=self.data_interface,
            calendar=self.calendar
        )

    def test_alt_tickers_not_in_outputs(self):
        strat = MixedDataStrategy(['A'], ['ALT:B'])
        results = self.backtester.run_backtest(strategy=strat)
        
        # Only regular tickers should appear in output DataFrames
        self.assertIn('A', results['signals_df'].columns)
        self.assertNotIn('ALT:B', results['signals_df'].columns)
        
        self.assertIn('A', results['tickers_returns'].columns)
        self.assertNotIn('ALT:B', results['tickers_returns'].columns)


if __name__ == '__main__':
    unittest.main()
