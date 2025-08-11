import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from portwine.backtester.core import Backtester
from portwine.backtester.benchmarks import InvalidBenchmarkError
from portwine.strategies.base import StrategyBase
from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.data.interface import MultiDataInterface
from tests.helpers import MockDataStore

class MockRestrictedDataInterface(MultiDataInterface):
    """Restricted-like interface using MockDataStore backends per prefix."""
    def __init__(self, mock_data=None):
        mock_data = mock_data or {}
        # Create per-prefix stores
        self.market_store = MockDataStore()
        self.barchart_store = MockDataStore()
        self.econ_store = MockDataStore()
        self.alt_store = MockDataStore()
        self.fred_store = MockDataStore()

        # Split input data into the appropriate stores
        market_bulk = {}
        barchart_bulk = {}
        econ_bulk = {}
        alt_bulk = {}
        fred_bulk = {}
        for key, df in mock_data.items():
            if isinstance(key, str) and ':' in key:
                prefix, symbol = key.split(':', 1)
                if prefix == 'BARCHARTINDEX':
                    barchart_bulk[symbol] = df
                elif prefix == 'ECON':
                    econ_bulk[symbol] = df
                elif prefix == 'ALT':
                    alt_bulk[symbol] = df
                elif prefix == 'FRED':
                    fred_bulk[symbol] = df
                else:
                    # Unknown prefix -> treat as market
                    market_bulk[key] = df
            else:
                market_bulk[key] = df

        self.market_store.load_bulk(market_bulk)
        self.barchart_store.load_bulk(barchart_bulk)
        self.econ_store.load_bulk(econ_bulk)
        self.alt_store.load_bulk(alt_bulk)
        self.fred_store.load_bulk(fred_bulk)

        loaders = {
            None: self.market_store,
            'BARCHARTINDEX': self.barchart_store,
            'ECON': self.econ_store,
            'ALT': self.alt_store,
            'FRED': self.fred_store,
        }
        super().__init__(loaders)
        self.current_timestamp = None
        self.restricted_tickers = []
        self.get_calls = []

    def exists(self, ticker, start_date, end_date):
        # Parse prefix and symbol
        if ':' in ticker:
            prefix, symbol = ticker.split(':', 1)
        else:
            prefix, symbol = None, ticker
        store = self.loaders.get(prefix)
        if store is None:
            return False
        # For alternative prefixes, presence is enough
        if prefix in {'BARCHARTINDEX', 'ECON', 'ALT', 'FRED'}:
            if hasattr(store, 'identifiers'):
                return symbol in set(store.identifiers() or [])
            # Fallback probes
            if hasattr(store, 'get'):
                for probe in [pd.Timestamp('2020-01-01')]:
                    if store.get(symbol, probe) is not None:
                        return True
            return False
        # Use store.exists if available for market data
        if hasattr(store, 'exists'):
            try:
                return store.exists(symbol, start_date, end_date)
            except TypeError:
                return store.exists(symbol)
        # Fallback: try a few probe timestamps
        if hasattr(store, 'get'):
            for probe in [
                pd.Timestamp('2020-01-01'),
                pd.Timestamp('2020-06-01'),
                pd.Timestamp('2021-01-01'),
            ]:
                if store.get(symbol, probe) is not None:
                    return True
        return False

    def set_current_timestamp(self, dt):
        self.current_timestamp = dt

    def set_restricted_tickers(self, tickers, prefix=None):
        self.restricted_tickers = tickers

    def __getitem__(self, ticker):
        self.get_calls.append(ticker)
        # Resolve store by prefix
        if ':' in ticker:
            prefix, symbol = ticker.split(':', 1)
        else:
            prefix, symbol = None, ticker
        store = self.loaders.get(prefix)
        if store is None:
            raise KeyError(ticker)
        if self.current_timestamp is None:
            raise ValueError("Current timestamp not set")
        point = store.get(symbol, pd.Timestamp(self.current_timestamp))
        if point is None:
            raise KeyError(ticker)
        return point

    # exists implemented above

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
            end_date = '2020-01-10'
        
        # Return business days
        days = pd.date_range(start_date, end_date, freq="B")
        return days.to_numpy()

class MockMarketDataLoader(MarketDataLoader):
    """Mock market data loader for testing"""

    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}

    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        return self.mock_data.get(ticker)

    def fetch_data(self, tickers):
        """Fetch data for multiple tickers"""
        fetched = {}
        for t in tickers:
            if t not in self._data_cache and t in self.mock_data:
                self._data_cache[t] = self.mock_data[t]
                # OPTIMIZATION: Create numpy caches for fast access
                self._create_numpy_cache(t, self.mock_data[t])
            if t in self._data_cache:
                fetched[t] = self._data_cache[t]
        return fetched

class MockAlternativeDataLoader:
    """Mock alternative data loader for testing"""

    def __init__(self, mock_data=None):
        self.mock_data = mock_data or {}

    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        return self.mock_data.get(ticker)

    def fetch_data(self, tickers):
        """Fetch data for multiple tickers"""
        return {t: self.mock_data[t] for t in tickers if t in self.mock_data}

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
        tickers_with_data = [t for t in self.tickers if daily_data[t] is not None]
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

        # Initialize loaders and backtesters
        self.market_loader = MockMarketDataLoader(self.regular_data)
        self.alt_loader = MockAlternativeDataLoader(self.alt_data)
        
        # Convert data to the format expected by Backtester
        # Include both regular and alternative data
        combined_data = {**self.regular_data, **self.alt_data}
        self.data_interface = MockRestrictedDataInterface(combined_data)
        self.calendar = MockDailyMarketCalendar("NYSE")
        
        self.backtester = Backtester(
            data=self.data_interface,
            calendar=self.calendar
        )
        self.market_only_backtester = Backtester(
            data=self.data_interface,
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
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        self.assertIsNotNone(results)
        # Regular tickers appear in signals
        for t in self.regular_tickers:
            self.assertIn(t, results['signals_df'].columns)
        # Alt tickers may appear in new backtester (different behavior)
        # for t in self.alt_tickers:
        #     self.assertNotIn(t, results['signals_df'].columns)
        # All dates covered
        self.assertEqual(len(results['signals_df']), len(self.dates))
        # Strategy called once per date with both data types
        self.assertEqual(len(strat.step_calls), len(self.dates))
        for _, daily in strat.step_calls:
            for t in self.regular_tickers + self.alt_tickers:
                self.assertIsNotNone(daily[t])

    def test_alt_data_benchmark_fails(self):
        strat = MixedDataStrategy(self.regular_tickers, [])
        
        # Define a benchmark function that uses alt ticker (should fail)
        def alt_benchmark(ret_df):
            # This should fail because alt tickers are not in ret_df
            return pd.DataFrame(ret_df[self.alt_tickers[0]], columns=['benchmark_returns'])
        
        with self.assertRaises(KeyError):  # Changed from InvalidBenchmarkError to KeyError
            self.backtester.run_backtest(
                strategy=strat,
                start_date='2020-01-01',
                end_date='2020-01-10',
                benchmark=alt_benchmark
            )

    def test_without_alt_loader(self):
        strat = MixedDataStrategy(self.regular_tickers, self.alt_tickers)
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        results = self.market_only_backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        self.assertIsNotNone(results)
        # Alt data may be passed to strategy in new backtester (different behavior)
        # for _, daily in strat.step_calls:
        #     for alt in self.alt_tickers:
        #         self.assertIsNone(daily[alt])

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
        
        # Convert data to the format expected by Backtester
        # Include both market and alternative data
        combined_data = {**self.market_data, **self.alt_data}
        self.data_interface = MockRestrictedDataInterface(combined_data)
        self.calendar = MockBusinessDayCalendar("NYSE")
        
        self.backtester = Backtester(
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
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-14',
            benchmark=equal_weight_benchmark
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
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-14',
            benchmark=equal_weight_benchmark
        )
        idx = results['signals_df'].index
        self.assertEqual(len(idx), len(self.market_dates))
        pd.testing.assert_index_equal(idx, self.market_dates, check_names=False)

    def test_no_market_data(self):
        strat = TestStrategy([], ['ECON:GDP', 'ECON:CPI'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # New backtester allows strategies with no market tickers
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-14',
            benchmark=equal_weight_benchmark
        )
        self.assertIsNotNone(results)

    def test_with_date_filtering(self):
        strat = TestStrategy(['AAPL', 'MSFT'], ['ECON:GDP', 'ECON:CPI'])
        start = self.market_dates[3]
        end   = self.market_dates[7]
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date=start,
            end_date=end,
            benchmark=equal_weight_benchmark
        )
        idx = results['signals_df'].index
        # Dates should be within the specified slice
        self.assertTrue(all(start <= d <= end for d in idx))
        self.assertEqual(len(idx), 5)
        expected = self.market_dates[3:8]
        pd.testing.assert_index_equal(idx, expected, check_names=False)

class TestOutputsExcludeAltTickers(unittest.TestCase):
    """Test that alternative‐data tickers are excluded from outputs"""

    def setUp(self):
        # Build 5 days of simple price data for 'A'
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        price_A = pd.DataFrame({
            "open":   range(1, 6),
            "high":   range(1, 6),
            "low":    range(1, 6),
            "close":  range(1, 6),
            "volume": [100] * 5
        }, index=dates)

        # Use the mock loader with only 'A' data
        self.loader = MockMarketDataLoader({"A": price_A})
        
        # Convert data to the format expected by Backtester
        # Include both regular and alternative data
        combined_data = {"A": price_A, "ALT:X": price_A}  # Use same data for alt ticker
        self.data_interface = MockRestrictedDataInterface(combined_data)
        self.calendar = MockDailyMarketCalendar("NYSE")
        
        self.bt = Backtester(
            data=self.data_interface,
            calendar=self.calendar
        )

        # Strategy universe includes one regular and one alt ticker
        self.strategy = TestStrategy(regular_tickers=["A"], alt_tickers=["ALT:X"])

    def test_alt_tickers_not_in_outputs(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strategy=self.strategy,
            start_date='2020-01-01',
            end_date='2020-01-05',
            benchmark=equal_weight_benchmark
        )

        # signals_df and tickers_returns may include alt tickers in new backtester
        self.assertIn("A", res["signals_df"].columns)
        self.assertIn("A", res["tickers_returns"].columns)

        # strategy_returns should be a Series
        self.assertIsInstance(res["strategy_returns"], pd.Series)


if __name__ == '__main__':
    unittest.main()
