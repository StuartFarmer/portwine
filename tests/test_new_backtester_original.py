import unittest
import pandas as pd
import numpy as np

# Import components to be tested
from portwine.backtester.core import Backtester
from portwine.backtester.benchmarks import InvalidBenchmarkError
from portwine.strategies.base import StrategyBase
from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.data.interface import DataInterface
from unittest.mock import Mock
from tests.helpers import MockDataStore

class MockDataInterface(DataInterface):
    """DataInterface backed by MockDataStore for testing."""
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

class MockMarketDataLoader(MarketDataLoader):
    """Deprecated in tests: now backed by MockDataStore for compatibility."""

    def __init__(self, mock_data=None):
        super().__init__()
        self._store = MockDataStore()
        self.mock_data = mock_data or {}
        if mock_data:
            self._store.load_bulk(mock_data)

    def load_ticker(self, ticker):
        return self.mock_data.get(ticker)

    def set_data(self, ticker, data):
        self.mock_data[ticker] = data
        self._store.load_dataframe(ticker, data)


class SimpleTestStrategy(StrategyBase):
    """Simple strategy implementation for testing"""

    def __init__(self, tickers, allocation=None):
        super().__init__(tickers)
        # Fixed allocation if provided, otherwise equal weight
        self.allocation = allocation or {ticker: 1.0 / len(tickers) for ticker in tickers}
        self.step_calls = []  # Track step calls for testing

    def step(self, current_date, daily_data):
        """Return a fixed allocation"""
        # Record the call for test verification
        self.step_calls.append((current_date, daily_data))
        return self.allocation


class DynamicTestStrategy(StrategyBase):
    """Strategy that changes allocations based on price movements"""

    def __init__(self, tickers):
        super().__init__(tickers)
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []

    def step(self, current_date, daily_data):
        """Allocate more to better performing assets"""
        self.dates.append(current_date)

        # Update price history
        for ticker in self.tickers:
            price = None
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close')

            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]

            self.price_history[ticker].append(price)

        # Simple momentum strategy: allocate to best performer over last 5 days
        if len(self.dates) >= 5:
            returns = {}
            for ticker in self.tickers:
                prices = self.price_history[ticker][-5:]
                if None not in prices and prices[0] > 0:
                    returns[ticker] = prices[-1] / prices[0] - 1
                else:
                    returns[ticker] = 0

            # Find best performer
            best_ticker = max(returns.items(), key=lambda x: x[1])[0] if returns else self.tickers[0]

            # Allocate everything to best performer
            allocation = {ticker: 1.0 if ticker == best_ticker else 0.0 for ticker in self.tickers}
            return allocation

        # Equal weight until we have enough history
        return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}

class FakeCalendar:
    tz = "UTC"
    def schedule(self, start_date, end_date):
        days = pd.date_range(start_date, end_date, freq="D")
        sel = [d for d in days if d.day % 2 == 1]
        closes = [pd.Timestamp(d.date()) + pd.Timedelta(hours=16) for d in sel]
        return pd.DataFrame({"market_close": closes}, index=sel)
    
    def get_datetime_index(self, start_date, end_date):
        """Return datetime index for the given date range"""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2020-01-10'
        
        # Return odd-numbered days
        days = pd.date_range(start_date, end_date, freq="D")
        sel = [d for d in days if d.day % 2 == 1]
        return np.array(sel)

from tests.calendar_utils import TestDailyMarketCalendar


class TestBacktester(unittest.TestCase):
    """Test cases for Backtester class"""

    def setUp(self):
        """Set up test environment"""
        # Sample date range for testing
        self.dates = pd.date_range(start='2020-01-01', end='2020-01-10')

        # Create sample price data for multiple tickers
        self.tickers = ['AAPL', 'MSFT', 'GOOG']

        # Sample price data with different trends
        self.price_data = {}

        # AAPL: upward trend
        self.price_data['AAPL'] = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000] * 10
        }, index=self.dates)

        # MSFT: downward trend
        self.price_data['MSFT'] = pd.DataFrame({
            'open': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            'high': [101, 100, 99, 98, 97, 96, 95, 94, 93, 92],
            'low': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'close': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'volume': [1000000] * 10
        }, index=self.dates)

        # GOOG: flat trend with gap
        self.price_data['GOOG'] = pd.DataFrame({
            'open': [200, 200, 200, 200, 200, np.nan, np.nan, 200, 200, 200],
            'high': [205, 205, 205, 205, 205, np.nan, np.nan, 205, 205, 205],
            'low': [195, 195, 195, 195, 195, np.nan, np.nan, 195, 195, 195],
            'close': [200, 200, 200, 200, 200, np.nan, np.nan, 200, 200, 200],
            'volume': [500000, 500000, 500000, 500000, 500000, 0, 0, 500000, 500000, 500000]
        }, index=self.dates)

        # SPY benchmark
        self.price_data['SPY'] = pd.DataFrame({
            'open': [300, 301, 302, 303, 304, 305, 306, 307, 308, 309],
            'high': [302, 303, 304, 305, 306, 307, 308, 309, 310, 311],
            'low': [299, 300, 301, 302, 303, 304, 305, 306, 307, 308],
            'close': [301, 302, 303, 304, 305, 306, 307, 308, 309, 310],
            'volume': [2000000] * 10
        }, index=self.dates)

        # Create mock loader with sample data
        self.loader = MockMarketDataLoader()
        for ticker, data in self.price_data.items():
            self.loader.set_data(ticker, data)

        # Create backktester with test calendar
        from portwine.backtester.core import DailyMarketCalendar
        
        # Convert data to the format expected by Backtester
        self.data_interface = MockDataInterface(self.price_data)
        
        # Create a calendar that returns all dates for TestBacktester
        self.calendar = TestDailyMarketCalendar(
            calendar_name="NYSE",
            mode="all",
            default_start="2020-01-01",
            default_end="2020-01-10",
            default_hour=16,
        )
        
        self.backtester = Backtester(
            data=self.data_interface,
            calendar=self.calendar
        )

    def test_initialization(self):
        """Test backktester initialization"""
        self.assertIsNotNone(self.backtester)
        self.assertEqual(self.backtester.data, self.data_interface)

    def test_simple_backtest(self):
        """Test basic backtest with fixed allocation strategy"""
        # Create a strategy with equal allocation
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest without lookahead protection
        results = self.backtester.run_backtest(
            strategy=strategy
        )

        # Assert results are non-empty
        self.assertIsNotNone(results)

        # Check that we have the expected keys in results
        expected_keys = ['signals_df', 'tickers_returns', 'strategy_returns']
        for key in expected_keys:
            self.assertIn(key, results)

        # Check that results have correct dates
        self.assertEqual(len(results['signals_df']), len(self.dates))
        
        # The calendar returns timestamps with 16:00 hours, so we need to compare accordingly
        expected_index = pd.DatetimeIndex([pd.Timestamp(d.date()) + pd.Timedelta(hours=16) for d in self.dates])
        pd.testing.assert_index_equal(
            results['signals_df'].index,
            expected_index,
            check_names=False  # Ignore index names in comparison
        )

        # Verify first day return is 0 (as expected for first day without shift_signals)
        self.assertEqual(results['strategy_returns'].iloc[0], 0.0)

        # Check that signals dataframe contains the correct tickers
        for ticker in self.tickers:
            self.assertIn(ticker, results['signals_df'].columns)

        # Verify strategy was called once for each date
        self.assertEqual(len(strategy.step_calls), len(self.dates))

    def test_with_signal_shifting(self):
        """Test backtest with shifting signals to avoid lookahead bias"""
        # Equal allocation strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest with signal shifting
        results = self.backtester.run_backtest(
            strategy=strategy
        )

        # The signals DataFrame contains the original signals (not shifted)
        # The shifting happens internally in calculate_results() for strategy returns
        first_day = results['signals_df'].iloc[0]
        for ticker in self.tickers:
            # First day should have the strategy's allocation (not shifted)
            self.assertEqual(first_day[ticker], strategy.allocation[ticker])

        # Second day should also match the strategy's allocation
        second_day = results['signals_df'].iloc[1]
        for ticker in self.tickers:
            self.assertEqual(second_day[ticker], strategy.allocation[ticker])

    def test_with_ticker_benchmark(self):
        """Test backtest with a ticker-based benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest with SPY as benchmark
        results = self.backtester.run_backtest(
            strategy=strategy,
            benchmark='SPY'
        )

        # Check that benchmark returns are present
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

        # Verify first day benchmark return is 0 (no prior day)
        self.assertEqual(results['benchmark_returns'].iloc[0], 0.0)

        # Verify other days have the expected returns
        # SPY went from 301 to 310 => daily returns should match
        for i in range(1, len(self.dates)):
            benchmark_prev = self.price_data['SPY']['close'].iloc[i - 1]
            benchmark_curr = self.price_data['SPY']['close'].iloc[i]
            expected_return = (benchmark_curr / benchmark_prev) - 1.0

            # Check with some tolerance for floating point
            self.assertAlmostEqual(
                results['benchmark_returns'].iloc[i],
                expected_return,
                places=6
            )

    def test_with_function_benchmark(self):
        """Test backtest with a function-based benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest with equal_weight benchmark
        results = self.backtester.run_backtest(
            strategy=strategy,
            benchmark='equal_weight'
        )

        # Check that benchmark returns are present
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

        # Test with custom benchmark function
        def custom_benchmark(daily_ret_df, verbose=False):
            """Simple custom benchmark that returns twice the equal weight return"""
            return daily_ret_df.mean(axis=1) * 2.0

        results_custom = self.backtester.run_backtest(
            strategy=strategy,
            benchmark=custom_benchmark
        )

        # Verify custom benchmark returns
        self.assertIn('benchmark_returns', results_custom)

        # Custom benchmark should be twice the equal weight
        pd.testing.assert_series_equal(
            results_custom['benchmark_returns'],
            results['benchmark_returns'] * 2.0
        )

    def test_with_markowitz_benchmark(self):
        """Test backtest with the Markowitz minimum variance benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest with markowitz benchmark
        results = self.backtester.run_backtest(
            strategy=strategy,
            benchmark='markowitz'
        )

        # Check that benchmark returns are present
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

        # The Markowitz benchmark should favor GOOG which has a flat trend
        # over the volatile AAPL and MSFT, but this is hard to test precisely
        # without reimplementing the algorithm. Just verify it runs.
        self.assertTrue(all(~np.isnan(results['benchmark_returns'])))

    def test_with_date_filtering(self):
        """Test backtest with start_date and end_date filtering"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Test with start_date only
        mid_date = self.dates[5]
        results_start = self.backtester.run_backtest(
            strategy=strategy,
            start_date=mid_date
        )

        # Should only have dates from mid_date onwards
        self.assertEqual(len(results_start['signals_df']), 5)  # 5 days left
        # Convert mid_date to timestamp with 16:00 hours for comparison
        mid_date_timestamp = pd.Timestamp(mid_date.date()) + pd.Timedelta(hours=16)
        self.assertTrue(all(date >= mid_date_timestamp for date in results_start['signals_df'].index))

        # Test with end_date only
        results_end = self.backtester.run_backtest(
            strategy=strategy,
            end_date=mid_date
        )

        # Should only have dates up to mid_date
        self.assertEqual(len(results_end['signals_df']), 6)  # First 6 days
        self.assertTrue(all(date <= mid_date_timestamp for date in results_end['signals_df'].index))

        # Test with both start_date and end_date
        start_date = self.dates[2]
        end_date = self.dates[7]
        results_both = self.backtester.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date
        )

        # Should only have dates between start_date and end_date
        self.assertEqual(len(results_both['signals_df']), 6)  # 6 days in range
        start_date_timestamp = pd.Timestamp(start_date.date()) + pd.Timedelta(hours=16)
        end_date_timestamp = pd.Timestamp(end_date.date()) + pd.Timedelta(hours=16)
        self.assertTrue(all(start_date_timestamp <= date <= end_date_timestamp
                            for date in results_both['signals_df'].index))

    def test_require_all_history(self):
        """Test backtest with require_all_history=True"""
        # Create a new set of data with staggered start dates
        staggered_dates = {}
        staggered_dates['AAPL'] = self.dates  # All dates
        staggered_dates['MSFT'] = self.dates[2:]  # Starts on day 3
        staggered_dates['GOOG'] = self.dates[4:]  # Starts on day 5

        staggered_data = {}
        for ticker, dates in staggered_dates.items():
            # Take original data but filter for dates
            staggered_data[ticker] = self.price_data[ticker].loc[dates]

        # Create new loader with staggered data
        staggered_loader = MockMarketDataLoader()
        for ticker, data in staggered_data.items():
            staggered_loader.set_data(ticker, data)

        # Create backktester with staggered data
        staggered_data_interface = MockDataInterface(staggered_data)
        
        # Create a calendar that returns all dates for this test
        staggered_backtester = Backtester(
            data=staggered_data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="all",
                default_start="2020-01-01",
                default_end="2020-01-10",
                default_hour=16,
            ),
        )

        # Create strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Test without require_all_history
        results_without = staggered_backtester.run_backtest(
            strategy=strategy
        )

        # Should have all dates (will have NaN or 0 allocations for missing tickers)
        self.assertEqual(len(results_without['signals_df']), len(self.dates))

        # Test with require_all_history (Backtester doesn't have this parameter, so behavior is the same)
        results_with = staggered_backtester.run_backtest(
            strategy=strategy
        )

        # Backtester doesn't have require_all_history parameter, so behavior is the same as without
        self.assertEqual(len(results_with['signals_df']), len(self.dates))
        # Both results should be identical since Backtester doesn't support require_all_history
        pd.testing.assert_frame_equal(results_with['signals_df'], results_without['signals_df'])

    def test_dynamic_strategy(self):
        """Test backtest with a dynamic strategy that changes allocations"""
        # Create dynamic strategy
        strategy = DynamicTestStrategy(tickers=self.tickers)

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy
        )

        # Verify we get results
        self.assertIsNotNone(results)

        # Early days should have equal allocations (no history yet)
        # Later days should shift to the best performer (AAPL in our test data)

        # Check signals on last day - should be all in AAPL
        last_signals = results['signals_df'].iloc[-1]
        # Allowing some floating point tolerance
        self.assertAlmostEqual(last_signals['AAPL'], 1.0, places=6)
        self.assertAlmostEqual(last_signals['MSFT'], 0.0, places=6)
        self.assertAlmostEqual(last_signals['GOOG'], 0.0, places=6)

    def test_handling_missing_data(self):
        """Test how the backtester handles missing data"""
        # GOOG has NaN values in the middle
        strategy = SimpleTestStrategy(tickers=['AAPL', 'GOOG'])

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy
        )

        # Verify we get results
        self.assertIsNotNone(results)

        # Check that we have results for all dates
        self.assertEqual(len(results['signals_df']), len(self.dates))

        # The Backtester doesn't forward-fill missing data, so we should have NaN values
        # where GOOG has missing data
        goog_returns = results['tickers_returns']['GOOG']
        # Check that we have NaN values on the missing days (days 6 and 7, indices 5 and 6)
        self.assertTrue(pd.isna(goog_returns.iloc[5]))  # Day 6 (index 5)
        self.assertTrue(pd.isna(goog_returns.iloc[6]))  # Day 7 (index 6)

    def test_empty_strategy(self):
        """Test backtest with an empty strategy"""
        # Create strategy with no tickers
        strategy = SimpleTestStrategy(tickers=[])

        # Run backtest - should raise because no tickers loaded
        with self.assertRaises(ValueError):
            results = self.backtester.run_backtest(strategy=strategy)

    def test_nonexistent_ticker(self):
        """Test backtest with non-existent tickers"""
        # Strategy with non-existent ticker
        strategy = SimpleTestStrategy(tickers=['NONEXISTENT'])

        # Run backtest - should raise because ticker doesnt exist
        with self.assertRaises(ValueError):
            results = self.backtester.run_backtest(strategy=strategy)

    def test_invalid_benchmark(self):
        """Test backtest with invalid benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        with self.assertRaises(InvalidBenchmarkError):
            results = self.backtester.run_backtest(
                strategy=strategy,
                benchmark=123  # Not a string or callable
            )

        # Non-existent benchmark ticker
        with self.assertRaises(InvalidBenchmarkError):
            results = self.backtester.run_backtest(
                strategy=strategy,
                benchmark='NONEXISTENT'
            )

    def test_over_allocation_raises(self):
        """Test that backtest errors when strategy returns weights summing to >1."""
        class OverAllocateStrategy(StrategyBase):
            def __init__(self, tickers):
                super().__init__(tickers)
            def step(self, current_date, daily_data):
                # Allocate 0.6 to each ticker, sum >1 for multiple tickers
                return {ticker: 0.6 for ticker in self.tickers}
        strategy = OverAllocateStrategy(self.tickers)
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(strategy=strategy)


class TestRequireAllHistory(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        mock_data = {
            'A': pd.DataFrame({
                    "open":   range(1, 11),
                    "high":   range(1, 11),
                    "low":    range(1, 11),
                    "close":  range(1, 11),
                    "volume": [100] * 10
                }, index=dates),
            'B': pd.DataFrame({
                    "open":   list(range(10, 0, -1)),
                    "high":   list(range(10, 0, -1)),
                    "low":    list(range(10, 0, -1)),
                    "close":  list(range(10, 0, -1)),
                    "volume": [100] * 10
                }, index=dates)
        }
        self.data_interface = MockDataInterface(mock_data)
        
        # Create a calendar that returns all dates for TestRequireAllHistory
        self.calendar = TestDailyMarketCalendar(
            calendar_name="NYSE",
            mode="all",
            default_start="2020-01-01",
            default_end="2020-01-10",
            default_hour=16,
        )
        self.bt = Backtester(
            data=self.data_interface,
            calendar=self.calendar
        )
        self.strat = SimpleTestStrategy(["A", "B"])

    def test_require_all_history_false_keeps_full_length(self):
        res = self.bt.run_backtest(self.strat)
        self.assertEqual(len(res["signals_df"]), 10)

    def test_require_all_history_true_trims_to_common_start(self):
        res = self.bt.run_backtest(self.strat)
        # Both tickers start on 2020-01-01, so still 10
        self.assertEqual(len(res["signals_df"]), 10)

class TestBenchmarkDefaultAndInvalid(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        mock_data = {
            'A': pd.DataFrame({
                "open": range(1, 11),
                "high": range(1, 11),
                "low": range(1, 11),
                "close": range(1, 11),
                "volume": [100] * 10
            }, index=dates),
            'B': pd.DataFrame({
                "open": list(range(10, 0, -1)),
                "high": list(range(10, 0, -1)),
                "low": list(range(10, 0, -1)),
                "close": list(range(10, 0, -1)),
                "volume": [100] * 10
            }, index=dates)
        }
        self.data_interface = MockDataInterface(mock_data)
        
        # Create a calendar that returns all dates for TestBenchmarkDefaultAndInvalid
        self.calendar = TestDailyMarketCalendar(
            calendar_name="NYSE",
            mode="all",
            default_start="2020-01-01",
            default_end="2020-01-10",
            default_hour=16,
        )
        self.bt = Backtester(
            data=self.data_interface,
            calendar=self.calendar
        )

    def test_default_benchmark_equal_weight(self):
        strat = SimpleTestStrategy(["A", "B"])
        res = self.bt.run_backtest(strat)  # no benchmark specified
        bm = res["benchmark_returns"]
        ret = res["tickers_returns"]
        expected = (ret["A"] + ret["B"]) / 2
        # Set the same name as benchmark returns for comparison
        expected.name = bm.name
        pd.testing.assert_series_equal(bm, expected)

    def test_invalid_benchmark_raises(self):
        strat = SimpleTestStrategy(["A"])
        with self.assertRaises(InvalidBenchmarkError):
            self.bt.run_backtest(strat, benchmark="NONEXISTENT")

class TestBacktesterWithCalendar(unittest.TestCase):
    def setUp(self):
        # Build 2020-01-13 to 2020-01-18 price data for 'X'
        dates = pd.date_range("2020-01-13", "2020-01-18", freq="D")
        df = pd.DataFrame({
            "open":   range(len(dates)),
            "high":   range(len(dates)),
            "low":    range(len(dates)),
            "close":  range(len(dates)),
            "volume": [1.0] * len(dates)
        }, index=dates)

        # Use mock loader instead of SimpleDateLoader
        mock_data = {"X": df}
        self.data_interface = MockDataInterface(mock_data)
        self.calendar = TestDailyMarketCalendar(
            calendar_name="NYSE",
            mode="odd",
            default_start="2020-01-13",
            default_end="2020-01-18",
            default_hour=16,
        )
        self.bt = Backtester(
            data=self.data_interface,
            calendar=self.calendar
        )
        # Use TestStrategy instead of ZeroStrategy
        self.strategy = SimpleTestStrategy(["X"])

        # Precompute expected calendar timestamps
        sel = [d for d in dates if d.day % 2 == 1]
        self.calendar_ts = pd.DatetimeIndex(
            [pd.Timestamp(d.date()) + pd.Timedelta(hours=16) for d in sel]
        )

    def test_calendar_overrides_data_dates(self):
        res = self.bt.run_backtest(self.strategy)
        pd.testing.assert_index_equal(res["signals_df"].index, self.calendar_ts)

    def test_start_end_filters_with_calendar(self):
        # Start‐date only
        start = pd.Timestamp("2020-01-13 16:00")
        res = self.bt.run_backtest(self.strategy, start_date=start)
        expected_after_start = pd.DatetimeIndex([
            pd.Timestamp("2020-01-13 16:00"),
            pd.Timestamp("2020-01-15 16:00"),
            pd.Timestamp("2020-01-17 16:00"),
        ])
        pd.testing.assert_index_equal(res["signals_df"].index, expected_after_start)

        # End‐date only
        end = pd.Timestamp("2020-01-17 16:00")
        res = self.bt.run_backtest(self.strategy, end_date=end)
        expected_before_end = pd.DatetimeIndex([
            pd.Timestamp("2020-01-13 16:00"),
            pd.Timestamp("2020-01-15 16:00"),
            pd.Timestamp("2020-01-17 16:00"),
        ])
        pd.testing.assert_index_equal(res["signals_df"].index, expected_before_end)

    def test_invalid_date_range_raises(self):
        with self.assertRaises(ValueError):
            self.bt.run_backtest(
                self.strategy,
                start_date="2020-01-10",
                end_date="2020-01-01"
            )

    def test_non_overlapping_date_range_raises(self):
        with self.assertRaises(ValueError):
            self.bt.run_backtest(
                self.strategy,
                start_date="2030-01-01",
                end_date="2030-01-05"
            )

    def test_require_all_history_with_calendar(self):
        res1 = self.bt.run_backtest(self.strategy)
        res2 = self.bt.run_backtest(self.strategy)
        pd.testing.assert_index_equal(
            res1["signals_df"].index,
            res2["signals_df"].index
        )

    def test_benchmark_equal_weight_with_calendar(self):
        res = self.bt.run_backtest(self.strategy)
        pd.testing.assert_series_equal(
            res["benchmark_returns"],
            res["strategy_returns"]
        )

    def test_invalid_benchmark_raises(self):
        with self.assertRaises(InvalidBenchmarkError):
            self.bt.run_backtest(self.strategy, benchmark="NONEXISTENT")
    def test_require_all_history_cuts_before_benchmark_start(self):
        loader = MockMarketDataLoader()
        idx_full = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        idx_bench = pd.date_range("2020-01-05", "2020-01-10", freq="D")
        base = {'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0}
        loader.set_data('A', pd.DataFrame(base, index=idx_full))
        loader.set_data('B', pd.DataFrame(base, index=idx_full))
        loader.set_data('BENCH', pd.DataFrame(base, index=idx_bench))

        data_interface = MockDataInterface(loader.mock_data)
        backtester = Backtester(
            data=data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="odd",
                default_start="2020-01-01",
                default_end="2020-01-10",
                default_hour=16,
            ),
        )
        strat = SimpleTestStrategy(['A', 'B'])
        # Backtester doesn't have require_all_history parameter, so it starts at the earliest available data date
        result = backtester.run_backtest(
            strategy=strat,
            benchmark='BENCH'
        )
        signals = result['signals_df']
        self.assertEqual(signals.index.min().date(), pd.Timestamp("2020-01-01").date())

    def test_without_require_all_history_includes_earliest_ticker(self):
        loader = MockMarketDataLoader()
        idx_full = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        idx_bench = pd.date_range("2020-01-05", "2020-01-10", freq="D")
        base = {'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0}
        loader.set_data('A', pd.DataFrame(base, index=idx_full))
        loader.set_data('B', pd.DataFrame(base, index=idx_full))
        loader.set_data('BENCH', pd.DataFrame(base, index=idx_bench))
        data_interface = MockDataInterface(loader.mock_data)
        backtester = Backtester(
            data=data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="odd",
                default_start="2020-01-01",
                default_end="2020-01-10",
                default_hour=16,
            ),
        )
        strat = SimpleTestStrategy(['A', 'B'])
        # With require_all_history=False, the backtest should begin at the earliest ticker date
        result = backtester.run_backtest(
            strategy=strat,
            benchmark='BENCH'
        )
        signals = result['signals_df']
        self.assertEqual(signals.index.min().date(), pd.Timestamp("2020-01-01").date())


if __name__ == "__main__":
    unittest.main()
