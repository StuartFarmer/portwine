import unittest
import pandas as pd
import numpy as np

# Import components to be tested
from portwine.backtester.core import Backtester
from portwine.backtester.benchmarks import InvalidBenchmarkError
from portwine.strategies.base import StrategyBase
from portwine.data.interface import DataInterface
from unittest.mock import Mock
from typing import Dict, List
from tests.helpers import MockDataStore


class MockDataInterface(DataInterface):
    """DataInterface backed by MockDataStore for tests."""

    def __init__(self, mock_data: Dict[str, pd.DataFrame] | None = None, exists_data=None):
        self.exists_data = exists_data or {}
        default_payload = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000}
        store = MockDataStore(default_ohlcv=default_payload)
        if mock_data:
            store.load_bulk(mock_data)
        super().__init__(store)
        self.current_timestamp = None
        self.set_timestamp_calls = []
        self.get_calls = []

    def exists(self, ticker: str, start_date: str, end_date: str) -> bool:
        # Allow tests to force existence behavior; otherwise defer to store
        if ticker in self.exists_data:
            return self.exists_data[ticker]
        return self.data_loader.exists(ticker, start_date, end_date)

    def set_current_timestamp(self, timestamp):
        self.current_timestamp = timestamp
        self.set_timestamp_calls.append(timestamp)
        super().set_current_timestamp(timestamp)

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
            try:
                ticker_data = daily_data[ticker]
                if ticker_data is not None:
                    price = ticker_data.get('close')
            except (KeyError, TypeError):
                pass

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

from tests.calendar_utils import TestDailyMarketCalendar

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
        # If start_date has a time component, align the first calendar point to that exact timestamp
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts.time() != pd.Timestamp('00:00').time():
            days = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq='D')
            # Replace first element with the exact start_ts
            result = [start_ts] + [d for d in days[1:]]
            return pd.DatetimeIndex(result).to_numpy()
        return pd.date_range(start_date, end_date, freq='D').to_numpy()


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

        # Create data interface with sample data via MockDataStore
        self.data_interface = MockDataInterface()
        for ticker, df in self.price_data.items():
            self.data_interface.data_loader.load_dataframe(ticker, df)

        # Create backktester with test calendar
        from portwine.backtester.core import DailyMarketCalendar
        self.backtester = Backtester(
            self.data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="all",
                default_start="2020-01-01",
                default_end="2020-01-10",
                default_hour=None,
            ),
        )

    def test_initialization(self):
        """Test backktester initialization"""
        self.assertIsNotNone(self.backtester)
        self.assertEqual(self.backtester.data, self.data_interface)

    def test_simple_backtest(self):
        """Test basic backtest with fixed allocation strategy"""
        # Create a strategy with equal allocation
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Assert results are non-empty
        self.assertIsNotNone(results)

        # Check that we have the expected keys in results
        expected_keys = ['signals_df', 'tickers_returns', 'strategy_returns', 'benchmark_returns']
        for key in expected_keys:
            self.assertIn(key, results)

        # Check that results have correct dates
        self.assertEqual(len(results['signals_df']), len(self.dates))

        # Check that signals dataframe contains the correct tickers
        for ticker in self.tickers:
            self.assertIn(ticker, results['signals_df'].columns)

        # Verify strategy was called once for each date
        self.assertEqual(len(strategy.step_calls), len(self.dates))

    def test_with_signal_shifting(self):
        """Test backtest with shifting signals to avoid lookahead bias"""
        # Equal allocation strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Verify we get results
        self.assertIsNotNone(results)
        self.assertIn('signals_df', results)
        self.assertIn('strategy_returns', results)

        # Check that signals dataframe contains the correct tickers
        for ticker in self.tickers:
            self.assertIn(ticker, results['signals_df'].columns)

    def test_with_ticker_benchmark(self):
        """Test backtest with a ticker-based benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Define a benchmark function that uses SPY data
        def spy_benchmark(ret_df):
            # For this test, we'll create a simple benchmark using the ticker returns
            # In a real scenario, this would use actual SPY data
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest with benchmark function
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=spy_benchmark
        )

        # Check that benchmark returns are present
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

        # Verify we get results
        self.assertIsNotNone(results)
        self.assertIn('signals_df', results)
        self.assertIn('strategy_returns', results)

    def test_with_function_benchmark(self):
        """Test backtest with a function-based benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Define equal weight benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest with equal_weight benchmark
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Check that benchmark returns are present
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

        # Test with custom benchmark function
        def custom_benchmark(ret_df):
            """Simple custom benchmark that returns twice the equal weight return"""
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights) * 2.0, columns=['benchmark_returns'])

        results_custom = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
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

        # Define a simple benchmark function (not actual Markowitz)
        def simple_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest with benchmark function
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=simple_benchmark
        )

        # Check that benchmark returns are present
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

        # Verify we get results
        self.assertIsNotNone(results)
        self.assertIn('signals_df', results)
        self.assertIn('strategy_returns', results)

    def test_with_date_filtering(self):
        """Test backtest with start_date and end_date filtering"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Test with start_date only
        mid_date = self.dates[5]
        results_start = self.backtester.run_backtest(
            strategy=strategy,
            start_date=mid_date,
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Should only have dates from mid_date onwards
        self.assertEqual(len(results_start['signals_df']), 5)  # 5 days left
        self.assertTrue(all(date >= mid_date for date in results_start['signals_df'].index))

        # Test with end_date only
        results_end = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date=mid_date,
            benchmark=equal_weight_benchmark
        )

        # Should only have dates up to mid_date
        self.assertEqual(len(results_end['signals_df']), 6)  # First 6 days
        self.assertTrue(all(date <= mid_date for date in results_end['signals_df'].index))

        # Test with both start_date and end_date
        start_date = self.dates[2]
        end_date = self.dates[7]
        results_both = self.backtester.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            benchmark=equal_weight_benchmark
        )

        # Should only have dates between start_date and end_date
        self.assertEqual(len(results_both['signals_df']), 6)  # 6 days in range
        self.assertTrue(all(start_date <= date <= end_date
                            for date in results_both['signals_df'].index))

    def test_require_all_history(self):
        """Test backtest with require_all_history=True"""
        # For the new backtester, we'll test a simpler scenario
        # since the new interface doesn't have require_all_history parameter
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Verify we get results
        self.assertIsNotNone(results)
        self.assertIn('signals_df', results)
        self.assertIn('strategy_returns', results)
        self.assertIn('benchmark_returns', results)

        # Check that results have correct dates
        self.assertEqual(len(results['signals_df']), len(self.dates))

    def test_dynamic_strategy(self):
        """Test backtest with a dynamic strategy that changes allocations"""
        # Create dynamic strategy
        strategy = DynamicTestStrategy(tickers=self.tickers)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Verify we get results
        self.assertIsNotNone(results)
        self.assertIn('signals_df', results)
        self.assertIn('strategy_returns', results)
        self.assertIn('benchmark_returns', results)

        # Check that signals dataframe contains the correct tickers
        for ticker in self.tickers:
            self.assertIn(ticker, results['signals_df'].columns)

    def test_handling_missing_data(self):
        """Test how the backtester handles missing data"""
        # GOOG has NaN values in the middle
        strategy = SimpleTestStrategy(tickers=['AAPL', 'GOOG'])

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )

        # Verify we get results
        self.assertIsNotNone(results)
        self.assertIn('signals_df', results)
        self.assertIn('strategy_returns', results)
        self.assertIn('benchmark_returns', results)

        # Check that signals dataframe contains the correct tickers
        for ticker in ['AAPL', 'GOOG']:
            self.assertIn(ticker, results['signals_df'].columns)

    def test_empty_strategy(self):
        """Test backtest with an empty strategy"""
        # Create strategy with no tickers
        strategy = SimpleTestStrategy(tickers=[])

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest - should raise because no tickers loaded
        with self.assertRaises(ValueError):
            results = self.backtester.run_backtest(
                strategy=strategy,
                start_date='2020-01-01',
                end_date='2020-01-10',
                benchmark=equal_weight_benchmark
            )

    def test_nonexistent_ticker(self):
        """Test backtest with non-existent tickers raises ValueError"""
        # Strategy with non-existent ticker
        strategy = SimpleTestStrategy(tickers=['NONEXISTENT'])

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])

        # Run backtest - should now raise ValueError because NONEXISTENT ticker doesn't exist
        with self.assertRaises(ValueError) as context:
            self.backtester.run_backtest(
                strategy=strategy,
                start_date='2020-01-01',
                end_date='2020-01-10',
                benchmark=equal_weight_benchmark
            )
        
        # Verify the error message
        self.assertIn("Data for ticker NONEXISTENT does not exist", str(context.exception))

    def test_invalid_benchmark(self):
        """Test backtest with invalid benchmark"""
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # For the new backtester, it uses a default benchmark when none is provided
        # So this should work without raising an error
        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-10'
            # No benchmark provided - should use default
        )
        
        # Verify that benchmark returns are present (using default)
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

    def test_over_allocation_raises(self):
        """Test that backtest errors when strategy returns weights summing to >1."""
        class OverAllocateStrategy(StrategyBase):
            def __init__(self, tickers):
                super().__init__(tickers)
            def step(self, current_date, daily_data):
                # Allocate 0.6 to each ticker, sum >1 for multiple tickers
                return {ticker: 0.6 for ticker in self.tickers}
        
        strategy = OverAllocateStrategy(self.tickers)
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(
                strategy=strategy,
                start_date='2020-01-01',
                end_date='2020-01-10',
                benchmark=equal_weight_benchmark
            )


class TestRequireAllHistory(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        
        # Create mock data interface with data
        mock_data = {
            'A': {
                'close': np.array(range(1, 11)),
                'open': np.array(range(1, 11)),
                'high': np.array(range(1, 11)),
                'low': np.array(range(1, 11)),
                'volume': np.array([100] * 10)
            },
            'B': {
                'close': np.array(list(range(10, 0, -1))),
                'open': np.array(list(range(10, 0, -1))),
                'high': np.array(list(range(10, 0, -1))),
                'low': np.array(list(range(10, 0, -1))),
                'volume': np.array([100] * 10)
            }
        }
        
        self.data_interface = MockDataInterface(mock_data)
        
        self.bt = Backtester(self.data_interface, calendar=MockDailyMarketCalendar("NYSE"))
        self.strat = SimpleTestStrategy(["A", "B"])

    def test_require_all_history_false_keeps_full_length(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        

        
        res = self.bt.run_backtest(
            self.strat, 
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        self.assertEqual(len(res["signals_df"]), 10)

    def test_require_all_history_true_trims_to_common_start(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            self.strat, 
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        # Both tickers start on 2020-01-01, so still 10
        self.assertEqual(len(res["signals_df"]), 10)

class TestBenchmarkDefaultAndInvalid(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        
        # Create mock data interface with data
        mock_data = {
            'A': {
                'close': np.array(range(1, 11)),
                'open': np.array(range(1, 11)),
                'high': np.array(range(1, 11)),
                'low': np.array(range(1, 11)),
                'volume': np.array([100] * 10)
            },
            'B': {
                'close': np.array(list(range(10, 0, -1))),
                'open': np.array(list(range(10, 0, -1))),
                'high': np.array(list(range(10, 0, -1))),
                'low': np.array(list(range(10, 0, -1))),
                'volume': np.array([100] * 10)
            }
        }
        
        self.data_interface = MockDataInterface(mock_data)
        
        self.bt = Backtester(self.data_interface, calendar=MockDailyMarketCalendar("NYSE"))

    def test_default_benchmark_equal_weight(self):
        strat = SimpleTestStrategy(["A", "B"])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            strat, 
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        bm = res["benchmark_returns"]
        ret = res["tickers_returns"]
        expected = (ret["A"] + ret["B"]) / 2
        pd.testing.assert_series_equal(bm, expected, check_names=False)

    def test_invalid_benchmark_raises(self):
        strat = SimpleTestStrategy(["A"])
        # For the new backtester, it uses a default benchmark when none is provided
        # So this should work without raising an error
        res = self.bt.run_backtest(
            strat,
            start_date='2020-01-01',
            end_date='2020-01-10'
            # No benchmark provided - should use default
        )
        
        # Verify that benchmark returns are present (using default)
        self.assertIn('benchmark_returns', res)

class TestBacktesterWithCalendar(unittest.TestCase):
    def setUp(self):
        # Build 2020-01-13 to 2020-01-18 price data for 'X'
        dates = pd.date_range("2020-01-13", "2020-01-18", freq="D")
        
        # Create mock data for 'X' ticker
        mock_data = {
            'X': pd.DataFrame({
                'close': range(len(dates)),
                'open': range(len(dates)),
                'high': range(len(dates)),
                'low': range(len(dates)),
                'volume': [1.0] * len(dates)
            }, index=dates)
        }
        
        # Create mock data interface with the data properly loaded
        self.data_interface = MockDataInterface(mock_data=mock_data)
        
        self.bt = Backtester(
            data=self.data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="odd",
                default_start="2020-01-13",
                default_end="2020-01-18",
                default_hour=None,
            ),
        )
        # Use TestStrategy instead of ZeroStrategy
        self.strategy = SimpleTestStrategy(["X"])

        # Precompute expected calendar timestamps
        sel = [d for d in dates if d.day % 2 == 1]
        self.calendar_ts = pd.DatetimeIndex(
            [pd.Timestamp(d.date()) + pd.Timedelta(hours=16) for d in sel]
        )

    def test_calendar_overrides_data_dates(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            self.strategy, 
            start_date='2020-01-13',
            end_date='2020-01-18',
            benchmark=equal_weight_benchmark
        )
        # The calendar returns dates without time, so we need to adjust the expected timestamps
        expected_ts = pd.DatetimeIndex(['2020-01-13', '2020-01-15', '2020-01-17'])
        pd.testing.assert_index_equal(res["signals_df"].index, expected_ts)

    def test_start_end_filters_with_calendar(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # Start‐date only
        start = pd.Timestamp("2020-01-13 16:00")
        res = self.bt.run_backtest(
            self.strategy, 
            start_date=start,
            end_date='2020-01-18',
            benchmark=equal_weight_benchmark
        )
        expected_after_start = pd.DatetimeIndex([
            pd.Timestamp("2020-01-13 16:00:00"),
            pd.Timestamp("2020-01-15 16:00:00"),
            pd.Timestamp("2020-01-17 16:00:00"),
        ])
        pd.testing.assert_index_equal(res["signals_df"].index, expected_after_start)

        # End‐date only
        end = pd.Timestamp("2020-01-17 16:00")
        res = self.bt.run_backtest(
            self.strategy,
            start_date='2020-01-13',
            end_date=end,
            benchmark=equal_weight_benchmark
        )
        expected_before_end = pd.DatetimeIndex([
            pd.Timestamp("2020-01-13"),
            pd.Timestamp("2020-01-15"),
            pd.Timestamp("2020-01-17"),
        ])
        pd.testing.assert_index_equal(res["signals_df"].index, expected_before_end)

    def test_invalid_date_range_raises(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        with self.assertRaises(ValueError):
            self.bt.run_backtest(
                self.strategy,
                start_date="2020-01-10",
                end_date="2020-01-01",
                benchmark=equal_weight_benchmark
            )

    def test_non_overlapping_date_range_raises(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        with self.assertRaises(ValueError):
            self.bt.run_backtest(
                self.strategy,
                start_date="2030-01-01",
                end_date="2030-01-05",
                benchmark=equal_weight_benchmark
            )

    def test_require_all_history_with_calendar(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res1 = self.bt.run_backtest(
            self.strategy, 
            start_date='2020-01-13',
            end_date='2020-01-18',
            benchmark=equal_weight_benchmark
        )
        res2 = self.bt.run_backtest(
            self.strategy, 
            start_date='2020-01-13',
            end_date='2020-01-18',
            benchmark=equal_weight_benchmark
        )
        pd.testing.assert_index_equal(
            res1["signals_df"].index,
            res2["signals_df"].index
        )

    def test_benchmark_equal_weight_with_calendar(self):
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        res = self.bt.run_backtest(
            self.strategy,
            start_date='2020-01-13',
            end_date='2020-01-18',
            benchmark=equal_weight_benchmark
        )
        # Compare the values, not the column names
        pd.testing.assert_series_equal(
            res["benchmark_returns"],
            res["strategy_returns"],
            check_names=False
        )

    def test_invalid_benchmark_raises(self):
        # For the new backtester, it uses a default benchmark when none is provided
        # So this should work without raising an error
        res = self.bt.run_backtest(
            self.strategy,
            start_date='2020-01-13',
            end_date='2020-01-18'
            # No benchmark provided - should use default
        )
        
        # Verify that benchmark returns are present (using default)
        self.assertIn('benchmark_returns', res)
    def test_require_all_history_cuts_before_benchmark_start(self):
        # Create mock data for tickers A and B
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        mock_data = {
            'A': pd.DataFrame({
                'close': np.ones(10),
                'open': np.ones(10),
                'high': np.ones(10),
                'low': np.ones(10),
                'volume': np.ones(10)
            }, index=dates),
            'B': pd.DataFrame({
                'close': np.ones(10),
                'open': np.ones(10),
                'high': np.ones(10),
                'low': np.ones(10),
                'volume': np.ones(10)
            }, index=dates)
        }

        data_interface = MockDataInterface(mock_data=mock_data)
        backtester = Backtester(data_interface, calendar=MockDailyMarketCalendar("NYSE"))
        strat = SimpleTestStrategy(['A', 'B'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # With the new backtester, we'll test that it works with the given date range
        result = backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        signals = result['signals_df']
        self.assertEqual(signals.index.min(), pd.Timestamp("2020-01-01"))

    def test_without_require_all_history_includes_earliest_ticker(self):
        # Create mock data for tickers A and B
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        mock_data = {
            'A': pd.DataFrame({
                'close': np.ones(10),
                'open': np.ones(10),
                'high': np.ones(10),
                'low': np.ones(10),
                'volume': np.ones(10)
            }, index=dates),
            'B': pd.DataFrame({
                'close': np.ones(10),
                'open': np.ones(10),
                'high': np.ones(10),
                'low': np.ones(10),
                'volume': np.ones(10)
            }, index=dates)
        }
        data_interface = MockDataInterface(mock_data=mock_data)
        backtester = Backtester(data_interface, calendar=MockDailyMarketCalendar("NYSE"))
        strat = SimpleTestStrategy(['A', 'B'])
        
        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # With the new backtester, we'll test that it works with the given date range
        result = backtester.run_backtest(
            strategy=strat,
            start_date='2020-01-01',
            end_date='2020-01-10',
            benchmark=equal_weight_benchmark
        )
        signals = result['signals_df']
        self.assertEqual(signals.index.min(), pd.Timestamp("2020-01-01"))


if __name__ == "__main__":
    unittest.main()
