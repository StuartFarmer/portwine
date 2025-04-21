import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import shutil

# Import components to be tested
from portwine.backtester import Backtester, STANDARD_BENCHMARKS, InvalidBenchmarkError
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class MockMarketDataLoader(MarketDataLoader):
    """Mock market data loader for testing purposes"""

    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}

    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        return self.mock_data.get(ticker)

    def set_data(self, ticker, data):
        """Set mock data for a ticker"""
        self.mock_data[ticker] = data


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

        # Create backktester
        self.backtester = Backtester(self.loader)

    def test_initialization(self):
        """Test backktester initialization"""
        self.assertIsNotNone(self.backtester)
        self.assertEqual(self.backtester.market_data_loader, self.loader)

    def test_simple_backtest(self):
        """Test basic backtest with fixed allocation strategy"""
        # Create a strategy with equal allocation
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest without lookahead protection
        results = self.backtester.run_backtest(
            strategy=strategy,
            shift_signals=False,
        )

        # Assert results are non-empty
        self.assertIsNotNone(results)

        # Check that we have the expected keys in results
        expected_keys = ['signals_df', 'tickers_returns', 'strategy_returns']
        for key in expected_keys:
            self.assertIn(key, results)

        # Check that results have correct dates
        self.assertEqual(len(results['signals_df']), len(self.dates))

        # Fix: Compare index values, not the entire index object which includes metadata like name
        pd.testing.assert_index_equal(
            results['signals_df'].index,
            self.dates,
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
            strategy=strategy,
            shift_signals=True
        )

        # Verify shifting - first day should have zeros
        first_day = results['signals_df'].iloc[0]
        for ticker in self.tickers:
            self.assertEqual(first_day[ticker], 0.0)

        # Second day should match the strategy's allocation
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
        self.assertTrue(all(date >= mid_date for date in results_start['signals_df'].index))

        # Test with end_date only
        results_end = self.backtester.run_backtest(
            strategy=strategy,
            end_date=mid_date
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
            end_date=end_date
        )

        # Should only have dates between start_date and end_date
        self.assertEqual(len(results_both['signals_df']), 6)  # 6 days in range
        self.assertTrue(all(start_date <= date <= end_date
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
        staggered_backtester = Backtester(staggered_loader)

        # Create strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Test without require_all_history
        results_without = staggered_backtester.run_backtest(
            strategy=strategy,
            require_all_history=False
        )

        # Should have all dates (will have NaN or 0 allocations for missing tickers)
        self.assertEqual(len(results_without['signals_df']), len(self.dates))

        # Test with require_all_history
        results_with = staggered_backtester.run_backtest(
            strategy=strategy,
            require_all_history=True
        )

        # Should only start when all tickers have data (day 5)
        self.assertEqual(len(results_with['signals_df']), len(self.dates[4:]))
        self.assertEqual(results_with['signals_df'].index[0], self.dates[4])

    def test_dynamic_strategy(self):
        """Test backtest with a dynamic strategy that changes allocations"""
        # Create dynamic strategy
        strategy = DynamicTestStrategy(tickers=self.tickers)

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            shift_signals=True
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
            strategy=strategy,
            shift_signals=True
        )

        # Verify we get results
        self.assertIsNotNone(results)

        # Check that price data has been forward-filled in the results
        price_df = results['tickers_returns'].copy()
        self.assertFalse(price_df.isnull().any().any())

        # Specifically check GOOG returns on the NaN days
        # (should be 0.0 since we're forward-filling prices)
        goog_returns = results['tickers_returns']['GOOG']
        self.assertEqual(goog_returns.iloc[6], 0.0)  # Day 7 (index 6)

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
        # Invalid benchmark type
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

if __name__ == '__main__':
    unittest.main()