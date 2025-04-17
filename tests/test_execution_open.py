import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components to be tested
from portwine.backtester import Backtester
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


class OpenToOpenStrategy(StrategyBase):
    """
    Strategy specifically designed to test open-to-open execution timing.
    Makes decisions based on the open price relative to the previous open.
    """

    def __init__(self, tickers):
        super().__init__(tickers)
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []

    def step(self, current_date, daily_data):
        """Allocate based on today's open vs yesterday's open"""
        self.dates.append(current_date)

        # Initialize allocations
        allocations = {ticker: 0.0 for ticker in self.tickers}

        # Update price history
        for ticker in self.tickers:
            if daily_data.get(ticker) is None:
                continue

            # Store today's open price
            open_price = daily_data[ticker].get('open')
            self.price_history[ticker].append(open_price)

            # Need at least two days of data to make a decision
            if len(self.price_history[ticker]) >= 2:
                # Get yesterday's and today's open prices
                yesterday_open = self.price_history[ticker][-2]
                today_open = self.price_history[ticker][-1]

                # Skip if data is missing or invalid
                if yesterday_open is None or today_open is None or yesterday_open <= 0:
                    continue

                # Calculate open-to-open return
                open_to_open_return = (today_open / yesterday_open) - 1.0

                # Allocate based on return: positive -> long, negative -> short
                if open_to_open_return > 0:
                    allocations[ticker] = 1.0 / len(self.tickers)  # Long
                elif open_to_open_return < 0:
                    allocations[ticker] = -1.0 / len(self.tickers)  # Short

        return allocations


class TestOpenToOpenTiming(unittest.TestCase):
    """Test cases for the open-to-open execution timing"""

    def setUp(self):
        """Set up test environment"""
        # Sample date range for testing
        self.dates = pd.date_range(start='2020-01-01', end='2020-01-10')

        # Create sample price data with specific patterns for testing
        self.tickers = ['AAPL', 'MSFT']
        self.price_data = {}

        # AAPL:
        # - Open prices increase steadily (+2 each day)
        # - Close prices more volatile
        self.price_data['AAPL'] = pd.DataFrame({
            'open': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],  # +2 each day
            'high': [105, 107, 109, 111, 113, 115, 117, 119, 121, 123],  # +5 from open
            'low': [98, 100, 102, 104, 106, 108, 110, 112, 114, 116],  # -2 from open
            'close': [103, 105, 107, 109, 111, 113, 115, 117, 119, 121],  # +3 from open
            'volume': [1000000] * 10
        }, index=self.dates)

        # MSFT:
        # - Open prices decrease steadily (-2 each day)
        # - Close prices less volatile
        self.price_data['MSFT'] = pd.DataFrame({
            'open': [100, 98, 96, 94, 92, 90, 88, 86, 84, 82],  # -2 each day
            'high': [102, 100, 98, 96, 94, 92, 90, 88, 86, 84],  # +2 from open
            'low': [99, 97, 95, 93, 91, 89, 87, 85, 83, 81],  # -1 from open
            'close': [101, 99, 97, 95, 93, 91, 89, 87, 85, 83],  # +1 from open
            'volume': [1000000] * 10
        }, index=self.dates)

        # Create mock loader with sample data
        self.loader = MockMarketDataLoader()
        for ticker, data in self.price_data.items():
            self.loader.set_data(ticker, data)

        # Create backtester
        self.backtester = Backtester(self.loader)

    def test_open_to_open_timing(self):
        """Test open-to-open execution timing"""
        # Create a simple equal-weight strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtest with open-to-open timing
        results = self.backtester.run_backtest(
            strategy=strategy,
            shift_signals=False,
            execution_timing='open-to-open'
        )

        # Verify execution_timing in results
        self.assertEqual(results['execution_timing'], 'open-to-open')

        # Verify returns calculation
        # AAPL open prices: [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        # MSFT open prices: [100, 98, 96, 94, 92, 90, 88, 86, 84, 82]

        # Expected returns for each ticker (day 2 onwards since day 1 has no previous open)
        expected_aapl_returns = [0.0] + [(self.price_data['AAPL']['open'].iloc[i] /
                                          self.price_data['AAPL']['open'].iloc[i - 1] - 1)
                                         for i in range(1, len(self.dates))]

        expected_msft_returns = [0.0] + [(self.price_data['MSFT']['open'].iloc[i] /
                                          self.price_data['MSFT']['open'].iloc[i - 1] - 1)
                                         for i in range(1, len(self.dates))]

        # Verify each day's returns
        for i in range(len(self.dates)):
            # Allow for small floating-point differences
            self.assertAlmostEqual(results['tickers_returns']['AAPL'].iloc[i],
                                   expected_aapl_returns[i], places=6)
            self.assertAlmostEqual(results['tickers_returns']['MSFT'].iloc[i],
                                   expected_msft_returns[i], places=6)

    def test_open_to_open_strategy(self):
        """Test a strategy specifically designed for open-to-open trading"""
        # Create a strategy based on open-to-open price movements
        strategy = OpenToOpenStrategy(tickers=self.tickers)

        # Run backtest with open-to-open timing
        results = self.backtester.run_backtest(
            strategy=strategy,
            execution_timing='open-to-open',
            shift_signals=False
        )

        # Verify results
        signals_df = results['signals_df']

        # AAPL open prices increase each day, so strategy should be long AAPL
        # MSFT open prices decrease each day, so strategy should be short MSFT

        # Skip first day (no previous open to compare)
        for i in range(1, len(signals_df)):
            # Check AAPL signals - should be positive (long) after day 1
            if abs(signals_df['AAPL'].iloc[i]) > 1e-6:  # Not zero
                self.assertGreater(signals_df['AAPL'].iloc[i], 0)

            # Check MSFT signals - should be negative (short) after day 1
            if abs(signals_df['MSFT'].iloc[i]) > 1e-6:  # Not zero
                self.assertLess(signals_df['MSFT'].iloc[i], 0)

    def test_benchmark_open_to_open(self):
        """Test benchmark with open-to-open execution timing"""
        # Create simple strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Add benchmark data with specific pattern
        benchmark_ticker = 'SPY'
        benchmark_data = pd.DataFrame({
            'open': [300, 303, 306, 309, 312, 315, 318, 321, 324, 327],  # +3 each day
            'high': [305, 308, 311, 314, 317, 320, 323, 326, 329, 332],  # +5 from open
            'low': [298, 301, 304, 307, 310, 313, 316, 319, 322, 325],  # -2 from open
            'close': [302, 305, 308, 311, 314, 317, 320, 323, 326, 329],  # +2 from open
            'volume': [5000000] * 10
        }, index=self.dates)

        # Add benchmark to loader
        self.loader.set_data(benchmark_ticker, benchmark_data)

        # Run backtest with open-to-open timing
        results = self.backtester.run_backtest(
            strategy=strategy,
            benchmark=benchmark_ticker,
            execution_timing='open-to-open',
            shift_signals=False
        )

        # Verify benchmark returns
        benchmark_returns = results['benchmark_returns']

        # Expected benchmark returns (open-to-open)
        expected_returns = [0.0]  # First day has no previous open
        for i in range(1, len(self.dates)):
            expected_returns.append(
                (benchmark_data['open'].iloc[i] / benchmark_data['open'].iloc[i - 1] - 1)
            )

        # Verify each day's benchmark return
        for i in range(len(self.dates)):
            self.assertAlmostEqual(
                benchmark_returns.iloc[i],
                expected_returns[i],
                places=6
            )

    def test_comparison_all_timings(self):
        """Compare all four execution timing methods"""
        # Create simple strategy
        strategy = SimpleTestStrategy(tickers=self.tickers)

        # Run backtests with all four timing methods
        timing_options = ['close-to-close', 'open-to-close', 'close-to-open', 'open-to-open']
        timing_results = {}

        for timing in timing_options:
            results = self.backtester.run_backtest(
                strategy=strategy,
                execution_timing=timing,
                shift_signals=False
            )
            timing_results[timing] = results

        # Verify we have results for all timing options
        self.assertEqual(len(timing_results), 4)

        # For AAPL:
        # - Open to Open: Steady 2% gains
        # - Close to Close: Steady 2% gains (equal to open-to-open in our data)
        # - Open to Close: Open at 100, close at 103 (3% gain)
        # - Close to Open: Close at 103, next open at 102 (-1% loss overnight)

        # For MSFT:
        # - Open to Open: Steady 2% losses
        # - Close to Close: Steady 2% losses (equal to open-to-open in our data)
        # - Open to Close: Open at 100, close at 101 (1% gain)
        # - Close to Open: Close at 101, next open at 98 (-3% loss overnight)

        # Calculate mean returns for each timing method
        mean_returns = {}
        for timing, results in timing_results.items():
            mean_returns[timing] = {
                'AAPL': results['tickers_returns']['AAPL'].mean(),
                'MSFT': results['tickers_returns']['MSFT'].mean()
            }

        # Verify pattern for AAPL
        # Open-to-close should be positive
        self.assertGreater(mean_returns['open-to-close']['AAPL'], 0)

        # Verify pattern for MSFT
        # Open-to-open should be negative
        self.assertLess(mean_returns['open-to-open']['MSFT'], 0)

        # Check relationships between different timing methods for MSFT
        # Our test data is designed so that:
        # - MSFT consistently loses value from open to open (-2% each day)
        # - but gains value within each day from open to close (+1% each day)
        self.assertGreater(mean_returns['open-to-close']['MSFT'], mean_returns['open-to-open']['MSFT'])

    def test_special_cases(self):
        """Test edge cases for open-to-open timing"""
        # Create data with missing values
        missing_data = pd.DataFrame({
            'open': [100, np.nan, 104, 106, np.nan, 110, 112, np.nan, 116, 118],
            'high': [105, np.nan, 109, 111, np.nan, 115, 117, np.nan, 121, 123],
            'low': [98, np.nan, 102, 104, np.nan, 108, 110, np.nan, 114, 116],
            'close': [103, np.nan, 107, 109, np.nan, 113, 115, np.nan, 119, 121],
            'volume': [1000000] * 10
        }, index=self.dates)

        # Set missing data for a test ticker
        test_ticker = 'TEST'
        self.loader.set_data(test_ticker, missing_data)

        # Create strategy with the test ticker
        strategy = SimpleTestStrategy(tickers=[test_ticker])

        # Run backtest with open-to-open timing
        results = self.backtester.run_backtest(
            strategy=strategy,
            execution_timing='open-to-open',
            shift_signals=False
        )

        # Verify that missing values are handled properly (filled with 0)
        ticker_returns = results['tickers_returns'][test_ticker]

        # Calculate expected returns with proper handling of NaN values
        expected_returns = [0.0]  # First day always 0
        for i in range(1, len(self.dates)):
            if np.isnan(missing_data['open'].iloc[i]) or np.isnan(missing_data['open'].iloc[i - 1]):
                # NaN in current or previous open should result in 0 return
                expected_returns.append(0.0)
            else:
                expected_returns.append(
                    (missing_data['open'].iloc[i] / missing_data['open'].iloc[i - 1] - 1)
                )

        # Verify each day's return
        for i in range(len(self.dates)):
            self.assertAlmostEqual(ticker_returns.iloc[i], expected_returns[i], places=6)

    def test_zero_price_handling(self):
        """Test handling of zero prices in open-to-open timing"""
        # Create data with zero prices
        zero_price_data = pd.DataFrame({
            'open': [100, 0, 104, 0, 108, 0, 112, 0, 116, 0],
            'high': [105, 0, 109, 0, 113, 0, 117, 0, 121, 0],
            'low': [98, 0, 102, 0, 106, 0, 110, 0, 114, 0],
            'close': [103, 0, 107, 0, 111, 0, 115, 0, 119, 0],
            'volume': [1000000] * 10
        }, index=self.dates)

        # Set zero price data for a test ticker
        test_ticker = 'ZERO'
        self.loader.set_data(test_ticker, zero_price_data)

        # Create strategy with the test ticker
        strategy = SimpleTestStrategy(tickers=[test_ticker])

        # Run backtest with open-to-open timing
        results = self.backtester.run_backtest(
            strategy=strategy,
            execution_timing='open-to-open',
            shift_signals=False
        )

        # Verify that zero prices are handled properly
        ticker_returns = results['tickers_returns'][test_ticker]

        # Second day should have return of 0 due to current 0 price
        self.assertEqual(ticker_returns.iloc[1], 0.0)

        # Third day would normally have very large return (104/0), but should be 0
        self.assertEqual(ticker_returns.iloc[2], 0.0)

    def test_complex_scenario(self):
        """Test a complex scenario with multiple tickers and varying patterns"""
        # Create strategy with both AAPL and MSFT
        strategy = OpenToOpenStrategy(tickers=self.tickers)

        # Run backtest with each timing method
        timing_options = ['close-to-close', 'open-to-close', 'close-to-open', 'open-to-open']
        strategy_returns = {}

        for timing in timing_options:
            results = self.backtester.run_backtest(
                strategy=strategy,
                execution_timing=timing,
                shift_signals=False
            )
            strategy_returns[timing] = results['strategy_returns']

        # Calculate cumulative returns for each timing method
        cumulative_returns = {}
        for timing, returns in strategy_returns.items():
            cumulative_returns[timing] = (1 + returns).prod() - 1

        # In this scenario, the open-to-open strategy should perform best with open-to-open timing
        # since the strategy is specifically designed for that timing method
        self.assertGreater(cumulative_returns['open-to-open'], cumulative_returns['close-to-close'])
        self.assertGreater(cumulative_returns['open-to-open'], cumulative_returns['open-to-close'])


if __name__ == '__main__':
    unittest.main()