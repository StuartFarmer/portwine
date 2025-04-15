import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components to be tested
from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class MockMarketDataLoader(MarketDataLoader):
    """Mock regular market data loader for testing"""

    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}

    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        return self.mock_data.get(ticker)

    def fetch_data(self, tickers):
        """Fetch data for multiple tickers"""
        result = {}
        for ticker in tickers:
            data = self.load_ticker(ticker)
            if data is not None:
                result[ticker] = data
        return result


class MockAlternativeDataLoader:
    """Mock alternative data loader for testing"""

    def __init__(self, mock_data=None):
        self.mock_data = mock_data or {}

    def load_ticker(self, ticker):
        """Return pre-defined mock data for a ticker"""
        # Assume ticker is already in SOURCE:TICKER format
        return self.mock_data.get(ticker)

    def fetch_data(self, tickers):
        """Fetch data for multiple tickers"""
        result = {}
        for ticker in tickers:
            data = self.load_ticker(ticker)
            if data is not None:
                result[ticker] = data
        return result


class MixedDataStrategy(StrategyBase):
    """Strategy that uses both regular and alternative data"""

    def __init__(self, regular_tickers, alt_tickers):
        """Initialize with separate regular and alternative tickers"""
        combined_tickers = list(regular_tickers) + list(alt_tickers)
        super().__init__(combined_tickers)

        self.regular_tickers = regular_tickers
        self.alt_tickers = alt_tickers
        self.step_calls = []  # Track step calls for testing

    def step(self, current_date, daily_data):
        """Return allocation based on available data"""
        # Record the call for test verification
        self.step_calls.append((current_date, daily_data))

        # Total number of tickers with data
        tickers_with_data = [t for t in self.tickers if daily_data.get(t) is not None]
        n_with_data = len(tickers_with_data)

        if n_with_data == 0:
            return {}

        # Equal weight allocation to all tickers with data
        allocation = {ticker: 1.0 / n_with_data if ticker in tickers_with_data else 0.0
                      for ticker in self.tickers}

        return allocation


class TestBacktesterWithAltData(unittest.TestCase):
    """Test cases for enhanced Backtester with alternative data support"""

    def setUp(self):
        """Set up test environment"""
        # Sample date range for testing
        self.dates = pd.date_range(start='2020-01-01', end='2020-01-10')

        # Create sample price data for multiple tickers
        self.regular_tickers = ['AAPL', 'MSFT']
        self.alt_tickers = ['FRED:GDP', 'FRED:FEDFUNDS', 'BARCHARTINDEX:ADDA']

        # Generate sample data
        self.regular_data = self._create_price_data(self.regular_tickers)
        self.alt_data = self._create_alt_data(self.alt_tickers)

        # Create mock loaders
        self.market_loader = MockMarketDataLoader(self.regular_data)
        self.alt_loader = MockAlternativeDataLoader(self.alt_data)

        # Create backtester with both loaders
        self.backtester = Backtester(
            market_data_loader=self.market_loader,
            alternative_data_loader=self.alt_loader
        )

        # Create backtester with only market loader for comparison
        self.market_only_backtester = Backtester(
            market_data_loader=self.market_loader
        )

    def _create_price_data(self, tickers):
        """Create sample price data for regular tickers"""
        data = {}

        for i, ticker in enumerate(tickers):
            # Create data with different starting values for each ticker
            base_price = 100.0 * (i + 1)

            # Simple upward trend
            prices = [base_price + j for j in range(len(self.dates))]

            data[ticker] = pd.DataFrame({
                'open': prices,
                'high': [p + 2 for p in prices],
                'low': [p - 2 for p in prices],
                'close': prices,
                'volume': [1000000] * len(self.dates)
            }, index=self.dates)

        return data

    def _create_alt_data(self, tickers):
        """Create sample alternative data"""
        data = {}

        for i, ticker in enumerate(tickers):
            # Alternative data - could be rates, economic indicators, etc.
            base_value = 10.0 * (i + 1)

            # Flat or slightly varying values
            values = [base_value + np.random.normal(0, 0.1) for _ in range(len(self.dates))]

            data[ticker] = pd.DataFrame({
                'open': values,
                'high': values,
                'low': values,
                'close': values,
                'volume': [0] * len(self.dates)
            }, index=self.dates)

        return data

    def test_initialization(self):
        """Test proper initialization with alternative data loader"""
        self.assertEqual(self.backtester.market_data_loader, self.market_loader)
        self.assertEqual(self.backtester.alternative_data_loader, self.alt_loader)

    def test_ticker_parsing(self):
        """Test separation of regular and alternative tickers"""
        regular, alternative = self.backtester._parse_tickers(
            self.regular_tickers + self.alt_tickers
        )

        self.assertEqual(set(regular), set(self.regular_tickers))
        self.assertEqual(set(alternative), set(self.alt_tickers))

    def test_mixed_data_strategy(self):
        """Test backtest with strategy using both regular and alternative data"""
        strategy = MixedDataStrategy(
            regular_tickers=self.regular_tickers,
            alt_tickers=self.alt_tickers
        )

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            shift_signals=False
        )

        # Verify results
        self.assertIsNotNone(results)

        # Check that signals dataframe contains all tickers
        # all_tickers = self.regular_tickers + self.alt_tickers
        for ticker in self.regular_tickers:
            self.assertIn(ticker, results['signals_df'].columns)

        for ticker in self.alt_tickers:
            self.assertNotIn(ticker, results['signals_df'].columns)

        # Check that we have data for all dates
        self.assertEqual(len(results['signals_df']), len(self.dates))

        # Verify strategy was called once for each date
        self.assertEqual(len(strategy.step_calls), len(self.dates))

        # Verify each call to step included both regular and alternative data
        for date, daily_data in strategy.step_calls:
            # Check for regular data
            for ticker in self.regular_tickers:
                self.assertIsNotNone(daily_data.get(ticker))

            # Check for alternative data
            for ticker in self.alt_tickers:
                self.assertIsNotNone(daily_data.get(ticker))

    def test_alt_data_benchmark(self):
        """Test using alternative data as benchmark"""
        strategy = MixedDataStrategy(
            regular_tickers=self.regular_tickers,
            alt_tickers=[]  # No alt data in strategy, just using for benchmark
        )

        # Run backtest with alternative data benchmark
        benchmark_ticker = self.alt_tickers[0]  # Use FRED:GDP as benchmark
        results = self.backtester.run_backtest(
            strategy=strategy,
            benchmark=benchmark_ticker,
            shift_signals=False
        )

        # Verify benchmark results
        self.assertIsNotNone(results)
        self.assertIn('benchmark_returns', results)
        self.assertEqual(len(results['benchmark_returns']), len(self.dates))

    def test_without_alt_loader(self):
        """Test behavior when alternative tickers are used without alt loader"""
        strategy = MixedDataStrategy(
            regular_tickers=self.regular_tickers,
            alt_tickers=self.alt_tickers
        )

        # Run backtest with market-only backtester
        results = self.market_only_backtester.run_backtest(
            strategy=strategy,
            shift_signals=False
        )

        # Should still work but only have data for regular tickers
        self.assertIsNotNone(results)

        # Strategy returns should only reflect regular tickers
        # Check step calls to verify alt data was missing
        any_alt_data_present = False
        for _, daily_data in strategy.step_calls:
            for alt_ticker in self.alt_tickers:
                if daily_data.get(alt_ticker) is not None:
                    any_alt_data_present = True
                    break

        self.assertFalse(any_alt_data_present, "Alt data should be None without alt loader")


if __name__ == '__main__':
    unittest.main()