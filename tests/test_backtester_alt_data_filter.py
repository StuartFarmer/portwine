import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components to be tested
from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


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
        return self.mock_data.get(ticker)

    def fetch_data(self, tickers):
        """Fetch data for multiple tickers"""
        result = {}
        for ticker in tickers:
            data = self.load_ticker(ticker)
            if data is not None:
                result[ticker] = data
        return result


class TestStrategy(StrategyBase):
    """Strategy that uses both regular and alternative data tickers"""

    def __init__(self, regular_tickers, alt_tickers):
        # Combine all tickers
        combined_tickers = list(regular_tickers) + list(alt_tickers)
        super().__init__(combined_tickers)

        # Keep track of calls for testing
        self.step_calls = []
        self.dates_seen = []

    def step(self, current_date, daily_data):
        """Record the call and return equal allocation to regular tickers"""
        self.step_calls.append((current_date, daily_data))
        self.dates_seen.append(current_date)

        # Create signals dict - only allocate to regular tickers
        regular_tickers = [t for t in self.tickers if ":" not in t]
        if not regular_tickers:
            return {}

        n = len(regular_tickers)
        weight = 1.0 / n if n > 0 else 0.0

        signals = {}
        for ticker in self.tickers:
            if ":" not in ticker:  # Regular ticker gets allocation
                signals[ticker] = weight
            else:  # Alternative ticker gets zero
                signals[ticker] = 0.0

        return signals


class TestAltDataDateFiltering(unittest.TestCase):
    """Test proper date filtering with alternative data"""

    def setUp(self):
        """Set up test data with different date ranges"""
        # Create date ranges
        # Market data: 10 trading days (weekdays)
        self.market_dates = pd.date_range(start='2020-01-01', end='2020-01-14', freq='B')  # 10 business days

        # Alternative data: Monthly data for same period plus extra dates
        # Including weekends and extends beyond market data range
        self.alt_dates_monthly = pd.date_range(start='2019-12-15', end='2020-02-15', freq='MS')  # Monthly data

        # Alternative data: Weekly data that includes non-trading days
        self.alt_dates_weekly = pd.date_range(start='2019-12-15', end='2020-02-15', freq='W')  # Weekly data

        # Create sample data
        self.market_data = {
            'AAPL': self._create_price_data(self.market_dates, 100),
            'MSFT': self._create_price_data(self.market_dates, 200),
            'SPY': self._create_price_data(self.market_dates, 300),  # For benchmark
        }

        self.alt_data = {
            'ECON:GDP': self._create_alt_data(self.alt_dates_monthly, 1000),
            'ECON:CPI': self._create_alt_data(self.alt_dates_monthly, 200),
            'ECON:RATES': self._create_alt_data(self.alt_dates_weekly, 3),
        }

        # Create loaders
        self.market_loader = MockMarketDataLoader(self.market_data)
        self.alt_loader = MockAlternativeDataLoader(self.alt_data)

        # Create backtester
        self.backtester = Backtester(
            market_data_loader=self.market_loader,
            alternative_data_loader=self.alt_loader
        )

    def _create_price_data(self, dates, base_price):
        """Create sample price data for market tickers"""
        prices = [base_price + i for i in range(len(dates))]
        return pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000000] * len(dates)
        }, index=dates)

    def _create_alt_data(self, dates, base_value):
        """Create sample alternative data"""
        values = [base_value + (i * 0.1) for i in range(len(dates))]
        return pd.DataFrame({
            'open': values,
            'high': values,
            'low': values,
            'close': values,
            'volume': [0] * len(dates)
        }, index=dates)

    def test_dates_from_market_data_only(self):
        """Test that backtest dates come only from market data, not alternative data"""
        # Create strategy with both types of tickers
        regular_tickers = ['AAPL', 'MSFT']
        alt_tickers = ['ECON:GDP', 'ECON:CPI', 'ECON:RATES']

        strategy = TestStrategy(regular_tickers, alt_tickers)

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            shift_signals=False,  # Easier to test without shifting
            verbose=False
        )

        # Verify we got results
        self.assertIsNotNone(results)

        # Check if dates in results match market dates
        result_dates = results['signals_df'].index
        self.assertEqual(len(result_dates), len(self.market_dates))

        # Dates should match market dates, not include any alt-only dates
        # Use check_names=False to ignore index name differences
        pd.testing.assert_index_equal(result_dates, self.market_dates, check_names=False)

        # Verify strategy was called exactly once for each market date
        self.assertEqual(len(strategy.step_calls), len(self.market_dates))

        # Ensure strategy dates match market dates
        strategy_dates = strategy.dates_seen
        self.assertEqual(len(strategy_dates), len(self.market_dates))
        for i, date in enumerate(self.market_dates):
            self.assertEqual(strategy_dates[i], date)

        # Verify signals_df only contains regular tickers
        for ticker in regular_tickers:
            self.assertIn(ticker, results['signals_df'].columns)

        # Verify signals_df does NOT contain alternative tickers
        for ticker in alt_tickers:
            self.assertNotIn(ticker, results['signals_df'].columns)

        # Verify tickers_returns only contains regular tickers
        for ticker in regular_tickers:
            self.assertIn(ticker, results['tickers_returns'].columns)

        # Verify tickers_returns does NOT contain alternative tickers
        for ticker in alt_tickers:
            self.assertNotIn(ticker, results['tickers_returns'].columns)


    def test_alternative_data_does_not_affect_trading_calendar(self):
        """
        Alternative‐data tickers in the strategy universe must not
        expand or shrink the backtest’s trading calendar.
        """
        # Strategy includes two regular tickers plus one alt‐data ticker
        regular_tickers = ['AAPL', 'MSFT']
        alt_ticker      = 'ECON:GDP'
        strategy = TestStrategy(regular_tickers, [alt_ticker])

        # Run backtest with no special benchmark
        results = self.backtester.run_backtest(
            strategy=strategy,
            shift_signals=False,
            verbose=False
        )

        # We must have a result
        self.assertIsNotNone(results)

        # signals_df dates must exactly match the market_dates fixture
        result_dates = results['signals_df'].index
        self.assertEqual(len(result_dates), len(self.market_dates))
        pd.testing.assert_index_equal(
            result_dates,
            self.market_dates,
            check_names=False
        )


    def test_no_market_data(self):
        """Test handling when there's only alternative data but no market data"""
        # Create a strategy with only alternative tickers
        alt_tickers = ['ECON:GDP', 'ECON:CPI']
        strategy = TestStrategy([], alt_tickers)

        # Run backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            verbose=False
        )

        # Should fail because there's no market data to determine trading dates
        self.assertIsNone(results)

    def test_with_date_filtering(self):
        """Test date filtering with both market and alternative data"""
        # Create strategy with both types of tickers
        regular_tickers = ['AAPL', 'MSFT']
        alt_tickers = ['ECON:GDP', 'ECON:CPI']
        strategy = TestStrategy(regular_tickers, alt_tickers)

        # Run backtest with date filtering - should still respect market dates
        start_date = self.market_dates[3]  # 4th market date
        end_date = self.market_dates[7]  # 8th market date

        results = self.backtester.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            shift_signals=False,
            verbose=False
        )

        # Verify we got results
        self.assertIsNotNone(results)

        # Check dates are within specified range
        result_dates = results['signals_df'].index
        self.assertTrue(all(start_date <= date <= end_date for date in result_dates))

        # Should have exactly 5 dates (4th through 8th market dates)
        self.assertEqual(len(result_dates), 5)

        # Dates should match corresponding market dates
        expected_dates = self.market_dates[3:8]
        pd.testing.assert_index_equal(result_dates, expected_dates, check_names=False)


if __name__ == '__main__':
    unittest.main()