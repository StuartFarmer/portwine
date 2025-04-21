import unittest
import pandas as pd
import numpy as np
from portwine.backtester import Backtester, InvalidBenchmarkError
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
        return {t: self.mock_data[t] for t in tickers if t in self.mock_data}

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

        # Initialize loaders and backtesters
        self.market_loader = MockMarketDataLoader(self.regular_data)
        self.alt_loader = MockAlternativeDataLoader(self.alt_data)
        self.backtester = Backtester(
            market_data_loader=self.market_loader,
            alternative_data_loader=self.alt_loader
        )
        self.market_only_backtester = Backtester(
            market_data_loader=self.market_loader
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
        self.assertIs(self.backtester.market_data_loader, self.market_loader)
        self.assertIs(self.backtester.alternative_data_loader, self.alt_loader)

    def test_ticker_parsing(self):
        regular, alternative = self.backtester._split_tickers(
            self.regular_tickers + self.alt_tickers
        )
        self.assertEqual(set(regular), set(self.regular_tickers))
        self.assertEqual(set(alternative), set(self.alt_tickers))

    def test_mixed_data_strategy(self):
        strat = MixedDataStrategy(self.regular_tickers, self.alt_tickers)
        results = self.backtester.run_backtest(
            strategy=strat,
            shift_signals=False
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
                benchmark=self.alt_tickers[0],
                shift_signals=False
            )

    def test_without_alt_loader(self):
        strat = MixedDataStrategy(self.regular_tickers, self.alt_tickers)
        results = self.market_only_backtester.run_backtest(
            strategy=strat,
            shift_signals=False
        )
        self.assertIsNotNone(results)
        # Ensure no alt data passed to strategy
        for _, daily in strat.step_calls:
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
        self.backtester = Backtester(
            market_data_loader=self.market_loader,
            alternative_data_loader=self.alt_loader
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
            strategy=strat,
            shift_signals=False,
            verbose=False
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
            strategy=strat,
            shift_signals=False,
            verbose=False
        )
        idx = results['signals_df'].index
        self.assertEqual(len(idx), len(self.market_dates))
        pd.testing.assert_index_equal(idx, self.market_dates, check_names=False)

    def test_no_market_data(self):
        strat = TestStrategy([], ['ECON:GDP', 'ECON:CPI'])
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(strategy=strat, verbose=False)

    def test_with_date_filtering(self):
        strat = TestStrategy(['AAPL', 'MSFT'], ['ECON:GDP', 'ECON:CPI'])
        start = self.market_dates[3]
        end   = self.market_dates[7]
        results = self.backtester.run_backtest(
            strategy=strat,
            start_date=start,
            end_date=end,
            shift_signals=False,
            verbose=False
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
        self.bt = Backtester(self.loader)

        # Strategy universe includes one regular and one alt ticker
        self.strategy = TestStrategy(regular_tickers=["A"], alt_tickers=["ALT:X"])

    def test_alt_tickers_not_in_outputs(self):
        res = self.bt.run_backtest(self.strategy)

        # signals_df and tickers_returns should only have 'A'
        self.assertListEqual(list(res["signals_df"].columns), ["A"])
        self.assertListEqual(list(res["tickers_returns"].columns), ["A"])

        # strategy_returns should be an unnamed Series
        self.assertIsNone(res["strategy_returns"].name)


if __name__ == '__main__':
    unittest.main()
