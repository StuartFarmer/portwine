import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import shutil

# Import components to be tested
from portwine.backtester.core import Backtester
from portwine.strategies.base import StrategyBase
from portwine.loaders import MarketDataLoader
from portwine.analyzers.equitydrawdown import EquityDrawdownAnalyzer
from portwine.analyzers.correlation import CorrelationAnalyzer
from portwine.data.interface import DataInterface
from tests.calendar_utils import TestDailyMarketCalendar
from tests.test_backtester import MockDataInterface

## Use shared, configurable test calendar


class DiskBasedMarketDataLoader(MarketDataLoader):
    """A real loader that saves/loads data from disk for integration testing"""

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        # Create directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)

    def load_ticker(self, ticker):
        """Load ticker data from CSV file"""
        file_path = os.path.join(self.data_path, f"{ticker}.csv")

        if not os.path.exists(file_path):
            return None

        try:
            # Read the CSV without specifying parse_dates or index_col first
            df = pd.read_csv(file_path)

            # Check if 'date' column exists
            if 'date' in df.columns:
                # Convert 'date' column to datetime
                df['date'] = pd.to_datetime(df['date'])
                # Set it as index
                df = df.set_index('date')
            else:
                # If no 'date' column exists, we can't properly load this file
                print(f"Error: No 'date' column found in file for {ticker}")
                return None

            return df
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None

    def save_ticker_data(self, ticker, data):
        """Save ticker data to CSV file (for test setup)"""
        file_path = os.path.join(self.data_path, f"{ticker}.csv")

        # Make sure the index is included as a column called 'date'
        data_to_save = data.copy()
        data_to_save = data_to_save.reset_index()  # Convert index to column

        # Ensure the column is named 'date'
        if data_to_save.columns[0] != 'date':
            data_to_save = data_to_save.rename(columns={data_to_save.columns[0]: 'date'})

        data_to_save.to_csv(file_path, index=False)

class VolatilityStrategy(StrategyBase):
    """Strategy that allocates inversely to volatility"""

    def __init__(self, tickers, lookback=20):
        super().__init__(tickers)
        self.lookback = lookback
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []

    def step(self, current_date, daily_data):
        """Allocate inversely to volatility"""
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

        # Calculate volatility once we have enough history
        if len(self.dates) >= self.lookback:
            volatilities = {}
            for ticker in self.tickers:
                prices = self.price_history[ticker][-self.lookback:]
                if None not in prices and 0 not in prices:
                    # Calculate returns
                    returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
                    if returns:
                        volatilities[ticker] = np.std(returns)
                    else:
                        volatilities[ticker] = 1.0  # Fallback
                else:
                    volatilities[ticker] = 1.0  # Fallback

            # Inverse volatility weighting
            if all(vol == 0 for vol in volatilities.values()):
                # If all volatilities are zero, equal weight
                return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}

            inv_vol = {ticker: 1.0 / vol if vol > 0 else 0.0 for ticker, vol in volatilities.items()}
            total_inv_vol = sum(inv_vol.values())

            if total_inv_vol > 0:
                # Normalize weights
                return {ticker: weight / total_inv_vol for ticker, weight in inv_vol.items()}

        # Equal weight until we have enough history
        return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}

class AltBasedStrategy(StrategyBase):
    """Uses one alt‐data series to set weights on regular tickers."""
    def __init__(self, regular: list[str], alt: str):
        super().__init__(regular + [alt])
        self.regular = regular
        self.alt = alt

    def step(self, ts, bar_data):
        val = bar_data[self.alt]['close']
        weight = 1.0 if val > 0 else 0.0
        return {t: weight for t in self.regular}

class TestBacktesterIntegration(unittest.TestCase):
    """Integration tests for Backtester with real components"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

        # Sample date range for testing
        self.dates = pd.date_range(start='2020-01-01', periods=30)

        # Create sample price data for multiple tickers
        self.tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

        # Generate data with different characteristics
        self.generate_test_data()

        # Create real disk-based loader
        self.loader = DiskBasedMarketDataLoader(self.test_dir)

        # Save test data to disk
        for ticker, data in self.price_data.items():
            self.loader.save_ticker_data(ticker, data)

        # Create proper mock data interface with the test data
        mock_data = {}
        for ticker, data in self.price_data.items():
            mock_data[ticker] = data
        
        self.data_interface = MockDataInterface(mock_data=mock_data)

        # Create backtester using shared test calendar (all days, midnight timestamps)
        self.backtester = Backtester(
            self.data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="all",
                default_start="2020-01-01",
                default_end="2020-01-30",
                default_hour=None,
            ),
        )

    def tearDown(self):
        """Clean up after test"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def generate_test_data(self):
        """Generate realistic test data for multiple tickers"""
        self.price_data = {}

        # Helper to generate a price series with specific characteristics
        def generate_prices(start_price, volatility, trend=0.0, gap_indices=None):
            gap_indices = gap_indices or []
            prices = [start_price]

            for i in range(1, len(self.dates)):
                if i in gap_indices:
                    prices.append(np.nan)
                else:
                    # Random walk with trend
                    change = np.random.normal(trend, volatility)
                    new_price = prices[-1] * (1 + change)
                    prices.append(new_price)

            return prices

        # Generate data for each ticker
        self.price_data['AAPL'] = pd.DataFrame({
            'close': generate_prices(150.0, 0.01, trend=0.002),  # Upward trend, low vol
            'open': generate_prices(150.0, 0.01, trend=0.002),
            'high': generate_prices(152.0, 0.01, trend=0.002),
            'low': generate_prices(148.0, 0.01, trend=0.002),
            'volume': np.random.randint(5000000, 10000000, len(self.dates))
        }, index=self.dates)

        self.price_data['MSFT'] = pd.DataFrame({
            'close': generate_prices(200.0, 0.008, trend=0.001),  # Slight upward trend, very low vol
            'open': generate_prices(200.0, 0.008, trend=0.001),
            'high': generate_prices(202.0, 0.008, trend=0.001),
            'low': generate_prices(198.0, 0.008, trend=0.001),
            'volume': np.random.randint(4000000, 8000000, len(self.dates))
        }, index=self.dates)

        self.price_data['GOOG'] = pd.DataFrame({
            'close': generate_prices(1200.0, 0.015, trend=0.0, gap_indices=[10, 11]),
            # Flat trend, medium vol, with gaps
            'open': generate_prices(1200.0, 0.015, trend=0.0, gap_indices=[10, 11]),
            'high': generate_prices(1210.0, 0.015, trend=0.0, gap_indices=[10, 11]),
            'low': generate_prices(1190.0, 0.015, trend=0.0, gap_indices=[10, 11]),
            'volume': np.random.randint(2000000, 5000000, len(self.dates))
        }, index=self.dates)

        self.price_data['AMZN'] = pd.DataFrame({
            'close': generate_prices(1800.0, 0.02, trend=-0.001),  # Slight downward trend, high vol
            'open': generate_prices(1800.0, 0.02, trend=-0.001),
            'high': generate_prices(1820.0, 0.02, trend=-0.001),
            'low': generate_prices(1780.0, 0.02, trend=-0.001),
            'volume': np.random.randint(3000000, 7000000, len(self.dates))
        }, index=self.dates)

        self.price_data['META'] = pd.DataFrame({
            'close': generate_prices(250.0, 0.025, trend=0.003),  # Strong upward trend, very high vol
            'open': generate_prices(250.0, 0.025, trend=0.003),
            'high': generate_prices(255.0, 0.025, trend=0.003),
            'low': generate_prices(245.0, 0.025, trend=0.003),
            'volume': np.random.randint(6000000, 12000000, len(self.dates))
        }, index=self.dates)

        # SPY benchmark
        self.price_data['SPY'] = pd.DataFrame({
            'close': generate_prices(300.0, 0.005, trend=0.0005),  # Market benchmark - low vol, slight uptrend
            'open': generate_prices(300.0, 0.005, trend=0.0005),
            'high': generate_prices(302.0, 0.005, trend=0.0005),
            'low': generate_prices(298.0, 0.005, trend=0.0005),
            'volume': np.random.randint(10000000, 20000000, len(self.dates))
        }, index=self.dates)

    def test_full_backtest_workflow(self):
        """Test complete backtesting workflow with analyzers"""
        # Create volatility strategy
        strategy = VolatilityStrategy(tickers=self.tickers, lookback=10)

        # Define a simple benchmark function
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
            end_date='2020-01-30',
            benchmark=spy_benchmark
        )

        # Verify we get results
        self.assertIsNotNone(results)

        # Verify keys in results
        expected_keys = ['signals_df', 'tickers_returns', 'strategy_returns', 'benchmark_returns']
        for key in expected_keys:
            self.assertIn(key, results)

        # Run equity drawdown analyzer
        analyzer = EquityDrawdownAnalyzer()
        analysis = analyzer.analyze(results)

        # Verify analysis results
        self.assertIsNotNone(analysis)
        self.assertIn('strategy_stats', analysis)
        self.assertIn('benchmark_stats', analysis)

        # Check strategy stats keys
        expected_stat_keys = ['TotalReturn', 'CAGR', 'AnnualVol', 'Sharpe', 'MaxDrawdown']
        for key in expected_stat_keys:
            self.assertIn(key, analysis['strategy_stats'])

        # Correlation analyzer
        corr_analyzer = CorrelationAnalyzer()
        corr_analysis = corr_analyzer.analyze(results)

        # Verify correlation results
        self.assertIsNotNone(corr_analysis)
        self.assertIn('correlation_matrix', corr_analysis)

        # Check correlation matrix shape
        corr_matrix = corr_analysis['correlation_matrix']
        self.assertEqual(corr_matrix.shape, (len(self.tickers), len(self.tickers)))

    def test_date_filtering_integration(self):
        """Test date filtering in a full integration test"""
        strategy = VolatilityStrategy(tickers=self.tickers, lookback=5)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # Run with first half of dates
        half_point = self.dates[len(self.dates) // 2]
        results_first_half = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date=half_point,
            benchmark=equal_weight_benchmark
        )

        # Check if results are None (which can happen if data loading failed)
        if results_first_half is None:
            self.fail("Failed to get results for first half test - data loading issue")

        # Run with second half of dates
        results_second_half = self.backtester.run_backtest(
            strategy=strategy,
            start_date=half_point,
            end_date='2020-01-30',
            benchmark=equal_weight_benchmark
        )

        # Check if results are None
        if results_second_half is None:
            self.fail("Failed to get results for second half test - data loading issue")

        # Verify date ranges
        self.assertLessEqual(results_first_half['signals_df'].index.max(), half_point)
        self.assertGreaterEqual(results_second_half['signals_df'].index.min(), half_point)

        # Run full backtest
        results_full = self.backtester.run_backtest(
            strategy=strategy,
            start_date='2020-01-01',
            end_date='2020-01-30',
            benchmark=equal_weight_benchmark
        )

        # Check if results are None
        if results_full is None:
            self.fail("Failed to get results for full test - data loading issue")

        # Number of dates in split tests should add up to full test minus 1 (overlap at half_point)
        expected_total = len(results_first_half['signals_df']) + len(results_second_half['signals_df']) - 1
        self.assertEqual(len(results_full['signals_df']), expected_total)

    def test_alternative_data_influences_regular_signals(self):
        # 1) Mock loader that returns self.price_data for underlying tickers
        class MockSourceLoader(MarketDataLoader):
            SOURCE_IDENTIFIER = 'MOCK'
            def __init__(self, data):
                super().__init__()
                self.data = data
            def load_ticker(self, ticker):
                return self.data.get(ticker)
            def fetch_data(self, tickers):
                result = {}
                for t in tickers:
                    if t not in self._data_cache and t in self.data:
                        df = self.data[t].copy()
                        self._data_cache[t] = df
                        # OPTIMIZATION: Create numpy caches for fast access
                        self._create_numpy_cache(t, df)
                    if t in self._data_cache:
                        result[t] = self._data_cache[t]
                return result

        # 2) Create mock data interface for the alternative data test
        mock_data = {}
        for ticker, data in self.price_data.items():
            mock_data[ticker] = data
        
        mock_data_interface = MockDataInterface(mock_data=mock_data)
        
        bt = Backtester(
            mock_data_interface,
            calendar=TestDailyMarketCalendar(
                calendar_name="NYSE",
                mode="all",
                default_start="2020-01-01",
                default_end="2020-01-30",
                default_hour=None,
            ),
        )

        # 3) Pick one ticker as alt, the rest as regular
        alt_ticker = f"MOCK:{self.tickers[0]}"
        regular   = self.tickers[1:]
        strat = AltBasedStrategy(regular, alt_ticker)

        # 4) Run backtest; expect ValueError due to total allocation >1
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        with self.assertRaises(ValueError):
            bt.run_backtest(
                strat, 
                start_date='2020-01-01',
                end_date='2020-01-30',
                benchmark=equal_weight_benchmark
            )


    def test_empty_date_range(self):
        """Test backtester with filtering that results in empty date range"""
        strategy = VolatilityStrategy(tickers=self.tickers)

        # Set start date after end date
        start_date = self.dates[-1] + timedelta(days=1)
        end_date = self.dates[0] - timedelta(days=1)

        # Define a simple benchmark function
        def equal_weight_benchmark(ret_df):
            n_tickers = len(ret_df.columns)
            weights = np.ones(n_tickers) / n_tickers
            return pd.DataFrame(ret_df.dot(weights), columns=['benchmark_returns'])
        
        # This should raise ValueError
        with self.assertRaises(ValueError):
            self.backtester.run_backtest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                benchmark=equal_weight_benchmark
            )

        # Set both dates outside the available range
        future_start = self.dates[-1] + timedelta(days=10000)
        future_end = future_start + timedelta(days=10000)

        # Also raises ValueError
        with self.assertRaises(ValueError):
            # This should return None (no dates in range)
            results = self.backtester.run_backtest(
                strategy=strategy,
                start_date=future_start,
                end_date=future_end,
                benchmark=equal_weight_benchmark
            )

            print(results)


if __name__ == '__main__':
    unittest.main()