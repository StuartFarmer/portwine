import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import pytest
from portwine.vectorized import (
    VectorizedBacktester,
    VectorizedStrategyBase,
    create_price_dataframe,
    benchmark_equal_weight
)
from portwine.strategies import StrategyBase


class MockMarketDataLoader:
    def fetch_data(self, tickers):
        # Mock data loader that returns predetermined data
        result = {}
        dates = pd.date_range(start='2020-01-01', periods=10)
        for ticker in tickers:
            # Create a dataframe with dates as index and 'close' column
            result[ticker] = pd.DataFrame(
                {'close': np.linspace(100, 110, len(dates))},
                index=dates
            )
        return result


class MockVectorizedStrategy(VectorizedStrategyBase):
    def batch(self, prices_df):
        # Mock implementation that returns equal weights
        weights = pd.DataFrame(index=prices_df.index, columns=prices_df.columns)
        for col in weights.columns:
            weights[col] = 1.0 / len(weights.columns)
        return weights


class TestVectorizedStrategyBase(unittest.TestCase):
    def setUp(self):
        self.tickers = ['AAPL', 'MSFT', 'GOOG']
        self.strategy = MockVectorizedStrategy(self.tickers)
        self.dates = pd.date_range(start='2020-01-01', periods=10)
        self.prices_df = pd.DataFrame(
            np.random.rand(10, 3),
            index=self.dates,
            columns=self.tickers
        )
        
    def test_init(self):
        # Test initialization
        strategy = MockVectorizedStrategy(self.tickers)
        self.assertEqual(strategy.tickers, self.tickers)
        self.assertIsNone(strategy.prices_df)
        self.assertIsNone(strategy.weights_df)
    
    def test_batch(self):
        # Test batch method implementation
        weights = self.strategy.batch(self.prices_df)
        self.assertEqual(weights.shape, self.prices_df.shape)
        self.assertTrue(all(weights.sum(axis=1).round(8) == 1.0))
    
    def test_batch_not_implemented(self):
        # Test that NotImplementedError is raised for base class
        strategy = VectorizedStrategyBase(self.tickers)
        with self.assertRaises(NotImplementedError):
            strategy.batch(self.prices_df)
    
    def test_step_with_weights_df(self):
        # Test step method with weights_df set
        self.strategy.weights_df = pd.DataFrame(
            np.random.rand(10, 3),
            index=self.dates,
            columns=self.tickers
        )
        # Normalize weights to sum to 1
        self.strategy.weights_df = self.strategy.weights_df.div(
            self.strategy.weights_df.sum(axis=1), axis=0
        )
        
        # Test step method with date in weights_df
        test_date = self.dates[5]
        expected_weights = {
            t: float(self.strategy.weights_df.loc[test_date, t])
            for t in self.tickers
        }
        result = self.strategy.step(test_date, None)
        for ticker in self.tickers:
            self.assertAlmostEqual(result[ticker], expected_weights[ticker])
    
    def test_step_without_weights_df(self):
        # Test step method without weights_df
        test_date = self.dates[0]
        result = self.strategy.step(test_date, None)
        expected_value = 1.0 / len(self.tickers)
        for ticker in self.tickers:
            self.assertEqual(result[ticker], expected_value)
    
    def test_step_date_not_in_weights_df(self):
        # Test step method with date not in weights_df
        self.strategy.weights_df = pd.DataFrame(
            np.random.rand(10, 3),
            index=self.dates,
            columns=self.tickers
        )
        test_date = pd.to_datetime('2021-01-01')  # Date not in weights_df
        result = self.strategy.step(test_date, None)
        expected_value = 1.0 / len(self.tickers)
        for ticker in self.tickers:
            self.assertEqual(result[ticker], expected_value)


class TestVectorizedBacktester(unittest.TestCase):
    def setUp(self):
        self.market_data_loader = MockMarketDataLoader()
        self.backtester = VectorizedBacktester(self.market_data_loader)
        self.tickers = ['AAPL', 'MSFT', 'GOOG']
        self.strategy = MockVectorizedStrategy(self.tickers)
    
    def test_init(self):
        # Test initialization
        backtester = VectorizedBacktester()
        self.assertIsNone(backtester.market_data_loader)
        
        backtester = VectorizedBacktester(self.market_data_loader)
        self.assertEqual(backtester.market_data_loader, self.market_data_loader)
    
    def test_run_backtest_basic(self):
        # Test basic backtest without options
        result = self.backtester.run_backtest(self.strategy)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
    
    def test_run_backtest_type_error(self):
        # Test with wrong strategy type
        class MockWrongStrategy(StrategyBase):
            pass
        
        wrong_strategy = MockWrongStrategy(self.tickers)
        with self.assertRaises(TypeError):
            self.backtester.run_backtest(wrong_strategy)
    
    def test_run_backtest_with_date_range(self):
        # Test with date range
        start_date = '2020-01-03'
        end_date = '2020-01-07'
        result = self.backtester.run_backtest(
            self.strategy,
            start_date=start_date,
            end_date=end_date
        )
        # Check that dates are within range
        self.assertTrue(all(result['signals_df'].index >= pd.to_datetime(start_date)))
        self.assertTrue(all(result['signals_df'].index <= pd.to_datetime(end_date)))
    
    def test_run_backtest_no_shift_signals(self):
        # Test with shift_signals=False
        result = self.backtester.run_backtest(self.strategy, shift_signals=False)
        # No direct way to test this without comparing to shifted version
        self.assertIn('signals_df', result)
    
    def test_run_backtest_require_all_history(self):
        # Test with require_all_history=True
        result = self.backtester.run_backtest(
            self.strategy,
            require_all_history=True
        )
        self.assertIn('signals_df', result)
    
    def test_run_backtest_require_all_history_error(self):
        # Test error case for require_all_history
        # Create a market data loader that returns None for some tickers
        class ErrorMockDataLoader:
            def fetch_data(self, tickers):
                result = {}
                dates = pd.date_range(start='2020-01-01', periods=10)
                for i, ticker in enumerate(tickers):
                    if i % 2 == 0:  # Every other ticker has no data
                        result[ticker] = None
                    else:
                        result[ticker] = pd.DataFrame(
                            {'close': np.linspace(100, 110, len(dates))},
                            index=dates
                        )
                return result
        
        backtester = VectorizedBacktester(ErrorMockDataLoader())
        with self.assertRaises(ValueError):
            backtester.run_backtest(
                self.strategy,
                require_all_history=True
            )
    
    def test_run_backtest_verbose(self):
        # Test with verbose=True
        # This is mostly about coverage, would need to capture stdout
        # to test the actual output
        result = self.backtester.run_backtest(self.strategy, verbose=True)
        self.assertIn('signals_df', result)
    
    def test_run_backtest_standard_benchmark(self):
        # Test with standard benchmark
        from portwine.backtester.benchmarks import STANDARD_BENCHMARKS
        
        # Mock the STANDARD_BENCHMARKS
        original = STANDARD_BENCHMARKS.copy()
        try:
            STANDARD_BENCHMARKS['test_benchmark'] = lambda x: x.mean(axis=1) * 2
            result = self.backtester.run_backtest(
                self.strategy,
                benchmark='test_benchmark'
            )
            self.assertIsNotNone(result['benchmark_returns'])
        finally:
            # Restore original
            STANDARD_BENCHMARKS.clear()
            STANDARD_BENCHMARKS.update(original)
    
    def test_run_backtest_custom_benchmark_ticker(self):
        # Test with custom benchmark ticker
        result = self.backtester.run_backtest(
            self.strategy,
            benchmark='SPY'
        )
        self.assertIsNotNone(result['benchmark_returns'])
    
    def test_run_backtest_callable_benchmark(self):
        # Test with callable benchmark
        def custom_benchmark(returns_df):
            return returns_df.mean(axis=1) * 1.5
        
        result = self.backtester.run_backtest(
            self.strategy,
            benchmark=custom_benchmark
        )
        self.assertIsNotNone(result['benchmark_returns'])
    
    def test_run_backtest_no_benchmark(self):
        # Test with benchmark=None
        result = self.backtester.run_backtest(
            self.strategy,
            benchmark=None
        )
        self.assertIsNone(result['benchmark_returns'])


class TestCreatePriceDataframe(unittest.TestCase):
    def setUp(self):
        self.market_data_loader = MockMarketDataLoader()
        self.tickers = ['AAPL', 'MSFT', 'GOOG']
    
    def test_create_price_dataframe_basic(self):
        # Test basic functionality
        df = create_price_dataframe(self.market_data_loader, self.tickers)
        self.assertEqual(set(df.columns), set(self.tickers))
        self.assertEqual(len(df), 10)  # Based on MockMarketDataLoader
    
    def test_create_price_dataframe_with_date_range(self):
        # Test with date range
        start_date = '2020-01-03'
        end_date = '2020-01-07'
        df = create_price_dataframe(
            self.market_data_loader,
            self.tickers,
            start_date=start_date,
            end_date=end_date
        )
        self.assertTrue(all(df.index >= pd.to_datetime(start_date)))
        self.assertTrue(all(df.index <= pd.to_datetime(end_date)))
    
    def test_create_price_dataframe_missing_data(self):
        # Test with missing data
        class MissingDataLoader:
            def fetch_data(self, tickers):
                result = {}
                dates = pd.date_range(start='2020-01-01', periods=10)
                for i, ticker in enumerate(tickers):
                    if i == 0:  # First ticker has missing dates
                        result[ticker] = pd.DataFrame(
                            {'close': np.linspace(100, 105, 6)},
                            index=dates[:6]
                        )
                    else:
                        result[ticker] = pd.DataFrame(
                            {'close': np.linspace(100, 110, len(dates))},
                            index=dates
                        )
                return result
        
        loader = MissingDataLoader()
        df = create_price_dataframe(loader, self.tickers)
        # Should forward-fill the missing dates
        self.assertEqual(len(df), 10)
        # The first ticker should have the last value repeated
        self.assertEqual(df.iloc[6][self.tickers[0]], df.iloc[5][self.tickers[0]])
    
    def test_create_price_dataframe_all_missing(self):
        # Test with all data missing for some dates
        class AllMissingDataLoader:
            def fetch_data(self, tickers):
                result = {}
                dates1 = pd.date_range(start='2020-01-01', periods=5)
                dates2 = pd.date_range(start='2020-01-06', periods=5)
                
                result[tickers[0]] = pd.DataFrame(
                    {'close': np.linspace(100, 105, 5)},
                    index=dates1
                )
                result[tickers[1]] = pd.DataFrame(
                    {'close': np.linspace(200, 205, 5)},
                    index=dates2
                )
                return result
        
        loader = AllMissingDataLoader()
        df = create_price_dataframe(loader, self.tickers[:2])
        
        # Should drop dates where all data is missing
        self.assertTrue(all(~df.isna().all(axis=1)))


class TestBenchmarkEqualWeight(unittest.TestCase):
    def test_benchmark_equal_weight(self):
        # Test the benchmark_equal_weight function
        dates = pd.date_range(start='2020-01-01', periods=5)
        returns_df = pd.DataFrame(
            np.random.rand(5, 3),
            index=dates,
            columns=['A', 'B', 'C']
        )
        
        result = benchmark_equal_weight(returns_df)
        expected = returns_df.mean(axis=1)
        
        pd.testing.assert_series_equal(result, expected) 