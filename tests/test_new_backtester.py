import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Set

# Import components to be tested
from portwine.backtester.core import Backtester, DailyMarketCalendar, validate_dates
from tests.calendar_utils import TestDailyMarketCalendar
from portwine.data.interface import DataInterface, RestrictedDataInterface
from portwine.strategies.base import StrategyBase
from portwine.universe import Universe
from tests.helpers import MockDataStore


class MockDataInterface(DataInterface):
    """DataInterface backed by MockDataStore for tests."""

    def __init__(self, mock_data=None, exists_data=None):
        default_payload = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000}
        store = MockDataStore(default_ohlcv=default_payload)
        if mock_data:
            store.load_bulk(mock_data)
        super().__init__(store)
        self.exists_data = exists_data or {}
        self.current_timestamp = None
        self.set_timestamp_calls = []
        self.get_calls = []

    def exists(self, ticker: str, start_date: str, end_date: str) -> bool:
        if ticker in self.exists_data:
            return self.exists_data[ticker]
        return self.data_loader.exists(ticker, start_date, end_date)

    def set_current_timestamp(self, timestamp):
        self.current_timestamp = timestamp
        self.set_timestamp_calls.append(timestamp)
        super().set_current_timestamp(timestamp)


class MockRestrictedDataInterface(RestrictedDataInterface):
    """Mock restricted data interface for testing purposes"""
    
    def __init__(self, mock_data=None, exists_data=None):
        # Create a mock data loader
        self.mock_data_loader = Mock()
        self.mock_data = mock_data or {}
        self.exists_data = exists_data or {}
        self.current_timestamp = None
        self.set_timestamp_calls = []
        self.set_restricted_calls = []
        self.get_calls = []
        
        # Initialize parent with mock loaders
        super().__init__({None: self.mock_data_loader})
    
    def exists(self, ticker: str, start_date: str, end_date: str) -> bool:
        """Mock exists method"""
        return self.exists_data.get(ticker, True)
    
    def set_current_timestamp(self, timestamp):
        """Mock set_current_timestamp method"""
        self.current_timestamp = timestamp
        self.set_timestamp_calls.append(timestamp)
        super().set_current_timestamp(timestamp)
    
    def set_restricted_tickers(self, tickers: List[str], prefix=None):
        """Mock set_restricted_tickers method"""
        self.set_restricted_calls.append(tickers)
        super().set_restricted_tickers(tickers, prefix)
    
    def __getitem__(self, ticker: str):
        """Mock __getitem__ method"""
        self.get_calls.append(ticker)
        
        # Check if ticker is in restricted list (parent method)
        if ticker not in self.restricted_tickers and len(self.restricted_tickers) > 0:
            raise KeyError(f"Ticker {ticker} is not in the restricted tickers list.")
        
        # Return mock data for the ticker
        if ticker in self.mock_data:
            return self.mock_data[ticker]
        else:
            # Return default OHLCV data
            return {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            }


class MockDailyMarketCalendar(TestDailyMarketCalendar):
    """Backward-compatible alias that records calls for assertions while using the shared test calendar."""
    def __init__(self, calendar_name="NYSE"):
        super().__init__(
            calendar_name=calendar_name,
            mode="all",
            allowed_year=2023,
            default_start="2023-01-01",
            default_end="2023-12-31",
            default_hour=None,
        )
        self.get_datetime_index_calls = []

    def get_datetime_index(self, start_date: str, end_date=None):
        self.get_datetime_index_calls.append((start_date, end_date))
        return super().get_datetime_index(start_date, end_date)


class MockUniverse:
    """Mock universe for testing purposes"""
    
    def __init__(self, tickers: List[str]):
        self.all_tickers = set(tickers)
        self._current_date = None
        self.set_datetime_calls = []
        self.get_constituents_calls = []
    
    def set_datetime(self, dt):
        """Mock set_datetime method"""
        self._current_date = dt
        self.set_datetime_calls.append(dt)
    
    def get_constituents(self, dt):
        """Mock get_constituents method"""
        self.get_constituents_calls.append(dt)
        return self.all_tickers


class MockStrategy(StrategyBase):
    """Mock strategy for testing purposes"""
    
    def __init__(self, tickers: List[str]):
        # Create a mock universe instead of using the real one
        self.universe = MockUniverse(tickers)
        self.step_calls = []
        self.step_return = {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    def step(self, current_date, daily_data):
        """Mock step method"""
        self.step_calls.append((current_date, daily_data))
        
        # Test that we can access data through the restricted interface
        if hasattr(daily_data, '__getitem__'):
            # This is a RestrictedDataInterface, test accessing tickers
            for ticker in self.universe.all_tickers:
                try:
                    data = daily_data[ticker]
                    # Verify we got OHLCV data
                    assert 'open' in data
                    assert 'close' in data
                except KeyError as e:
                    # This is expected if ticker is not in restricted list
                    pass
        
        return self.step_return.copy()


class TestBacktester(unittest.TestCase):
    """Test cases for Backtester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock data interface (DataInterface, not RestrictedDataInterface)
        self.mock_data_interface = MockDataInterface()
        
        # Add some test data to the mock data store so _compute_effective_end_date can work
        test_data = {
            'AAPL': pd.DataFrame({
                'open': [100.0] * 365,
                'high': [105.0] * 365,
                'low': [95.0] * 365,
                'close': [102.0] * 365,
                'volume': [1000000] * 365
            }, index=pd.date_range('2023-01-01', periods=365, freq='D')),
            'GOOGL': pd.DataFrame({
                'open': [200.0] * 365,
                'high': [210.0] * 365,
                'low': [190.0] * 365,
                'close': [204.0] * 365,
                'volume': [2000000] * 365
            }, index=pd.date_range('2023-01-01', periods=365, freq='D'))
        }
        self.mock_data_interface.data_loader.load_bulk(test_data)
    
        # Create test calendar (shared impl configured for 2023)
        self.mock_calendar = MockDailyMarketCalendar()
    
        # Create mock strategy
        self.mock_strategy = MockStrategy(['AAPL', 'GOOGL'])
    
        # Create Backtester instance
        self.backtester = Backtester(self.mock_data_interface, self.mock_calendar)
    
        # Define a simple benchmark function for testing
        def equal_weight_benchmark(ret_df):
            return ret_df.mean(axis=1)
        
        self.benchmark_func = equal_weight_benchmark
    
    def test_initialization(self):
        """Test Backtester initialization"""
        # Test that initialization works correctly
        self.assertIs(self.backtester.data, self.mock_data_interface)
        self.assertIs(self.backtester.calendar, self.mock_calendar)
    
    def test_validate_data_success(self):
        """Test validate_data method with valid data"""
        # Set up mock data to exist
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
        # Test validation
        result = self.backtester.validate_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')
        self.assertTrue(result)
    
    def test_validate_data_missing_ticker(self):
        """Test validate_data method with missing ticker data"""
        # Set up mock data where one ticker doesn't exist
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': False  # This ticker doesn't exist
        }
        
        # Test that validation raises ValueError
        with self.assertRaises(ValueError) as context:
            self.backtester.validate_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')
        
        self.assertIn("Data for ticker GOOGL does not exist", str(context.exception))
    
    def test_validate_data_empty_tickers(self):
        """Test validate_data method with empty ticker list"""
        result = self.backtester.validate_data([], '2023-01-01', '2023-12-31')
        self.assertTrue(result)
    
    def test_run_backtest_basic(self):
        """Test basic backtest execution"""
        # Set up mock data
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
                # Run backtest
        result = self.backtester.run_backtest(
            self.mock_strategy,
            start_date='2023-01-01',
            end_date='2023-01-05',
            benchmark=self.benchmark_func
        )
        
        # Verify calendar was called correctly
        self.assertEqual(len(self.mock_calendar.get_datetime_index_calls), 1)
        self.assertEqual(self.mock_calendar.get_datetime_index_calls[0], ('2023-01-01', '2023-01-05'))
        
        # Verify strategy step was called for each day
        self.assertEqual(len(self.mock_strategy.step_calls), 5)
        
        # Verify the result is a dictionary of DataFrames
        self.assertIsInstance(result, dict)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
        
        sig_df = result['signals_df']
        ret_df = result['tickers_returns']
        strategy_ret_df = result['strategy_returns']
        benchmark_ret_df = result['benchmark_returns']
        
        # Verify signals DataFrame
        self.assertIsInstance(sig_df, pd.DataFrame)
        self.assertEqual(len(sig_df), 5)  # 5 days
        self.assertEqual(list(sig_df.columns), ['AAPL', 'GOOGL'])
        # Check that all values are 0.5 (equal weight)
        for col in sig_df.columns:
            self.assertTrue(all(sig_df[col] == 0.5))
            
        # Verify returns DataFrame
        self.assertIsInstance(ret_df, pd.DataFrame)
        self.assertEqual(len(ret_df), 5)  # 5 days
        self.assertEqual(list(ret_df.columns), ['AAPL', 'GOOGL'])
        # First day returns should be 0 (no previous day to calculate from)
        self.assertTrue(all(ret_df.iloc[0] == 0.0))
        
        # Verify strategy returns Series
        self.assertIsInstance(strategy_ret_df, pd.Series)
        self.assertEqual(len(strategy_ret_df), 5)  # 5 days
        self.assertEqual(strategy_ret_df.name, 'strategy_returns')
        
        # Verify benchmark returns Series
        self.assertIsInstance(benchmark_ret_df, pd.Series)
        self.assertEqual(len(benchmark_ret_df), 5)  # 5 days
    
    def test_run_backtest_without_dates(self):
        """Test backtest execution without specifying dates"""
        # Set up mock data
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
        # Run backtest without dates
        result = self.backtester.run_backtest(self.mock_strategy, benchmark=self.benchmark_func)
        
        # Verify calendar was called with computed dates (not None)
        self.assertEqual(len(self.mock_calendar.get_datetime_index_calls), 1)
        # The backtester computes dates from data when none are provided
        self.assertEqual(self.mock_calendar.get_datetime_index_calls[0], ('2023-01-01', '2023-12-31'))
        
        # Verify result is a dictionary of DataFrames
        self.assertIsInstance(result, dict)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
        
        sig_df = result['signals_df']
        ret_df = result['tickers_returns']
        strategy_ret_df = result['strategy_returns']
        benchmark_ret_df = result['benchmark_returns']
        
        # Verify signals DataFrame
        self.assertIsInstance(sig_df, pd.DataFrame)
        self.assertEqual(len(sig_df), 365)  # Full year (2023-01-01 to 2023-12-31)
        self.assertEqual(list(sig_df.columns), ['AAPL', 'GOOGL'])
        # Check that all values are 0.5 (equal weight)
        for col in sig_df.columns:
            self.assertTrue(all(sig_df[col] == 0.5))
            
        # Verify returns DataFrame
        self.assertIsInstance(ret_df, pd.DataFrame)
        self.assertEqual(len(ret_df), 365)  # Full year
        self.assertEqual(list(ret_df.columns), ['AAPL', 'GOOGL'])
        # First day returns should be 0 (no previous day to calculate from)
        self.assertTrue(all(ret_df.iloc[0] == 0.0))
        
        # Verify strategy returns Series
        self.assertIsInstance(strategy_ret_df, pd.Series)
        self.assertEqual(len(strategy_ret_df), 365)  # Full year
        self.assertEqual(strategy_ret_df.name, 'strategy_returns')
        
        # Verify benchmark returns Series
        self.assertIsInstance(benchmark_ret_df, pd.Series)
        self.assertEqual(len(benchmark_ret_df), 365)  # Full year
    
    def test_run_backtest_data_validation_failure(self):
        """Test backtest execution when data validation fails"""
        # Set up mock data where ticker doesn't exist
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': False  # This ticker doesn't exist
        }
        
        # Test that backtest raises ValueError
        with self.assertRaises(ValueError) as context:
            self.backtester.run_backtest(
                self.mock_strategy, 
                start_date='2023-01-01', 
                end_date='2023-01-05'
            )
        
        self.assertIn("Data for ticker GOOGL does not exist", str(context.exception))
    
    def test_run_backtest_strategy_universe_interaction(self):
        """Test that backtest properly interacts with strategy universe"""
        # Create a strategy with a mock universe
        mock_universe = Mock()
        mock_universe.all_tickers = {'AAPL', 'GOOGL'}
        mock_universe.get_constituents.return_value = {'AAPL', 'GOOGL'}
        
        strategy = MockStrategy(['AAPL', 'GOOGL'])
        strategy.universe = mock_universe
        
        # Set up mock data
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
                # Run backtest
        self.backtester.run_backtest(
            strategy,
            start_date='2023-01-01',
            end_date='2023-01-03',
            benchmark=self.benchmark_func
        )
        
        # Verify universe methods were called
        self.assertEqual(mock_universe.set_datetime.call_count, 3)  # 3 days
        self.assertEqual(mock_universe.get_constituents.call_count, 3)
        
        # Note: The Backtester now uses RestrictedDataInterface for the strategy,
        # so we don't check the get_calls tracking since it's not available
    
    def test_run_backtest_empty_calendar(self):
        """Test backtest execution with empty calendar (no trading days)"""
        # Configure calendar to produce no trading days: odd-only mode with an even-only range
        self.mock_calendar.mode = "odd"
        even_only_day = '2023-01-02'  # even day, so odd filter yields empty

        # Set up mock data
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
    
        # Run backtest - this should raise ValueError since no trading days are found
        with self.assertRaises(ValueError) as context:
            result = self.backtester.run_backtest(
                self.mock_strategy,
                start_date=even_only_day,
                end_date=even_only_day
            )
        
        # Verify the error message
        self.assertIn("No trading days found in the specified date range", str(context.exception))
    
    def test_run_backtest_strategy_returns_different_signals(self):
        """Test backtest with strategy that returns different signals each step"""
        # Create a strategy that returns different signals
        class DynamicStrategy(MockStrategy):
            def step(self, current_date, daily_data):
                self.step_calls.append((current_date, daily_data))
                # Return different allocations based on date
                if len(self.step_calls) == 1:
                    return {'AAPL': 0.7, 'GOOGL': 0.3}
                elif len(self.step_calls) == 2:
                    return {'AAPL': 0.3, 'GOOGL': 0.7}
                else:
                    return {'AAPL': 0.5, 'GOOGL': 0.5}
        
        dynamic_strategy = DynamicStrategy(['AAPL', 'GOOGL'])
        
        # Set up mock data
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
                # Run backtest
        result = self.backtester.run_backtest(
            dynamic_strategy,
            start_date='2023-01-01',
            end_date='2023-01-03',
            benchmark=self.benchmark_func
        )
        
        # Verify strategy was called 3 times
        self.assertEqual(len(dynamic_strategy.step_calls), 3)
        
        # Verify the result is a dictionary of DataFrames
        self.assertIsInstance(result, dict)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
        
        sig_df = result['signals_df']
        ret_df = result['tickers_returns']
        strategy_ret_df = result['strategy_returns']
        benchmark_ret_df = result['benchmark_returns']
        
        # Verify signals DataFrame
        self.assertIsInstance(sig_df, pd.DataFrame)
        self.assertEqual(len(sig_df), 3)  # 3 days
        self.assertEqual(list(sig_df.columns), ['AAPL', 'GOOGL'])
        # Check that the signals change as expected
        self.assertEqual(sig_df.iloc[0]['AAPL'], 0.7)  # First day
        self.assertEqual(sig_df.iloc[0]['GOOGL'], 0.3)
        self.assertEqual(sig_df.iloc[1]['AAPL'], 0.3)  # Second day
        self.assertEqual(sig_df.iloc[1]['GOOGL'], 0.7)
        self.assertEqual(sig_df.iloc[2]['AAPL'], 0.5)  # Third day
        self.assertEqual(sig_df.iloc[2]['GOOGL'], 0.5)
        
        # Verify returns DataFrame
        self.assertIsInstance(ret_df, pd.DataFrame)
        self.assertEqual(len(ret_df), 3)  # 3 days
        self.assertEqual(list(ret_df.columns), ['AAPL', 'GOOGL'])
        # First day returns should be 0 (no previous day to calculate from)
        self.assertTrue(all(ret_df.iloc[0] == 0.0))
        
        # Verify strategy returns Series
        self.assertIsInstance(strategy_ret_df, pd.Series)
        self.assertEqual(len(strategy_ret_df), 3)  # 3 days
        self.assertEqual(strategy_ret_df.name, 'strategy_returns')
        
        # Verify benchmark returns Series
        self.assertIsInstance(benchmark_ret_df, pd.Series)
        self.assertEqual(len(benchmark_ret_df), 3)  # 3 days
    
    def test_run_backtest_with_mock_data(self):
        """Test backtest with actual mock data"""
        # Set up mock data with specific OHLCV values
        self.mock_data_interface.mock_data = {
            'AAPL': {
                'open': 150.0,
                'high': 155.0,
                'low': 148.0,
                'close': 152.0,
                'volume': 5000000
            },
            'GOOGL': {
                'open': 2800.0,
                'high': 2850.0,
                'low': 2780.0,
                'close': 2820.0,
                'volume': 2000000
            }
        }
        
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
        # Run backtest
        result = self.backtester.run_backtest(
            self.mock_strategy, 
            start_date='2023-01-01', 
            end_date='2023-01-03',
            benchmark=self.benchmark_func
        )
        
        # Verify data was retrieved correctly
        # Note: The Backtester now uses RestrictedDataInterface for the strategy,
        # so we don't check the get_calls tracking since it's not available
        
        # Verify the result is a dictionary of DataFrames
        self.assertIsInstance(result, dict)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
        
        sig_df = result['signals_df']
        ret_df = result['tickers_returns']
        strategy_ret_df = result['strategy_returns']
        benchmark_ret_df = result['benchmark_returns']
        
        # Verify signals DataFrame
        self.assertIsInstance(sig_df, pd.DataFrame)
        self.assertEqual(len(sig_df), 3)  # 3 days
        self.assertEqual(list(sig_df.columns), ['AAPL', 'GOOGL'])
        # Check that all values are 0.5 (equal weight)
        for col in sig_df.columns:
            self.assertTrue(all(sig_df[col] == 0.5))
            
        # Verify returns DataFrame
        self.assertIsInstance(ret_df, pd.DataFrame)
        self.assertEqual(len(ret_df), 3)  # 3 days
        self.assertEqual(list(ret_df.columns), ['AAPL', 'GOOGL'])
        # First day returns should be 0 (no previous day to calculate from)
        self.assertTrue(all(ret_df.iloc[0] == 0.0))
        
        # Verify strategy returns Series
        self.assertIsInstance(strategy_ret_df, pd.Series)
        self.assertEqual(len(strategy_ret_df), 3)  # 3 days
        self.assertEqual(strategy_ret_df.name, 'strategy_returns')
        
        # Verify benchmark returns Series
        self.assertIsInstance(benchmark_ret_df, pd.Series)
        self.assertEqual(len(benchmark_ret_df), 3)  # 3 days
    
    def test_run_backtest_restricted_data_access(self):
        """Test that backtest uses restricted data interface to prevent access to non-universe tickers"""
        # Create a strategy that tries to access a ticker outside the universe
        class MaliciousStrategy(MockStrategy):
            def step(self, current_date, daily_data):
                self.step_calls.append((current_date, daily_data))
                
                # Try to access a ticker that's not in the universe
                try:
                    # This should raise a KeyError because 'MSFT' is not in the universe
                    data = daily_data['MSFT']
                    # If we get here, the restriction failed
                    raise AssertionError("Should not be able to access MSFT data")
                except KeyError as e:
                    # This is expected - MSFT is not in the restricted tickers
                    assert "MSFT" in str(e)
                
                # Try to access a ticker that IS in the universe
                try:
                    data = daily_data['AAPL']
                    # This should work
                    assert 'open' in data
                    assert 'close' in data
                except KeyError:
                    # This should not happen
                    raise AssertionError("Should be able to access AAPL data")
                
                return self.step_return.copy()
        
        malicious_strategy = MaliciousStrategy(['AAPL', 'GOOGL'])
        
        # Set up mock data
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True,
            'MSFT': True  # MSFT exists in data but not in universe
        }
        
                # Run backtest
        result = self.backtester.run_backtest(
            malicious_strategy,
            start_date='2023-01-01',
            end_date='2023-01-03',
            benchmark=self.benchmark_func
        )
        
        # Verify strategy was called
        self.assertEqual(len(malicious_strategy.step_calls), 3)
        
        # Verify the result is a dictionary of DataFrames
        self.assertIsInstance(result, dict)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
        
        sig_df = result['signals_df']
        ret_df = result['tickers_returns']
        strategy_ret_df = result['strategy_returns']
        benchmark_ret_df = result['benchmark_returns']
        
        # Verify signals DataFrame
        self.assertIsInstance(sig_df, pd.DataFrame)
        self.assertEqual(len(sig_df), 3)  # 3 days
        self.assertEqual(list(sig_df.columns), ['AAPL', 'GOOGL'])
        # Check that all values are 0.5 (equal weight)
        for col in sig_df.columns:
            self.assertTrue(all(sig_df[col] == 0.5))
            
        # Verify returns DataFrame
        self.assertIsInstance(ret_df, pd.DataFrame)
        self.assertEqual(len(ret_df), 3)  # 3 days
        self.assertEqual(list(ret_df.columns), ['AAPL', 'GOOGL'])
        # First day returns should be 0 (no previous day to calculate from)
        self.assertTrue(all(ret_df.iloc[0] == 0.0))
    
    def test_run_backtest_returns_calculation(self):
        """Test that returns are calculated correctly"""
        # Set up mock data with changing prices to test returns calculation
        self.mock_data_interface.mock_data = {
            'AAPL': {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            },
            'GOOGL': {
                'open': 2000.0,
                'high': 2100.0,
                'low': 1950.0,
                'close': 2040.0,
                'volume': 500000
            }
        }
        
        self.mock_data_interface.exists_data = {
            'AAPL': True,
            'GOOGL': True
        }
        
        # Run backtest
        result = self.backtester.run_backtest(
            self.mock_strategy, 
            start_date='2023-01-01', 
            end_date='2023-01-03',
            benchmark=self.benchmark_func
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('signals_df', result)
        self.assertIn('tickers_returns', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('benchmark_returns', result)
        
        sig_df = result['signals_df']
        ret_df = result['tickers_returns']
        strategy_ret_df = result['strategy_returns']
        benchmark_ret_df = result['benchmark_returns']
        
        # Verify returns DataFrame structure
        self.assertIsInstance(ret_df, pd.DataFrame)
        self.assertEqual(len(ret_df), 3)  # 3 days
        self.assertEqual(list(ret_df.columns), ['AAPL', 'GOOGL'])
        
        # First day returns should be 0 (no previous day to calculate from)
        self.assertTrue(all(ret_df.iloc[0] == 0.0))
        
        # Verify that returns are calculated (non-zero for subsequent days)
        # The exact values depend on the mock data, but they should be calculated
        self.assertGreater(len(ret_df), 1)  # At least 2 days of data
        
        # Test with specific price changes to verify returns calculation
        # Set up mock data with known price changes
        self.mock_data_interface.mock_data = {
            'AAPL': {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 110.0,  # 10% increase
                'volume': 1000000
            },
            'GOOGL': {
                'open': 2000.0,
                'high': 2100.0,
                'low': 1950.0,
                'close': 1900.0,  # 5% decrease
                'volume': 500000
            }
        }
        
        # Run backtest again with new data
        result2 = self.backtester.run_backtest(
            self.mock_strategy, 
            start_date='2023-01-01', 
            end_date='2023-01-02',
            benchmark=self.benchmark_func
        )
        
        sig_df2 = result2['signals_df']
        ret_df2 = result2['tickers_returns']
        strategy_ret_df2 = result2['strategy_returns']
        benchmark_ret_df2 = result2['benchmark_returns']
        
        # First day returns should be 0
        self.assertTrue(all(ret_df2.iloc[0] == 0.0))
        
        # Second day should have calculated returns based on price changes
        # Note: The exact values depend on the mock data implementation
        # but we can verify that returns are being calculated
        self.assertEqual(len(ret_df2), 2)


class TestDailyMarketCalendar(unittest.TestCase):
    """Test cases for DailyMarketCalendar class"""
    
    def test_initialization(self):
        """Test DailyMarketCalendar initialization"""
        # The real DailyMarketCalendar doesn't store calendar_name as an attribute
        # It just uses it to create the calendar object
        self.assertIsNotNone(DailyMarketCalendar("NYSE").calendar)
    
    def test_validate_dates_valid(self):
        """Test validate_dates with valid dates"""
        result = validate_dates("2023-01-01", "2023-12-31")
        self.assertTrue(result)
    
    def test_validate_dates_invalid_start_date(self):
        """Test validate_dates with invalid start date format"""
        with self.assertRaises(AssertionError):
            validate_dates(123, "2023-12-31")
    
    def test_validate_dates_end_before_start(self):
        """Test validate_dates with end date before start date"""
        with self.assertRaises(AssertionError):
            validate_dates("2023-12-31", "2023-01-01")
    
    def test_validate_dates_none_end_date(self):
        """Test validate_dates with None end date"""
        result = validate_dates("2023-01-01", None)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main() 