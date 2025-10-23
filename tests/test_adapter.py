import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from collections import OrderedDict

from portwine.data.stores.adapter import MarketDataLoaderAdapter
from portwine.data.providers.loader_adapters import MarketDataLoader


class MockMarketDataLoader:
    """
    Mock loader that simulates the behavior of MarketDataLoader
    for testing the adapter interface compatibility.
    """
    
    def __init__(self, data_map):
        self.data_map = data_map
        self._data_cache = {}
        self._numpy_cache = {}
        self._date_cache = {}
        self._create_caches()
    
    def _create_caches(self):
        """Create numpy caches like the real MarketDataLoader"""
        for ticker, df in self.data_map.items():
            if df is not None:
                self._data_cache[ticker] = df
                self._date_cache[ticker] = df.index.values.astype('datetime64[ns]')
                self._numpy_cache[ticker] = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float64)
    
    def load_ticker(self, ticker):
        """Simulate load_ticker method"""
        return self.data_map.get(ticker)
    
    def fetch_data(self, tickers):
        """Simulate fetch_data method"""
        fetched = {}
        for t in tickers:
            if t in self._data_cache:
                fetched[t] = self._data_cache[t]
        return fetched
    
    def next(self, tickers, ts):
        """Simulate next method"""
        data = self.fetch_data(tickers)
        bar_dict = {}
        
        for t in data.keys():
            if t in self._numpy_cache:
                # Use numpy-based search like real loader
                date_array = self._date_cache[t]
                if len(date_array) == 0:
                    bar_dict[t] = None
                    continue
                
                # Convert timestamp to numpy datetime64
                if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    ts_utc = ts.tz_convert('UTC')
                    ts_np = np.datetime64(ts_utc.replace(tzinfo=None))
                else:
                    ts_np = np.datetime64(ts)
                
                pos = np.searchsorted(date_array, ts_np, side="right") - 1
                if pos < 0:
                    bar_dict[t] = None
                else:
                    row = self._numpy_cache[t][pos]
                    bar_dict[t] = {
                        'open': float(row[0]),
                        'high': float(row[1]),
                        'low': float(row[2]),
                        'close': float(row[3]),
                        'volume': float(row[4])
                    }
            else:
                bar_dict[t] = None
        
        return bar_dict


class MarketDataLoaderAdapterTests(unittest.TestCase):
    """
    Test suite for MarketDataLoaderAdapter to ensure it has the same interface
    as MarketDataLoader and can be used as a drop-in replacement.
    """
    
    def setUp(self):
        """Set up test data and mock loader"""
        # Create sample DataFrames with different date indices
        idx1 = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-05'])
        self.df1 = pd.DataFrame({
            'open': [1.0, 2.0, 3.0],
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9],
            'close': [1.0, 2.0, 3.0],
            'volume': [100, 200, 300],
        }, index=idx1)

        idx2 = pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-06'])
        self.df2 = pd.DataFrame({
            'open': [5.0, 6.0, 7.0],
            'high': [5.1, 6.1, 7.1],
            'low': [4.9, 5.9, 6.9],
            'close': [5.0, 6.0, 7.0],
            'volume': [500, 600, 700],
        }, index=idx2)
        
        # Create mock loader with test data
        self.mock_loader = MockMarketDataLoader({
            'AAPL': self.df1,
            'MSFT': self.df2,
            'INVALID': None
        })
        
        # Create adapter instance
        self.adapter = MarketDataLoaderAdapter(self.mock_loader)
    
    def test_adapter_initialization(self):
        """Test that adapter initializes correctly with a loader"""
        self.assertIsInstance(self.adapter, MarketDataLoaderAdapter)
        self.assertEqual(self.adapter.loader, self.mock_loader)
        self.assertIsNone(self.adapter._current_timestamp)
    
    def test_add_method_silently_ignores(self):
        """Test that add method silently ignores operations (read-only behavior)"""
        # Should not raise any exceptions
        self.adapter.add('AAPL', {'open': 10.0, 'close': 11.0})
        # Verify no data was actually stored
        result = self.adapter.get('AAPL', pd.Timestamp('2025-01-01'))
        # Should still return the original data, not the added data
        self.assertEqual(result['open'], 1.0)
    
    def test_get_method_single_ticker_single_timestamp(self):
        """Test get method returns correct data for single ticker at specific timestamp"""
        # Test exact timestamp match
        result = self.adapter.get('AAPL', pd.Timestamp('2025-01-02'))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 2.0)
        self.assertEqual(result['close'], 2.0)
        self.assertEqual(result['volume'], 200)
        
        # Test timestamp between data points (should get earlier data)
        result = self.adapter.get('AAPL', pd.Timestamp('2025-01-03'))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 2.0)  # Should get 2025-01-02 data
        
        # Test timestamp before all data (should return None)
        result = self.adapter.get('AAPL', pd.Timestamp('2024-12-31'))
        self.assertIsNone(result)
        
        # Test timestamp after all data (should get last available data)
        result = self.adapter.get('AAPL', pd.Timestamp('2025-01-10'))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 3.0)  # Should get 2025-01-05 data
    
    def test_get_method_with_timezone_aware_timestamps(self):
        """Test get method handles timezone-aware timestamps correctly"""
        # Test with timezone-aware timestamp
        tz_aware_ts = pd.Timestamp('2025-01-02', tz='UTC')
        result = self.adapter.get('AAPL', tz_aware_ts)
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 2.0)
    
    def test_get_all_method_date_range(self):
        """Test get_all method returns correct data for date range"""
        start_date = pd.Timestamp('2025-01-02')
        end_date = pd.Timestamp('2025-01-05')
        
        result = self.adapter.get_all('AAPL', start_date, end_date)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, OrderedDict)
        
        # Should have 2 data points: 2025-01-02 and 2025-01-05
        self.assertEqual(len(result), 2)
        
        # Check first data point
        first_ts = list(result.keys())[0]
        self.assertEqual(first_ts, pd.Timestamp('2025-01-02'))
        self.assertEqual(result[first_ts]['open'], 2.0)
        
        # Check second data point
        second_ts = list(result.keys())[1]
        self.assertEqual(second_ts, pd.Timestamp('2025-01-05'))
        self.assertEqual(result[second_ts]['open'], 3.0)
    
    def test_get_all_method_start_date_only(self):
        """Test get_all method with only start_date specified"""
        start_date = pd.Timestamp('2025-01-02')
        
        result = self.adapter.get_all('AAPL', start_date)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # 2025-01-02 and 2025-01-05
        
        # Check that all dates are >= start_date
        for ts in result.keys():
            self.assertGreaterEqual(ts, start_date)
    
    def test_get_all_method_end_date_only(self):
        """Test get_all method with only end_date specified"""
        end_date = pd.Timestamp('2025-01-02')
        
        result = self.adapter.get_all('AAPL', None, end_date)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # 2025-01-01 and 2025-01-02
        
        # Check that all dates are <= end_date
        for ts in result.keys():
            self.assertLessEqual(ts, end_date)
    
    def test_get_all_method_no_date_constraints(self):
        """Test get_all method with no date constraints"""
        result = self.adapter.get_all('AAPL', None, None)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # All data points
    
    def test_get_all_method_empty_dataframe(self):
        """Test get_all method handles empty dataframes correctly"""
        # Test with ticker that has no data
        result = self.adapter.get_all('INVALID', pd.Timestamp('2025-01-01'))
        self.assertIsNone(result)
    
    def test_get_all_method_filtered_empty_result(self):
        """Test get_all method returns None when date filtering results in empty data"""
        # Test with date range that has no data
        start_date = pd.Timestamp('2026-01-01')
        result = self.adapter.get_all('AAPL', start_date)
        self.assertIsNone(result)
    
    def test_get_latest_method(self):
        """Test get_latest method returns the most recent data point"""
        result = self.adapter.get_latest('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 3.0)  # Should be 2025-01-05 data
        self.assertEqual(result['close'], 3.0)
        self.assertEqual(result['volume'], 300)
    
    def test_get_latest_method_no_data(self):
        """Test get_latest method handles tickers with no data"""
        result = self.adapter.get_latest('INVALID')
        self.assertIsNone(result)
    
    def test_latest_method(self):
        """Test latest method returns the most recent timestamp"""
        result = self.adapter.latest('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(result, pd.Timestamp('2025-01-05'))
    
    def test_latest_method_no_data(self):
        """Test latest method handles tickers with no data"""
        result = self.adapter.latest('INVALID')
        self.assertIsNone(result)
    
    def test_exists_method_with_date_range(self):
        """Test exists method with date range constraints"""
        # Test with valid date range
        start_date = pd.Timestamp('2025-01-02')
        end_date = pd.Timestamp('2025-01-05')
        result = self.adapter.exists('AAPL', start_date, end_date)
        self.assertTrue(result)
        
        # Test with date range that has no data
        start_date = pd.Timestamp('2026-01-01')
        result = self.adapter.exists('AAPL', start_date)
        self.assertFalse(result)
    
    def test_exists_method_no_date_constraints(self):
        """Test exists method with no date constraints"""
        result = self.adapter.exists('AAPL')
        self.assertTrue(result)
        
        result = self.adapter.exists('INVALID')
        self.assertFalse(result)
    
    def test_identifiers_method(self):
        """Test identifiers method returns empty list (as documented)"""
        result = self.adapter.identifiers()
        self.assertEqual(result, [])
    
    def test_data_format_consistency(self):
        """Test that all data returned by adapter has consistent format"""
        # Test get method
        result = self.adapter.get('AAPL', pd.Timestamp('2025-01-01'))
        self.assertIsNotNone(result)
        expected_keys = {'open', 'high', 'low', 'close', 'volume'}
        self.assertEqual(set(result.keys()), expected_keys)
        
        # Test get_all method
        result = self.adapter.get_all('AAPL', pd.Timestamp('2025-01-01'))
        self.assertIsNotNone(result)
        for ts, data in result.items():
            self.assertEqual(set(data.keys()), expected_keys)
        
        # Test get_latest method
        result = self.adapter.get_latest('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(set(result.keys()), expected_keys)
    
    def test_data_type_consistency(self):
        """Test that all numeric data is returned as float"""
        result = self.adapter.get('AAPL', pd.Timestamp('2025-01-01'))
        self.assertIsNotNone(result)
        
        for key in ['open', 'high', 'low', 'close', 'volume']:
            self.assertIsInstance(result[key], float)
    
    def test_adapter_interface_compatibility_with_loader(self):
        """Test that adapter provides the same interface as MarketDataLoader.next()"""
        # Test that adapter.get() produces the same result as loader.next()
        timestamp = pd.Timestamp('2025-01-02')
        
        # Get data via adapter
        adapter_result = self.adapter.get('AAPL', timestamp)
        
        # Get data via loader
        loader_result = self.mock_loader.next(['AAPL'], timestamp)
        
        # Results should be identical
        self.assertEqual(adapter_result, loader_result['AAPL'])
    
    def test_multiple_tickers_consistency(self):
        """Test that adapter works consistently across multiple tickers"""
        timestamp = pd.Timestamp('2025-01-02')
        
        # Test both tickers
        aapl_result = self.adapter.get('AAPL', timestamp)
        msft_result = self.adapter.get('MSFT', timestamp)
        
        self.assertIsNotNone(aapl_result)
        self.assertIsNotNone(msft_result)
        
        # Verify different data for different tickers
        self.assertNotEqual(aapl_result['open'], msft_result['open'])
        self.assertEqual(aapl_result['open'], 2.0)
        self.assertEqual(msft_result['open'], 5.0)
    
    def test_edge_cases(self):
        """Test various edge cases and error conditions"""
        # Test with None values
        self.assertIsNone(self.adapter.get('INVALID', pd.Timestamp('2025-01-01')))
        self.assertIsNone(self.adapter.get_all('INVALID', pd.Timestamp('2025-01-01')))
        self.assertIsNone(self.adapter.get_latest('INVALID'))
        self.assertIsNone(self.adapter.latest('INVALID'))
        self.assertFalse(self.adapter.exists('INVALID'))
        
        # Test with empty string ticker
        self.assertIsNone(self.adapter.get('', pd.Timestamp('2025-01-01')))
        
        # Test with very old timestamp
        old_ts = pd.Timestamp('1900-01-01')
        self.assertIsNone(self.adapter.get('AAPL', old_ts))
        
        # Test with very future timestamp
        future_ts = pd.Timestamp('2100-01-01')
        result = self.adapter.get('AAPL', future_ts)
        self.assertIsNotNone(result)  # Should get last available data
        self.assertEqual(result['open'], 3.0)  # Last data point
    
    def test_adapter_preserves_loader_state(self):
        """Test that adapter doesn't modify the underlying loader's state"""
        # Store original state
        original_cache = self.mock_loader._data_cache.copy()
        
        # Use adapter
        _ = self.adapter.get('AAPL', pd.Timestamp('2025-01-01'))
        _ = self.adapter.get_all('MSFT', pd.Timestamp('2025-01-01'))
        
        # Verify loader state is unchanged
        self.assertEqual(self.mock_loader._data_cache, original_cache)
    
    def test_adapter_handles_datetime_objects(self):
        """Test that adapter accepts both datetime and pd.Timestamp objects"""
        # Test with datetime object
        dt = datetime(2025, 1, 2)
        result = self.adapter.get('AAPL', dt)
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 2.0)
        
        # Test with pd.Timestamp
        ts = pd.Timestamp('2025-01-02')
        result = self.adapter.get('AAPL', ts)
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 2.0)
        
        # Results should be identical
        self.assertEqual(result['open'], 2.0)


if __name__ == '__main__':
    unittest.main()
