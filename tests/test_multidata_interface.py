import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from portwine.data.interface import MultiDataInterface, RestrictedDataInterface
from tests.helpers import MockDataStore


class MockStore(MockDataStore):
    def __init__(self, available_tickers):
        super().__init__()
        # Load a constant OHLCV series for each available ticker
        for t in available_tickers:
            self.load_date_dict(t, {"2020-01-01": {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000000
            }})


class TestMultiDataInterface(unittest.TestCase):
    """Comprehensive test suite for MultiDataInterface"""
    
    def setUp(self):
        # Create mock stores
        self.market_store = MockStore(['AAPL', 'MSFT'])
        self.alt_store = MockStore(['GDP', 'FEDFUNDS'])  # Note: just the symbol part
        
        # Create MultiDataInterface with stores
        self.loaders = {
            None: self.market_store,  # Default store for regular tickers
            'FRED': self.alt_store,   # FRED store for alternative data
        }
        self.data_interface = MultiDataInterface(self.loaders)
        self.data_interface.set_current_timestamp(pd.Timestamp('2020-01-01'))
    
    def test_regular_ticker_access(self):
        """Test accessing regular tickers (no prefix)"""
        # Should work for regular tickers
        result = self.data_interface['AAPL']
        self.assertIsNotNone(result)
        self.assertEqual(result['close'], 102.0)
        
        result = self.data_interface['MSFT']
        self.assertIsNotNone(result)
        self.assertEqual(result['close'], 102.0)
    
    def test_alternative_ticker_access(self):
        """Test accessing alternative tickers (with prefix)"""
        # Should work for alternative tickers
        result = self.data_interface['FRED:GDP']
        self.assertIsNotNone(result)
        self.assertEqual(result['close'], 102.0)
        
        result = self.data_interface['FRED:FEDFUNDS']
        self.assertIsNotNone(result)
        self.assertEqual(result['close'], 102.0)
    
    def test_nonexistent_ticker(self):
        """Test accessing tickers that don't exist"""
        # Should raise KeyError for tickers that don't exist
        with self.assertRaises(KeyError):
            self.data_interface['INVALID']
        
        with self.assertRaises(KeyError):
            self.data_interface['FRED:INVALID']
    
    def test_unknown_prefix(self):
        """Test accessing tickers with unknown prefix"""
        with self.assertRaises(ValueError):
            self.data_interface['UNKNOWN:TICKER']
    
    def test_get_method(self):
        """Test the get method for safe access"""
        # MultiDataInterface doesn't have a get method, only RestrictedDataInterface does
        # This test should be moved to TestRestrictedDataInterface
        pass


class TestRestrictedDataInterface(unittest.TestCase):
    """Test RestrictedDataInterface functionality"""
    
    def setUp(self):
        # Create mock stores
        self.market_store = MockStore(['AAPL', 'MSFT'])
        self.alt_store = MockStore(['GDP', 'FEDFUNDS'])  # Note: just the symbol part
        
        # Create RestrictedDataInterface
        self.loaders = {
            None: self.market_store,
            'FRED': self.alt_store,
        }
        self.data_interface = RestrictedDataInterface(self.loaders)
        self.data_interface.set_current_timestamp(pd.Timestamp('2020-01-01'))
    
    def test_no_restrictions(self):
        """Test behavior when no restrictions are set"""
        # Should work normally when no restrictions
        result = self.data_interface['AAPL']
        self.assertIsNotNone(result)
        
        result = self.data_interface['FRED:GDP']
        self.assertIsNotNone(result)
    
    def test_restrict_regular_tickers(self):
        """Test restricting regular tickers"""
        # Restrict to only AAPL
        self.data_interface.set_restricted_tickers(['AAPL'], prefix=None)
        
        # Should work for allowed ticker
        result = self.data_interface['AAPL']
        self.assertIsNotNone(result)
        
        # Should raise KeyError for restricted ticker
        with self.assertRaises(KeyError):
            self.data_interface['MSFT']
        
        # Alternative tickers should still work
        result = self.data_interface['FRED:GDP']
        self.assertIsNotNone(result)
    
    def test_restrict_alternative_tickers(self):
        """Test restricting alternative tickers"""
        # Restrict FRED tickers to only GDP
        self.data_interface.set_restricted_tickers(['GDP'], prefix='FRED')
        
        # Regular tickers should still work
        result = self.data_interface['AAPL']
        self.assertIsNotNone(result)
        
        # Should work for allowed alternative ticker
        result = self.data_interface['FRED:GDP']
        self.assertIsNotNone(result)
        
        # Should raise KeyError for restricted alternative ticker
        with self.assertRaises(KeyError):
            self.data_interface['FRED:FEDFUNDS']


if __name__ == '__main__':
    unittest.main()
