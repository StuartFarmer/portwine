#!/usr/bin/env python3
"""
Test suite for loader adapters to verify backward compatibility.

This test suite ensures that the new loader adapters maintain the exact same
API as the original loaders, while internally using the new provider system.

Key tests:
- Constructor compatibility (data_path parameter)
- load_ticker method functionality
- fetch_data method functionality  
- Legacy import compatibility
"""

import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import MagicMock

# Test the new adapters
from portwine.data.providers.loader_adapters import (
    EODHDMarketDataLoader,
    AlpacaMarketDataLoader,
    BrokerDataLoader,
)


class TestEODHDLoaderCompatibility(unittest.TestCase):
    """Test that EODHDMarketDataLoader maintains original API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_csv_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def create_test_csv_files(self):
        """Create test CSV files for EODHD loader tests."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = {
            'date': dates,
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'adjusted_close': [102.5, 103.5, 104.5, 105.5, 106.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }
        
        df = pd.DataFrame(data)
        
        # Save test files
        for ticker in ['AAPL', 'MSFT']:
            file_path = os.path.join(self.test_dir, f"{ticker}.US.csv")
            df.to_csv(file_path, index=False)
    
    def test_constructor_compatibility(self):
        """Test that constructor maintains original API."""
        # Should accept data_path parameter
        loader = EODHDMarketDataLoader(data_path=self.test_dir)
        self.assertEqual(loader.data_path, self.test_dir)
        self.assertEqual(loader.exchange_code, "US")
        
        # Should accept custom exchange_code
        loader = EODHDMarketDataLoader(data_path=self.test_dir, exchange_code="LSE")
        self.assertEqual(loader.exchange_code, "LSE")
    
    def test_load_ticker_method(self):
        """Test that load_ticker method works correctly."""
        loader = EODHDMarketDataLoader(data_path=self.test_dir)
        
        # Test loading AAPL data
        df = loader.load_ticker('AAPL')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)  # 5 days of data
        self.assertListEqual(list(df.columns), ['open', 'high', 'low', 'close', 'volume'])
        
        # Test loading MSFT data
        df = loader.load_ticker('MSFT')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)
        
        # Test loading non-existent ticker
        df = loader.load_ticker('INVALID')
        self.assertIsNone(df)
    
    def test_fetch_data_method(self):
        """Test that fetch_data method works correctly."""
        loader = EODHDMarketDataLoader(data_path=self.test_dir)
        
        # Test fetching multiple tickers
        data = loader.fetch_data(['AAPL', 'MSFT'])
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 2)
        self.assertIn('AAPL', data)
        self.assertIn('MSFT', data)
    
    def test_legacy_import_compatibility(self):
        """Test that legacy imports work and maintain the same API."""
        # Test that legacy imports work
        from portwine.loaders import EODHDMarketDataLoader as LegacyEODHDLoader
        
        # Should accept data_path parameter
        loader = LegacyEODHDLoader(data_path=self.test_dir)
        self.assertEqual(loader.data_path, self.test_dir)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
