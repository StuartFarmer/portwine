import os
import tempfile
import shutil

# Import the data loaders to test
from portwine.loaders import AlternativeMarketDataLoader
from portwine.loaders import FREDMarketDataLoader
from portwine.loaders import BarchartIndicesMarketDataLoader


class MockFREDLoader(FREDMarketDataLoader):
    """Mock FRED loader that returns predefined data without requiring API access"""

    def __init__(self, data_path):
        super().__init__(data_path, api_key=None, save_missing=False)
        self.mock_data = {}

    def add_mock_data(self, ticker, data):
        """Add mock data for a ticker"""
        self.mock_data[ticker] = data

    def load_ticker(self, ticker):
        """Override to return mock data without file access"""
        if ticker in self.mock_data:
            return self.mock_data[ticker].copy()  # Return a copy to prevent modification issues

        # Fall back to parent implementation if mock data not found
        return super().load_ticker(ticker)


class MockBarchartLoader(BarchartIndicesMarketDataLoader):
    """Mock Barchart loader that returns predefined data without file access"""

    def __init__(self, data_path):
        super().__init__(data_path)
        self.mock_data = {}

    def add_mock_data(self, ticker, data):
        """Add mock data for a ticker"""
        self.mock_data[ticker] = data

    def load_ticker(self, ticker):
        """Override to return mock data without file access"""
        if ticker in self.mock_data:
            return self.mock_data[ticker].copy()  # Return a copy to prevent modification issues

        # Fall back to parent implementation if mock data not found
        return super().load_ticker(ticker)


import unittest
import pandas as pd
import numpy as np
from datetime import datetime


# This is a simplified test case just for the caching functionality
class SimpleMockLoader:
    """A very simple mock loader for testing caching"""

    SOURCE_IDENTIFIER = 'MOCK'

    def __init__(self):
        self.data = {}

    def set_data(self, ticker, data):
        """Set the data for a ticker"""
        self.data[ticker] = data

    def load_ticker(self, ticker):
        """Return the data for a ticker (or None if not found)"""
        return self.data.get(ticker)

class TestAlternativeMarketDataLoader(unittest.TestCase):
    """Test cases for the AlternativeMarketDataLoader"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.fred_dir = os.path.join(self.test_dir, 'fred')
        self.barchart_dir = os.path.join(self.test_dir, 'barchart')

        os.makedirs(self.fred_dir, exist_ok=True)
        os.makedirs(self.barchart_dir, exist_ok=True)

        # Create mock data
        dates = pd.date_range(start='2020-01-01', end='2020-01-10')

        # FRED mock data
        self.fred_data = pd.DataFrame({
            'close': np.random.randn(len(dates)) + 3.0,  # Random data centered around 3
        }, index=dates)
        self.fred_data['open'] = self.fred_data['close']
        self.fred_data['high'] = self.fred_data['close']
        self.fred_data['low'] = self.fred_data['close']
        self.fred_data['volume'] = 0

        # Barchart mock data
        self.barchart_data = pd.DataFrame({
            'close': np.random.randn(len(dates)) * 10 + 1000,  # Random data centered around 1000
            'open': np.random.randn(len(dates)) * 10 + 1000,
            'high': np.random.randn(len(dates)) * 10 + 1010,
            'low': np.random.randn(len(dates)) * 10 + 990,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        # Create mock loaders
        self.fred_loader = MockFREDLoader(self.fred_dir)
        self.barchart_loader = MockBarchartLoader(self.barchart_dir)

        # Add mock data to loaders
        self.fred_loader.add_mock_data('FEDFUNDS', self.fred_data)
        self.fred_loader.add_mock_data('GDP', self.fred_data * 2)  # Different data

        self.barchart_loader.add_mock_data('ADDA', self.barchart_data)
        self.barchart_loader.add_mock_data('BTCX', self.barchart_data * 3)  # Different data

        # Create the alternative loader
        self.alt_loader = AlternativeMarketDataLoader([
            self.fred_loader,
            self.barchart_loader
        ])

    def tearDown(self):
        """Clean up after each test"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_simple_caching(self):
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=5)
        original_data = pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 4.0, 5.0],
            'open': [1.0, 2.0, 3.0, 4.0, 5.0],
            'high': [1.0, 2.0, 3.0, 4.0, 5.0],
            'low': [1.0, 2.0, 3.0, 4.0, 5.0],
            'volume': [0, 0, 0, 0, 0]
        }, index=dates)

        modified_data = pd.DataFrame({
            'close': [100.0, 200.0, 300.0, 400.0, 500.0],
            'open': [100.0, 200.0, 300.0, 400.0, 500.0],
            'high': [100.0, 200.0, 300.0, 400.0, 500.0],
            'low': [100.0, 200.0, 300.0, 400.0, 500.0],
            'volume': [0, 0, 0, 0, 0]
        }, index=dates)

        # Set up the mock loader
        mock_loader = SimpleMockLoader()
        mock_loader.set_data('TEST', original_data)

        # Create the alternative loader
        alt_loader = AlternativeMarketDataLoader([mock_loader])

        # First load should get original data
        first_result = alt_loader.load_ticker('MOCK:TEST')
        self.assertIsNotNone(first_result)
        pd.testing.assert_frame_equal(first_result, original_data)

        # Change the data in the mock loader
        mock_loader.set_data('TEST', modified_data)

        # Second load should still get original data (from cache)
        second_result = alt_loader.load_ticker('MOCK:TEST')
        self.assertIsNotNone(second_result)
        pd.testing.assert_frame_equal(second_result, original_data)

        # Verify that modified data is indeed different
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(original_data, modified_data)

        # Clear the cache
        alt_loader._data_cache = {}

        # Third load should get the modified data
        third_result = alt_loader.load_ticker('MOCK:TEST')
        self.assertIsNotNone(third_result)
        pd.testing.assert_frame_equal(third_result, modified_data)

    def test_initialization(self):
        """Test proper initialization of the loader"""
        # Check that source loaders are properly mapped
        self.assertEqual(len(self.alt_loader.source_loaders), 2)
        self.assertIn('FRED', self.alt_loader.source_loaders)
        self.assertIn('BARCHARTINDEX', self.alt_loader.source_loaders)

        # Check get_available_sources
        sources = self.alt_loader.get_available_sources()
        self.assertEqual(len(sources), 2)
        self.assertIn('FRED', sources)
        self.assertIn('BARCHARTINDEX', sources)

    def test_load_fred_ticker(self):
        """Test loading a ticker from FRED"""
        # Test loading a FRED ticker
        result = self.alt_loader.load_ticker('FRED:FEDFUNDS')

        # Verify the result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.fred_data))
        pd.testing.assert_frame_equal(result, self.fred_data)

    def test_load_barchart_ticker(self):
        """Test loading a ticker from Barchart"""
        # Test loading a Barchart ticker
        result = self.alt_loader.load_ticker('BARCHARTINDEX:ADDA')

        # Verify the result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.barchart_data))
        pd.testing.assert_frame_equal(result, self.barchart_data)

    def test_load_nonexistent_ticker(self):
        """Test loading a ticker that doesn't exist"""
        # Test loading a non-existent FRED ticker
        result = self.alt_loader.load_ticker('FRED:NONEXISTENT')
        self.assertIsNone(result)

        # Test loading a non-existent Barchart ticker
        result = self.alt_loader.load_ticker('BARCHARTINDEX:NONEXISTENT')
        self.assertIsNone(result)

    def test_load_invalid_source(self):
        """Test loading a ticker with an invalid source"""
        # Test loading a ticker with an invalid source
        result = self.alt_loader.load_ticker('INVALID:TICKER')
        self.assertIsNone(result)

    def test_load_invalid_format(self):
        """Test loading a ticker with invalid format"""
        # Test loading a ticker without a source prefix
        result = self.alt_loader.load_ticker('FEDFUNDS')
        self.assertIsNone(result)

    def test_fetch_data_single(self):
        """Test fetching a single ticker"""
        # Test fetching a single ticker
        result = self.alt_loader.fetch_data(['FRED:FEDFUNDS'])

        # Verify the result
        self.assertIn('FRED:FEDFUNDS', result)
        pd.testing.assert_frame_equal(result['FRED:FEDFUNDS'], self.fred_data)

    def test_fetch_data_multiple(self):
        """Test fetching multiple tickers"""
        # Test fetching multiple tickers
        result = self.alt_loader.fetch_data([
            'FRED:FEDFUNDS',
            'BARCHARTINDEX:ADDA',
            'FRED:GDP'
        ])

        # Verify the result
        self.assertEqual(len(result), 3)
        self.assertIn('FRED:FEDFUNDS', result)
        self.assertIn('BARCHARTINDEX:ADDA', result)
        self.assertIn('FRED:GDP', result)

        pd.testing.assert_frame_equal(result['FRED:FEDFUNDS'], self.fred_data)
        pd.testing.assert_frame_equal(result['BARCHARTINDEX:ADDA'], self.barchart_data)
        pd.testing.assert_frame_equal(result['FRED:GDP'], self.fred_data * 2)

    def test_fetch_data_mixed_valid_invalid(self):
        """Test fetching a mix of valid and invalid tickers"""
        # Test fetching a mix of valid and invalid tickers
        result = self.alt_loader.fetch_data([
            'FRED:FEDFUNDS',
            'INVALID:TICKER',
            'BARCHARTINDEX:NONEXISTENT'
        ])

        # Verify the result - should only contain valid tickers
        self.assertEqual(len(result), 1)
        self.assertIn('FRED:FEDFUNDS', result)
        pd.testing.assert_frame_equal(result['FRED:FEDFUNDS'], self.fred_data)

    def test_loader_without_source_identifier(self):
        """Test adding a loader without a SOURCE_IDENTIFIER"""

        # Create a loader class without SOURCE_IDENTIFIER
        class InvalidLoader:
            def __init__(self):
                pass

            def load_ticker(self, ticker):
                return None

        # Initialize with the invalid loader
        invalid_loader = InvalidLoader()
        alt_loader = AlternativeMarketDataLoader([invalid_loader])

        # Should have no source loaders
        self.assertEqual(len(alt_loader.source_loaders), 0)

    def test_fetch_data_real_file(self):
        """Test fetching data from an actual file"""
        # Create a real CSV file
        barchart_file = os.path.join(self.barchart_dir, 'TEST.csv')
        dates = pd.date_range(start='2020-01-01', end='2020-01-05')
        test_data = pd.DataFrame({
            'date': dates,
            'open': [1000, 1010, 1020, 1030, 1040],
            'high': [1050, 1060, 1070, 1080, 1090],
            'low': [990, 980, 970, 960, 950],
            'close': [1020, 1030, 1040, 1050, 1060],
            'volume': [5000, 6000, 7000, 8000, 9000]
        })
        test_data.to_csv(barchart_file, index=False)

        # Load the data through the alternative loader
        result = self.alt_loader.load_ticker('BARCHARTINDEX:TEST')

        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(dates))
        self.assertEqual(result.loc[dates[0], 'close'], 1020)


if __name__ == '__main__':
    unittest.main()