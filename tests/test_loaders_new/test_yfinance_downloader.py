import unittest
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from portwine.loaders_new.yfinance_downloader import YFinanceDownloader
import shutil

class TestYFinanceDownloader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.start_date = datetime(2020, 1, 1)
        self.downloader = YFinanceDownloader(self.test_dir, start_date=self.start_date)
        
        # Create mock data for successful downloads
        # Note: yf.download() returns a DataFrame with Date as the index
        dates = pd.date_range('2024-01-01', periods=3)
        self.mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200],
            'Adj Close': [103.0, 104.0, 105.0]
        }, index=dates)
        self.mock_data.index.name = 'Date'  # yf.download() uses 'Date' as index name

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.downloader.data_path, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_init_creates_directory(self):
        """Test that the data directory is created on initialization"""
        test_path = os.path.join(self.test_dir, "subdir")
        downloader = YFinanceDownloader(test_path)
        self.assertTrue(os.path.exists(test_path))

    def test_init_default_start_date(self):
        """Test that default start date is set correctly"""
        downloader = YFinanceDownloader(self.test_dir)
        expected_date = datetime.now() - timedelta(days=5*365)
        self.assertAlmostEqual(
            downloader.start_date.timestamp(),
            expected_date.timestamp(),
            delta=5  # Allow 5 seconds difference due to test execution time
        )

    @patch('yfinance.Ticker')
    def test_download_ticker_success(self, mock_ticker):
        """Test successful ticker download"""
        # Create mock data
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [101.0],
            'Low': [99.0],
            'Close': [100.5],
            'Volume': [1000000]
        }, index=pd.DatetimeIndex([datetime(2020, 1, 1)], name='timestamp'))

        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        # Test download
        result = self.downloader.download_ticker('AAPL')

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.index.name, 'timestamp')
        self.assertEqual(result.iloc[0]['open'], 100.0)
        self.assertEqual(result.iloc[0]['high'], 101.0)
        self.assertEqual(result.iloc[0]['low'], 99.0)
        self.assertEqual(result.iloc[0]['close'], 100.5)
        self.assertEqual(result.iloc[0]['volume'], 1000000)

        # Verify file was saved
        parquet_path = os.path.join(self.test_dir, 'AAPL.parquet')
        self.assertTrue(os.path.exists(parquet_path))

    @patch('yfinance.Ticker')
    def test_download_ticker_empty_data(self, mock_ticker):
        """Test download with empty data"""
        # Setup mock to return empty DataFrame
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        result = self.downloader.download_ticker('AAPL')
        self.assertIsNone(result)

    @patch('yfinance.Ticker')
    def test_download_ticker_missing_columns(self, mock_ticker):
        """Test download with missing columns"""
        # Create mock data with missing columns
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'Close': [100.5],
            'Volume': [1000000]
        }, index=pd.DatetimeIndex([datetime(2020, 1, 1)]))

        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        result = self.downloader.download_ticker('AAPL')
        self.assertIsNone(result)

    def test_load_ticker_from_parquet(self):
        """Test loading ticker data from existing parquet file"""
        # Create test data
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        }, index=pd.DatetimeIndex([datetime(2020, 1, 1)], name='timestamp'))

        # Save test data
        parquet_path = os.path.join(self.test_dir, 'AAPL.parquet')
        df.to_parquet(parquet_path)

        # Test loading
        result = self.downloader.load_ticker('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(result.index.name, 'timestamp')
        self.assertEqual(result.iloc[0]['open'], 100.0)

    @patch('yfinance.Ticker')
    def test_load_ticker_download_fallback(self, mock_ticker):
        """Test that load_ticker falls back to download if parquet doesn't exist"""
        # Create mock data
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [101.0],
            'Low': [99.0],
            'Close': [100.5],
            'Volume': [1000000]
        }, index=pd.DatetimeIndex([datetime(2020, 1, 1)], name='timestamp'))

        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance

        # Test loading non-existent file
        result = self.downloader.load_ticker('AAPL')
        self.assertIsNotNone(result)
        mock_ticker.assert_called_once_with('AAPL')

if __name__ == '__main__':
    unittest.main() 