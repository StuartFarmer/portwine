import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import pandas as pd

from portwine.loaders_new.base import MarketDataLoader


class TestMarketDataLoaderImpl(MarketDataLoader):
    """A concrete implementation of MarketDataLoader for testing."""
    def __init__(self, data_path: str = None, data: dict[str, pd.DataFrame] = None):
        super().__init__(data_path)
        self._test_data = data or {}

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        return self._test_data.get(ticker)


class TestMarketDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = TestMarketDataLoaderImpl(data_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init_creates_directory(self):
        """Test that the data directory is created if it doesn't exist"""
        test_path = os.path.join(self.test_dir, "subdir")
        loader = TestMarketDataLoaderImpl(data_path=test_path)
        self.assertTrue(os.path.exists(test_path))

    def test_init_no_data_path(self):
        """Test that loader works without a data path"""
        loader = TestMarketDataLoaderImpl()
        self.assertIsNone(loader.data_path)

    def test_load_ticker_not_implemented(self):
        """Test that base class load_ticker raises NotImplementedError"""
        loader = MarketDataLoader()  # Use base class for this test
        with self.assertRaises(NotImplementedError):
            loader.load_ticker("AAPL")

    def test_fetch_data_empty(self):
        """Test fetch_data returns empty dict when no data available"""
        result = self.loader.fetch_data(["AAPL", "MSFT"])
        self.assertEqual(result, {})

    def test_get_all_dates_empty(self):
        """Test get_all_dates returns empty list when no data available"""
        result = self.loader.get_all_dates(["AAPL", "MSFT"])
        self.assertEqual(result, [])

    def test_next_empty(self):
        """Test next returns empty dict when no data available"""
        result = self.loader.next(["AAPL"], pd.Timestamp("2021-01-01"))
        self.assertEqual(result, {"AAPL": None})

    def test_get_bar_at_or_before_empty_df(self):
        """Test _get_bar_at_or_before handles empty DataFrame"""
        df = pd.DataFrame()
        result = self.loader._get_bar_at_or_before(df, pd.Timestamp("2021-01-01"))
        self.assertIsNone(result)

    def test_get_bar_at_or_before_before_first(self):
        """Test _get_bar_at_or_before returns None for timestamp before first bar"""
        df = pd.DataFrame({
            'open': [1.0],
            'high': [2.0],
            'low': [0.5],
            'close': [1.5],
            'volume': [1000]
        }, index=[pd.Timestamp("2021-01-02")])
        
        result = self.loader._get_bar_at_or_before(df, pd.Timestamp("2021-01-01"))
        self.assertIsNone(result)

    def test_get_bar_at_or_before_exact(self):
        """Test _get_bar_at_or_before returns correct bar for exact timestamp"""
        df = pd.DataFrame({
            'open': [1.0],
            'high': [2.0],
            'low': [0.5],
            'close': [1.5],
            'volume': [1000]
        }, index=[pd.Timestamp("2021-01-01")])
        
        result = self.loader._get_bar_at_or_before(df, pd.Timestamp("2021-01-01"))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 1.0)
        self.assertEqual(result['high'], 2.0)
        self.assertEqual(result['low'], 0.5)
        self.assertEqual(result['close'], 1.5)
        self.assertEqual(result['volume'], 1000)

    def test_get_bar_at_or_before_between(self):
        """Test _get_bar_at_or_before returns previous bar for timestamp between bars"""
        df = pd.DataFrame({
            'open': [1.0, 2.0],
            'high': [2.0, 3.0],
            'low': [0.5, 1.5],
            'close': [1.5, 2.5],
            'volume': [1000, 2000]
        }, index=[pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-03")])
        
        result = self.loader._get_bar_at_or_before(df, pd.Timestamp("2021-01-02"))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 1.0)
        self.assertEqual(result['close'], 1.5)

    def test_fetch_data_with_data(self):
        """Test fetch_data returns data for available tickers"""
        test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0],
                'high': [2.0],
                'low': [0.5],
                'close': [1.5],
                'volume': [1000]
            }, index=[pd.Timestamp("2021-01-01")])
        }
        loader = TestMarketDataLoaderImpl(data=test_data)
        result = loader.fetch_data(['AAPL', 'MSFT'])
        self.assertEqual(len(result), 1)
        self.assertIn('AAPL', result)
        self.assertEqual(result['AAPL'].iloc[0]['close'], 1.5)

    def test_get_all_dates_with_data(self):
        """Test get_all_dates returns sorted dates from all tickers"""
        test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0],
                'high': [2.0],
                'low': [0.5],
                'close': [1.5],
                'volume': [1000]
            }, index=[pd.Timestamp("2021-01-01")]),
            'MSFT': pd.DataFrame({
                'open': [2.0],
                'high': [3.0],
                'low': [1.5],
                'close': [2.5],
                'volume': [2000]
            }, index=[pd.Timestamp("2021-01-02")])
        }
        loader = TestMarketDataLoaderImpl(data=test_data)
        result = loader.get_all_dates(['AAPL', 'MSFT'])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], pd.Timestamp("2021-01-01"))
        self.assertEqual(result[1], pd.Timestamp("2021-01-02"))


if __name__ == '__main__':
    unittest.main() 