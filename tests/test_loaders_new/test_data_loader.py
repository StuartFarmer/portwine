from datetime import datetime
from typing import List, Optional

import pandas as pd
import unittest

from portwine.loaders_new.data_loader import DataLoader
from portwine.loaders_new.data_source import DataSource


class TestDataSource(DataSource):
    """A test implementation of DataSource."""
    
    def __init__(self, source_id: str):
        self.source_id = source_id
        self.data = {}

    def _fetch_historical(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Not implemented for this test."""
        raise NotImplementedError

    def get_latest(self, ticker: str) -> Optional[pd.DataFrame]:
        """Not implemented for this test."""
        raise NotImplementedError

    def download_historical(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Not implemented for this test."""
        raise NotImplementedError


class InMemoryDataLoader(DataLoader):
    """A test implementation of DataLoader that uses an in-memory dictionary."""
    
    def __init__(self, market_data: DataSource, alternative_data: List[DataSource]):
        """Initialize the test data loader with an empty data dictionary."""
        super().__init__(market_data, alternative_data)
        self.data = {}

    def store(self, data: pd.DataFrame, ticker: str, source: Optional[str] = None) -> bool:
        """
        Store data in the in-memory dictionary.

        Args:
            data: DataFrame with OHLCV data
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            True if storage was successful, False otherwise
        """
        try:
            source = source or self.market_data.source_id
            if source not in self.data:
                self.data[source] = {}
            self.data[source][ticker] = data.copy()
            return True
        except Exception:
            return False

    def load(
        self,
        ticker: str,
        source: Optional[str] = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> pd.DataFrame | None:
        """
        Load data from the in-memory dictionary with optional date filtering.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        try:
            source = source or self.market_data.source_id
            if source not in self.data or ticker not in self.data[source]:
                return None
                
            df = self.data[source][ticker].copy()
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            return df if not df.empty else None
            
        except Exception:
            return None

    def has(self, ticker: str, source: Optional[str] = None) -> bool:
        """
        Check if data exists for a ticker, optionally specifying a source.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            True if data exists, False otherwise
        """
        try:
            source = source or self.market_data.source_id
            return source in self.data and ticker in self.data[source]
        except Exception:
            return False

    def get_date_index(self, ticker: str, source: Optional[str] = None) -> Optional[List[datetime]]:
        """
        Get a sorted list of timestamps available for a ticker, optionally specifying a source.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            List of sorted timestamps or None if no data exists
        """
        try:
            source = source or self.market_data.source_id
            if not self.has(ticker, source):
                return None
            return sorted(self.data[source][ticker].index.tolist())
        except Exception:
            return None


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.market_data = TestDataSource("MARKET")
        self.alternative_data = [TestDataSource("ALT1"), TestDataSource("ALT2")]
        self.loader = InMemoryDataLoader(self.market_data, self.alternative_data)
        self.dates = pd.date_range('2020-01-01', periods=3)
        self.test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        }, index=self.dates)

    def test_initialization(self):
        """Test DataLoader initialization with market and alternative data sources."""
        self.assertEqual(self.loader.market_data.source_id, "MARKET")
        self.assertEqual(len(self.loader.alternative_data), 2)
        self.assertEqual(self.loader.alternative_data[0].source_id, "ALT1")
        self.assertEqual(self.loader.alternative_data[1].source_id, "ALT2")

    def test_store_data(self):
        """Test storing data in the test loader."""
        # Store data without specifying source (uses market_data)
        success = self.loader.store(self.test_data, 'AAPL')
        self.assertTrue(success)

        # Verify data was stored correctly
        self.assertIn('MARKET', self.loader.data)
        self.assertIn('AAPL', self.loader.data['MARKET'])
        
        stored_data = self.loader.data['MARKET']['AAPL']
        self.assertIsInstance(stored_data, pd.DataFrame)
        self.assertEqual(len(stored_data), 3)
        self.assertEqual(stored_data.iloc[0]['open'], 100.0)
        self.assertEqual(stored_data.iloc[2]['close'], 105.0)

    def test_store_multiple_sources(self):
        """Test storing data for multiple sources."""
        # Store data for first source
        success1 = self.loader.store(self.test_data, 'AAPL', 'ALT1')
        self.assertTrue(success1)

        # Store data for second source
        success2 = self.loader.store(self.test_data, 'MSFT', 'ALT2')
        self.assertTrue(success2)

        # Verify both sources were stored
        self.assertIn('ALT1', self.loader.data)
        self.assertIn('ALT2', self.loader.data)
        self.assertIn('AAPL', self.loader.data['ALT1'])
        self.assertIn('MSFT', self.loader.data['ALT2'])

    def test_store_multiple_tickers(self):
        """Test storing multiple tickers for the same source."""
        # Store first ticker
        success1 = self.loader.store(self.test_data, 'AAPL')
        self.assertTrue(success1)

        # Store second ticker
        success2 = self.loader.store(self.test_data, 'MSFT')
        self.assertTrue(success2)

        # Verify both tickers were stored
        self.assertIn('AAPL', self.loader.data['MARKET'])
        self.assertIn('MSFT', self.loader.data['MARKET'])

    def test_store_overwrite(self):
        """Test overwriting existing data."""
        # Store initial data
        self.loader.store(self.test_data, 'AAPL')

        # Create new data
        new_data = pd.DataFrame({
            'open': [200.0],
            'high': [205.0],
            'low': [195.0],
            'close': [203.0],
            'volume': [2000]
        }, index=[datetime(2020, 1, 1)])

        # Overwrite existing data
        success = self.loader.store(new_data, 'AAPL')
        self.assertTrue(success)

        # Verify data was overwritten
        stored_data = self.loader.data['MARKET']['AAPL']
        self.assertEqual(len(stored_data), 1)
        self.assertEqual(stored_data.iloc[0]['open'], 200.0)
        self.assertEqual(stored_data.iloc[0]['close'], 203.0)

    def test_load_nonexistent(self):
        """Test loading nonexistent data"""
        df = self.loader.load('GOOGL')
        self.assertIsNone(df)

    def test_load_with_date_range(self):
        """Test loading data with date range filters"""
        # Store data
        self.loader.store(self.test_data, 'AAPL')

        # Test start date filter
        start_date = datetime(2020, 1, 2)
        df = self.loader.load('AAPL', start_date=start_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['open'], 101.0)

        # Test end date filter
        end_date = datetime(2020, 1, 2)
        df = self.loader.load('AAPL', end_date=end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['close'], 104.0)

        # Test both filters
        df = self.loader.load('AAPL', start_date=start_date, end_date=end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['open'], 101.0)
        self.assertEqual(df.iloc[0]['close'], 104.0)

    def test_load_empty_result(self):
        """Test loading data with date range that results in empty data"""
        # Store data
        self.loader.store(self.test_data, 'AAPL')

        # Test date range with no data
        start_date = datetime(2021, 1, 1)
        df = self.loader.load('AAPL', start_date=start_date)
        self.assertIsNone(df)

    def test_has_data(self):
        """Test checking for data existence."""
        # Initially no data exists
        self.assertFalse(self.loader.has('AAPL'))
        self.assertFalse(self.loader.has('MSFT'))

        # Store data for AAPL
        self.loader.store(self.test_data, 'AAPL')
        
        # Verify AAPL exists but MSFT doesn't
        self.assertTrue(self.loader.has('AAPL'))
        self.assertFalse(self.loader.has('MSFT'))
        
        # Store data for MSFT
        self.loader.store(self.test_data, 'MSFT')
        
        # Verify both exist
        self.assertTrue(self.loader.has('AAPL'))
        self.assertTrue(self.loader.has('MSFT'))
        
        # Check nonexistent source
        self.assertFalse(self.loader.has('AAPL', 'NONEXISTENT'))

    def test_has_after_store_error(self):
        """Test has method behavior after failed store operation."""
        # Create data that might cause store to fail
        bad_data = None
        
        # Attempt to store invalid data
        success = self.loader.store(bad_data, 'AAPL')
        self.assertFalse(success)
        
        # Verify has returns False
        self.assertFalse(self.loader.has('AAPL'))

    def test_get_date_index(self):
        """Test getting date index for data."""
        # Initially no data exists
        self.assertIsNone(self.loader.get_date_index('AAPL'))

        # Store data
        self.loader.store(self.test_data, 'AAPL')
        
        # Get date index
        dates = self.loader.get_date_index('AAPL')
        self.assertIsNotNone(dates)
        self.assertEqual(len(dates), 3)
        self.assertEqual(dates, sorted(self.test_data.index.tolist()))
        
        # Verify dates are in correct order
        self.assertEqual(dates[0], datetime(2020, 1, 1))
        self.assertEqual(dates[1], datetime(2020, 1, 2))
        self.assertEqual(dates[2], datetime(2020, 1, 3))

    def test_get_date_index_nonexistent(self):
        """Test getting date index for nonexistent data."""
        # Check nonexistent source
        self.assertIsNone(self.loader.get_date_index('AAPL', 'NONEXISTENT'))
        
        # Check nonexistent ticker
        self.loader.store(self.test_data, 'AAPL')
        self.assertIsNone(self.loader.get_date_index('MSFT'))

    def test_get_date_index_after_store_error(self):
        """Test get_date_index behavior after failed store operation."""
        # Create data that might cause store to fail
        bad_data = None
        
        # Attempt to store invalid data
        success = self.loader.store(bad_data, 'AAPL')
        self.assertFalse(success)
        
        # Verify get_date_index returns None
        self.assertIsNone(self.loader.get_date_index('AAPL'))


if __name__ == '__main__':
    unittest.main() 