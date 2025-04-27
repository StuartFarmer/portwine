import unittest
from datetime import datetime, timedelta
from typing import Optional, Any

import pandas as pd

from portwine.loaders_new.data_source import DataSource


class TestDataSource(DataSource):
    """A test data source that uses a dictionary to simulate data."""
    
    def __init__(self, name: str, test_data: dict[str, pd.DataFrame]):
        """
        Initialize the test data source.

        Args:
            name: Source name
            test_data: Dictionary mapping tickers to their test data
        """
        super().__init__(name)
        self._test_data = test_data

    def _fetch_historical(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        store: bool = True
    ) -> pd.DataFrame | None:
        """
        Get test data for a ticker.

        Args:
            ticker: The ticker symbol
            start_date: Optional start date (ignored in test implementation)
            end_date: Optional end date (ignored in test implementation)
            store: Ignored for test source

        Returns:
            DataFrame with OHLCV data or None if ticker not found
        """
        if ticker not in self._test_data:
            return None

        return self._test_data[ticker].copy()

    def get_latest(self, ticker: str) -> dict[str, Any] | None:
        """
        Get the latest available data for a ticker.

        Args:
            ticker: The ticker symbol

        Returns:
            Dictionary with OHLCV data and timestamp, or None if not available
        """
        if ticker not in self._test_data:
            return None

        df = self._test_data[ticker]
        if df.empty:
            return None

        # Get the latest row
        latest_row = df.iloc[-1]
        return {
            'timestamp': df.index[-1],
            'open': float(latest_row['open']),
            'high': float(latest_row['high']),
            'low': float(latest_row['low']),
            'close': float(latest_row['close']),
            'volume': float(latest_row['volume'])
        }

    def get(self, ticker: str, timestamp: datetime) -> dict[str, float] | None:
        """Not implemented for this test class."""
        raise NotImplementedError

    def sync(self, ticker: str) -> bool:
        """Not implemented for this test class."""
        return True


class TestDataSourceValidation(unittest.TestCase):
    def setUp(self):
        # Create a concrete class just for testing the base methods
        class TestSource(DataSource):
            def _fetch_historical(self, ticker, start_date=None, end_date=None, store=True):
                pass
            def get_latest(self, ticker):
                pass
            def sync(self, ticker):
                pass
        
        self.source = TestSource('TEST')

    def test_validate_timestamp(self):
        """Test timestamp validation logic"""
        # Test past timestamp
        past = datetime.now() - timedelta(days=1)
        self.assertTrue(self.source._validate_timestamp(past))

        # Test current timestamp
        now = datetime.now()
        self.assertTrue(self.source._validate_timestamp(now))

        # Test future timestamp
        future = datetime.now() + timedelta(days=1)
        self.assertFalse(self.source._validate_timestamp(future))

    def test_validate_date_range(self):
        """Test date range validation logic"""
        # Test valid range
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        self.assertTrue(self.source._validate_date_range(start, end))

        # Test invalid range (start after end)
        self.assertFalse(self.source._validate_date_range(end, start))

        # Test with None values
        self.assertTrue(self.source._validate_date_range(None, end))
        self.assertTrue(self.source._validate_date_range(start, None))
        self.assertTrue(self.source._validate_date_range(None, None))

        # Test with same start and end
        same = datetime(2020, 1, 1)
        self.assertTrue(self.source._validate_date_range(same, same))


class TestDataSourceDownload(unittest.TestCase):
    def setUp(self):
        # Create test data
        dates = pd.date_range('2020-01-01', periods=3)
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [100.0, 101.0, 102.0],
                'high': [105.0, 106.0, 107.0],
                'low': [95.0, 96.0, 97.0],
                'close': [103.0, 104.0, 105.0],
                'volume': [1000, 1100, 1200]
            }, index=dates),
            'MSFT': pd.DataFrame({
                'open': [200.0, 201.0, 202.0],
                'high': [205.0, 206.0, 207.0],
                'low': [195.0, 196.0, 197.0],
                'close': [203.0, 204.0, 205.0],
                'volume': [2000, 2100, 2200]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)

    def test_download_historical_full(self):
        """Test downloading full historical data"""
        # Test AAPL
        df = self.source.download_historical('AAPL')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]['open'], 100.0)
        self.assertEqual(df.iloc[2]['close'], 105.0)

        # Test MSFT
        df = self.source.download_historical('MSFT')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]['open'], 200.0)
        self.assertEqual(df.iloc[2]['close'], 205.0)

        # Test non-existent ticker
        df = self.source.download_historical('GOOGL')
        self.assertIsNone(df)

    def test_download_historical_date_range(self):
        """Test downloading data with date range filters"""
        # Test start date filter
        start = datetime(2020, 1, 2)
        df = self.source.download_historical('AAPL', start_date=start)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['open'], 101.0)

        # Test end date filter
        end = datetime(2020, 1, 2)
        df = self.source.download_historical('AAPL', end_date=end)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['close'], 104.0)

        # Test both filters
        df = self.source.download_historical('AAPL', start_date=start, end_date=end)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['open'], 101.0)
        self.assertEqual(df.iloc[0]['close'], 104.0)

        # Test empty result
        start = datetime(2021, 1, 1)
        df = self.source.download_historical('AAPL', start_date=start)
        self.assertIsNone(df)


class TestDataSourceLongRange(unittest.TestCase):
    def setUp(self):
        # Create test data with a longer date range
        dates = pd.date_range('2000-01-01', '2020-12-31', freq='M')
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0] * len(dates),
                'high': [2.0] * len(dates),
                'low': [0.5] * len(dates),
                'close': [1.5] * len(dates),
                'volume': [1000] * len(dates)
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)

    def test_download_historical_start_date(self):
        """Test downloading data with a specific start date"""
        # Test start date in the middle of the range
        start_date = datetime(2010, 1, 1)
        df = self.source.download_historical('AAPL', start_date=start_date)
        
        self.assertIsNotNone(df)
        self.assertEqual(df.index[0], pd.Timestamp('2010-01-31'))
        self.assertEqual(df.index[-1], pd.Timestamp('2020-12-31'))
        self.assertEqual(len(df), 132)  # 11 years * 12 months

        # Test start date at the beginning
        start_date = datetime(2000, 1, 1)
        df = self.source.download_historical('AAPL', start_date=start_date)
        self.assertIsNotNone(df)
        self.assertEqual(df.index[0], pd.Timestamp('2000-01-31'))
        self.assertEqual(len(df), 252)  # 21 years * 12 months

        # Test start date at the end
        start_date = datetime(2020, 12, 1)
        df = self.source.download_historical('AAPL', start_date=start_date)
        self.assertIsNotNone(df)
        self.assertEqual(df.index[0], pd.Timestamp('2020-12-31'))
        self.assertEqual(len(df), 1)


class TestDataSourceNonContinuous(unittest.TestCase):
    def setUp(self):
        # Create test data with non-continuous dates
        dates = [
            datetime(2000, 1, 1),
            datetime(2000, 3, 1),
            datetime(2000, 5, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0],
                'high': [2.0, 3.0, 4.0],
                'low': [0.5, 1.5, 2.5],
                'close': [1.5, 2.5, 3.5],
                'volume': [1000, 2000, 3000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)

    def test_download_historical_start_date(self):
        """Test downloading data with a start date between data points"""
        # Test start date between first and second data points
        start_date = datetime(2000, 2, 1)
        df = self.source.download_historical('AAPL', start_date=start_date)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], pd.Timestamp('2000-03-01'))
        self.assertEqual(df.index[1], pd.Timestamp('2000-05-01'))
        self.assertEqual(df.iloc[0]['open'], 2.0)
        self.assertEqual(df.iloc[1]['open'], 3.0)


class TestDataSourceEarlyStart(unittest.TestCase):
    def setUp(self):
        # Create test data starting from 2000-03-01
        dates = [
            datetime(2000, 3, 1),
            datetime(2000, 4, 1),
            datetime(2000, 5, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0],
                'high': [2.0, 3.0, 4.0],
                'low': [0.5, 1.5, 2.5],
                'close': [1.5, 2.5, 3.5],
                'volume': [1000, 2000, 3000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)

    def test_download_historical_early_start(self):
        """Test downloading data when requested start date is before data begins"""
        # Test start date before data begins
        start_date = datetime(2000, 1, 1)
        df = self.source.download_historical('AAPL', start_date=start_date)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index[0], pd.Timestamp('2000-03-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-05-01'))
        self.assertEqual(df.iloc[0]['open'], 1.0)
        self.assertEqual(df.iloc[-1]['open'], 3.0)


class TestDataSourceEndDate(unittest.TestCase):
    def setUp(self):
        # Create test data with monthly dates
        dates = [
            datetime(2000, 1, 1),
            datetime(2000, 2, 1),
            datetime(2000, 3, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0],
                'high': [2.0, 3.0, 4.0],
                'low': [0.5, 1.5, 2.5],
                'close': [1.5, 2.5, 3.5],
                'volume': [1000, 2000, 3000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)

    def test_download_historical_no_end_date(self):
        """Test downloading data with no end date (should return all data)"""
        df = self.source.download_historical('AAPL')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index[0], pd.Timestamp('2000-01-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-03-01'))
        self.assertEqual(df.iloc[0]['open'], 1.0)
        self.assertEqual(df.iloc[-1]['open'], 3.0)

    def test_download_historical_exact_end_date(self):
        """Test downloading data with an end date that matches a data point"""
        end_date = datetime(2000, 2, 1)
        df = self.source.download_historical('AAPL', end_date=end_date)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], pd.Timestamp('2000-01-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-02-01'))
        self.assertEqual(df.iloc[0]['open'], 1.0)
        self.assertEqual(df.iloc[-1]['open'], 2.0)

    def test_download_historical_missing_end_date(self):
        """Test downloading data with an end date between data points"""
        end_date = datetime(2000, 2, 15)  # Between Feb 1 and Mar 1
        df = self.source.download_historical('AAPL', end_date=end_date)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], pd.Timestamp('2000-01-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-02-01'))
        self.assertEqual(df.iloc[0]['open'], 1.0)
        self.assertEqual(df.iloc[-1]['open'], 2.0)


class TestDataSourceCombinedDates(unittest.TestCase):
    def setUp(self):
        self.start_date = datetime(2000, 2, 1)
        self.end_date = datetime(2000, 5, 1)

    def test_case1_continuous_data(self):
        """Test Case 1: Continuous data covering entire range"""
        dates = [
            datetime(2000, 1, 1),
            datetime(2000, 2, 1),
            datetime(2000, 3, 1),
            datetime(2000, 4, 1),
            datetime(2000, 5, 1),
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'high': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                'low': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                'close': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                'volume': [1000, 2000, 3000, 4000, 5000, 6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)
        self.assertEqual(df.index[0], pd.Timestamp('2000-02-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-05-01'))
        self.assertEqual(df.iloc[0]['open'], 2.0)
        self.assertEqual(df.iloc[-1]['open'], 5.0)

    def test_case2_late_start(self):
        """Test Case 2: Data starts after requested start date"""
        dates = [
            datetime(2000, 3, 1),
            datetime(2000, 4, 1),
            datetime(2000, 5, 1),
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [3.0, 4.0, 5.0, 6.0],
                'high': [4.0, 5.0, 6.0, 7.0],
                'low': [2.5, 3.5, 4.5, 5.5],
                'close': [3.5, 4.5, 5.5, 6.5],
                'volume': [3000, 4000, 5000, 6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index[0], pd.Timestamp('2000-03-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-05-01'))
        self.assertEqual(df.iloc[0]['open'], 3.0)
        self.assertEqual(df.iloc[-1]['open'], 5.0)

    def test_case3_missing_data(self):
        """Test Case 3: Missing data point in the middle"""
        dates = [
            datetime(2000, 1, 1),
            datetime(2000, 2, 1),
            datetime(2000, 3, 1),
            datetime(2000, 5, 1),
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0, 5.0, 6.0],
                'high': [2.0, 3.0, 4.0, 6.0, 7.0],
                'low': [0.5, 1.5, 2.5, 4.5, 5.5],
                'close': [1.5, 2.5, 3.5, 5.5, 6.5],
                'volume': [1000, 2000, 3000, 5000, 6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index[0], pd.Timestamp('2000-02-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-05-01'))
        self.assertEqual(df.iloc[0]['open'], 2.0)
        self.assertEqual(df.iloc[-1]['open'], 5.0)

    def test_case4_early_end(self):
        """Test Case 4: Data ends before requested end date"""
        dates = [
            datetime(2000, 3, 1),
            datetime(2000, 4, 1),
            datetime(2000, 5, 1),
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [3.0, 4.0, 5.0, 6.0],
                'high': [4.0, 5.0, 6.0, 7.0],
                'low': [2.5, 3.5, 4.5, 5.5],
                'close': [3.5, 4.5, 5.5, 6.5],
                'volume': [3000, 4000, 5000, 6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index[0], pd.Timestamp('2000-03-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-05-01'))
        self.assertEqual(df.iloc[0]['open'], 3.0)
        self.assertEqual(df.iloc[-1]['open'], 5.0)

    def test_case5_short_data(self):
        """Test Case 5: Data ends before requested end date"""
        dates = [
            datetime(2000, 3, 1),
            datetime(2000, 4, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [3.0, 4.0],
                'high': [4.0, 5.0],
                'low': [2.5, 3.5],
                'close': [3.5, 4.5],
                'volume': [3000, 4000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], pd.Timestamp('2000-03-01'))
        self.assertEqual(df.index[-1], pd.Timestamp('2000-04-01'))
        self.assertEqual(df.iloc[0]['open'], 3.0)
        self.assertEqual(df.iloc[-1]['open'], 4.0)

    def test_case6_early_data(self):
        """Test Case 6: All data is before requested start date"""
        dates = [
            datetime(2000, 1, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0],
                'high': [2.0],
                'low': [0.5],
                'close': [1.5],
                'volume': [1000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNone(df)

    def test_case7_late_data(self):
        """Test Case 7: All data is after requested end date"""
        dates = [
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [6.0],
                'high': [7.0],
                'low': [5.5],
                'close': [6.5],
                'volume': [6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date, end_date=self.end_date)
        self.assertIsNone(df)

    def test_case8_end_date_only_late_data(self):
        """Test Case 8: Only end date provided, but data is after end date"""
        dates = [
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [6.0],
                'high': [7.0],
                'low': [5.5],
                'close': [6.5],
                'volume': [6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', end_date=self.end_date)
        self.assertIsNone(df)

    def test_case9_no_dates_no_data(self):
        """Test Case 9: No dates provided and no data available"""
        self.test_data = {
            'AAPL': pd.DataFrame()
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL')
        self.assertIsNone(df)

    def test_case10_start_date_only_early_data(self):
        """Test Case 10: Only start date provided, but data is before start date"""
        dates = [
            datetime(2000, 1, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0],
                'high': [2.0],
                'low': [0.5],
                'close': [1.5],
                'volume': [1000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        df = self.source.download_historical('AAPL', start_date=self.start_date)
        self.assertIsNone(df)


class TestDataSourceGetLatest(unittest.TestCase):
    def test_case1_continuous_data(self):
        """Test Case 1: Get latest data from continuous data"""
        dates = [
            datetime(2000, 1, 1),
            datetime(2000, 2, 1),
            datetime(2000, 3, 1),
            datetime(2000, 4, 1),
            datetime(2000, 5, 1),
            datetime(2000, 6, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'high': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                'low': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                'close': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                'volume': [1000, 2000, 3000, 4000, 5000, 6000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        latest = self.source.get_latest('AAPL')
        self.assertIsNotNone(latest)
        self.assertEqual(latest['timestamp'], pd.Timestamp('2000-06-01'))
        self.assertEqual(latest['open'], 6.0)
        self.assertEqual(latest['high'], 7.0)
        self.assertEqual(latest['low'], 5.5)
        self.assertEqual(latest['close'], 6.5)
        self.assertEqual(latest['volume'], 6000.0)

    def test_case2_short_data(self):
        """Test Case 2: Get latest data from shorter dataset"""
        dates = [
            datetime(2000, 1, 1),
            datetime(2000, 2, 1),
            datetime(2000, 3, 1),
            datetime(2000, 4, 1),
            datetime(2000, 5, 1)
        ]
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [1.0, 2.0, 3.0, 4.0, 5.0],
                'high': [2.0, 3.0, 4.0, 5.0, 6.0],
                'low': [0.5, 1.5, 2.5, 3.5, 4.5],
                'close': [1.5, 2.5, 3.5, 4.5, 5.5],
                'volume': [1000, 2000, 3000, 4000, 5000]
            }, index=dates)
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        latest = self.source.get_latest('AAPL')
        self.assertIsNotNone(latest)
        self.assertEqual(latest['timestamp'], pd.Timestamp('2000-05-01'))
        self.assertEqual(latest['open'], 5.0)
        self.assertEqual(latest['high'], 6.0)
        self.assertEqual(latest['low'], 4.5)
        self.assertEqual(latest['close'], 5.5)
        self.assertEqual(latest['volume'], 5000.0)

    def test_case3_no_data(self):
        """Test Case 3: Get latest data when no data is available"""
        self.test_data = {
            'AAPL': pd.DataFrame()
        }
        self.source = TestDataSource('TEST', self.test_data)
        
        latest = self.source.get_latest('AAPL')
        self.assertIsNone(latest)

    def test_case4_nonexistent_ticker(self):
        """Test Case 4: Get latest data for nonexistent ticker"""
        self.test_data = {}
        self.source = TestDataSource('TEST', self.test_data)
        
        latest = self.source.get_latest('AAPL')
        self.assertIsNone(latest)


if __name__ == '__main__':
    unittest.main() 