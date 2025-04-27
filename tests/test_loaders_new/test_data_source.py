import unittest
from datetime import datetime, timedelta

from portwine.loaders_new.data_source import DataSource


class TestDataSourceValidation(unittest.TestCase):
    def setUp(self):
        # Create a concrete class just for testing the base methods
        class TestSource(DataSource):
            def download_historical(self, ticker, start_date=None, end_date=None, store=True):
                pass
            def get(self, ticker, timestamp):
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


if __name__ == '__main__':
    unittest.main() 