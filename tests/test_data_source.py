import unittest
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Union

import numpy as np

from portwine.data.stores.parquet import ParquetDataStore
from portwine.data.source import DataSource
from portwine.data.provider import DataProvider


class DummyProvider(DataProvider):
    def __init__(self):
        self.calls = []
        # Preload some simple data
        self.data = {
            'AAPL': {
                datetime(2023, 1, 1): {'close': 101.0},
                datetime(2023, 1, 2): {'close': 102.0},
            }
        }

    def get_data(self, identifier: str, start_date: datetime, end_date: Optional[datetime] = None):
        self.calls.append((identifier, start_date, end_date))
        series = self.data.get(identifier, {})
        # filter by range inclusive
        end_date = end_date or start_date
        out = {}
        for dt, fields in series.items():
            if start_date <= dt <= end_date:
                out[dt] = dict(fields)
        return out


class TestDataSource(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = ParquetDataStore(self.temp_dir)
        self.provider = DummyProvider()
        self.source = DataSource(self.provider, self.store)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_cache_miss_fetches_and_populates_store(self):
        dt = np.datetime64('2023-01-01')
        # Initially missing
        self.assertIsNone(self.store.get('AAPL', dt))
        # DataSource should fetch and then return
        rec = self.source.get('AAPL', dt)
        self.assertIsInstance(rec, dict)
        self.assertEqual(rec.get('close'), 101.0)
        # Subsequent store get now hits
        rec2 = self.store.get('AAPL', dt)
        self.assertIsNotNone(rec2)

    def test_get_cache_hit_avoids_provider_call(self):
        dt = np.datetime64('2023-01-02')
        # Pre-populate store directly
        self.store.add('AAPL', {datetime(2023, 1, 2): {'close': 102.0}})
        call_count_before = len(self.provider.calls)
        rec = self.source.get('AAPL', dt)
        self.assertEqual(rec.get('close'), 102.0)
        self.assertEqual(len(self.provider.calls), call_count_before)

    def test_get_all_miss_fetches_and_populates_store(self):
        start = np.datetime64('2023-01-01')
        end = np.datetime64('2023-01-02')
        # Ensure empty
        self.assertIsNone(self.store.get_all('AAPL', start, end))
        result = self.source.get_all('AAPL', start, end)
        self.assertIsNotNone(result)
        dates = list(result.keys())
        self.assertEqual(dates[0], datetime(2023, 1, 1))
        self.assertEqual(dates[-1], datetime(2023, 1, 2))

    def test_identifiers_passthrough(self):
        self.store.add('AAPL', {datetime(2023, 1, 1): {'close': 101.0}})
        self.assertIn('AAPL', self.source.identifiers())

    def test_latest_passthrough(self):
        self.store.add('AAPL', {
            datetime(2023, 1, 1): {'close': 101.0},
            datetime(2023, 1, 2): {'close': 102.0},
        })
        latest_dt = self.source.latest('AAPL')
        self.assertEqual(latest_dt, datetime(2023, 1, 2))
        latest_row = self.source.get_latest('AAPL')
        self.assertEqual(latest_row.get('close'), 102.0)


if __name__ == '__main__':
    unittest.main()


