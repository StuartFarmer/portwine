import os
import tempfile
import unittest
from pathlib import Path
from typing import List

import pandas as pd

from portwine.loaders_new.flat_file_loader import FlatFileDataLoader
from portwine.sources.base import DataSource


class TestDataSource(DataSource):
    """A test implementation of DataSource."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.data = {}

    def _fetch_historical(self, ticker: str, start_date=None, end_date=None, store=True):
        raise NotImplementedError

    def get_latest(self, ticker: str):
        raise NotImplementedError


class TestFlatFileDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test data sources and temporary directory."""
        self.market_data = TestDataSource("POLYGON")
        self.alternative_data = [TestDataSource("FRED"), TestDataSource("BARCHART")]
        self.temp_dir = tempfile.mkdtemp()
        self.loader = FlatFileDataLoader(
            market_data=self.market_data,
            alternative_data=self.alternative_data,
            data_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_directory_structure(self):
        """Test that the directory structure is created correctly."""
        # Remove the directories created in setUp
        import shutil
        shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

        # Test that the folders do not exist initially
        market_dir = Path(self.temp_dir) / "market" / "polygon"
        alt_dir = Path(self.temp_dir) / "alternative" / "fred"
        self.assertFalse(market_dir.exists())
        self.assertFalse(alt_dir.exists())

        # Create a new loader which will create the directories
        loader = FlatFileDataLoader(
            market_data=self.market_data,
            alternative_data=self.alternative_data,
            data_dir=self.temp_dir
        )

        # Test that the folders exist
        self.assertTrue(market_dir.exists())
        self.assertTrue(alt_dir.exists())

    def test_get_file_path_source_is_none(self):
        """Test that the returned path assumes the market data source name."""
        path = self.loader._get_file_path("AAPL")
        expected = Path(self.temp_dir) / "market" / "polygon" / "AAPL.parquet"
        self.assertEqual(path, expected)

    def test_get_file_path_source_is_market_data_source(self):
        """Test that the returned path is the base dir / market data source name / ticker."""
        path = self.loader._get_file_path("AAPL", "polygon")
        expected = Path(self.temp_dir) / "market" / "polygon" / "AAPL.parquet"
        self.assertEqual(path, expected)

    def test_get_file_path_source_is_not_market_data_source(self):
        """Test that the returned path is the base dir / alternative data source name / ticker."""
        path = self.loader._get_file_path("VIX", "barchart")
        expected = Path(self.temp_dir) / "alternative" / "barchart" / "VIX.parquet"
        self.assertEqual(path, expected)

    def test_get_file_path_source_is_invalid_file_name(self):
        """Test that an exception is raised for invalid source names."""
        with self.assertRaises(ValueError):
            self.loader._get_file_path("AAPL", "invalid/source")

    def test_get_file_path_ticker_is_none(self):
        """Test that an exception is raised when ticker is None."""
        with self.assertRaises(ValueError):
            self.loader._get_file_path(None)

    def test_get_file_path_ticker_is_invalid_file_name(self):
        """Test that an exception is raised for invalid ticker names."""
        with self.assertRaises(ValueError):
            self.loader._get_file_path("invalid/ticker")


if __name__ == '__main__':
    unittest.main() 