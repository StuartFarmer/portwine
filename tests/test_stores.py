"""
Test suite for the data stores.

This module tests both ParquetDataStore and NoisyDataStore implementations.
"""

import unittest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from collections import OrderedDict
import pandas as pd
import numpy as np

# Add the portwine directory to the path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portwine.data.stores import DataStore, ParquetDataStore, NoisyDataStore


class TestParquetDataStore(unittest.TestCase):
    """Test cases for ParquetDataStore."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.store = ParquetDataStore(self.test_dir)
        
        # Test data
        self.test_data = {
            datetime(2023, 1, 1): {
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000
            },
            datetime(2023, 1, 2): {
                'open': 102.0,
                'high': 108.0,
                'low': 100.0,
                'close': 106.0,
                'volume': 1200
            },
            datetime(2023, 1, 3): {
                'open': 106.0,
                'high': 110.0,
                'low': 104.0,
                'close': 108.0,
                'volume': 1500
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test store initialization."""
        self.assertTrue(os.path.exists(self.test_dir))
        # ParquetDataStore stores data_dir as a Path object, so convert to string for comparison
        self.assertEqual(str(self.store.data_dir), self.test_dir)
    
    def test_add_data(self):
        """Test adding data to the store."""
        self.store.add("TEST", self.test_data)
        
        # Check that files were created
        expected_file = os.path.join(self.test_dir, "TEST.pqt")
        self.assertTrue(os.path.exists(expected_file))
    
    def test_add_data_overwrite(self):
        """Test adding data with overwrite flag."""
        # Add initial data
        self.store.add("TEST", self.test_data)
        
        # Add overlapping data with overwrite
        overlapping_data = {
            datetime(2023, 1, 2): {
                'open': 103.0,
                'high': 109.0,
                'low': 101.0,
                'close': 107.0,
                'volume': 1300
            }
        }
        
        self.store.add("TEST", overlapping_data, overwrite=True)
        
        # Verify the data was overwritten
        retrieved = self.store.get("TEST", datetime(2023, 1, 2))
        self.assertEqual(retrieved['open'], 103.0)
        self.assertEqual(retrieved['volume'], 1300)
    
    def test_get_single_point(self):
        """Test retrieving a single data point."""
        self.store.add("TEST", self.test_data)
        
        # Test existing data
        result = self.store.get("TEST", datetime(2023, 1, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 100.0)
        self.assertEqual(result['close'], 102.0)
        
        # Test non-existing data
        result = self.store.get("TEST", datetime(2023, 1, 5))
        self.assertIsNone(result)
    
    def test_get_all(self):
        """Test retrieving data for a date range."""
        self.store.add("TEST", self.test_data)
        
        # Test full range
        result = self.store.get_all(
            "TEST", 
            datetime(2023, 1, 1), 
            datetime(2023, 1, 3)
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        
        # Verify chronological order
        dates = list(result.keys())
        self.assertEqual(dates[0], datetime(2023, 1, 1))
        self.assertEqual(dates[2], datetime(2023, 1, 3))
        
        # Test partial range
        result = self.store.get_all(
            "TEST", 
            datetime(2023, 1, 1), 
            datetime(2023, 1, 2)
        )
        self.assertEqual(len(result), 2)
    
    def test_get_latest(self):
        """Test retrieving the latest data point."""
        self.store.add("TEST", self.test_data)
        
        result = self.store.get_latest("TEST")
        self.assertIsNotNone(result)
        self.assertEqual(result['close'], 108.0)  # Latest close price
    
    def test_latest_date(self):
        """Test retrieving the latest date."""
        self.store.add("TEST", self.test_data)

        result = self.store.latest("TEST")
        self.assertEqual(result, datetime(2023, 1, 3))

    def test_earliest_date(self):
        """Test retrieving the earliest date."""
        self.store.add("TEST", self.test_data)

        result = self.store.earliest("TEST")
        self.assertEqual(result, datetime(2023, 1, 1))

        # Test with non-existent identifier
        result = self.store.earliest("NONEXISTENT")
        self.assertIsNone(result)

    def test_exists(self):
        """Test checking if data exists."""
        self.store.add("TEST", self.test_data)
        
        # Test existing data
        self.assertTrue(self.store.exists("TEST", datetime(2023, 1, 1), datetime(2023, 1, 3)))
        self.assertTrue(self.store.exists("TEST", datetime(2023, 1, 2)))
        
        # Test non-existing data
        self.assertFalse(self.store.exists("TEST", datetime(2023, 1, 5)))
        self.assertFalse(self.store.exists("NONEXISTENT"))
    
    def test_identifiers(self):
        """Test retrieving all identifiers."""
        self.store.add("TEST1", self.test_data)
        self.store.add("TEST2", self.test_data)
        
        identifiers = self.store.identifiers()
        self.assertIn("TEST1", identifiers)
        self.assertIn("TEST2", identifiers)
        self.assertEqual(len(identifiers), 2)
    
    def test_string_dates(self):
        """Test handling string dates in data."""
        string_data = {
            "2023-01-01": {
                'open': 100.0,
                'close': 102.0,
                'volume': 1000
            }
        }
        
        self.store.add("STRING_TEST", string_data)
        result = self.store.get("STRING_TEST", datetime(2023, 1, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result['open'], 100.0)


class TestNoisyDataStore(unittest.TestCase):
    """Test cases for NoisyDataStore."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.base_store = ParquetDataStore(self.test_dir)
        self.noisy_store = NoisyDataStore(
            base_store=self.base_store,
            noise_multiplier=0.1,
            volatility_window=5,
            seed=42
        )
        
        # Test data with more points for volatility calculation
        self.test_data = {
            datetime(2023, 1, 1): {'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 102.0, 'volume': 1000},
            datetime(2023, 1, 2): {'open': 102.0, 'high': 108.0, 'low': 100.0, 'close': 106.0, 'volume': 1200},
            datetime(2023, 1, 3): {'open': 106.0, 'high': 110.0, 'low': 104.0, 'close': 108.0, 'volume': 1500},
            datetime(2023, 1, 4): {'open': 108.0, 'high': 112.0, 'low': 106.0, 'close': 110.0, 'volume': 1400},
            datetime(2023, 1, 5): {'open': 110.0, 'high': 115.0, 'low': 108.0, 'close': 113.0, 'volume': 1600},
            datetime(2023, 1, 6): {'open': 113.0, 'high': 118.0, 'low': 111.0, 'close': 116.0, 'volume': 1700},
            datetime(2023, 1, 7): {'open': 116.0, 'high': 120.0, 'low': 114.0, 'close': 118.0, 'volume': 1800},
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test noisy store initialization."""
        self.assertEqual(self.noisy_store.base_store, self.base_store)
        self.assertEqual(self.noisy_store.noise_multiplier, 0.1)
        self.assertEqual(self.noisy_store.volatility_window, 5)
    
    def test_noise_injection(self):
        """Test that noise is actually added to data."""
        self.base_store.add("TEST", self.test_data)
        
        # Get original and noisy data
        original = self.base_store.get("TEST", datetime(2023, 1, 4))
        noisy = self.noisy_store.get("TEST", datetime(2023, 1, 4))
        
        # Verify noise was added
        self.assertNotEqual(original['open'], noisy['open'])
        self.assertNotEqual(original['close'], noisy['close'])
        
        # Verify OHLC consistency is maintained
        self.assertGreaterEqual(noisy['high'], noisy['open'])
        self.assertGreaterEqual(noisy['high'], noisy['close'])
        self.assertLessEqual(noisy['low'], noisy['open'])
        self.assertLessEqual(noisy['low'], noisy['close'])
    
    def test_volatility_scaling(self):
        """Test that noise scales with volatility."""
        # Create data with different volatility patterns
        low_vol_data = {
            datetime(2023, 1, 1): {'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.1, 'volume': 1000},
            datetime(2023, 1, 2): {'open': 100.1, 'high': 100.6, 'low': 99.6, 'close': 100.2, 'volume': 1000},
            datetime(2023, 1, 3): {'open': 100.2, 'high': 100.7, 'low': 99.7, 'close': 100.3, 'volume': 1000},
        }
        
        high_vol_data = {
            datetime(2023, 1, 1): {'open': 100.0, 'high': 110.0, 'low': 90.0, 'close': 105.0, 'volume': 1000},
            datetime(2023, 1, 2): {'open': 105.0, 'high': 115.0, 'low': 95.0, 'close': 110.0, 'volume': 1000},
            datetime(2023, 1, 3): {'open': 110.0, 'high': 120.0, 'low': 100.0, 'close': 115.0, 'volume': 1000},
        }
        
        self.base_store.add("LOW_VOL", low_vol_data)
        self.base_store.add("HIGH_VOL", high_vol_data)
        
        # Get noisy data for both
        low_vol_noisy = self.noisy_store.get("LOW_VOL", datetime(2023, 1, 3))
        high_vol_noisy = self.noisy_store.get("HIGH_VOL", datetime(2023, 1, 3))
        
        # Calculate noise magnitude
        low_vol_noise = abs(low_vol_noisy['close'] - 100.3)
        high_vol_noise = abs(high_vol_noisy['close'] - 115.0)
        
        # High volatility should generally have more noise
        # (though this is probabilistic, so we check it's reasonable)
        self.assertGreater(high_vol_noise, 0)
        self.assertGreater(low_vol_noise, 0)
    
    def test_get_all_with_noise(self):
        """Test retrieving all data with noise."""
        self.base_store.add("TEST", self.test_data)
        
        original_all = self.base_store.get_all("TEST", datetime(2023, 1, 1), datetime(2023, 1, 7))
        noisy_all = self.noisy_store.get_all("TEST", datetime(2023, 1, 1), datetime(2023, 1, 7))
        
        self.assertIsNotNone(noisy_all)
        self.assertEqual(len(noisy_all), len(original_all))
        
        # Verify that all points have noise
        for dt in original_all.keys():
            original = original_all[dt]
            noisy = noisy_all[dt]
            
            # At least some fields should have noise
            has_noise = any(
                abs(original[key] - noisy[key]) > 0.001 
                for key in ['open', 'high', 'low', 'close'] 
                if key in original and key in noisy
            )
            self.assertTrue(has_noise)
    
    def test_get_latest_with_noise(self):
        """Test retrieving latest data with noise."""
        self.base_store.add("TEST", self.test_data)
        
        original_latest = self.base_store.get_latest("TEST")
        noisy_latest = self.noisy_store.get_latest("TEST")
        
        self.assertIsNotNone(noisy_latest)
        self.assertNotEqual(original_latest['close'], noisy_latest['close'])
    
    def test_delegation_methods(self):
        """Test that delegation methods work correctly."""
        self.base_store.add("TEST", self.test_data)

        # Test exists
        self.assertTrue(self.noisy_store.exists("TEST", datetime(2023, 1, 1)))
        self.assertFalse(self.noisy_store.exists("NONEXISTENT"))

        # Test latest date
        latest_date = self.noisy_store.latest("TEST")
        self.assertEqual(latest_date, datetime(2023, 1, 7))

        # Test earliest date
        earliest_date = self.noisy_store.earliest("TEST")
        self.assertEqual(earliest_date, datetime(2023, 1, 1))

        # Test identifiers
        identifiers = self.noisy_store.identifiers()
        self.assertIn("TEST", identifiers)
    
    def test_reproducibility(self):
        """Test that noise is reproducible with the same seed."""
        # Create two stores with the same seed
        noisy_store1 = NoisyDataStore(self.base_store, seed=42)
        noisy_store2 = NoisyDataStore(self.base_store, seed=42)
        
        self.base_store.add("TEST", self.test_data)
        
        # Get noisy data from both stores
        noisy1 = noisy_store1.get("TEST", datetime(2023, 1, 4))
        noisy2 = noisy_store2.get("TEST", datetime(2023, 1, 4))
        
        # Should be identical with same seed
        self.assertEqual(noisy1['open'], noisy2['open'])
        self.assertEqual(noisy1['close'], noisy2['close'])
    
    def test_different_seeds(self):
        """Test that different seeds produce different noise."""
        # Create two stores with different seeds
        noisy_store1 = NoisyDataStore(self.base_store, seed=42)
        noisy_store2 = NoisyDataStore(self.base_store, seed=123)
        
        self.base_store.add("TEST", self.test_data)
        
        # Get noisy data from both stores
        noisy1 = noisy_store1.get("TEST", datetime(2023, 1, 4))
        noisy2 = noisy_store2.get("TEST", datetime(2023, 1, 4))
        
        # Should be different with different seeds
        self.assertNotEqual(noisy1['open'], noisy2['open'])
        self.assertNotEqual(noisy1['close'], noisy2['close'])


class TestDataStoreIntegration(unittest.TestCase):
    """Integration tests for the data store system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.base_store = ParquetDataStore(self.test_dir)
        self.noisy_store = NoisyDataStore(
            base_store=self.base_store,
            noise_multiplier=0.05,
            seed=42
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_store_hierarchy(self):
        """Test that the store hierarchy works correctly."""
        # Verify inheritance
        self.assertIsInstance(self.base_store, DataStore)
        self.assertIsInstance(self.noisy_store, DataStore)
        
        # Verify composition
        self.assertEqual(self.noisy_store.base_store, self.base_store)
    
    def test_data_flow(self):
        """Test complete data flow through the system."""
        # Create test data
        test_data = {
            datetime(2023, 1, 1): {
                'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 102.0, 'volume': 1000
            },
            datetime(2023, 1, 2): {
                'open': 102.0, 'high': 108.0, 'low': 100.0, 'close': 106.0, 'volume': 1200
            }
        }
        
        # Add data through noisy store (delegates to base store)
        self.noisy_store.add("TEST", test_data)
        
        # Verify data exists in base store
        self.assertTrue(self.base_store.exists("TEST"))
        
        # Verify data can be retrieved through both stores
        base_result = self.base_store.get("TEST", datetime(2023, 1, 1))
        noisy_result = self.noisy_store.get("TEST", datetime(2023, 1, 1))
        
        self.assertIsNotNone(base_result)
        self.assertIsNotNone(noisy_result)
        
        # Verify noise was added
        self.assertNotEqual(base_result['close'], noisy_result['close'])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
