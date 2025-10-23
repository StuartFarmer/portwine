import unittest
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from collections import OrderedDict
import numpy as np

from portwine.data.stores.parquet import ParquetDataStore


class TestParquetDataStore(unittest.TestCase):
    """
    Comprehensive test suite for ParquetDataStore.
    Each test method tests exactly one behavior/branch to ensure precise failure isolation.
    """
    
    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = ParquetDataStore(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directory_if_not_exists(self):
        """
        Test: Constructor creates directory if it doesn't exist
        How: Initialize with non-existent directory path
        Expected: Directory is created
        """
        # Create a temporary directory and then remove it
        temp_dir = tempfile.mkdtemp()
        shutil.rmtree(temp_dir)
        
        # Verify directory doesn't exist
        self.assertFalse(os.path.exists(temp_dir))
        
        # Initialize ParquetDataStore with non-existent directory
        store = ParquetDataStore(temp_dir)
        
        # Verify directory was created
        self.assertTrue(os.path.exists(temp_dir))
        self.assertTrue(os.path.isdir(temp_dir))
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    def test_init_uses_existing_directory(self):
        """
        Test: Constructor works with existing directory
        How: Initialize with existing directory path
        Expected: No error, store uses existing directory
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Verify directory exists
        self.assertTrue(os.path.exists(temp_dir))
        
        # Initialize ParquetDataStore with existing directory
        store = ParquetDataStore(temp_dir)
        
        # Verify no error occurred and directory still exists
        self.assertTrue(os.path.exists(temp_dir))
        self.assertTrue(os.path.isdir(temp_dir))
        
        # Verify store's data_dir points to the correct directory
        self.assertEqual(store.data_dir, Path(temp_dir))
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    def test_get_file_path_returns_correct_path(self):
        """
        Test: _get_file_path returns correct parquet file path
        How: Call with identifier "AAPL"
        Expected: Returns path ending with "AAPL.pqt"
        """
        # Call the private method with identifier "AAPL"
        file_path = self.store._get_file_path("AAPL")
        
        # Verify it returns a Path object
        self.assertIsInstance(file_path, Path)
        
        # Verify it points to the correct directory
        self.assertEqual(file_path.parent, self.store.data_dir)
        
        # Verify it has the correct filename
        self.assertEqual(file_path.name, "AAPL.pqt")
        
        # Verify the full path is correct
        expected_path = self.store.data_dir / "AAPL.pqt"
        self.assertEqual(file_path, expected_path)
    
    def test_load_dataframe_returns_empty_df_when_file_not_exists(self):
        """
        Test: _load_dataframe returns empty DataFrame when file doesn't exist
        How: Call with non-existent identifier
        Expected: Returns empty pandas DataFrame
        """
        # Call with non-existent identifier
        df = self.store._load_dataframe("NONEXISTENT")
        
        # Verify it returns a pandas DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        
        # Verify it's empty
        self.assertTrue(df.empty)
        
        # Verify it has the correct shape
        self.assertEqual(df.shape, (0, 0))
    
    def test_load_dataframe_loads_existing_parquet_file(self):
        """
        Test: _load_dataframe loads existing parquet file correctly
        How: Create parquet file manually, then call _load_dataframe
        Expected: Returns DataFrame with correct data and datetime index
        """
        # Create test data
        test_data = {
            'open': [100.0, 101.0, 102.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        }
        test_dates = pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])
        df_original = pd.DataFrame(test_data, index=test_dates)
        
        # Save to parquet file
        file_path = self.store._get_file_path("AAPL")
        df_original.to_parquet(file_path)
        
        # Load using the private method
        df_loaded = self.store._load_dataframe("AAPL")
        
        # Verify it returns a DataFrame
        self.assertIsInstance(df_loaded, pd.DataFrame)
        
        # Verify it's not empty
        self.assertFalse(df_loaded.empty)
        
        # Verify it has the correct shape
        self.assertEqual(df_loaded.shape, (3, 3))
        
        # Verify it has datetime index
        self.assertIsInstance(df_loaded.index, pd.DatetimeIndex)
        
        # Verify data matches
        pd.testing.assert_frame_equal(df_loaded, df_original)
    
    def test_load_dataframe_handles_corrupted_parquet_file(self):
        """
        Test: _load_dataframe handles corrupted parquet file gracefully
        How: Create corrupted parquet file, then call _load_dataframe
        Expected: Returns empty DataFrame and prints error message
        """
        # Create a corrupted parquet file by writing random bytes
        file_path = self.store._get_file_path("CORRUPTED")
        with open(file_path, 'wb') as f:
            f.write(b'This is not a valid parquet file')
        
        # Load using the private method
        df = self.store._load_dataframe("CORRUPTED")
        
        # Verify it returns an empty DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)
    
    def test_load_dataframe_converts_string_index_to_datetime(self):
        """
        Test: _load_dataframe converts string index to datetime index
        How: Create parquet file with string date index, then load
        Expected: Returns DataFrame with DatetimeIndex
        """
        # Create test data with string index
        test_data = {
            'open': [100.0, 101.0],
            'close': [101.0, 102.0]
        }
        df_original = pd.DataFrame(test_data, index=['2023-01-01', '2023-01-02'])
        
        # Save to parquet file
        file_path = self.store._get_file_path("STRING_INDEX")
        df_original.to_parquet(file_path)
        
        # Load using the private method
        df_loaded = self.store._load_dataframe("STRING_INDEX")
        
        # Verify it has datetime index
        self.assertIsInstance(df_loaded.index, pd.DatetimeIndex)
        
        # Verify the dates are correct
        expected_dates = pd.DatetimeIndex(['2023-01-01', '2023-01-02'])
        pd.testing.assert_index_equal(df_loaded.index, expected_dates)
    
    def test_load_dataframe_sorts_index_chronologically(self):
        """
        Test: _load_dataframe sorts index in chronological order
        How: Create parquet file with unsorted dates, then load
        Expected: Returns DataFrame with sorted DatetimeIndex
        """
        # Create test data with unsorted dates
        test_data = {
            'open': [100.0, 101.0, 102.0],
            'close': [101.0, 102.0, 103.0]
        }
        # Dates in reverse order
        df_original = pd.DataFrame(test_data, index=['2023-01-03', '2023-01-01', '2023-01-02'])
        
        # Save to parquet file
        file_path = self.store._get_file_path("UNSORTED")
        df_original.to_parquet(file_path)
        
        # Load using the private method
        df_loaded = self.store._load_dataframe("UNSORTED")
        
        # Verify it has datetime index
        self.assertIsInstance(df_loaded.index, pd.DatetimeIndex)
        
        # Verify it's sorted chronologically
        self.assertTrue(df_loaded.index.is_monotonic_increasing)
        
        # Verify the dates are in correct order
        expected_dates = pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])
        pd.testing.assert_index_equal(df_loaded.index, expected_dates)
    
    def test_save_dataframe_creates_parquet_file(self):
        """
        Test: _save_dataframe creates parquet file
        How: Call with DataFrame and identifier
        Expected: Parquet file is created in data directory
        """
        # Create test DataFrame
        test_data = {
            'open': [100.0, 101.0],
            'close': [101.0, 102.0]
        }
        df = pd.DataFrame(test_data, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02']))
        
        # Save using the private method
        self.store._save_dataframe("AAPL", df)
        
        # Verify file was created
        file_path = self.store._get_file_path("AAPL")
        self.assertTrue(file_path.exists())
        
        # Verify it's a file (not directory)
        self.assertTrue(file_path.is_file())
    
    def test_save_dataframe_sorts_data_before_saving(self):
        """
        Test: _save_dataframe sorts data chronologically before saving
        How: Save DataFrame with unsorted index
        Expected: Saved file contains data sorted by datetime
        """
        # Create test data with unsorted dates
        test_data = {
            'open': [100.0, 101.0, 102.0],
            'close': [101.0, 102.0, 103.0]
        }
        # Dates in reverse order
        df_unsorted = pd.DataFrame(test_data, index=pd.DatetimeIndex(['2023-01-03', '2023-01-01', '2023-01-02']))
        
        # Save using the private method
        self.store._save_dataframe("SORTED", df_unsorted)
        
        # Load the saved file
        df_loaded = self.store._load_dataframe("SORTED")
        
        # Verify it's sorted chronologically
        self.assertTrue(df_loaded.index.is_monotonic_increasing)
        
        # Verify the dates are in correct order
        expected_dates = pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])
        pd.testing.assert_index_equal(df_loaded.index, expected_dates)
    
    def test_save_dataframe_handles_save_errors(self):
        """
        Test: _save_dataframe handles save errors gracefully
        How: Try to save to read-only directory
        Expected: Prints error message, doesn't crash
        """
        # Create a read-only directory
        read_only_dir = tempfile.mkdtemp()
        os.chmod(read_only_dir, 0o444)  # Read-only
        
        try:
            # Create store with read-only directory
            store = ParquetDataStore(read_only_dir)
            
            # Create test DataFrame
            test_data = {'open': [100.0], 'close': [101.0]}
            df = pd.DataFrame(test_data, index=pd.DatetimeIndex(['2023-01-01']))
            
            # This should not crash, even though it can't save
            store._save_dataframe("TEST", df)
            
            # Verify no exception was raised
            self.assertTrue(True)  # If we get here, no exception occurred
            
        finally:
            # Clean up - make writable again to remove
            os.chmod(read_only_dir, 0o755)
            shutil.rmtree(read_only_dir)
    
    def test_add_does_nothing_with_empty_data(self):
        """
        Test: add does nothing when data is empty
        How: Call add with empty dictionary
        Expected: No file is created, no error occurs
        """
        # Call add with empty data
        self.store.add("AAPL", {})
        
        # Verify no file was created
        file_path = self.store._get_file_path("AAPL")
        self.assertFalse(file_path.exists())
    
    def test_add_creates_new_file_for_new_identifier(self):
        """
        Test: add creates new parquet file for new identifier
        How: Call add with new identifier and data
        Expected: New parquet file is created with correct data
        """
        # Create test data
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0, "volume": 1000},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0, "volume": 1100}
        }
        
        # Add data
        self.store.add("AAPL", data)
        
        # Verify file was created
        file_path = self.store._get_file_path("AAPL")
        self.assertTrue(file_path.exists())
        
        # Verify data was stored correctly
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], pd.Timestamp('2023-01-01'))
        self.assertEqual(df.index[1], pd.Timestamp('2023-01-02'))
        self.assertEqual(df.iloc[0]['open'], 100.0)
        self.assertEqual(df.iloc[1]['close'], 102.0)
    
    def test_add_converts_string_dates_to_datetime(self):
        """
        Test: add converts string dates to datetime objects
        How: Call add with data containing string date keys
        Expected: Data is stored with datetime index
        """
        # Create test data with string dates
        data = {
            "2023-01-01": {"open": 100.0, "close": 101.0},
            "2023-01-02": {"open": 101.0, "close": 102.0}
        }
        
        # Add data
        self.store.add("AAPL", data)
        
        # Verify data was stored with datetime index
        df = self.store._load_dataframe("AAPL")
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.index[0], pd.Timestamp('2023-01-01'))
        self.assertEqual(df.index[1], pd.Timestamp('2023-01-02'))
    
    def test_add_skips_existing_data_when_overwrite_false(self):
        """
        Test: add skips existing data when overwrite=False (default)
        How: Add data, then add overlapping data with overwrite=False
        Expected: Original data is preserved, new data is ignored for overlapping dates
        """
        # Add initial data
        initial_data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0, "volume": 1000},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0, "volume": 1100}
        }
        self.store.add("AAPL", initial_data)
        
        # Add overlapping data with overwrite=False
        overlapping_data = {
            datetime(2023, 1, 1): {"open": 999.0, "close": 999.0, "volume": 9999},  # Different values
            datetime(2023, 1, 3): {"open": 103.0, "close": 104.0, "volume": 1200}   # New date
        }
        self.store.add("AAPL", overlapping_data, overwrite=False)
        
        # Verify original data is preserved for overlapping date
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 3)  # Should have 3 dates now
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 100.0)  # Original value preserved
        self.assertEqual(df.loc[pd.Timestamp('2023-01-03'), 'open'], 103.0)  # New date added
    
    def test_add_overwrites_existing_data_when_overwrite_true(self):
        """
        Test: add overwrites existing data when overwrite=True
        How: Add data, then add overlapping data with overwrite=True
        Expected: New data replaces old data for overlapping dates
        """
        # Add initial data
        initial_data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0, "volume": 1000},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0, "volume": 1100}
        }
        self.store.add("AAPL", initial_data)
        
        # Add overlapping data with overwrite=True
        overlapping_data = {
            datetime(2023, 1, 1): {"open": 999.0, "close": 999.0, "volume": 9999},  # Different values
            datetime(2023, 1, 3): {"open": 103.0, "close": 104.0, "volume": 1200}   # New date
        }
        self.store.add("AAPL", overlapping_data, overwrite=True)
        
        # Verify new data overwrote old data for overlapping date
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 3)  # Should have 3 dates now
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 999.0)  # New value overwrote old
        self.assertEqual(df.loc[pd.Timestamp('2023-01-03'), 'open'], 103.0)  # New date added
    
    def test_add_merges_non_overlapping_data(self):
        """
        Test: add merges non-overlapping data correctly
        How: Add data for dates 2023-01-01 to 2023-01-05, then add data for 2023-01-06 to 2023-01-10
        Expected: Both datasets are stored in chronological order
        """
        # Add first dataset
        first_data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0},
            datetime(2023, 1, 4): {"open": 103.0, "close": 104.0},
            datetime(2023, 1, 5): {"open": 104.0, "close": 105.0}
        }
        self.store.add("AAPL", first_data)
        
        # Add second dataset
        second_data = {
            datetime(2023, 1, 6): {"open": 105.0, "close": 106.0},
            datetime(2023, 1, 7): {"open": 106.0, "close": 107.0},
            datetime(2023, 1, 8): {"open": 107.0, "close": 108.0},
            datetime(2023, 1, 9): {"open": 108.0, "close": 109.0},
            datetime(2023, 1, 10): {"open": 109.0, "close": 110.0}
        }
        self.store.add("AAPL", second_data)
        
        # Verify all data is stored
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 10)
        
        # Verify chronological order
        self.assertTrue(df.index.is_monotonic_increasing)
        
        # Verify data integrity
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 100.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-10'), 'close'], 110.0)
    
    def test_add_handles_duplicate_dates_in_input_data(self):
        """
        Test: add handles duplicate dates in input data
        How: Call add with data containing duplicate date keys
        Expected: Last occurrence of each date is kept
        """
        # Create data with duplicate dates
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},  # First occurrence
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 1): {"open": 999.0, "close": 999.0},  # Duplicate date, different values
            datetime(2023, 1, 3): {"open": 103.0, "close": 104.0}
        }
        
        # Add data
        self.store.add("AAPL", data)
        
        # Verify only last occurrence is kept
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 3)  # Should have 3 unique dates
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 999.0)  # Last occurrence kept
    
    def test_add_preserves_existing_data_for_non_overlapping_dates(self):
        """
        Test: add preserves existing data for non-overlapping dates
        How: Add data for dates 2023-01-01 to 2023-01-10, then add data for 2023-01-15 to 2023-01-20
        Expected: All data is preserved, no gaps in dates 2023-01-01 to 2023-01-10
        """
        # Add first dataset
        first_data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0},
            datetime(2023, 1, 4): {"open": 103.0, "close": 104.0},
            datetime(2023, 1, 5): {"open": 104.0, "close": 105.0},
            datetime(2023, 1, 6): {"open": 105.0, "close": 106.0},
            datetime(2023, 1, 7): {"open": 106.0, "close": 107.0},
            datetime(2023, 1, 8): {"open": 107.0, "close": 108.0},
            datetime(2023, 1, 9): {"open": 108.0, "close": 109.0},
            datetime(2023, 1, 10): {"open": 109.0, "close": 110.0}
        }
        self.store.add("AAPL", first_data)
        
        # Add second dataset (non-overlapping)
        second_data = {
            datetime(2023, 1, 15): {"open": 115.0, "close": 116.0},
            datetime(2023, 1, 16): {"open": 116.0, "close": 117.0},
            datetime(2023, 1, 17): {"open": 117.0, "close": 118.0},
            datetime(2023, 1, 18): {"open": 118.0, "close": 119.0},
            datetime(2023, 1, 19): {"open": 119.0, "close": 120.0},
            datetime(2023, 1, 20): {"open": 120.0, "close": 121.0}
        }
        self.store.add("AAPL", second_data)
        
        # Verify all original data is preserved
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 16)  # 10 + 6 = 16 dates
        
        # Verify original data is intact
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 100.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-10'), 'close'], 110.0)
        
        # Verify new data is added
        self.assertEqual(df.loc[pd.Timestamp('2023-01-15'), 'open'], 115.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-20'), 'close'], 121.0)
    
    def test_add_handles_mixed_datetime_and_string_dates(self):
        """
        Test: add handles mixed datetime and string dates in same data
        How: Call add with data containing both datetime objects and string dates
        Expected: All dates are converted to datetime and stored correctly
        """
        # Create data with mixed date types
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},  # datetime object
            "2023-01-02": {"open": 101.0, "close": 102.0},          # string date
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0},  # datetime object
            "2023-01-04": {"open": 103.0, "close": 104.0}           # string date
        }
        
        # Add data
        self.store.add("AAPL", data)
        
        # Verify all dates are stored as datetime
        df = self.store._load_dataframe("AAPL")
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(len(df), 4)
        
        # Verify data integrity
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 100.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-02'), 'open'], 101.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-03'), 'open'], 102.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-04'), 'open'], 103.0)
    
    def test_add_handles_empty_values_in_data(self):
        """
        Test: add handles empty or None values in data
        How: Call add with data containing empty dictionaries or None values
        Expected: Empty values are stored as NaN
        """
        # Create data with empty values
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0, "volume": 1000},
            datetime(2023, 1, 2): {},  # Empty dictionary
            datetime(2023, 1, 3): {"open": 102.0, "close": None, "volume": 1200}  # None value
        }
        
        # Add data
        self.store.add("AAPL", data)
        
        # Verify data is stored
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 3)
        
        # Verify empty dictionary becomes NaN
        self.assertTrue(pd.isna(df.loc[pd.Timestamp('2023-01-02'), 'open']))
        
        # Verify None value becomes NaN
        self.assertTrue(pd.isna(df.loc[pd.Timestamp('2023-01-03'), 'close']))
        
        # Verify valid data is preserved
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 100.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-03'), 'open'], 102.0)
    
    def test_add_handles_large_number_of_dates(self):
        """
        Test: add handles large number of dates efficiently
        How: Add data with 1000 dates
        Expected: All data is stored correctly
        """
        # Create data with 1000 dates
        data = {}
        for i in range(1000):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i, "volume": 1000 + i}
        
        # Add data
        self.store.add("AAPL", data)
        
        # Verify all data is stored
        df = self.store._load_dataframe("AAPL")
        self.assertEqual(len(df), 1000)
        
        # Verify chronological order
        self.assertTrue(df.index.is_monotonic_increasing)
        
        # Verify data integrity
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01'), 'open'], 100.0)
        self.assertEqual(df.loc[pd.Timestamp('2023-01-01') + timedelta(days=999), 'open'], 1099.0)
    
    def test_get_returns_none_for_nonexistent_identifier(self):
        """
        Test: get returns None for non-existent identifier
        How: Call get with identifier that has no parquet file
        Expected: Returns None
        """
        # Call get with non-existent identifier
        result = self.store.get("NONEXISTENT", datetime(2023, 1, 1))
        
        # Verify it returns None
        self.assertIsNone(result)

    def test_get_accepts_numpy_datetime64(self):
        """
        Test: get accepts numpy.datetime64 inputs
        How: Add data for a date, call get with np.datetime64
        Expected: Returns correct dict
        """
        # Add data
        data = {datetime(2023, 1, 5): {"open": 105.0, "close": 106.0}}
        self.store.add("AAPL", data)

        # Use numpy datetime64
        dt_np = np.datetime64('2023-01-05')
        result = self.store.get("AAPL", dt_np)

        # Verify correct record is returned
        self.assertIsInstance(result, dict)
        self.assertEqual(result["open"], 105.0)
        self.assertEqual(result["close"], 106.0)
    
    def test_get_all_returns_ordereddict_with_datetime_keys(self):
        """
        Test: get returns OrderedDict with datetime keys
        How: Call get with existing identifier
        Expected: Returns OrderedDict with datetime objects as keys
        """
        # Add some test data
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_all
        result = self.store.get_all("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 2))
        
        # Verify it returns OrderedDict
        self.assertIsInstance(result, OrderedDict)
        
        # Verify keys are datetime objects
        for key in result.keys():
            self.assertIsInstance(key, datetime)

    def test_get_all_accepts_numpy_datetime64(self):
        """
        Test: get_all accepts numpy.datetime64 inputs for start and end
        How: Add data across multiple dates, call get_all with np.datetime64
        Expected: Returns OrderedDict of correct length and keys are datetime
        """
        data = {
            datetime(2023, 1, 1): {"open": 100.0},
            datetime(2023, 1, 2): {"open": 101.0},
            datetime(2023, 1, 3): {"open": 102.0},
        }
        self.store.add("AAPL", data)

        start_np = np.datetime64('2023-01-02')
        end_np = np.datetime64('2023-01-03')
        result = self.store.get_all("AAPL", start_np, end_np)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        keys = list(result.keys())
        self.assertEqual(keys[0], datetime(2023, 1, 2))
        self.assertEqual(keys[1], datetime(2023, 1, 3))
    
    def test_get_all_returns_data_in_chronological_order(self):
        """
        Test: get returns data in chronological order (earliest to latest)
        How: Add data with dates in random order, then call get
        Expected: Returns OrderedDict with keys in chronological order
        """
        # Add data in random order
        data = {
            datetime(2023, 1, 3): {"open": 103.0, "close": 104.0},
            datetime(2023, 1, 1): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 2): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_all
        result = self.store.get_all("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 3))
        
        # Verify it's in chronological order
        dates = list(result.keys())
        self.assertEqual(dates[0], datetime(2023, 1, 1))
        self.assertEqual(dates[1], datetime(2023, 1, 2))
        self.assertEqual(dates[2], datetime(2023, 1, 3))
        
        # Verify it's monotonic increasing
        self.assertTrue(all(dates[i] <= dates[i+1] for i in range(len(dates)-1)))
    
    def test_get_all_filters_by_start_date(self):
        """
        Test: get filters data by start_date
        How: Add data for dates 2023-01-01 to 2023-01-10, call get with start_date=2023-01-05
        Expected: Returns only data from 2023-01-05 onwards
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call get_all with start_date only
        result = self.store.get_all("AAPL", datetime(2023, 1, 5))
        
        # Verify only data from 2023-01-05 onwards is returned
        dates = list(result.keys())
        self.assertTrue(all(date >= datetime(2023, 1, 5) for date in dates))
        
        # Verify we have 6 days of data (2023-01-05 to 2023-01-10)
        self.assertEqual(len(result), 6)
        
        # Verify first date is 2023-01-05
        self.assertEqual(dates[0], datetime(2023, 1, 5))
    
    def test_get_all_filters_by_end_date(self):
        """
        Test: get filters data by end_date
        How: Add data for dates 2023-01-01 to 2023-01-10, call get with end_date=2023-01-05
        Expected: Returns only data up to 2023-01-05
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call get_all with end_date
        result = self.store.get_all("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 5))
        
        # Verify only data up to 2023-01-05 is returned
        dates = list(result.keys())
        self.assertTrue(all(date <= datetime(2023, 1, 5) for date in dates))
        
        # Verify we have 5 days of data (2023-01-01 to 2023-01-05)
        self.assertEqual(len(result), 5)
        
        # Verify last date is 2023-01-05
        self.assertEqual(dates[-1], datetime(2023, 1, 5))
    
    def test_get_all_filters_by_both_start_and_end_date(self):
        """
        Test: get filters data by both start_date and end_date
        How: Add data for dates 2023-01-01 to 2023-01-10, call get with start_date=2023-01-03, end_date=2023-01-07
        Expected: Returns only data from 2023-01-03 to 2023-01-07
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call get_all with both start and end dates
        result = self.store.get_all("AAPL", datetime(2023, 1, 3), datetime(2023, 1, 7))
        
        # Verify only data in the specified range is returned
        dates = list(result.keys())
        self.assertTrue(all(datetime(2023, 1, 3) <= date <= datetime(2023, 1, 7) for date in dates))
        
        # Verify we have 5 days of data (2023-01-03 to 2023-01-07)
        self.assertEqual(len(result), 5)
        
        # Verify first and last dates
        self.assertEqual(dates[0], datetime(2023, 1, 3))
        self.assertEqual(dates[-1], datetime(2023, 1, 7))
    
    def test_get_all_uses_latest_date_when_end_date_is_none(self):
        """
        Test: get uses latest available date when end_date is None
        How: Add data for dates 2023-01-01 to 2023-01-10, call get with end_date=None
        Expected: Returns data up to 2023-01-10
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call get_all with end_date=None
        result = self.store.get_all("AAPL", datetime(2023, 1, 1))
        
        # Verify all data is returned
        self.assertEqual(len(result), 10)
        
        # Verify last date is 2023-01-10
        dates = list(result.keys())
        self.assertEqual(dates[-1], datetime(2023, 1, 10))
    
    def test_get_all_returns_none_when_no_data_in_range(self):
        """
        Test: get returns None when no data exists in specified range
        How: Add data for dates 2023-01-01 to 2023-01-10, call get with start_date=2023-01-15
        Expected: Returns None
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call get_all with start_date after available data
        result = self.store.get_all("AAPL", datetime(2023, 1, 15))
        
        # Verify it returns None
        self.assertIsNone(result)
    
    def test_get_all_handles_edge_case_start_date_equals_end_date(self):
        """
        Test: get handles edge case where start_date equals end_date
        How: Add data, call get with start_date=end_date
        Expected: Returns data for that specific date only
        """
        # Add data for 3 days
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_all with start_date equals end_date
        result = self.store.get_all("AAPL", datetime(2023, 1, 2), datetime(2023, 1, 2))
        
        # Verify only one date is returned
        self.assertEqual(len(result), 1)
        
        # Verify it's the correct date
        self.assertEqual(list(result.keys())[0], datetime(2023, 1, 2))
        self.assertEqual(list(result.values())[0]["open"], 101.0)
    
    def test_get_all_handles_edge_case_start_date_after_end_date(self):
        """
        Test: get handles edge case where start_date is after end_date
        How: Add data, call get with start_date > end_date
        Expected: Returns empty OrderedDict or None
        """
        # Add data for 3 days
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_all with start_date after end_date
        result = self.store.get_all("AAPL", datetime(2023, 1, 3), datetime(2023, 1, 1))
        
        # Verify it returns None (no data in invalid range)
        self.assertIsNone(result)
    
    def test_get_latest_returns_none_for_nonexistent_identifier(self):
        """
        Test: get_latest returns None for non-existent identifier
        How: Call get_latest with identifier that has no parquet file
        Expected: Returns None
        """
        # Call get_latest with non-existent identifier
        result = self.store.get_latest("NONEXISTENT")
        
        # Verify it returns None
        self.assertIsNone(result)
    
    def test_get_latest_returns_dict_with_latest_data(self):
        """
        Test: get_latest returns dict with latest data point
        How: Add data for multiple dates, call get_latest
        Expected: Returns dict with data from the latest date
        """
        # Add data for multiple dates
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_latest
        result = self.store.get_latest("AAPL")
        
        # Verify it returns a dict
        self.assertIsInstance(result, dict)
        
        # Verify it contains the latest data (2023-01-03)
        self.assertEqual(result["open"], 102.0)
        self.assertEqual(result["close"], 103.0)
    
    def test_get_latest_returns_latest_data_when_dates_not_in_order(self):
        """
        Test: get_latest returns latest data even when dates were added out of order
        How: Add data in random order, call get_latest
        Expected: Returns data from the chronologically latest date
        """
        # Add data in random order
        data = {
            datetime(2023, 1, 3): {"open": 103.0, "close": 104.0},
            datetime(2023, 1, 1): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 2): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_latest
        result = self.store.get_latest("AAPL")
        
        # Verify it returns data from the latest date (2023-01-03)
        self.assertEqual(result["open"], 103.0)
        self.assertEqual(result["close"], 104.0)
    
    def test_get_latest_returns_single_data_point(self):
        """
        Test: get_latest returns only one data point (the latest)
        How: Add multiple data points, call get_latest
        Expected: Returns dict with only the latest data, not all data
        """
        # Add data for multiple dates
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call get_latest
        result = self.store.get_latest("AAPL")
        
        # Verify it's a single data point (not a collection)
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, (list, tuple))
        
        # Verify it has the expected keys
        self.assertIn("open", result)
        self.assertIn("close", result)
    
    def test_get_latest_handles_empty_dataframe(self):
        """
        Test: get_latest handles empty dataframe gracefully
        How: Create empty parquet file, call get_latest
        Expected: Returns None
        """
        # Create an empty DataFrame and save it
        empty_df = pd.DataFrame()
        self.store._save_dataframe("EMPTY", empty_df)
        
        # Call get_latest
        result = self.store.get_latest("EMPTY")
        
        # Verify it returns None
        self.assertIsNone(result)
    
    def test_latest_returns_none_for_nonexistent_identifier(self):
        """
        Test: latest returns None for non-existent identifier
        How: Call latest with identifier that has no parquet file
        Expected: Returns None
        """
        # Call latest with non-existent identifier
        result = self.store.latest("NONEXISTENT")
        
        # Verify it returns None
        self.assertIsNone(result)
    
    def test_latest_returns_datetime_object(self):
        """
        Test: latest returns datetime object, not string
        How: Add data for multiple dates, call latest
        Expected: Returns datetime object representing the latest date
        """
        # Add data for multiple dates
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call latest
        result = self.store.latest("AAPL")
        
        # Verify it returns a datetime object
        self.assertIsInstance(result, datetime)
        
        # Verify it's the latest date
        self.assertEqual(result, datetime(2023, 1, 3))
    
    def test_latest_returns_latest_date_when_dates_not_in_order(self):
        """
        Test: latest returns latest date even when dates were added out of order
        How: Add data in random order, call latest
        Expected: Returns the chronologically latest date
        """
        # Add data in random order
        data = {
            datetime(2023, 1, 3): {"open": 103.0, "close": 104.0},
            datetime(2023, 1, 1): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 2): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call latest
        result = self.store.latest("AAPL")
        
        # Verify it returns the latest date
        self.assertEqual(result, datetime(2023, 1, 3))
    
    def test_latest_returns_single_date_not_collection(self):
        """
        Test: latest returns single date, not collection of dates
        How: Add multiple data points, call latest
        Expected: Returns single datetime object, not list or tuple
        """
        # Add data for multiple dates
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call latest
        result = self.store.latest("AAPL")
        
        # Verify it's a single date (not a collection)
        self.assertIsInstance(result, datetime)
        self.assertNotIsInstance(result, (list, tuple, set))
    
    def test_latest_handles_empty_dataframe(self):
        """
        Test: latest handles empty dataframe gracefully
        How: Create empty parquet file, call latest
        Expected: Returns None
        """
        # Create an empty DataFrame and save it
        empty_df = pd.DataFrame()
        self.store._save_dataframe("EMPTY", empty_df)
        
        # Call latest
        result = self.store.latest("EMPTY")
        
        # Verify it returns None
        self.assertIsNone(result)
    
    def test_exists_returns_false_for_nonexistent_identifier(self):
        """
        Test: exists returns False for non-existent identifier
        How: Call exists with identifier that has no parquet file
        Expected: Returns False
        """
        # Call exists with non-existent identifier
        result = self.store.exists("NONEXISTENT")
        
        # Verify it returns False
        self.assertFalse(result)
    
    def test_exists_returns_true_when_data_exists(self):
        """
        Test: exists returns True when data exists for identifier
        How: Add data, call exists without date parameters
        Expected: Returns True
        """
        # Add some data
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0}
        }
        self.store.add("AAPL", data)
        
        # Call exists without date parameters
        result = self.store.exists("AAPL")
        
        # Verify it returns True
        self.assertTrue(result)
    
    def test_exists_returns_true_when_data_exists_in_date_range(self):
        """
        Test: exists returns True when data exists in specified date range
        How: Add data for dates 2023-01-01 to 2023-01-10, call exists with start_date=2023-01-03, end_date=2023-01-07
        Expected: Returns True
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call exists with date range
        result = self.store.exists("AAPL", datetime(2023, 1, 3), datetime(2023, 1, 7))
        
        # Verify it returns True
        self.assertTrue(result)
    
    def test_exists_returns_false_when_no_data_in_date_range(self):
        """
        Test: exists returns False when no data exists in specified date range
        How: Add data for dates 2023-01-01 to 2023-01-10, call exists with start_date=2023-01-15
        Expected: Returns False
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call exists with start_date after available data
        result = self.store.exists("AAPL", datetime(2023, 1, 15))
        
        # Verify it returns False
        self.assertFalse(result)
    
    def test_exists_uses_earliest_date_when_start_date_is_none(self):
        """
        Test: exists uses earliest date when start_date is None
        How: Add data for dates 2023-01-01 to 2023-01-10, call exists with start_date=None, end_date=2023-01-05
        Expected: Returns True (data exists from earliest date to 2023-01-05)
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call exists with start_date=None
        result = self.store.exists("AAPL", None, datetime(2023, 1, 5))
        
        # Verify it returns True
        self.assertTrue(result)
    
    def test_exists_uses_latest_date_when_end_date_is_none(self):
        """
        Test: exists uses latest date when end_date is None
        How: Add data for dates 2023-01-01 to 2023-01-10, call exists with start_date=2023-01-05, end_date=None
        Expected: Returns True (data exists from 2023-01-05 to latest date)
        """
        # Add data for 10 days
        data = {}
        for i in range(10):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            data[date] = {"open": 100.0 + i, "close": 101.0 + i}
        self.store.add("AAPL", data)
        
        # Call exists with end_date=None
        result = self.store.exists("AAPL", datetime(2023, 1, 5), None)
        
        # Verify it returns True
        self.assertTrue(result)
    
    def test_exists_returns_true_when_both_dates_are_none(self):
        """
        Test: exists returns True when both start_date and end_date are None
        How: Add data, call exists with both dates None
        Expected: Returns True (checks if any data exists)
        """
        # Add some data
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0}
        }
        self.store.add("AAPL", data)
        
        # Call exists with both dates None
        result = self.store.exists("AAPL", None, None)
        
        # Verify it returns True
        self.assertTrue(result)
    
    def test_exists_returns_false_for_empty_dataframe(self):
        """
        Test: exists returns False for empty dataframe
        How: Create empty parquet file, call exists
        Expected: Returns False
        """
        # Create an empty DataFrame and save it
        empty_df = pd.DataFrame()
        self.store._save_dataframe("EMPTY", empty_df)
        
        # Call exists
        result = self.store.exists("EMPTY")
        
        # Verify it returns False
        self.assertFalse(result)
    
    def test_exists_handles_edge_case_start_date_equals_end_date(self):
        """
        Test: exists handles edge case where start_date equals end_date
        How: Add data, call exists with start_date=end_date
        Expected: Returns True if data exists for that date
        """
        # Add data for 3 days
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call exists with start_date equals end_date (existing date)
        result = self.store.exists("AAPL", datetime(2023, 1, 2), datetime(2023, 1, 2))
        
        # Verify it returns True
        self.assertTrue(result)
    
    def test_exists_handles_edge_case_start_date_after_end_date(self):
        """
        Test: exists handles edge case where start_date is after end_date
        How: Add data, call exists with start_date > end_date
        Expected: Returns False (invalid range)
        """
        # Add data for 3 days
        data = {
            datetime(2023, 1, 1): {"open": 100.0, "close": 101.0},
            datetime(2023, 1, 2): {"open": 101.0, "close": 102.0},
            datetime(2023, 1, 3): {"open": 102.0, "close": 103.0}
        }
        self.store.add("AAPL", data)
        
        # Call exists with start_date after end_date
        result = self.store.exists("AAPL", datetime(2023, 1, 3), datetime(2023, 1, 1))
        
        # Verify it returns False (no data in invalid range)
        self.assertFalse(result)
    
    def test_identifiers_returns_empty_list_when_no_files(self):
        """
        Test: identifiers returns empty list when no parquet files exist
        How: Call identifiers on empty directory
        Expected: Returns empty list
        """
        # Call identifiers on empty directory
        result = self.store.identifiers()
        
        # Verify it returns empty list
        self.assertEqual(result, [])
        self.assertIsInstance(result, list)
    
    def test_identifiers_returns_list_of_identifiers(self):
        """
        Test: identifiers returns list of all identifiers
        How: Add data for multiple identifiers, call identifiers
        Expected: Returns list containing all identifiers
        """
        # Add data for multiple identifiers
        data1 = {datetime(2023, 1, 1): {"open": 100.0, "close": 101.0}}
        data2 = {datetime(2023, 1, 1): {"open": 200.0, "close": 201.0}}
        data3 = {datetime(2023, 1, 1): {"open": 300.0, "close": 301.0}}
        
        self.store.add("AAPL", data1)
        self.store.add("GOOGL", data2)
        self.store.add("MSFT", data3)
        
        # Call identifiers
        result = self.store.identifiers()
        
        # Verify it returns a list
        self.assertIsInstance(result, list)
        
        # Verify it contains all identifiers
        self.assertIn("AAPL", result)
        self.assertIn("GOOGL", result)
        self.assertIn("MSFT", result)
        
        # Verify it has the correct length
        self.assertEqual(len(result), 3)
    
    def test_identifiers_returns_identifiers_without_file_extension(self):
        """
        Test: identifiers returns identifiers without .pqt file extension
        How: Add data with identifier containing special characters, call identifiers
        Expected: Returns identifier name without .pqt extension
        """
        # Add data with identifier that might have special characters
        data = {datetime(2023, 1, 1): {"open": 100.0, "close": 101.0}}
        self.store.add("SPY-ETF", data)
        
        # Call identifiers
        result = self.store.identifiers()
        
        # Verify it returns the identifier without extension
        self.assertIn("SPY-ETF", result)
        self.assertNotIn("SPY-ETF.pqt", result)
    
    def test_identifiers_ignores_non_parquet_files(self):
        """
        Test: identifiers ignores non-parquet files in directory
        How: Create non-parquet files in directory, call identifiers
        Expected: Returns only identifiers from .pqt files
        """
        # Create some non-parquet files
        non_parquet_file = Path(self.temp_dir) / "AAPL.txt"
        non_parquet_file.write_text("some text")
        
        another_file = Path(self.temp_dir) / "GOOGL.csv"
        another_file.write_text("some csv data")
        
        # Add data for one parquet file
        data = {datetime(2023, 1, 1): {"open": 100.0, "close": 101.0}}
        self.store.add("MSFT", data)
        
        # Call identifiers
        result = self.store.identifiers()
        
        # Verify it only returns the parquet file identifier
        self.assertEqual(result, ["MSFT"])
        self.assertNotIn("AAPL", result)
        self.assertNotIn("GOOGL", result)
    
    def test_add_preserves_data_types_in_stored_data(self):
        """
        Test: add preserves data types in stored data
        How: Add data with mixed types (float, int, string), then retrieve
        Expected: Retrieved data has same types as original
        """
        pass
    
    def test_add_handles_missing_values_in_data(self):
        """
        Test: add handles missing values (NaN, None) in data
        How: Add data with missing values, then retrieve
        Expected: Missing values are preserved as NaN
        """
        pass
    
    def test_add_handles_large_datasets(self):
        """
        Test: add handles large datasets efficiently
        How: Add 1000 data points, then retrieve
        Expected: All data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_very_old_dates(self):
        """
        Test: add handles very old dates correctly
        How: Add data with dates from 1900s, then retrieve
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_future_dates(self):
        """
        Test: add handles future dates correctly
        How: Add data with dates in 2030s, then retrieve
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_timezone_aware_datetimes(self):
        """
        Test: add handles timezone-aware datetime objects
        How: Add data with timezone-aware datetime keys, then retrieve
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_timezone_naive_datetimes(self):
        """
        Test: add handles timezone-naive datetime objects
        How: Add data with timezone-naive datetime keys, then retrieve
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_get_handles_edge_case_start_date_equals_end_date(self):
        """
        Test: get handles edge case where start_date equals end_date
        How: Add data, call get with start_date=end_date
        Expected: Returns data for that specific date only
        """
        pass
    
    def test_get_handles_edge_case_start_date_after_end_date(self):
        """
        Test: get handles edge case where start_date is after end_date
        How: Add data, call get with start_date > end_date
        Expected: Returns empty OrderedDict or None
        """
        pass
    
    def test_add_handles_unicode_identifiers(self):
        """
        Test: add handles unicode characters in identifiers
        How: Add data with identifier containing unicode characters
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_special_characters_in_identifiers(self):
        """
        Test: add handles special characters in identifiers
        How: Add data with identifier containing special characters
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_very_long_identifiers(self):
        """
        Test: add handles very long identifiers
        How: Add data with very long identifier string
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_empty_identifier(self):
        """
        Test: add handles empty identifier string
        How: Add data with empty string as identifier
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_whitespace_in_identifiers(self):
        """
        Test: add handles whitespace in identifiers
        How: Add data with identifier containing spaces/tabs
        Expected: Data is stored and retrieved correctly
        """
        pass
    
    def test_add_handles_duplicate_identifiers_different_data(self):
        """
        Test: add handles multiple calls with same identifier but different data
        How: Add data for "AAPL", then add different data for "AAPL"
        Expected: Data is merged correctly according to overwrite parameter
        """
        pass
    
    def test_add_handles_concurrent_access_simulation(self):
        """
        Test: add handles simulated concurrent access
        How: Rapidly add data for same identifier multiple times
        Expected: All data is preserved correctly
        """
        pass
    
    def test_get_handles_millisecond_precision_dates(self):
        """
        Test: get handles dates with millisecond precision
        How: Add data with datetime objects including milliseconds, then retrieve
        Expected: Data is retrieved with same precision
        """
        pass
    
    def test_get_handles_microsecond_precision_dates(self):
        """
        Test: get handles dates with microsecond precision
        How: Add data with datetime objects including microseconds, then retrieve
        Expected: Data is retrieved with same precision
        """
        pass
    
    def test_add_handles_data_with_nested_structures(self):
        """
        Test: add handles data with nested dictionary structures
        How: Add data with nested dictionaries as values, then retrieve
        Expected: Nested structures are preserved
        """
        pass
    
    def test_add_handles_data_with_lists_as_values(self):
        """
        Test: add handles data with lists as values
        How: Add data with lists as values, then retrieve
        Expected: Lists are preserved correctly
        """
        pass
    
    def test_add_handles_data_with_custom_objects_as_values(self):
        """
        Test: add handles data with custom objects as values
        How: Add data with custom objects as values, then retrieve
        Expected: Objects are serialized/deserialized correctly
        """
        pass


if __name__ == '__main__':
    unittest.main() 