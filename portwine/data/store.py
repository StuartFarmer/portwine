'''
DataStore is a class that stores data fetched from a data provider.

It can store it in flat files, in a database, or in memory; whatever the developer wants.
'''

from datetime import datetime
from typing import Union, OrderedDict
from collections import OrderedDict as OrderedDictType
import abc
import os
import pandas as pd
from pathlib import Path

'''

Data format:

identifier: {
    datetime_str: {
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    },
    datetime_str: {
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    },
    ...
}

Could also be fundamental data, etc. like:

identifier: {
    datetime_str: {
        gdp: float,
        inflation: float,
        unemployment: float,
        interest_rate: float,
        etc.
    },
    datetime_str: {
        gdp: float,
        inflation: float,
        unemployment: float,
        interest_rate: float,
        etc.
    },
    ...
}

'''

class DataStore(abc.ABC):
    def __init__(self, *args, **kwargs):
        ...

    '''
    Adds data to the store. Assumes that data is immutable, and that if the data already exists for the given times, 
    it is skipped.
    '''
    def add(self, identifier: str, data: dict):
        ...

    '''
    Gets data from the store in chronological order (earliest to latest).
    If the data is not found, it returns None.
    '''
    def get(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        ...

    '''
    Gets the latest data point for the identifier.
    If the data is not found, it returns None.
    '''
    def get_latest(self, identifier: str) -> Union[dict, None]:
        ...

    '''
    Gets the latest date for a given identifier from the store.
    If the data is not found, it returns None.
    '''
    def latest(self, identifier: str) -> Union[datetime, None]:
        ...

    '''
    Checks if data exists for a given identifier, start_date, and end_date.

    If start_date is None, it assumes the earliest date for that piece of data.
    If end_date is None, it assumes the latest date for that piece of data.

    If start_date is not None and end_date is not None, it checks if the data exists
    for the given start_date until the end of the data.
    '''
    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        ...
    
    '''
    Gets all identifiers from the store.
    '''
    def identifiers(self):
        ...


class ParquetDataStore(DataStore):
    """
    A DataStore implementation that stores data in parquet files.
    
    File structure:
    data_dir/
    ├── <identifier_1>.pqt
    ├── <identifier_2>.pqt
    ├── <identifier_3>.pqt
    └── <identifier_4>.pqt
    
    Each parquet file contains a DataFrame with:
    - Index: datetime (chronologically ordered)
    - Columns: data fields (open, high, low, close, volume, etc.)
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the ParquetDataStore.
        
        Args:
            data_dir: Directory where parquet files are stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, identifier: str) -> Path:
        """Get the parquet file path for an identifier."""
        return self.data_dir / f"{identifier}.pqt"
    
    def _load_dataframe(self, identifier: str) -> pd.DataFrame:
        """Load DataFrame from parquet file, return empty DataFrame if file doesn't exist."""
        file_path = self._get_file_path(identifier)
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                return df.sort_index()  # Ensure chronological order
            except Exception as e:
                print(f"Error loading parquet file for {identifier}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save_dataframe(self, identifier: str, df: pd.DataFrame):
        """Save DataFrame to parquet file."""
        file_path = self._get_file_path(identifier)
        try:
            # Ensure chronological order before saving
            df_sorted = df.sort_index()
            df_sorted.to_parquet(file_path, index=True)
        except Exception as e:
            print(f"Error saving parquet file for {identifier}: {e}")
    
    def add(self, identifier: str, data: dict, overwrite: bool = False):
        """
        Adds data to the store.
        
        Args:
            identifier: The identifier for the data
            data: Dictionary with datetime keys and data dictionaries as values
            overwrite: If True, overwrite existing data for the same dates
        """
        if not data:
            return
        
        # Load existing data
        df_existing = self._load_dataframe(identifier)
        
        # Convert new data to DataFrame
        new_data = []
        for dt, values in data.items():
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            row_data = {'date': dt, **values}
            new_data.append(row_data)
        
        if not new_data:
            return
        
        df_new = pd.DataFrame(new_data)
        df_new.set_index('date', inplace=True)
        
        if df_existing.empty:
            # No existing data, just save new data
            self._save_dataframe(identifier, df_new)
        else:
            if overwrite:
                # Remove existing rows for the same dates, then concatenate
                df_existing = df_existing.drop(df_new.index, errors='ignore')
                df_combined = pd.concat([df_existing, df_new])
            else:
                # For non-overwrite mode, filter out dates that already exist
                existing_dates = df_existing.index
                df_new_filtered = df_new[~df_new.index.isin(existing_dates)]
                
                if df_new_filtered.empty:
                    # No new data to add
                    return
                
                # Concatenate existing data with only new dates
                df_combined = pd.concat([df_existing, df_new_filtered])
            
            self._save_dataframe(identifier, df_combined)
    
    def get(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        """
        Gets data from the store in chronological order (earliest to latest).
        If the data is not found, it returns None.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        
        # Filter by date range
        if end_date is None:
            end_date = df.index.max()
        
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df[mask]
        
        if df_filtered.empty:
            return None
        
        # Convert to OrderedDict with datetime keys
        result = OrderedDict()
        for dt, row in df_filtered.iterrows():
            result[dt] = row.to_dict()
        
        return result
    
    def get_latest(self, identifier: str) -> Union[dict, None]:
        """
        Gets the latest data point for the identifier.
        If the data is not found, it returns None.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        
        latest_row = df.iloc[-1]
        return latest_row.to_dict()
    
    def latest(self, identifier: str) -> Union[datetime, None]:
        """
        Gets the latest date for a given identifier from the store.
        If the data is not found, it returns None.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        
        return df.index.max()
    
    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        """
        Checks if data exists for a given identifier, start_date, and end_date.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return False
        
        if start_date is None:
            start_date = df.index.min()
        if end_date is None:
            end_date = df.index.max()
        
        # Check if any data exists in the specified range
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df[mask].shape[0] > 0
    
    def identifiers(self):
        """
        Gets all identifiers from the store.
        """
        identifiers = []
        for file_path in self.data_dir.glob("*.pqt"):
            # Extract identifier from filename (remove .pqt extension)
            identifier = file_path.stem
            identifiers.append(identifier)
        return identifiers

