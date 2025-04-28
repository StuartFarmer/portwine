import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

from portwine.loaders_new.data_loader import DataLoader
from portwine.sources.base import DataSource


class FlatFileDataLoader(DataLoader):
    """
    A DataLoader implementation that stores data in a tree structure on disk.
    
    The data is organized as follows:
    data/
    ├── market/
    │   ├── eodhd/
    │   └── polygon/
    │       └── AAPL.parquet
    └── alternative/
        ├── fred/
        └── barchart/
            ├── VIX.parquet
            └── SOME_INDEX.parquet
    """

    def __init__(
        self,
        market_data: DataSource,
        alternative_data: List[DataSource],
        data_dir: str = "data"
    ):
        """
        Initialize the FlatFileDataLoader.

        Args:
            market_data: The primary market data source
            alternative_data: List of alternative data sources
            data_dir: Root directory for storing data files
        """
        super().__init__(market_data, alternative_data)
        self.data_dir = Path(data_dir)
        self._create_directory_structure()
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _create_directory_structure(self):
        """Create the directory structure if it doesn't exist."""
        # Create market data directories
        market_dir = self.data_dir / "market"
        for source in [self.market_data.name.lower()]:
            (market_dir / source).mkdir(parents=True, exist_ok=True)

        # Create alternative data directories
        alt_dir = self.data_dir / "alternative"
        for source in [alt.name.lower() for alt in self.alternative_data]:
            (alt_dir / source).mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, ticker: str, source: Optional[str] = None) -> Path:
        """
        Get the file path for a ticker and source.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            Path to the parquet file

        Raises:
            ValueError: If ticker is None or contains invalid characters
            ValueError: If source contains invalid characters
        """
        if ticker is None:
            raise ValueError("Ticker cannot be None")

        # Validate ticker and source names
        for name, value in [("ticker", ticker), ("source", source or self.market_data.name.lower())]:
            if any(c in value for c in r'\/:*?"<>|'):
                raise ValueError(f"{name} contains invalid characters: {value}")

        source = source or self.market_data.name.lower()
        
        # Determine if this is a market or alternative data source
        if source == self.market_data.name.lower():
            base_dir = self.data_dir / "market"
        else:
            base_dir = self.data_dir / "alternative"
            
        return base_dir / source / f"{ticker}.parquet"

    def _get_cache_key(self, ticker: str, source: Optional[str] = None) -> str:
        """
        Get the cache key for a ticker and source.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            Cache key string
        """
        source = source or self.market_data.name.lower()
        return f"{source}:{ticker}"

    def store(self, data: pd.DataFrame, ticker: str, source: Optional[str] = None) -> bool:
        """
        Store data for a ticker in a parquet file and update the cache.

        Args:
            data: DataFrame with OHLCV data
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            True if storage was successful, False otherwise
        """
        try:
            file_path = self._get_file_path(ticker, source)
            data.to_parquet(file_path)
            
            # Update cache
            cache_key = self._get_cache_key(ticker, source)
            self._data_cache[cache_key] = data.copy()
            
            return True
        except Exception as e:
            print(f"Error storing data for {ticker}: {e}")
            return False

    def load(
        self,
        ticker: str,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker from cache or parquet file.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        try:
            cache_key = self._get_cache_key(ticker, source)
            
            # Try to get from cache first
            if cache_key in self._data_cache:
                df = self._data_cache[cache_key].copy()
            else:
                # If not in cache, load from file
                file_path = self._get_file_path(ticker, source)
                if not file_path.exists():
                    return None
                    
                df = pd.read_parquet(file_path)
                # Store in cache
                self._data_cache[cache_key] = df.copy()
            
            # Filter by date range if provided
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
                
            return df if not df.empty else None
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None

    def has(self, ticker: str, source: Optional[str] = None) -> bool:
        """
        Check if data exists for a ticker in cache or on disk.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            True if data exists, False otherwise
        """
        cache_key = self._get_cache_key(ticker, source)
        if cache_key in self._data_cache:
            return True
            
        file_path = self._get_file_path(ticker, source)
        return file_path.exists()

    def get_date_index(self, ticker: str, source: Optional[str] = None) -> Optional[List[datetime]]:
        """
        Get a sorted list of timestamps available for a ticker.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            List of sorted timestamps or None if no data exists
        """
        try:
            df = self.load(ticker, source)
            if df is None:
                return None
            return sorted(df.index.tolist())
        except Exception as e:
            print(f"Error getting date index for {ticker}: {e}")
            return None

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()

    def remove_from_cache(self, ticker: str, source: Optional[str] = None):
        """
        Remove a specific ticker from the cache.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.
        """
        cache_key = self._get_cache_key(ticker, source)
        self._data_cache.pop(cache_key, None) 