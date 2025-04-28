from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
import warnings

import pandas as pd


class DataSource(ABC):
    """
    Base class for market data sources (e.g., Yahoo Finance, Polygon, FRED).
    Each source implementation should handle its own data format and API interactions.
    """

    def __init__(self, name: str):
        """
        Initialize the data source.

        Args:
            name: A unique identifier for this data source (e.g., 'YAHOO', 'POLYGON', 'FRED')
        """
        self.name = name

    def download_historical(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        store: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a ticker.

        Args:
            ticker: The ticker symbol to download
            start_date: Optional start date. If None, download as far back as possible
            end_date: Optional end date. If None, download up to now
            store: If True, store the downloaded data using the configured storage

        Returns:
            DataFrame with OHLCV data or None if download fails
        """
        # Validate date range
        if not self._validate_date_range(start_date, end_date):
            return None

        # Fetch raw data from the concrete implementation
        df = self._fetch_historical(ticker, start_date, end_date, store)
        if df is None or df.empty:
            return None

        # Filter by date range if provided
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        return df if not df.empty else None

    @abstractmethod
    def _fetch_historical(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        store: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch raw historical data for a ticker.
        This method should be implemented by concrete classes to handle the actual data fetching.
        The implementation may use the start_date and end_date parameters to optimize data fetching,
        but should still return all available data if these parameters are None.

        Args:
            ticker: The ticker symbol to fetch
            start_date: Optional start date. If None, fetch as far back as possible
            end_date: Optional end date. If None, fetch up to now
            store: If True, store the downloaded data

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest available data for a ticker.
        This is used for live data and returns a single row of OHLCV data with a timestamp.

        Args:
            ticker: The ticker symbol

        Returns:
            Dictionary with OHLCV data and timestamp, or None if not available.
            The dictionary should have keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        """
        raise NotImplementedError

    def sync(self, ticker: str) -> bool:
        """
        Synchronize data for a ticker with the source.
        This method is deprecated and will be removed in a future version.
        Data synchronization should be handled by a separate object.

        Args:
            ticker: The ticker symbol to synchronize

        Returns:
            True if sync was successful, False otherwise
        """
        warnings.warn(
            "The sync method is deprecated and will be removed in a future version. "
            "Data synchronization should be handled by a separate object.",
            DeprecationWarning,
            stacklevel=2
        )
        return True

    def _validate_timestamp(self, timestamp: datetime) -> bool:
        """
        Validate that a timestamp is not in the future.

        Args:
            timestamp: The timestamp to validate

        Returns:
            True if timestamp is valid, False otherwise
        """
        return timestamp <= datetime.now()

    def _validate_date_range(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> bool:
        """
        Validate that a date range is valid.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            True if date range is valid, False otherwise
        """
        if start_date is not None and end_date is not None:
            return start_date <= end_date
        return True 