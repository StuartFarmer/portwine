from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any

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

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get(self, ticker: str, timestamp: datetime) -> Optional[Dict[str, float]]:
        """
        Get data for a specific ticker and timestamp.

        Args:
            ticker: The ticker symbol
            timestamp: The timestamp to get data for

        Returns:
            Dictionary with OHLCV data or None if not available
        """
        raise NotImplementedError

    @abstractmethod
    def sync(self, ticker: str) -> bool:
        """
        Synchronize data for a ticker with the source.
        This should download any missing data and update existing data.

        Args:
            ticker: The ticker symbol to synchronize

        Returns:
            True if sync was successful, False otherwise
        """
        raise NotImplementedError

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