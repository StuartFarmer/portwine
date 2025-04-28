from datetime import datetime
from typing import Optional, List

import pandas as pd

from portwine.sources.base import DataSource


class DataLoader:
    """
    A class for storing and loading market data from various sources.
    """

    def __init__(self, market_data: DataSource, alternative_data: List[DataSource]):
        """
        Initialize the DataLoader.

        Args:
            market_data: The primary market data source
            alternative_data: List of alternative data sources
        """
        self.market_data = market_data
        self.alternative_data = alternative_data

    def store(self, data: pd.DataFrame, ticker: str, source: Optional[str] = None) -> bool:
        """
        Store data for a ticker, optionally specifying a source.

        Args:
            data: DataFrame with OHLCV data
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            True if storage was successful, False otherwise
        """
        raise NotImplementedError

    def load(
        self,
        ticker: str,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker, optionally specifying a source and date range.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        raise NotImplementedError

    def has(self, ticker: str, source: Optional[str] = None) -> bool:
        """
        Check if data exists for a ticker, optionally specifying a source.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            True if data exists, False otherwise
        """
        raise NotImplementedError

    def get_date_index(self, ticker: str, source: Optional[str] = None) -> Optional[List[datetime]]:
        """
        Get a sorted list of timestamps available for a ticker, optionally specifying a source.

        Args:
            ticker: The ticker symbol
            source: Optional source identifier. If None, uses market_data source.

        Returns:
            List of sorted timestamps or None if no data exists
        """
        raise NotImplementedError