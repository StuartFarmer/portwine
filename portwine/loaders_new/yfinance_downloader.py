import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import MarketDataLoader


class YFinanceDownloader(MarketDataLoader):
    """
    Downloads market data from Yahoo Finance and saves it as parquet files.
    """

    def __init__(self, data_path: str, start_date: Optional[datetime] = None):
        """
        Initialize the YFinance downloader.

        Args:
            data_path: Path to store the parquet files
            start_date: Optional start date for historical data. If not provided,
                       defaults to 5 years ago.
        """
        super().__init__(data_path)
        self.start_date = start_date or (datetime.now() - timedelta(days=5*365))
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        if self.data_path is not None:
            os.makedirs(self.data_path, exist_ok=True)

    def _get_parquet_path(self, ticker: str) -> str:
        """Get the path for a ticker's parquet file."""
        return os.path.join(self.data_path, f"{ticker}.parquet")

    def download_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download data for a ticker from Yahoo Finance.

        Args:
            ticker: The ticker symbol to download

        Returns:
            DataFrame with OHLCV data or None if download fails
        """
        try:
            # Download data from yfinance
            ticker_data = yf.Ticker(ticker)
            df = ticker_data.history(start=self.start_date)

            if df.empty:
                print(f"No data found for {ticker}")
                return None

            # Ensure the index is named 'timestamp'
            df.index.name = 'timestamp'

            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns for {ticker}")
                return None

            # Rename columns to lowercase
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Save to parquet
            parquet_path = self._get_parquet_path(ticker)
            df.to_parquet(parquet_path)

            return df

        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            return None

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker from parquet file, downloading if necessary.

        Args:
            ticker: The ticker symbol to load

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        parquet_path = self._get_parquet_path(ticker)

        # Try to load from parquet first
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                return df
            except Exception as e:
                print(f"Error loading {ticker} from parquet: {str(e)}")

        # If not found or error loading, download fresh data
        return self.download_ticker(ticker) 