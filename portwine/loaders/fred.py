import os
import pandas as pd
from fredapi import Fred
from portwine.loaders.base import MarketDataLoader


class FREDMarketDataLoader(MarketDataLoader):
    """
    Loads data from the FRED (Federal Reserve Economic Data) system.

    This loader functions as a standard MarketDataLoader but with added capabilities:
    1. It can load existing parquet files from a specified directory
    2. If a file is not found and save_missing=True, it will attempt to download
       the data from FRED using the provided API key
    3. Downloaded data is saved as parquet for future use

    This is particularly useful for accessing economic indicators like interest rates,
    GDP, inflation metrics, and other macroeconomic data needed for advanced strategies.

    Parameters
    ----------
    data_path : str
        Directory where parquet files are stored
    api_key : str, optional
        FRED API key for downloading missing data
    save_missing : bool, default=False
        Whether to download and save missing data from FRED
    transform_to_daily : bool, default=True
        Convert non-daily data to daily frequency using forward-fill
    """

    # Source identifier for AlternativeMarketDataLoader
    SOURCE_IDENTIFIER = 'FRED'

    def __init__(self, data_path, api_key=None, save_missing=False, transform_to_daily=True):
        """
        Initialize the FRED market data loader.
        """
        super().__init__()
        self.data_path = data_path
        self.api_key = api_key
        self.save_missing = save_missing
        self.transform_to_daily = transform_to_daily

        self._fred_client = None

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    @property
    def fred_client(self):
        """
        Lazy initialization of the FRED client.
        """
        if self._fred_client is None and self.api_key:
            self._fred_client = Fred(api_key=self.api_key)
        return self._fred_client

    def load_ticker(self, ticker):
        """
        Load data for a specific ticker from parquet file or download from FRED.

        Parameters
        ----------
        ticker : str
            FRED series identifier (e.g., 'FEDFUNDS', 'DTB3', 'CPIAUCSL')

        Returns
        -------
        pd.DataFrame
            DataFrame with daily date index and appropriate columns for the portwine framework
        """
        file_path = os.path.join(self.data_path, f"{ticker}.parquet")

        # Check if file exists and load it
        if os.path.isfile(file_path):
            try:
                df = pd.read_parquet(file_path)
                return self._format_dataframe(df, ticker)
            except Exception as e:
                print(f"Error loading data for {ticker}: {str(e)}")

        # If file doesn't exist and save_missing is enabled, download from FRED
        if self.save_missing and self.fred_client:
            try:
                print(f"Downloading data for {ticker} from FRED...")
                # Get data from FRED
                series = self.fred_client.get_series(ticker)

                if series is not None and not series.empty:
                    # Convert to DataFrame with correct column name
                    df = pd.DataFrame(series, columns=['close'])

                    # Save to parquet for future use
                    df.to_parquet(file_path)

                    return self._format_dataframe(df, ticker)
                else:
                    print(f"No data found on FRED for ticker: {ticker}")
            except Exception as e:
                print(f"Error downloading data for {ticker} from FRED: {str(e)}")

        return None

    def _format_dataframe(self, df, ticker):
        """
        Format the dataframe to match the expected structure for portwine.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with appropriate columns
        """
        # Make sure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Make sure the index name is 'date'
        df.index.name = 'date'

        # Convert index to date (not datetime)
        if hasattr(df.index, 'normalize'):
            df.index = df.index.normalize().to_period('D').to_timestamp()

        # If we have only a Series, convert to DataFrame with 'close' column
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df, columns=['close'])

        # Handle the case where the first column might be the values
        if 'close' not in df.columns and len(df.columns) >= 1:
            # Rename the first column to 'close'
            df = df.rename(columns={df.columns[0]: 'close'})

        # If we have frequency that's not daily and transform_to_daily is True
        if self.transform_to_daily:
            # Reindex to daily frequency with forward fill
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(date_range, method='ffill')

        # Ensure we have all required columns for portwine
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df

    def get_fred_info(self, ticker):
        """
        Get information about a FRED series.

        Parameters
        ----------
        ticker : str
            FRED series identifier

        Returns
        -------
        pd.Series
            Series containing information about the FRED series
        """
        if self.fred_client:
            try:
                return self.fred_client.get_series_info(ticker)
            except Exception as e:
                print(f"Error getting info for {ticker}: {str(e)}")
        return None

    def search_fred(self, text, limit=10):
        """
        Search for FRED series by text.

        Parameters
        ----------
        text : str
            Text to search for
        limit : int, default=10
            Maximum number of results to return

        Returns
        -------
        pd.DataFrame
            DataFrame with search results
        """
        if self.fred_client:
            try:
                return self.fred_client.search(text, limit=limit)
            except Exception as e:
                print(f"Error searching FRED for '{text}': {str(e)}")
        return None
