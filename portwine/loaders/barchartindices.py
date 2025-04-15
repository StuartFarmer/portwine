import os
import pandas as pd
from portwine.loaders.base import MarketDataLoader


class BarchartIndicesMarketDataLoader(MarketDataLoader):
    """
    Loads data from local CSV files containing Barchart Indices data.

    This loader is designed to handle CSV files with index data following the format
    of Barchart indices. It reads from a local directory and does not download from
    any online source.

    Parameters
    ----------
    data_path : str
        Directory where index CSV files are stored
    """

    # Source identifier for AlternativeMarketDataLoader
    SOURCE_IDENTIFIER = 'BARCHARTINDEX'

    def __init__(self, data_path):
        """
        Initialize the Barchart Indices market data loader.
        """
        super().__init__()
        self.data_path = data_path

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def load_ticker(self, ticker):
        """
        Load data for a specific Barchart index from a CSV file.

        Parameters
        ----------
        ticker : str
            Barchart index code (e.g., 'ADDA')

        Returns
        -------
        pd.DataFrame
            DataFrame with daily date index and appropriate columns for the portwine framework
        """
        file_path = os.path.join(self.data_path, f"{ticker}.csv")

        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for Barchart index {ticker}: {file_path}")
            return None

        try:
            # Load CSV file with automatic date parsing
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            return self._format_dataframe(df)
        except Exception as e:
            print(f"Error loading data for Barchart index {ticker}: {str(e)}")
            return None

    def _format_dataframe(self, df):
        """
        Format the dataframe to match the expected structure for portwine.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format

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

        # Sort by date
        df = df.sort_index()

        # Ensure we have all required columns for portwine
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check if we have a close column, if not look for other possible names
        if 'close' not in df.columns:
            for alt_name in ['Close', 'CLOSE', 'price', 'Price', 'value', 'Value']:
                if alt_name in df.columns:
                    df['close'] = df[alt_name]
                    break
            # If still not found, use the first numeric column
            if 'close' not in df.columns and not df.empty:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df['close'] = df[numeric_cols[0]]

        # Fill missing required columns
        for col in required_columns:
            if col not in df.columns:
                if col in ['open', 'high', 'low'] and 'close' in df.columns:
                    df[col] = df['close']
                elif col == 'volume':
                    df[col] = 0
                else:
                    # If we can't determine a suitable close value, use NaN
                    df[col] = float('nan')

        return df