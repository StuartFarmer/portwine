import os
import pandas as pd
from portwine.loaders.base import MarketDataLoader


class EODHDMarketDataLoader(MarketDataLoader):
    """
    Loads historical market data for a list of tickers from CSV files.
    Each CSV must be located in data_path and named as TICKER.US.csv for each ticker.
    The CSV is assumed to have at least these columns:
        date, open, high, low, close, adjusted_close, volume
    The loaded data will be stored in a dictionary keyed by ticker symbol.

    Once loaded, data is cached in memory. Subsequent calls for the same ticker
    will not re-read from disk.
    """

    def __init__(self, data_path, exchange_code='US'):
        """
        Parameters
        ----------
        data_path : str
            The directory path where CSV files are located.
        """
        self.data_path = data_path
        self.exchange_code = exchange_code
        super().__init__()

    def load_ticker(self, ticker):
        file_path = os.path.join(self.data_path, f"{ticker}.{self.exchange_code}.csv")
        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for {ticker}: {file_path}")
            return None

        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        # Calculate adjusted prices
        adj_ratio = df['adjusted_close'] / df['close']

        df['open'] = df['open'] * adj_ratio
        df['high'] = df['high'] * adj_ratio
        df['low'] = df['low'] * adj_ratio
        df['close'] = df['adjusted_close']

        # Optional: reorder columns if needed
        df = df[[
            'open', 'high', 'low', 'close', 'volume',
        ]]
        df.sort_index(inplace=True)

        return df
