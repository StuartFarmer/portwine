import os
import pandas as pd
from portwine.loaders.base import MarketDataLoader


class PolygonMarketDataLoader(MarketDataLoader):
    """
    Assumes storage as parquet files.
    """

    def __init__(self, data_path):
        """
        Parameters
        ----------
        data_path : str
            The directory path where CSV files are located.
        """
        self.data_path = data_path
        super().__init__()

    def load_ticker(self, ticker):
        file_path = os.path.join(self.data_path, f"{ticker}.parquet")
        if not os.path.isfile(file_path):
            print(f"Warning: Parquet file not found for {ticker}: {file_path}")
            return None

        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df.set_index('date', inplace=True)
        df.drop(columns='timestamp', inplace=True)

        return df
