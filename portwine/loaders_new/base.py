import os
import pandas as pd
from typing import Optional, Dict, List


class MarketDataLoader:
    """
    Base loader. Override load_ticker; fetch_data remains unchanged.
    Adds:
      - get_all_dates: union calendar for any tickers
      - next: returns the bar at or immediately before a given ts via searchsorted
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the market data loader.
        
        Args:
            data_path: Optional path to store/load data files. If not provided,
                      a temporary directory will be used.
        """
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self.data_path = data_path
        if data_path is not None:
            os.makedirs(data_path, exist_ok=True)

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Must be overridden to load and return a DataFrame indexed by pd.Timestamp
        with columns ['open','high','low','close','volume'], or return None.
        """
        raise NotImplementedError

    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Exactly as before: caches & returns all requested tickers.
        """
        fetched = {}
        for t in tickers:
            if t not in self._data_cache:
                df = self.load_ticker(t)
                if df is not None:
                    self._data_cache[t] = df
            if t in self._data_cache:
                fetched[t] = self._data_cache[t]
        return fetched

    def get_all_dates(self, tickers: List[str]) -> List[pd.Timestamp]:
        """
        Build the *union* of all timestamps across these tickers.
        This is your intraday/daily trading calendar.
        """
        data = self.fetch_data(tickers)
        all_ts = {ts for df in data.values() for ts in df.index}
        return sorted(all_ts)

    def _get_bar_at_or_before(self, df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
        """
        Find the row whose index is <= ts, using searchsorted.
        Returns the row (a pd.Series) or None if ts is before the first index.
        """
        if df is None or df.empty:
            return None
            
        idx = df.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos >= 0:
            return df.iloc[pos]
        return None

    def next(self,
             tickers: List[str],
             ts: pd.Timestamp
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """
        For a given timestamp ts, return a dict:
          { ticker: {'open','high','low','close','volume'} }
        where the values come from the bar at or immediately before ts.
        """
        data = self.fetch_data(tickers)
        bar_dict: Dict[str, Optional[Dict[str, float]]] = {}

        # Initialize all tickers to None
        for t in tickers:
            bar_dict[t] = None

        # Update with available data
        for t, df in data.items():
            row = self._get_bar_at_or_before(df, ts)
            if row is not None:
                bar_dict[t] = {
                    'open':   float(row['open']),
                    'high':   float(row['high']),
                    'low':    float(row['low']),
                    'close':  float(row['close']),
                    'volume': float(row['volume'])
                }

        return bar_dict 