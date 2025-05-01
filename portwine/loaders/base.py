import pandas as pd
from typing import Optional

class MarketDataLoader:
    """
    Base loader. Override load_ticker; fetch_data remains unchanged.
    Adds:
      - get_all_dates: union calendar for any tickers
      - next: returns the bar at or immediately before a given ts via searchsorted
    """

    def __init__(self):
        self._data_cache = {}

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        """
        Must be overridden to load and return a DataFrame indexed by pd.Timestamp
        with columns ['open','high','low','close','volume'], or return None.
        """
        raise NotImplementedError

    def fetch_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
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

    def get_all_dates(self, tickers: list[str]) -> list[pd.Timestamp]:
        """
        Build the *union* of all timestamps across these tickers.
        This is your intraday/daily trading calendar.
        """
        data = self.fetch_data(tickers)
        all_ts = {ts for df in data.values() for ts in df.index}
        return sorted(all_ts)

    def _get_bar_at_or_before(self, df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
        """
        Get the bar at or immediately before the given timestamp.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        ts : pd.Timestamp
            Timestamp to get data for
            
        Returns
        -------
        pd.Series or None
            Series with OHLCV data if found, None otherwise
        """
        if df.empty:
            return None
            
        # Ensure both timestamp and index are timezone-aware and match
        if ts.tzinfo is None:
            ts = ts.tz_localize(df.index.tz)
        elif df.index.tz is None:
            df.index = df.index.tz_localize(ts.tz)
        elif str(ts.tz) != str(df.index.tz):
            ts = ts.tz_convert(df.index.tz)
            
        idx = df.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        return df.iloc[pos]

    def next(self,
             tickers: list[str],
             ts: pd.Timestamp
    ) -> dict[str, dict[str, float] | None]:
        """
        For a given timestamp ts, return a dict:
          { ticker: {'open','high','low','close','volume'} }
        where the values come from the bar at or immediately before ts.
        """
        data = self.fetch_data(tickers)
        bar_dict: dict[str, dict[str, float] | None] = {}

        for t, df in data.items():
            row = self._get_bar_at_or_before(df, ts)
            if row is None:
                bar_dict[t] = None
            else:
                bar_dict[t] = {
                    'open':   float(row['open']),
                    'high':   float(row['high']),
                    'low':    float(row['low']),
                    'close':  float(row['close']),
                    'volume': float(row['volume'])
                }

        return bar_dict
