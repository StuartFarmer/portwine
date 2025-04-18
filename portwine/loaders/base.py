import pandas as pd

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

    def _get_bar_at_or_before(self, df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
        """
        Find the row whose index is <= ts, using searchsorted.
        Returns the row (a pd.Series) or None if ts is before the first index.
        """
        idx = df.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos >= 0:
            return df.iloc[pos]
        return None

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
