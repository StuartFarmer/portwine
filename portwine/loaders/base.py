class MarketDataLoader:
    """
    Loads historical market data given a ticker. Override load_ticker to determine
    how a ticker is fetched from a file, database, online, etc.

    Price data tickers must be returned as a Pandas DataFrame with a daily date index,
    open, high, low, close, and volume columns that are adjusted.

    Once loaded, data is cached in memory. Subsequent calls for the same ticker
    will not re-read from disk.
    """

    def __init__(self):
        self._data_cache = {}  # Dictionary {ticker: DataFrame of historical data}

    def load_ticker(self, ticker):
        raise NotImplementedError

    def fetch_data(self, tickers):
        """
        Ensures we have data loaded for each ticker in 'tickers'.
        If a ticker is not in the cache, read from CSV and cache it.
        Returns a dictionary of {ticker -> DataFrame} for the requested tickers.

        Parameters
        ----------
        tickers : list
            Tickers to ensure data is loaded for.

        Returns
        -------
        data_dict : dict
            { ticker: DataFrame }
        """
        fetched_data = {}
        for ticker in tickers:
            if ticker not in self._data_cache:
                # Load the ticker
                df = self.load_ticker(ticker)

                # Ticker not found
                if df is None:
                    continue

                # Cache it
                self._data_cache[ticker] = df

            # Add to the returned dictionary
            if ticker in self._data_cache:
                fetched_data[ticker] = self._data_cache[ticker]

        return fetched_data
