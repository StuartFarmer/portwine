from portwine.loaders.base import MarketDataLoader


class AlternativeMarketDataLoader(MarketDataLoader):
    """
    A unified market data loader that routes requests to specialized data loaders
    based on source identifiers in ticker symbols.

    This loader expects tickers in the format "SOURCE:TICKER" where SOURCE matches
    a source identifier from one of the provided data loaders.

    Examples:
    - "FRED:SP500" would fetch SP500 data from the FRED data loader
    - "BARCHARTINDEX:ADDA" would fetch ADDA index from the Barchart Indices loader

    Parameters
    ----------
    loaders : list
        List of MarketDataLoader instances with SOURCE_IDENTIFIER class attributes
    """

    def __init__(self, loaders):
        """
        Initialize the alternative market data loader with a list of specialized loaders.
        """
        super().__init__()

        # Create a dictionary mapping source identifiers to loaders
        self.source_loaders = {}
        for loader in loaders:
            if hasattr(loader, 'SOURCE_IDENTIFIER'):
                self.source_loaders[loader.SOURCE_IDENTIFIER] = loader
            else:
                print(f"Warning: Loader {loader.__class__.__name__} does not have a SOURCE_IDENTIFIER")

    def load_ticker(self, ticker):
        """
        Load data for a specific ticker by routing to the appropriate data loader.

        Parameters
        ----------
        ticker : str
            Ticker in format "SOURCE:TICKER" (e.g., "FRED:SP500")

        Returns
        -------
        pd.DataFrame or None
            DataFrame with daily date index and appropriate columns, or None if load fails
        """
        # Check if it's already in the cache
        if ticker in self._data_cache:
            return self._data_cache[ticker].copy()  # Important: return a copy from cache

        # Parse the ticker to extract source and actual ticker
        if ":" not in ticker:
            print(f"Error: Invalid ticker format for {ticker}. Expected format is 'SOURCE:TICKER'.")
            return None

        source, actual_ticker = ticker.split(":", 1)

        # Find the appropriate loader
        if source in self.source_loaders:
            loader = self.source_loaders[source]
            data = loader.load_ticker(actual_ticker)

            # Cache the result if successful
            if data is not None:
                self._data_cache[ticker] = data.copy()  # Important: store a copy in cache

            return data
        else:
            print(f"Error: No loader found for source identifier '{source}'.")
            print(f"Available sources: {list(self.source_loaders.keys())}")
            return None

    def fetch_data(self, tickers):
        """
        Ensures we have data loaded for each ticker in 'tickers'.
        Routes each ticker to the appropriate specialized data loader.

        Parameters
        ----------
        tickers : list
            Tickers to ensure data is loaded for, in format "SOURCE:TICKER"

        Returns
        -------
        data_dict : dict
            { ticker: DataFrame }
        """
        fetched_data = {}

        for ticker in tickers:
            data = self.load_ticker(ticker)

            # Add to the returned dictionary if load was successful
            if data is not None:
                fetched_data[ticker] = data

        return fetched_data

    def get_available_sources(self):
        """
        Returns a list of available source identifiers.

        Returns
        -------
        list
            List of source identifiers
        """
        return list(self.source_loaders.keys())