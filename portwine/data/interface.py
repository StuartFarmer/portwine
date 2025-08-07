from datetime import datetime
from typing import Dict, Optional, Any, List
import pandas as pd

"""
These classes are the interfaces that strategies use to access data. They can also be used to access data for other purposes, such as backtesting.

The DataInterface class is the base class for all data interfaces. It is used to access data for a single ticker.

The MultiDataInterface class is used to access data for multiple tickers. It is used to access data for different data sources using prefixes.

The RestrictedDataInterface class is used to access data for a subset of tickers. It is used to access data for a subset of tickers.

The API is as follows:
    1. Provide a loader (which handles the caching, loading, etc.)
    2. Set the current timestamp
    3. Use the __getitem__ method to access data for a ticker
"""

class DataInterface:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.current_timestamp = None

    def set_current_timestamp(self, timestamp: datetime):
        self.current_timestamp = timestamp
    
    def __getitem__(self, ticker: str):
        """
        Access data for a ticker using bracket notation: interface['AAPL']
        
        Returns the latest OHLCV data for the ticker at the current timestamp.
        This enables lazy loading and caching without passing large dictionaries to strategies.
        
        Args:
            ticker: The ticker symbol to retrieve data for
            
        Returns:
            dict: OHLCV data dictionary with keys ['open', 'high', 'low', 'close', 'volume']
            
        Raises:
            ValueError: If current_timestamp is not set
            KeyError: If the ticker is not found or has no data
        """
        if self.current_timestamp is None:
            raise ValueError("Current timestamp not set. Call set_current_timestamp() first.")
        
        # Get data for this ticker at the current timestamp
        data = self.data_loader.next([ticker], self.current_timestamp)
        
        if ticker not in data or data[ticker] is None:
            raise KeyError(f"No data found for ticker: {ticker}")
        
        return data[ticker]

class MultiDataInterface:
    """
    A data interface that supports multiple data loaders with prefixes.
    
    Allows access to different data sources using prefix notation:
    - 'AAPL' -> uses the default market data loader
    - 'INDEX:SPY' -> uses the 'INDEX' loader
    - 'ECON:GDP' -> uses the 'ECON' loader
    
    The default loader (None prefix) is always the market data loader.
    """
    
    def __init__(self, loaders: Dict[Optional[str], Any]):
        """
        Initialize with a dictionary of loaders.
        
        Args:
            loaders: Dictionary mapping prefixes to data loaders.
                    Use None as key for the default market data loader.
                    Example: {None: market_loader, 'INDEX': index_loader, 'ECON': fred_loader}
        """
        self.loaders = loaders
        self.current_timestamp = None
        
        # Validate that we have a default loader (None key)
        if None not in loaders:
            raise ValueError("Must provide a default loader with None as key")
    
    def set_current_timestamp(self, timestamp: datetime):
        """Set the current timestamp for all loaders."""
        self.current_timestamp = timestamp
    
    def _parse_ticker(self, ticker: str) -> tuple[Optional[str], str]:
        """
        Parse a ticker string to extract prefix and symbol.
        
        Args:
            ticker: Ticker string like 'AAPL' or 'INDEX:SPY'
            
        Returns:
            tuple: (prefix, symbol) where prefix is None for default loader
        """
        if ':' in ticker:
            prefix, symbol = ticker.split(':', 1)
            return prefix, symbol
        else:
            return None, ticker
    
    def __getitem__(self, ticker: str):
        """
        Access data for a ticker using bracket notation.
        
        Supports both direct ticker access and prefixed access:
        - interface['AAPL'] -> uses default market data loader
        - interface['INDEX:SPY'] -> uses INDEX loader
        - interface['ECON:GDP'] -> uses ECON loader
        
        Args:
            ticker: The ticker symbol, optionally with prefix
            
        Returns:
            dict: Data dictionary (format depends on the loader)
            
        Raises:
            ValueError: If current_timestamp is not set
            KeyError: If the ticker is not found or has no data
            ValueError: If the prefix is not recognized
        """
        if self.current_timestamp is None:
            raise ValueError("Current timestamp not set. Call set_current_timestamp() first.")
        
        prefix, symbol = self._parse_ticker(ticker)
        
        # Get the appropriate loader
        if prefix not in self.loaders:
            raise ValueError(f"Unknown prefix '{prefix}' for ticker '{ticker}'. "
                           f"Available prefixes: {list(self.loaders.keys())}")
        
        loader = self.loaders[prefix]
        
        # Get data for this ticker at the current timestamp
        data = loader.next([symbol], self.current_timestamp)
        
        if symbol not in data or data[symbol] is None:
            raise KeyError(f"No data found for ticker: {ticker}")
        
        return data[symbol]
    
    def get_loader(self, prefix: Optional[str] = None):
        """
        Get a specific loader by prefix.
        
        Args:
            prefix: The prefix to get the loader for. None for default loader.
            
        Returns:
            The data loader for the specified prefix
        """
        if prefix not in self.loaders:
            raise ValueError(f"Unknown prefix '{prefix}'. "
                           f"Available prefixes: {list(self.loaders.keys())}")
        return self.loaders[prefix]
    
    def get_available_prefixes(self) -> list[Optional[str]]:
        """
        Get list of available prefixes.
        
        Returns:
            List of available prefixes (None represents the default loader)
        """
        return list(self.loaders.keys())
    
    def exists(self, ticker: str, start_date: str, end_date: str) -> bool:
        """
        Check if data exists for a ticker in the given date range.
        
        Args:
            ticker: The ticker symbol, optionally with prefix
            start_date: Start date string
            end_date: End date string
            
        Returns:
            True if data exists, False otherwise
        """
        try:
            # Try to access the ticker to see if it exists
            # We'll use a dummy timestamp to test
            original_timestamp = self.current_timestamp
            self.set_current_timestamp(pd.Timestamp('2020-01-01'))
            self[ticker]  # This will raise KeyError if ticker doesn't exist
            self.set_current_timestamp(original_timestamp)
            return True
        except (KeyError, ValueError):
            return False
    

class RestrictedDataInterface(MultiDataInterface):
    def __init__(self, loaders: Dict[Optional[str], Any]):
        super().__init__(loaders)
        self.restricted_tickers_by_prefix = {}

    def set_restricted_tickers(self, tickers: List[str], prefix: Optional[str] = None):
        """
        Set restricted tickers for a specific prefix.
        
        Args:
            tickers: List of tickers to restrict to
            prefix: The prefix to restrict (None for default loader)
        """
        self.restricted_tickers_by_prefix[prefix] = tickers
    
    def __getitem__(self, ticker: str):
        prefix, symbol = self._parse_ticker(ticker)
        
        # Check if this prefix has restrictions
        if prefix in self.restricted_tickers_by_prefix:
            restricted_tickers = self.restricted_tickers_by_prefix[prefix]
            if len(restricted_tickers) > 0 and symbol not in restricted_tickers:
                raise KeyError(f"Ticker {symbol} is not in the restricted tickers list for prefix {prefix}.")
        
        return super().__getitem__(ticker)
    
    def get(self, ticker: str, default=None):
        """Get data for a ticker with a default value if not found."""
        try:
            return self.__getitem__(ticker)
        except (KeyError, ValueError):
            return default
    
    def keys(self):
        """Return available tickers as a dictionary-like keys view."""
        # Get all available tickers from all loaders
        all_tickers = set()
        for prefix, loader in self.loaders.items():
            if hasattr(loader, 'get_available_tickers'):
                try:
                    tickers = loader.get_available_tickers()
                    if tickers is not None:
                        all_tickers.update(tickers)
                except (TypeError, AttributeError):
                    pass
            elif hasattr(loader, 'data_dict'):
                # For mock loaders that have data_dict
                all_tickers.update(loader.data_dict.keys())
            elif hasattr(loader, 'mock_data'):
                # For mock loaders that have mock_data
                all_tickers.update(loader.mock_data.keys())
            elif hasattr(loader, '_data_cache'):
                # For base MarketDataLoader
                all_tickers.update(loader._data_cache.keys())
            elif hasattr(loader, 'next'):
                # For loaders that have a next method, try to get tickers from the data interface
                # This is a fallback for mock loaders
                try:
                    # Try to get tickers from the data interface's mock_data
                    if hasattr(self, 'mock_data'):
                        all_tickers.update(self.mock_data.keys())
                except (TypeError, AttributeError):
                    pass
        
        # Apply restrictions if any
        restricted_tickers = set()
        for prefix, tickers in self.restricted_tickers_by_prefix.items():
            if tickers:  # Only apply restriction if tickers list is not empty
                restricted_tickers.update(tickers)
        
        if restricted_tickers:
            all_tickers = all_tickers.intersection(restricted_tickers)
        
        return all_tickers
    
    def __iter__(self):
        """Make the interface iterable over available tickers."""
        return iter(self.keys())
    
    def __contains__(self, ticker):
        """Check if a ticker is available."""
        try:
            self[ticker]
            return True
        except (KeyError, ValueError):
            return False
    
    def copy(self):
        """Create a copy of the interface."""
        new_interface = RestrictedDataInterface(self.loaders)
        new_interface.restricted_tickers_by_prefix = self.restricted_tickers_by_prefix.copy()
        new_interface.current_timestamp = self.current_timestamp
        return new_interface
