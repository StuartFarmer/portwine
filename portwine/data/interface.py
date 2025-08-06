from datetime import datetime
from typing import Dict, Optional, Any, List

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
    

class RestrictedDataInterface(MultiDataInterface):
    def __init__(self, loaders: Dict[Optional[str], Any]):
        super().__init__(loaders)
        self.restricted_tickers = []

    def set_restricted_tickers(self, tickers: List[str]):
        self.restricted_tickers = tickers
    
    def __getitem__(self, ticker: str):
        if ticker not in self.restricted_tickers and len(self.restricted_tickers) > 0:
            raise KeyError(f"Ticker {ticker} is not in the restricted tickers list.")
        return super().__getitem__(ticker)
