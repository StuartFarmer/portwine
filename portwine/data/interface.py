from datetime import datetime
from typing import Dict, Optional, Any, List
import pandas as pd
import pandas as pd

"""
These classes are the interfaces that strategies use to access data. They can also be used to access data for other purposes, such as backtesting.

The DataInterface class is the base class for all data interfaces. It is used to access data for a single ticker.

The MultiDataInterface class is used to access data for multiple data sources using prefixes. Each source is a store-like object
that implements the DataStore API (e.g., `get`, `add`, etc.). An optional DataSource (read-through cache orchestrator) can be
used anywhere a store is accepted as long as it conforms to the same API.

The RestrictedDataInterface class is used to access data for a subset of tickers.

API summary:
    1. Provide a store-like object implementing the DataStore API
    2. Set the current timestamp
    3. Use the __getitem__ method to access data for a ticker at the current timestamp via `store.get(symbol, dt)`
"""

class DataInterface:
    def __init__(self, data_loader):
        # `data_loader` may be a legacy loader (with `next`) or a store-like object (with `get`).
        # Keep attribute name for backwards-compatibility with existing call-sites.
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
        
        point = self.data_loader.get(ticker, self.current_timestamp)
        if point is None:
            raise KeyError(f"No data found for ticker: {ticker}")
        return point

    # ---------- Earliest-date discovery helpers ----------
    def _earliest_for_symbol(self, symbol: str) -> Optional[pd.Timestamp]:
        store = self.data_loader
        # Preferred: dedicated earliest() on store
        try:
            earliest_dt = getattr(store, "earliest")(symbol)
            if earliest_dt is not None:
                return pd.to_datetime(earliest_dt)
        except AttributeError:
            pass
        except Exception:
            pass

        # Fallback for legacy loaders: fetch_data and read index.min()
        try:
            df_map = getattr(store, "fetch_data")([symbol])
            df = df_map.get(symbol)
            if df is not None and not df.empty:
                return pd.to_datetime(df.index.min())
        except Exception:
            pass

        # Fallback to get_all across full history if available (may be expensive)
        try:
            get_all = getattr(store, "get_all")
            from datetime import datetime as _dt
            data = get_all(symbol, _dt(1900, 1, 1), None)
            if data:
                first_key = next(iter(data.keys()))
                return pd.to_datetime(first_key)
        except Exception:
            pass

        return None

    def earliest_any_date(self, tickers: List[str]) -> str:
        dates: List[pd.Timestamp] = []
        for t in tickers:
            dt = self._earliest_for_symbol(t)
            if dt is not None:
                dates.append(dt)
        if not dates:
            raise ValueError("Cannot determine earliest date: no data found for any requested tickers")
        return min(dates).strftime('%Y-%m-%d')

    def earliest_common_date(self, tickers: List[str]) -> str:
        dates: List[pd.Timestamp] = []
        missing: List[str] = []
        for t in tickers:
            dt = self._earliest_for_symbol(t)
            if dt is None:
                missing.append(t)
            else:
                dates.append(dt)
        if missing:
            raise ValueError(f"Cannot determine earliest common date; missing data for: {missing}")
        if not dates:
            raise ValueError("Cannot determine earliest common date: no dates computed")
        # Common means all tickers have data from this date forward → choose latest of their earliest
        return max(dates).strftime('%Y-%m-%d')

    def exists(self, ticker: str, start_date: str, end_date: str) -> bool:
        """
        Check if data exists for a ticker in the given date range.

        Args:
            ticker: The ticker symbol
            start_date: Start date string (YYYY-MM-DD format)
            end_date: End date string (YYYY-MM-DD format)

        Returns:
            True if data exists, False otherwise
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        return self.data_loader.exists(ticker, start_dt, end_dt)

class MultiDataInterface:
    """
    A data interface that supports multiple data stores with prefixes.
    
    Allows access to different data sources using prefix notation:
    - 'AAPL' -> uses the default market data store
    - 'INDEX:SPY' -> uses the 'INDEX' store
    - 'ECON:GDP' -> uses the 'ECON' store
    
    The default store (None prefix) is always the market data store.
    """
    
    def __init__(self, loaders: Dict[Optional[str], Any]):
        """
        Initialize with a dictionary of stores.
        
        Args:
            loaders: Dictionary mapping prefixes to data stores.
                    Use None as key for the default market data store.
                    Example: {None: market_store, 'INDEX': index_store, 'ECON': econ_store}
        """
        self.loaders = loaders
        self.current_timestamp = None
        
        # Validate that we have a default store (None key)
        if None not in loaders:
            raise ValueError("Must provide a default store with None as key")
    
    def set_current_timestamp(self, timestamp: datetime):
        """Set the current timestamp for all stores."""
        self.current_timestamp = timestamp
    
    def _parse_ticker(self, ticker: str) -> tuple[Optional[str], str]:
        """
        Parse a ticker string to extract prefix and symbol.
        
        Args:
            ticker: Ticker string like 'AAPL' or 'INDEX:SPY'
            
        Returns:
            tuple: (prefix, symbol) where prefix is None for default store
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
        - interface['AAPL'] -> uses default market data store
        - interface['INDEX:SPY'] -> uses INDEX store
        - interface['ECON:GDP'] -> uses ECON store
        
        Args:
            ticker: The ticker symbol, optionally with prefix
            
        Returns:
            dict: Data dictionary (format depends on the store)
            
        Raises:
            ValueError: If current_timestamp is not set
            KeyError: If the ticker is not found or has no data
            ValueError: If the prefix is not recognized
        """
        if self.current_timestamp is None:
            raise ValueError("Current timestamp not set. Call set_current_timestamp() first.")
        
        prefix, symbol = self._parse_ticker(ticker)
        
        # Get the appropriate store
        if prefix not in self.loaders:
            raise ValueError(f"Unknown prefix '{prefix}' for ticker '{ticker}'. "
                           f"Available prefixes: {list(self.loaders.keys())}")
        store = self.loaders[prefix]
        point = store.get(symbol, self.current_timestamp)
        
        if point is None:
            raise KeyError(f"No data found for ticker: {ticker}")
        return point

    # ---------- Earliest-date discovery for multi-source ----------
    def _earliest_for_symbol_on_store(self, prefix: Optional[str], symbol: str) -> Optional[pd.Timestamp]:
        if prefix not in self.loaders:
            return None
        store = self.loaders[prefix]

        # Preferred: dedicated earliest() on store
        try:
            earliest_dt = getattr(store, "earliest")(symbol)
            if earliest_dt is not None:
                return pd.to_datetime(earliest_dt)
        except AttributeError:
            pass
        except Exception:
            pass

        # Fallback for legacy loaders: fetch_data and read index.min()
        try:
            df_map = getattr(store, "fetch_data")([symbol])
            df = df_map.get(symbol)
            if df is not None and not df.empty:
                return pd.to_datetime(df.index.min())
        except Exception:
            pass

        # Fallback to get_all across full history if available (may be expensive)
        try:
            get_all = getattr(store, "get_all")
            from datetime import datetime as _dt
            data = get_all(symbol, _dt(1900, 1, 1), None)
            if data:
                first_key = next(iter(data.keys()))
                return pd.to_datetime(first_key)
        except Exception:
            pass

        return None

    def earliest_any_date(self, tickers: List[str]) -> str:
        dates: List[pd.Timestamp] = []
        for t in tickers:
            prefix, symbol = self._parse_ticker(t)
            dt = self._earliest_for_symbol_on_store(prefix, symbol)
            if dt is not None:
                dates.append(dt)
        if not dates:
            raise ValueError("Cannot determine earliest date: no data found for any requested tickers")
        return min(dates).strftime('%Y-%m-%d')

    def earliest_common_date(self, tickers: List[str]) -> str:
        dates: List[pd.Timestamp] = []
        missing: List[str] = []
        for t in tickers:
            prefix, symbol = self._parse_ticker(t)
            dt = self._earliest_for_symbol_on_store(prefix, symbol)
            if dt is None:
                missing.append(t)
            else:
                dates.append(dt)
        if missing:
            raise ValueError(f"Cannot determine earliest common date; missing data for: {missing}")
        if not dates:
            raise ValueError("Cannot determine earliest common date: no dates computed")
        return max(dates).strftime('%Y-%m-%d')
    
    def get_loader(self, prefix: Optional[str] = None):
        """
        Get a specific store by prefix.
        
        Args:
            prefix: The prefix to get the store for. None for default store.
            
        Returns:
            The data store for the specified prefix
        """
        if prefix not in self.loaders:
            raise ValueError(f"Unknown prefix '{prefix}'. "
                           f"Available prefixes: {list(self.loaders.keys())}")
        return self.loaders[prefix]
    
    def get_available_prefixes(self) -> list[Optional[str]]:
        """
        Get list of available prefixes.
        
        Returns:
            List of available prefixes (None represents the default store)
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
        # Get all available tickers from all stores
        all_tickers = set()
        for prefix, store in self.loaders.items():
            try:
                tickers = store.identifiers()
            except AttributeError:
                tickers = None
            if tickers:
                all_tickers.update(tickers)
        
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
