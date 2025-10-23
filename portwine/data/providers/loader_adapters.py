"""
Loader Adapters for Backward Compatibility

This module provides adapter classes that implement the old loader interface
but internally use the new provider system. This allows for gradual migration
from the old loaders to the new providers while maintaining backward compatibility.

The adapters implement the same interface as the old MarketDataLoader classes
but delegate data fetching to the appropriate DataProvider instances.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from collections import OrderedDict
import warnings
import os

from .base import DataProvider
from .alpaca import AlpacaProvider
from .eodhd import EODHDProvider
from .polygon import PolygonProvider
from .fred import FREDProvider


class ProviderBasedLoader:
    """
    Base adapter class that implements the old loader interface using providers.
    
    This class provides the same interface as the old MarketDataLoader but
    internally uses DataProvider instances for data fetching.
    """
    
    def __init__(self):
        self._data_cache = {}
        self._numpy_cache = {}
        self._date_cache = {}
        self._provider_cache = {}
        # Add data_loader attribute for compatibility with DataInterface
        self.data_loader = self
        # Add current_timestamp attribute for compatibility with DataInterface
        self.current_timestamp = None
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """
        Get the appropriate provider for a ticker.
        Override this method in subclasses to implement provider selection logic.
        """
        raise NotImplementedError("Subclasses must implement _get_provider")
    
    def _load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker using the appropriate provider.
        """
        provider = self._get_provider(ticker)
        if provider is None:
            return None
        
        try:
            # Get data for the last 5 years as a reasonable default
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            
            raw_data = provider.get_data(ticker, start_date, end_date)
            
            if not raw_data:
                return None
            
            # Convert to DataFrame
            data_list = []
            for dt, bar_data in raw_data.items():
                row = {
                    'open': bar_data.get('open', 0.0),
                    'high': bar_data.get('high', 0.0),
                    'low': bar_data.get('low', 0.0),
                    'close': bar_data.get('close', 0.0),
                    'volume': bar_data.get('volume', 0.0)
                }
                data_list.append(row)
            
            df = pd.DataFrame(data_list, index=list(raw_data.keys()))
            df.index.name = 'date'
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            warnings.warn(f"Failed to load data for {ticker}: {e}")
            return None
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Public method to load data for a ticker.
        This is the main interface method that subclasses should override.
        """
        raise NotImplementedError("Subclasses must implement load_ticker")
    
    def get(self, ticker: str, timestamp: datetime) -> Optional[Dict[str, float]]:
        """
        Get data for a ticker at a specific timestamp.
        This method makes the loader compatible with the new DataInterface system.
        
        Args:
            ticker: The ticker symbol
            timestamp: The timestamp to get data for
            
        Returns:
            dict: OHLCV data dictionary or None if no data found
        """
        # Convert timestamp to pandas Timestamp if needed
        if not isinstance(timestamp, pd.Timestamp):
            ts = pd.Timestamp(timestamp)
        else:
            ts = timestamp
            
        # Use the existing next method logic
        result = self.next([ticker], ts)
        return result.get(ticker)
    
    def exists(self, ticker: str, start_date: str, end_date: str) -> bool:
        """
        Check if data exists for a ticker in the given date range.
        This method is required by the backtester for validation.
        
        Args:
            ticker: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            # Convert dates to datetime objects
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            
            # Fetch data for the ticker
            df = self.fetch_data([ticker]).get(ticker)
            if df is None or df.empty:
                return False
            
            # Check if we have data in the date range
            ticker_dates = df.index
            if len(ticker_dates) == 0:
                return False
            
            # Check if any dates fall within the range
            mask = (ticker_dates >= start_dt) & (ticker_dates <= end_dt)
            return mask.any()
            
        except Exception:
            return False
    
    def set_current_timestamp(self, timestamp: datetime) -> None:
        """
        Set the current timestamp for data access.
        This method is required by the DataInterface system.
        
        Args:
            timestamp: The timestamp to set
        """
        # This method is required by DataInterface but not used by legacy loaders
        # The legacy loaders use the timestamp passed to next() method
        pass
    
    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Caches & returns all requested tickers.
        Maintains the same interface as the old MarketDataLoader.
        """
        fetched = {}
        for ticker in tickers:
            if ticker not in self._data_cache:
                df = self.load_ticker(ticker)
                if df is not None:
                    self._data_cache[ticker] = df
                    self._create_numpy_cache(ticker, df)
            if ticker in self._data_cache:
                fetched[ticker] = self._data_cache[ticker]
        return fetched
    
    def _create_numpy_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Create numpy arrays for fast data access.
        """
        self._date_cache[ticker] = df.index.values.astype('datetime64[ns]')
        self._numpy_cache[ticker] = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float64)
    
    def get_all_dates(self, tickers: List[str]) -> List[pd.Timestamp]:
        """
        Build the union of all timestamps across these tickers.
        """
        data = self.fetch_data(tickers)
        all_ts = {ts for df in data.values() for ts in df.index}
        return sorted(all_ts)
    
    def _get_bar_at_or_before_numpy(self, ticker: str, ts: pd.Timestamp) -> Optional[np.ndarray]:
        """
        Get the bar at or immediately before the given timestamp using numpy.
        """
        if ticker not in self._numpy_cache:
            return None
            
        date_array = self._date_cache[ticker]
        if len(date_array) == 0:
            return None
            
        # Convert timestamp to numpy datetime64 for comparison
        if ts.tzinfo is not None:
            ts_utc = ts.tz_convert('UTC')
            ts_np = np.datetime64(ts_utc.replace(tzinfo=None))
        else:
            ts_np = np.datetime64(ts)
        
        pos = np.searchsorted(date_array, ts_np, side="right") - 1
        if pos < 0:
            return None
            
        return self._numpy_cache[ticker][pos]
    
    def _get_bar_at_or_before(self, df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
        """
        Get the bar at or immediately before the given timestamp.
        This method works with pandas DataFrames and returns pandas Series.
        """
        if df.empty:
            return None
            
        # Convert timestamp to pandas Timestamp if needed
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
            
        # Handle timezone conversion
        if ts.tzinfo is not None:
            ts_utc = ts.tz_convert('UTC')
            ts = ts_utc.replace(tzinfo=None)
        
        # Find the position of the timestamp
        pos = df.index.get_indexer([ts], method='ffill')[0]
        
        if pos < 0:
            return None
            
        return df.iloc[pos]
    
    def next(self, tickers: List[str], ts: pd.Timestamp) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Get data for tickers at or immediately before timestamp.
        Maintains the same interface as the old MarketDataLoader.
        """
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
            
        result = {}
        for ticker in tickers:
            df = self.fetch_data([ticker]).get(ticker)
            if df is not None:
                bar = self._get_bar_at_or_before_numpy(ticker, ts)
                if bar is not None:
                    result[ticker] = {
                        "open": float(bar[0]),
                        "high": float(bar[1]),
                        "low": float(bar[2]),
                        "close": float(bar[3]),
                        "volume": float(bar[4]),
                    }
                else:
                    result[ticker] = None
            # Don't add ticker to result if df is None (ticker not found)
        return result


class AlpacaMarketDataLoader(ProviderBasedLoader):
    """
    Adapter for Alpaca data that uses AlpacaProvider internally.
    """
    
    def __init__(self, api_key: str, api_secret: str, data_url: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.data_url = data_url
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """Get Alpaca provider for any ticker."""
        if 'alpaca' not in self._provider_cache:
            self._provider_cache['alpaca'] = AlpacaProvider(
                self.api_key, 
                self.api_secret, 
                data_url=self.data_url
            )
        return self._provider_cache['alpaca']


class EODHDMarketDataLoader(ProviderBasedLoader):
    """
    Adapter for EODHD data that loads CSV files from disk.
    Maintains backward compatibility with the original API.
    """
    
    def __init__(self, data_path: str, exchange_code: str = "US"):
        super().__init__()
        self.data_path = data_path
        self.exchange_code = exchange_code
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """This loader doesn't use external providers - it loads from disk."""
        return None
    
    def _load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Override _load_ticker to use the disk-based load_ticker method.
        This ensures fetch_data works correctly.
        """
        return self.load_ticker(ticker)
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker from CSV file.
        This method maintains backward compatibility with the original loader API.
        """
        if not self.data_path:
            return None
            
        file_path = os.path.join(self.data_path, f"{ticker}.{self.exchange_code}.csv")
        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for {ticker}: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            # Calculate adjusted prices if columns exist
            if 'adjusted_close' in df.columns and 'close' in df.columns:
                adj_ratio = df['adjusted_close'] / df['close']
                df['open'] = df['open'] * adj_ratio
                df['high'] = df['high'] * adj_ratio
                df['low'] = df['low'] * adj_ratio
                df['close'] = df['adjusted_close']
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                df = df[required_cols]
                df.sort_index(inplace=True)
                return df
            else:
                print(f"Warning: Missing required columns in {file_path}")
                return None
                
        except Exception as e:
            print(f"Warning: Failed to load CSV for {ticker}: {e}")
            return None


class PolygonMarketDataLoader(ProviderBasedLoader):
    """
    Adapter for Polygon data that uses PolygonProvider internally.
    """
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """Get Polygon provider for any ticker."""
        if 'polygon' not in self._provider_cache:
            self._provider_cache['polygon'] = PolygonProvider(self.api_key)
        return self._provider_cache['polygon']


class FREDMarketDataLoader(ProviderBasedLoader):
    """
    Adapter for FRED data that uses FREDProvider internally.
    """
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """Get FRED provider for any ticker."""
        if 'fred' not in self._provider_cache:
            self._provider_cache['fred'] = FREDProvider(self.api_key)
        return self._provider_cache['fred']


class BrokerDataLoader(ProviderBasedLoader):
    """
    Adapter for broker data that maintains the same interface as the old BrokerDataLoader.
    """
    
    SOURCE_IDENTIFIER = "BROKER"
    
    def __init__(self, broker=None, initial_equity: Optional[float] = None):
        super().__init__()
        self.broker = broker
        self.equity = initial_equity
        
        if broker is None and initial_equity is None:
            raise ValueError("Give either a broker or an initial_equity")
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """Broker loader doesn't use external providers."""
        return None
    
    def next(self, tickers: List[str], ts: pd.Timestamp) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Return a dict for each ticker; if prefixed with 'BROKER', return {'equity': value}, else None.
        Maintains the same logic as the old BrokerDataLoader.
        """
        out: Dict[str, Optional[Dict[str, float]]] = {}
        for ticker in tickers:
            # Only handle tickers with a prefix; non-colon tickers are not for BROKER
            if ":" not in ticker:
                out[ticker] = None
                continue
            src, key = ticker.split(":", 1)
            if src != self.SOURCE_IDENTIFIER:
                out[ticker] = None
                continue

            # live vs. offline
            if self.broker is not None:
                account = self.broker.get_account()
                eq = account.equity
            else:
                eq = self.equity

            out[ticker] = {"equity": float(eq)}
        return out
    
    def update(self, ts: pd.Timestamp, raw_sigs: Dict[str, Any], 
               raw_rets: Dict[str, float], strat_ret: float) -> None:
        """
        Backtest-only hook: evolve self.equity by applying strategy return.
        """
        if self.broker is None and strat_ret is not None:
            self.equity *= (1 + strat_ret)


# Legacy import aliases for backward compatibility
# These will show deprecation warnings when imported
def _deprecated_import_warning(old_name: str, new_name: str):
    """Helper to show deprecation warnings for legacy imports."""
    warnings.warn(
        f"Importing {old_name} is deprecated. Use {new_name} instead. "
        f"This import will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )


# Legacy class aliases with deprecation warnings
class MarketDataLoader(ProviderBasedLoader):
    """Legacy alias for ProviderBasedLoader. Deprecated."""
    
    def __init__(self, *args, **kwargs):
        _deprecated_import_warning("MarketDataLoader", "ProviderBasedLoader")
        super().__init__(*args, **kwargs)


# Export the new classes and legacy aliases
__all__ = [
    'ProviderBasedLoader',
    'AlpacaMarketDataLoader', 
    'EODHDMarketDataLoader',
    'PolygonMarketDataLoader',
    'FREDMarketDataLoader',
    'BrokerDataLoader',
    'MarketDataLoader',  # Legacy alias
]
