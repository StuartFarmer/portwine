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
    
    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Caches & returns all requested tickers.
        Maintains the same interface as the old MarketDataLoader.
        """
        fetched = {}
        for ticker in tickers:
            if ticker not in self._data_cache:
                df = self._load_ticker(ticker)
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
            else:
                result[ticker] = None
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
    Adapter for EODHD data that uses EODHDProvider internally.
    """
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    def _get_provider(self, ticker: str) -> Optional[DataProvider]:
        """Get EODHD provider for any ticker."""
        if 'eodhd' not in self._provider_cache:
            self._provider_cache['eodhd'] = EODHDProvider(self.api_key)
        return self._provider_cache['eodhd']


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
