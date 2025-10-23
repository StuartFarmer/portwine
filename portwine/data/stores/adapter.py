from portwine.data.stores.base import DataStore
from datetime import datetime
from typing import Union, OrderedDict
from collections import OrderedDict as OrderedDictType

class MarketDataLoaderAdapter(DataStore):
    """
    Adapter class that makes MarketDataLoader compatible with the DataStore interface.
    
    This allows MarketDataLoader to be used with NoisyDataStore and other DataStore-based
    components.
    """
    
    def __init__(self, market_data_loader):
        self.loader = market_data_loader
        self._current_timestamp = None
    
    def add(self, identifier: str, data: dict):
        """Not implemented - MarketDataLoader is read-only"""
        # Instead of raising an error, we'll silently ignore add operations
        # This allows NoisyDataStore to work with our adapter
        pass
    
    def get(self, identifier: str, dt: datetime) -> Union[dict, None]:
        """Get data for a single ticker at a specific timestamp"""
        if self._current_timestamp is None:
            self._current_timestamp = dt
        
        # Use the loader's next method to get data
        data = self.loader.next([identifier], dt)
        return data.get(identifier)
    
    def get_all(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        """Get all data for a ticker in a date range"""
        # Fetch the data for the ticker
        ticker_data = self.loader.fetch_data([identifier])
        if identifier not in ticker_data:
            return None
        
        df = ticker_data[identifier]
        if df.empty:
            return None
        
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if df.empty:
            return None
        
        # Convert to OrderedDict format
        result = OrderedDict()
        for ts, row in df.iterrows():
            result[ts] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
        
        return result
    
    def get_latest(self, identifier: str) -> Union[dict, None]:
        """Get the latest data point for a ticker"""
        ticker_data = self.loader.fetch_data([identifier])
        if identifier not in ticker_data or ticker_data[identifier].empty:
            return None
        
        df = ticker_data[identifier]
        latest_row = df.iloc[-1]
        
        return {
            'open': float(latest_row['open']),
            'high': float(latest_row['high']),
            'low': float(latest_row['low']),
            'close': float(latest_row['close']),
            'volume': float(latest_row['volume'])
        }
    
    def latest(self, identifier: str) -> Union[datetime, None]:
        """Get the latest date for a ticker"""
        ticker_data = self.loader.fetch_data([identifier])
        if identifier not in ticker_data or ticker_data[identifier].empty:
            return None

        return ticker_data[identifier].index[-1]

    def earliest(self, identifier: str) -> Union[datetime, None]:
        """Get the earliest date for a ticker"""
        ticker_data = self.loader.fetch_data([identifier])
        if identifier not in ticker_data or ticker_data[identifier].empty:
            return None

        return ticker_data[identifier].index[0]

    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        """Check if data exists for a ticker in a date range"""
        ticker_data = self.loader.fetch_data([identifier])
        if identifier not in ticker_data or ticker_data[identifier].empty:
            return False
        
        df = ticker_data[identifier]
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return not df.empty
    
    def identifiers(self):
        """
        Get all available ticker identifiers.

        Returns identifiers from the underlying loader if available,
        otherwise returns an empty list.
        """
        if hasattr(self.loader, 'identifiers') and callable(getattr(self.loader, 'identifiers')):
            return self.loader.identifiers()
        return []

