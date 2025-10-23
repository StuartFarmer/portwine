"""
NoisyDataStore - A DataStore wrapper that injects noise before returning data.

This wrapper can be used with any DataStore implementation to add
realistic noise to market data, scaled by rolling volatility.
"""

from datetime import datetime
from typing import Union, OrderedDict, Optional, Dict
from collections import OrderedDict as OrderedDictType
import pandas as pd
import numpy as np

from .base import DataStore


class NoisyDataStore(DataStore):
    """
    A DataStore wrapper that injects noise before returning data.
    
    This wrapper can be used with any DataStore implementation to add
    realistic noise to market data, with the magnitude of noise
    proportional to the local volatility (measured as rolling standard deviation
    of returns). This ensures the noise adapts to different market regimes.
    
    Parameters
    ----------
    base_store : DataStore
        The base data store to wrap
    noise_multiplier : float, optional
        Base multiplier for the noise magnitude (default: 1.0)
    volatility_window : int, optional
        Window size in days for rolling volatility calculation (default: 21)
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        base_store: DataStore,
        noise_multiplier: float = 1.0,
        volatility_window: int = 21,
        seed: Optional[int] = None
    ):
        self.base_store = base_store
        self.noise_multiplier = noise_multiplier
        self.volatility_window = volatility_window
        
        # Create a local random number generator for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _calculate_volatility_from_data(self, data: OrderedDict) -> float:
        """
        Calculate volatility from historical data.
        
        Parameters
        ----------
        data : OrderedDict
            Historical data points with datetime keys
            
        Returns
        -------
        float
            Calculated volatility value
        """
        if len(data) < 2:
            return 0.01
        
        # Extract close prices and calculate returns
        closes = [point['close'] for point in data.values() if 'close' in point]
        if len(closes) < 2:
            return 0.01
        
        returns = [(closes[i] / closes[i-1]) - 1 for i in range(1, len(closes))]
        volatility = np.std(returns)
        
        return max(volatility, 0.01) if np.isfinite(volatility) else 0.01
    
    def _inject_noise_to_point(self, data_point: Dict, volatility: float) -> Dict:
        """
        Add noise to a single data point.
        
        Parameters
        ----------
        data_point : Dict
            Original data point
        volatility : float
            Volatility scaling factor
            
        Returns
        -------
        Dict
            Data point with noise added
        """
        noisy_point = data_point.copy()
        
        # Generate noise scaled by volatility
        noise_scale = volatility * self.noise_multiplier
        
        # Add noise to OHLC values
        for key in ['open', 'high', 'low', 'close']:
            if key in noisy_point:
                # Scale noise by the price value to maintain proportional noise
                noise = self.rng.normal(0, noise_scale * noisy_point[key])
                noisy_point[key] = noisy_point[key] + noise
        
        # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
        if all(k in noisy_point for k in ['open', 'high', 'low', 'close']):
            noisy_point['high'] = max(noisy_point['open'], noisy_point['high'], 
                                    noisy_point['low'], noisy_point['close'])
            noisy_point['low'] = min(noisy_point['open'], noisy_point['low'], 
                                   noisy_point['high'], noisy_point['close'])
        
        return noisy_point
    
    def get(self, identifier: str, dt: datetime) -> Union[dict, None]:
        """
        Get noisy data for a single point in time.
        
        Parameters
        ----------
        identifier : str
            The identifier for the data
        dt : datetime
            The timestamp to retrieve data for
            
        Returns
        -------
        dict or None
            Noisy data point or None if not found
        """
        original_data = self.base_store.get(identifier, dt)
        if original_data is None:
            return None
        
        # Get historical data for volatility calculation
        # Use a reasonable lookback period
        start_date = dt - pd.Timedelta(days=self.volatility_window)
        historical_data = self.base_store.get_all(identifier, start_date, dt)
        
        # Calculate volatility
        if historical_data:
            volatility = self._calculate_volatility_from_data(historical_data)
        else:
            volatility = 0.01
        
        # Add noise and return
        return self._inject_noise_to_point(original_data, volatility)
    
    def get_all(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        """
        Get noisy data for a date range.
        
        Parameters
        ----------
        identifier : str
            The identifier for the data
        start_date : datetime
            Start date for the range
        end_date : datetime, optional
            End date for the range (defaults to latest available)
            
        Returns
        -------
        OrderedDict or None
            Noisy data for the date range or None if not found
        """
        original_data = self.base_store.get_all(identifier, start_date, end_date)
        if original_data is None:
            return None
        
        # Calculate volatility from the data itself
        volatility = self._calculate_volatility_from_data(original_data)
        
        # Add noise to each data point
        noisy_data = OrderedDict()
        for dt, point in original_data.items():
            noisy_data[dt] = self._inject_noise_to_point(point, volatility)
        
        return noisy_data
    
    # Delegate other methods to base store
    def add(self, identifier: str, data: dict):
        """Add data to the base store."""
        return self.base_store.add(identifier, data)
    
    def get_latest(self, identifier: str) -> Union[dict, None]:
        """Get the latest noisy data point."""
        latest_data = self.base_store.get_latest(identifier)
        if latest_data is None:
            return None
        
        # For latest data, we need to get some historical context for volatility
        # Get the latest timestamp and calculate volatility from recent data
        latest_dt = self.base_store.latest(identifier)
        if latest_dt is None:
            return latest_data
        
        start_date = latest_dt - pd.Timedelta(days=self.volatility_window)
        historical_data = self.base_store.get_all(identifier, start_date, latest_dt)
        
        if historical_data:
            volatility = self._calculate_volatility_from_data(historical_data)
        else:
            volatility = 0.01
        
        return self._inject_noise_to_point(latest_data, volatility)
    
    def latest(self, identifier: str) -> Union[datetime, None]:
        """Get the latest date from the base store."""
        return self.base_store.latest(identifier)

    def earliest(self, identifier: str) -> Union[datetime, None]:
        """Get the earliest date from the base store."""
        return self.base_store.earliest(identifier)

    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        """Check if data exists in the base store."""
        return self.base_store.exists(identifier, start_date, end_date)
    
    def identifiers(self):
        """Get all identifiers from the base store."""
        return self.base_store.identifiers()
