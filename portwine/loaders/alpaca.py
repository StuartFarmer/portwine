"""
Alpaca market data loader for the portwine framework.

This module provides a MarketDataLoader implementation for fetching data
from the Alpaca Markets API, both for historical and real-time data.
Uses direct REST API calls instead of the Alpaca Python SDK.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests

import pandas as pd
import pytz

from portwine.loaders.base import MarketDataLoader

# Configure logging
logger = logging.getLogger(__name__)

# API URLs
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaMarketDataLoader(MarketDataLoader):
    """
    Market data loader for Alpaca Markets API.
    
    This loader fetches historical and real-time data from Alpaca Markets API
    using direct REST calls. It supports fetching OHLCV data for stocks and ETFs.
    
    Parameters
    ----------
    api_key : str, optional
        Alpaca API key. If not provided, attempts to read from ALPACA_API_KEY env var.
    api_secret : str, optional
        Alpaca API secret. If not provided, attempts to read from ALPACA_API_SECRET env var.
    start_date : Union[str, datetime], optional
        Start date for historical data, defaults to 2 years ago
    end_date : Union[str, datetime], optional
        End date for historical data, defaults to today
    cache_dir : str, optional
        Directory to cache data to. If not provided, data is not cached.
    paper_trading : bool, default True
        Whether to use paper trading mode (sandbox)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        cache_dir: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """Initialize Alpaca market data loader."""
        super().__init__()
        
        # Use environment variables if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not provided. "
                "Either pass as parameters or set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
        
        # Set up API URLs based on paper trading flag
        self.base_url = ALPACA_PAPER_URL if paper_trading else ALPACA_LIVE_URL
        self.data_url = ALPACA_DATA_URL
        
        # Auth headers for API requests
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        
        # Create requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Cache directory
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Date range for historical data
        today = datetime.now(pytz.UTC)
        if start_date is None:
            # Default to 2 years ago
            self.start_date = today - timedelta(days=365 * 2)
        else:
            self.start_date = pd.Timestamp(start_date).to_pydatetime()
            if self.start_date.tzinfo is None:
                self.start_date = pytz.UTC.localize(self.start_date)
        
        if end_date is None:
            self.end_date = today
        else:
            self.end_date = pd.Timestamp(end_date).to_pydatetime()
            if self.end_date.tzinfo is None:
                self.end_date = pytz.UTC.localize(self.end_date)
        
        # Latest data cache to avoid frequent API calls
        self._latest_data_cache: Dict[str, Dict] = {}
        self._latest_data_timestamp = None
    
    def _api_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Helper method to make authenticated GET requests to Alpaca API
        
        Parameters
        ----------
        url : str
            API endpoint URL (starting with /)
        params : Dict[str, Any], optional
            Query parameters for the request
            
        Returns
        -------
        Any
            JSON response data
            
        Raises
        ------
        Exception
            If API request fails
        """
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            raise
    
    def _get_cache_path(self, ticker: str) -> str:
        """
        Get path to cached data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        str
            Path to cached data
        """
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{ticker}.parquet")
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if data is cached, None otherwise
        """
        if not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Error loading cached data for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pd.DataFrame
            Data to cache
        """
        if not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(ticker)
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Error caching data for {ticker}: {e}")
    
    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for API requests"""
        return dt.isoformat()
    
    def _fetch_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Alpaca API.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            # Format parameters for API request
            params = {
                "symbols": ticker,
                "timeframe": "1Day",
                "start": self._format_datetime(self.start_date),
                "end": self._format_datetime(self.end_date),
                "limit": 10000,  # Maximum allowed by the API
                "adjustment": "raw"
            }
            
            # Make API request for bars
            url = f"{self.data_url}/v2/stocks/bars"
            response = self._api_get(url, params)
            
            # Extract data
            if 'bars' in response and ticker in response['bars']:
                bars = response['bars'][ticker]
                
                # Convert to dataframe
                df = pd.DataFrame(bars)
                
                # Convert timestamp string to datetime index
                df['t'] = pd.to_datetime(df['t'])
                df = df.set_index('t')
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                # Drop timezone for compatibility
                df.index = df.index.tz_localize(None)
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
        
        return None
    
    def _fetch_latest_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch latest data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        dict or None
            Latest OHLCV data or None if fetch fails
        """
        # Check if we have recent data in memory
        now = datetime.now()
        if (
            self._latest_data_timestamp 
            and (now - self._latest_data_timestamp).total_seconds() < 60
            and ticker in self._latest_data_cache
        ):
            return self._latest_data_cache[ticker]
        
        try:
            # Format parameters for API request
            params = {
                "symbols": ticker
            }
            
            # Make API request for latest bar
            url = f"{self.data_url}/v2/stocks/bars/latest"
            response = self._api_get(url, params)
            
            if 'bars' in response and ticker in response['bars']:
                bar = response['bars'][ticker]
                
                # Update cache
                self._latest_data_cache[ticker] = {
                    "open": float(bar['o']),
                    "high": float(bar['h']),
                    "low": float(bar['l']),
                    "close": float(bar['c']),
                    "volume": float(bar['v']),
                }
                self._latest_data_timestamp = now
                
                return self._latest_data_cache[ticker]
        
        except Exception as e:
            logger.error(f"Error fetching latest data for {ticker}: {e}")
        
        return None
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if load fails
        """
        # Check cache first
        df = self._load_from_cache(ticker)
        
        # Fetch from API if not cached or outdated
        if df is None or df.index.max() < self.end_date.replace(tzinfo=None):
            df_new = self._fetch_historical_data(ticker)
            
            if df_new is not None and not df_new.empty:
                if df is not None:
                    # Append new data
                    df = pd.concat([df[~df.index.isin(df_new.index)], df_new])
                    df = df.sort_index()
                else:
                    df = df_new
                
                # Save to cache
                self._save_to_cache(ticker, df)
        
        return df
    
    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        timestamp : pd.Timestamp
            Timestamp to get data for
            
        Returns
        -------
        Dict[str, dict]
            Dictionary mapping tickers to bar data
        """
        result = {}
        
        # If timestamp is close to now, get live data
        now = pd.Timestamp.now()
        if abs((now - timestamp).total_seconds()) < 86400:  # Within 24 hours
            for ticker in tickers:
                bar_data = self._fetch_latest_data(ticker)
                if bar_data:
                    result[ticker] = bar_data
                else:
                    result[ticker] = None
        else:
            # Otherwise use historical data
            for ticker in tickers:
                df = self.fetch_data([ticker]).get(ticker)
                if df is not None:
                    bar = self._get_bar_at_or_before(df, timestamp)
                    if bar is not None:
                        result[ticker] = {
                            "open": float(bar["open"]),
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "volume": float(bar["volume"]),
                        }
                    else:
                        result[ticker] = None
                else:
                    result[ticker] = None
        
        return result
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close() 