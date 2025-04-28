"""
Polygon market data loader for the portwine framework.

This module provides a MarketDataLoader implementation for fetching data
from the Polygon.io API, supporting both historical daily data and current
partial day data.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests
import pandas as pd
import pytz

from portwine.loaders.base import MarketDataLoader

# Configure logging
logger = logging.getLogger(__name__)

# API URLs
POLYGON_BASE_URL = "https://api.polygon.io"


class PolygonMarketDataLoader(MarketDataLoader):
    """
    Market data loader for Polygon.io API.
    
    This loader fetches historical daily data and current partial day data
    from Polygon.io API using direct REST calls.
    
    Parameters
    ----------
    api_key : str, optional
        Polygon API key. If not provided, attempts to read from POLYGON_API_KEY env var.
    data_dir : str
        Directory where historical data files are stored.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = "data",
    ):
        """Initialize Polygon market data loader."""
        super().__init__()
        
        # Use environment variable if not provided
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            logger.warning("Polygon API key not provided. Will raise error if fetching historical data.")
            raise ValueError(
                "Polygon API key not provided. "
                "Either pass as parameter or set POLYGON_API_KEY environment variable."
            )
        
        # Base URL for API requests
        self.base_url = POLYGON_BASE_URL
        
        # Create requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Data directory
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # In-memory cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Latest data cache for partial day data
        self._latest_data_cache: Dict[str, Dict] = {}
        self._latest_data_timestamp = None

    def _api_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Helper method to make authenticated GET requests to Polygon API
        
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
        if url is None:
            raise ValueError("URL cannot be None")
            
        response = None
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            raise

    def _get_data_path(self, ticker: str) -> str:
        """
        Get path to data file for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        str
            Path to data file
            
        Raises
        ------
        ValueError
            If ticker contains invalid characters for a filename
        """
        # Check for invalid characters in ticker
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in ticker for char in invalid_chars):
            raise ValueError(f"Ticker {ticker} contains invalid characters for a filename")
            
        return os.path.join(self.data_dir, f"{ticker}.parquet")
    
    def _load_from_disk(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from disk if available.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if data exists on disk, None otherwise
        """
        data_path = self._get_data_path(ticker)
        if os.path.exists(data_path):
            try:
                return pd.read_parquet(data_path)
            except Exception as e:
                logger.warning(f"Error loading data for {ticker}: {e}")
        else:
            logger.warning(f"Error loading data for {ticker}: File not found")
        
        return None
    
    def _save_to_disk(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save data to disk.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pd.DataFrame
            Data to save
        """
        data_path = self._get_data_path(ticker)
        try:
            df.to_parquet(data_path)
        except Exception as e:
            logger.warning(f"Error saving data for {ticker}: {e}")

    def fetch_historical_data(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a given ticker and date range from Polygon.io API.
        
        Parameters
        ----------
        ticker : str
            The stock ticker symbol to fetch data for
        from_date : str, optional
            Start date in YYYY-MM-DD format. If None, defaults to 2 years ago.
        to_date : str, optional
            End date in YYYY-MM-DD format. If None, defaults to today.
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data if successful, None if error occurs
            
        Raises
        ------
        ValueError
            If dates are malformed or if to_date is before from_date
            If API key is not provided
        """
        try:
            # Set default dates if not provided
            today = datetime.now()
            if from_date is None:
                from_date = (today - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
            if to_date is None:
                to_date = today.strftime("%Y-%m-%d")

            # Validate date formats and order
            try:
                from_date_obj = datetime.strptime(from_date, "%Y-%m-%d")
                to_date_obj = datetime.strptime(to_date, "%Y-%m-%d")
            except ValueError as e:
                logger.error(f"Invalid date format: {str(e)}")
                raise ValueError(f"Dates must be in YYYY-MM-DD format. Got from_date={from_date}, to_date={to_date}")

            if to_date_obj < from_date_obj:
                error_msg = f"to_date ({to_date}) must be after from_date ({from_date})"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Construct API endpoint
            endpoint = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
            
            # Make API request
            response_data = self._api_get(endpoint, params={"adjusted": "true", "sort": "asc"})

            # Process response
            if response_data and response_data.get("results"):
                # Convert to DataFrame
                df = pd.DataFrame(response_data["results"])
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    "v": "volume",
                    "o": "open",
                    "c": "close",
                    "h": "high",
                    "l": "low",
                    "t": "timestamp"
                })
                
                # Convert timestamp from milliseconds to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                # Cache the data
                self._data_cache[ticker] = df
                
                # Save to disk
                self._save_to_disk(ticker, df)
                
                logger.info(f"Successfully fetched historical data for {ticker} from {from_date} to {to_date}")
                return df
            else:
                logger.warning(f"No data returned for {ticker} from {from_date} to {to_date}")
                return None

        except ValueError:
            # Re-raise ValueError exceptions (invalid dates)
            raise
        except Exception as e:
            # Log and return None for all other exceptions
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def _fetch_partial_day_data(self, ticker: str) -> Optional[Dict]:
        """Fetch current day's partial data from Polygon API."""
        try:
            # Get today's date in EST
            est = pytz.timezone('US/Eastern')
            today = datetime.now(est).date()
            
            # Format parameters for API request
            params = {
                "ticker": ticker,
                "timespan": "minute",
                "from": today.strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"),
                "limit": 50000,
                "adjusted": "true"
            }
            
            # Make API request for today's minute bars
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute"
            response = self._api_get(url, params)
            
            if 'results' in response:
                # Get today's open from first bar
                first_bar = response['results'][0]
                today_open = first_bar['o']
                
                # Calculate high, low, close from all bars
                high = max(bar['h'] for bar in response['results'])
                low = min(bar['l'] for bar in response['results'])
                close = response['results'][-1]['c']
                volume = sum(bar['v'] for bar in response['results'])
                
                return {
                    "open": float(today_open),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume)
                }
        
        except Exception as e:
            logger.error(f"Error fetching partial day data for {ticker}: {e}")
        
        return None

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker from memory cache or disk.
        This method only reads from cache/disk and does not fetch new data.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if data is not found
        """
        # Check in-memory cache first
        if ticker in self._data_cache:
            return self._data_cache[ticker]
        
        # If not in memory, try loading from disk
        df = self._load_from_disk(ticker)
        if df is not None:
            # Update in-memory cache
            self._data_cache[ticker] = df
        
        return df

    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.
        For current day, returns partial day data.
        """
        result = {}
        
        # If timestamp is today, get partial day data
        now = pd.Timestamp.now()
        if timestamp.date() == now.date():
            for ticker in tickers:
                bar_data = self._fetch_partial_day_data(ticker)
                if bar_data:
                    result[ticker] = bar_data
                else:
                    result[ticker] = None
        else:
            # Otherwise use historical data from cache/disk
            for ticker in tickers:
                df = self.load_ticker(ticker)
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