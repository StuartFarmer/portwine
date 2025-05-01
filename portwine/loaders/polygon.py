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
    timezone : str
        Timezone for the data. Default is "America/New_York".
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = "data",
        timezone: str = "America/New_York"
    ):
        """Initialize Polygon market data loader."""
        super().__init__()
        
        # Use environment variable if not provided
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            logger.warning("Polygon API key not provided. Will raise error if fetching historical data.")
            raise ValueError(
                "Polygon API key must be provided either as argument or POLYGON_API_KEY env var.")
        
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
        
        # Cache for last valid data used in ffill
        self._last_valid_data: Optional[Dict] = None
        
        # Timezone
        self.timezone = timezone

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

    def _validate_and_convert_dates(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Validate and convert date inputs to proper format.
        
        Parameters
        ----------
        from_date : str, optional
            Start date in YYYY-MM-DD format or millisecond timestamp
        to_date : str, optional
            End date in YYYY-MM-DD format or millisecond timestamp
            
        Returns
        -------
        tuple[str, str]
            Tuple of (from_date, to_date) in proper format for API
            
        Raises
        ------
        ValueError
            If dates are malformed or if to_date is before from_date
        """
        # Set default dates if not provided
        today = datetime.now(pytz.UTC)
        if from_date is None:
            from_date = str(int(datetime.timestamp(today - timedelta(days=365 * 2)) * 1000))
        if to_date is None:
            to_date = str(int(datetime.timestamp(today) * 1000))
            
        # If date is in YYYY-MM-DD format, convert to milliseconds
        if "-" in str(from_date):
            try:
                dt = datetime.strptime(from_date, "%Y-%m-%d")
                dt = dt.replace(tzinfo=pytz.UTC)  # Make timezone-aware as UTC
                from_date = str(int(dt.timestamp() * 1000))
            except ValueError:
                raise ValueError(f"from_date must be in YYYY-MM-DD format or millisecond timestamp. Got {from_date}")
                
        if "-" in str(to_date):
            try:
                dt = datetime.strptime(to_date, "%Y-%m-%d")
                dt = dt.replace(tzinfo=pytz.UTC)  # Make timezone-aware as UTC
                to_date = str(int(dt.timestamp() * 1000))
            except ValueError:
                raise ValueError(f"to_date must be in YYYY-MM-DD format or millisecond timestamp. Got {to_date}")
        
        # Validate millisecond timestamps
        try:
            from_ms = int(from_date)
            to_ms = int(to_date)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid millisecond timestamps. Got from_date={from_date}, to_date={to_date}")
            
        # Check order
        if to_ms < from_ms:
            raise ValueError(f"to_date ({to_date}) must be after from_date ({from_date})")
            
        return from_date, to_date

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
            Start date in YYYY-MM-DD format or millisecond timestamp. If None, defaults to 2 years ago.
        to_date : str, optional
            End date in YYYY-MM-DD format or millisecond timestamp. If None, defaults to today.
            
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
            # Validate and convert dates
            from_date, to_date = self._validate_and_convert_dates(from_date, to_date)
            
            # Initialize list to store all results
            all_results = []
            
            # Construct initial API endpoint
            endpoint = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
            
            # Fetch data with pagination
            while endpoint:
                # Make API request
                response_data = self._api_get(endpoint, params={"adjusted": "true", "sort": "asc"})
                
                # Process response
                if response_data and response_data.get("results"):
                    all_results.extend(response_data["results"])
                    
                    # Get next URL for pagination
                    endpoint = response_data.get("next_url")
                else:
                    break
            
            if not all_results:
                logger.warning(f"No data returned for {ticker} from {from_date} to {to_date}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(all_results)
            
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(self.timezone)
            df.set_index("timestamp", inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Cache the data
            self._data_cache[ticker] = df
            
            # Save to disk
            self._save_to_disk(ticker, df)
            
            logger.info(f"Successfully fetched historical data for {ticker} from {from_date} to {to_date}")
            return df

        except ValueError:
            # Re-raise ValueError exceptions (invalid dates)
            raise
        except Exception as e:
            # Log and return None for all other exceptions
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def _fetch_partial_day_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch current day's partial data from Polygon API.
        """
        try:
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            now_ms = int(now.timestamp() * 1000)
            from_ms = now_ms - (24 * 60 * 60 * 1000)  # 24 hours ago
            # Use path parameters for from and to
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute/{from_ms}/{now_ms}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000
            }
            response = self._api_get(url, params)
            if 'results' in response:
                today_bars = [
                    bar for bar in response['results']
                    if est.localize(datetime.fromtimestamp(bar['t'] / 1000)).hour >= 9
                    and est.localize(datetime.fromtimestamp(bar['t'] / 1000)).hour < 16
                ]
                if not today_bars:
                    return None
                first_bar = today_bars[0]
                today_open = first_bar['o']
                high = max(bar['h'] for bar in today_bars)
                low = min(bar['l'] for bar in today_bars)
                close = today_bars[-1]['c']
                volume = sum(bar['v'] for bar in today_bars)
                return {
                    "open": float(today_open),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume)
                }
        except Exception as e:
            logger.error(f"Error fetching partial day data for {ticker}: {e}")
            if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 404:
                return None
            return None
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
            df = self._data_cache[ticker]
        else:
            df = self._load_from_disk(ticker)
            if df is not None:
                self._data_cache[ticker] = df
        if df is not None:
            # Ensure index is timezone-aware and matches self.timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone)
            elif str(df.index.tz) != str(pytz.timezone(self.timezone)):
                df.index = df.index.tz_convert(self.timezone)
        return df

    def next(self, tickers: List[str], timestamp: pd.Timestamp, ffill: bool = False) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.
        For current day, returns partial day data.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to get data for
        timestamp : pd.Timestamp or datetime
            Timestamp to get data for
        ffill : bool, optional
            If True, when a ticker has no data, use the last non-None ticker's data.
            If False, return None for tickers with no data.
            Default is False.
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping ticker symbols to their OHLCV data or None
        """
        # Convert timestamp to pandas Timestamp if needed
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
            
        # Ensure timestamp is timezone-aware and in correct timezone
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(self.timezone)
        else:
            timestamp = timestamp.tz_convert(self.timezone)
            
        result = {}
        now = pd.Timestamp.now(tz=self.timezone)
        
        if timestamp.date() == now.date():
            for ticker in tickers:
                bar_data = self._fetch_partial_day_data(ticker)
                if bar_data:
                    result[ticker] = bar_data
                    if ffill:
                        self._last_valid_data = bar_data
                else:
                    if ffill and self._last_valid_data is not None:
                        result[ticker] = self._last_valid_data
                    else:
                        result[ticker] = None
        else:
            for ticker in tickers:
                df = self.load_ticker(ticker)
                if df is None:
                    result[ticker] = self._last_valid_data if ffill else None
                    continue
                    
                # Ensure DataFrame index is timezone-aware and matches timestamp timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize(self.timezone)
                elif str(df.index.tz) != str(timestamp.tz):
                    df.index = df.index.tz_convert(self.timezone)
                    
                bar = self._get_bar_at_or_before(df, timestamp)
                if bar is None:
                    result[ticker] = self._last_valid_data if ffill else None
                    continue
                    
                result[ticker] = {
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar["volume"]),
                }
                if ffill:
                    self._last_valid_data = result[ticker]
        return result

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()