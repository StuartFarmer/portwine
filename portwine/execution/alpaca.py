"""
Alpaca execution implementation for the portwine framework.

This module provides an execution implementation for Alpaca Markets,
allowing live trading using the Alpaca API via direct REST calls.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests

import pandas as pd

from portwine.execution import AccountInfo, ExecutionBase, Order, OrderExecutionError, Position
from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase

# Configure logging
logger = logging.getLogger(__name__)

# API URLs
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"


class AlpacaExecution(ExecutionBase):
    """
    Alpaca execution implementation for live trading via direct REST calls.
    
    This implementation connects to the Alpaca API to execute trades and
    fetch account information using direct HTTP requests rather than an SDK.
    
    Parameters
    ----------
    strategy : StrategyBase
        Strategy implementation to use
    market_data_loader : MarketDataLoader
        Market data loader for price data
    alternative_data_loader : MarketDataLoader, optional
        Alternative data loader for other data sources
    api_key : str, optional
        Alpaca API key. If not provided, attempts to read from ALPACA_API_KEY env var.
    api_secret : str, optional
        Alpaca API secret. If not provided, attempts to read from ALPACA_API_SECRET env var.
    paper_trading : bool, default True
        Whether to use paper trading mode (sandbox)
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_data_loader: MarketDataLoader,
        alternative_data_loader: Optional[MarketDataLoader] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """Initialize Alpaca execution."""
        super().__init__(strategy, market_data_loader, alternative_data_loader)
        
        # Use environment variables if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not provided. "
                "Either pass as parameters or set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
        
        # Set up API URLs
        self.base_url = ALPACA_PAPER_URL if paper_trading else ALPACA_LIVE_URL
        self.paper_trading = paper_trading
        
        # Set up API headers
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        
        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Verify account
        account = self._get_account()
        logger.info(f"Connected to Alpaca account: {account['id']}")
        logger.info(f"Account status: {account['status']}")
        
        # Track executed orders
        self.executed_orders = []
        
        # Cache for available assets
        self._available_assets = {}
        self._update_available_assets()
    
    def _api_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a request to the Alpaca API.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, DELETE, etc.)
        endpoint : str
            API endpoint (starting with /)
        params : dict, optional
            Query parameters for GET requests
        data : dict, optional
            JSON data for POST, PUT, PATCH requests
            
        Returns
        -------
        dict
            JSON response data
            
        Raises
        ------
        Exception
            If API request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, params=params)
            elif method.upper() == "PATCH":
                response = self.session.patch(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url}")
            logger.error(f"Error: {str(e)}")
            if "response" in locals() and response is not None:
                logger.error(f"Response: {response.text}")
            raise
    
    def _get_account(self) -> Dict[str, Any]:
        """Get account information"""
        return self._api_request("GET", "/v2/account")
    
    def _update_available_assets(self) -> None:
        """Update the cache of available assets."""
        try:
            # Get all active US stocks
            params = {"status": "active", "asset_class": "us_equity"}
            assets = self._api_request("GET", "/v2/assets", params=params)
            
            # Update cache
            self._available_assets = {asset["symbol"]: asset for asset in assets}
            
            logger.info(f"Updated asset cache: {len(self._available_assets)} assets available")
        except Exception as e:
            logger.error(f"Error updating asset cache: {e}")
    
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.
        
        Returns
        -------
        AccountInfo
            Current account information
        """
        try:
            # Get account information
            account = self._get_account()
            
            # Get positions
            alpaca_positions = self._api_request("GET", "/v2/positions")
            
            # Convert to Position objects
            positions = {}
            for pos in alpaca_positions:
                positions[pos["symbol"]] = {
                    "symbol": pos["symbol"],
                    "qty": float(pos["qty"]),
                    "market_value": float(pos["market_value"]),
                    "avg_entry_price": float(pos["avg_entry_price"]),
                    "unrealized_pl": float(pos["unrealized_pl"]),
                }
            
            return {
                "cash": float(account["cash"]),
                "portfolio_value": float(account["portfolio_value"]),
                "positions": positions,
            }
        
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    def execute_order(self, order: Order) -> bool:
        """
        Execute an order with Alpaca.
        
        Parameters
        ----------
        order : Order
            Order to execute
            
        Returns
        -------
        bool
            True if order was executed successfully, False otherwise
            
        Raises
        ------
        OrderExecutionError
            If order execution fails
        """
        symbol = order["symbol"]
        qty = order["qty"]
        
        # Skip 0 quantity orders
        if qty == 0:
            return True
        
        # Determine order side
        side = "buy" if qty > 0 else "sell"
        
        try:
            # Check if asset is tradable
            if symbol not in self._available_assets:
                # Refresh assets and check again
                self._update_available_assets()
                if symbol not in self._available_assets:
                    raise OrderExecutionError(f"Asset {symbol} is not tradable on Alpaca")
            
            # Create market order
            order_data = {
                "symbol": symbol,
                "qty": abs(qty),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }
            
            # Submit order
            alpaca_order = self._api_request("POST", "/v2/orders", data=order_data)
            
            # Track order
            self.executed_orders.append({
                "symbol": symbol,
                "qty": qty,
                "order_type": "market",
                "time_in_force": "day",
                "limit_price": None,
                "order_id": alpaca_order["id"],
                "timestamp": datetime.now(),
            })
            
            logger.info(f"Order submitted: {alpaca_order['id']} - {side} {abs(qty)} {symbol}")
            return True
        
        except Exception as e:
            logger.error(f"Error executing order for {symbol}: {e}")
            raise OrderExecutionError(f"Failed to execute order for {symbol}: {str(e)}")
    
    def check_market_status(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns
        -------
        bool
            True if the market is open, False otherwise
        """
        try:
            clock = self._api_request("GET", "/v2/clock")
            return clock["is_open"]
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the status of an order.
        
        Parameters
        ----------
        order_id : str
            Order ID to check
            
        Returns
        -------
        str or None
            Order status if found, None otherwise
        """
        try:
            order = self._api_request("GET", f"/v2/orders/{order_id}")
            return order["status"]
        except Exception as e:
            logger.error(f"Error checking order status for {order_id}: {e}")
            return None
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self._api_request("DELETE", "/v2/orders")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self._api_request("DELETE", "/v2/positions", params={"cancel_orders": "true"})
            return True
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return False
    
    def step(self, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, bool]:
        """
        Execute a single step of the execution process.
        
        Parameters
        ----------
        timestamp : pd.Timestamp, optional
            Timestamp to execute step for, defaults to now
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping symbols to execution success
        """
        timestamp = timestamp or pd.Timestamp.now()
        self.last_step_time = timestamp
        
        # Check if market is open
        if not self.check_market_status():
            logger.warning("Market is closed, skipping execution step")
            return {}
        
        try:
            # Run the base step implementation
            results = super().step(timestamp)
            
            # Log summary
            account_info = self.get_account_info()
            logger.info(f"Step completed: {timestamp}")
            logger.info(f"Portfolio value: ${account_info['portfolio_value']:.2f}")
            logger.info(f"Cash: ${account_info['cash']:.2f}")
            logger.info(f"Positions: {len(account_info['positions'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in execution step: {e}")
            raise
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close() 