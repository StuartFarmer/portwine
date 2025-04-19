"""
Alpaca broker implementation.

This module provides a broker implementation for the Alpaca trading platform.
"""

import logging
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union

from portwine.execution_complex.broker import BrokerBase

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerBase):
    """
    Broker implementation for the Alpaca trading platform.
    
    This class provides an implementation of the BrokerBase interface
    for executing trades through the Alpaca API.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper_trading: bool = True,
        api_version: str = "v2"
    ):
        """
        Initialize the Alpaca broker.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper_trading: Whether to use paper trading (True) or live trading (False)
            api_version: API version to use
        """
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        self.api_version = api_version
        
        # Set up API endpoints
        self.base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        
        # Set up headers for API requests
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized AlpacaBroker with paper_trading={paper_trading}")
        
        # Verify account on initialization
        self._verify_account()
    
    def _verify_account(self):
        """Verify that the account is accessible with the provided credentials."""
        try:
            account_info = self.get_account_info()
            logger.info(f"Connected to Alpaca account: {account_info.get('id', 'unknown')}")
            logger.info(f"Account status: {account_info.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca account: {e}")
            raise ConnectionError(f"Could not connect to Alpaca: {e}")
    
    def _api_request(
        self,
        method: str,
        endpoint: str,
        base: str = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request to Alpaca.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            base: Base URL to use (defaults to self.base_url)
            params: Query parameters
            data: Request body data
            
        Returns:
            Response data as a dictionary
            
        Raises:
            Exception: If the request fails
        """
        base_url = base or self.base_url
        url = f"{base_url}/{self.api_version}/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            if response.text:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"API error: {error_data.get('message', str(e))}"
                except:
                    error_msg = f"API error: {e.response.text}"
            
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def check_market_status(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            response = self._api_request("GET", "clock")
            return response.get("is_open", False)
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            # Default to closed if we can't determine status
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information from Alpaca.
        
        Returns:
            Dict containing account information
        """
        try:
            account = self._api_request("GET", "account")
            positions = self.get_positions()
            
            # Convert positions to a dictionary by symbol
            positions_dict = {
                position["symbol"]: position for position in positions
            }
            
            return {
                "id": account.get("id", ""),
                "status": account.get("status", ""),
                "cash": float(account.get("cash", 0)),
                "portfolio_value": float(account.get("portfolio_value", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "positions": positions_dict
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from Alpaca.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions_raw = self._api_request("GET", "positions")
            positions = []
            
            for pos in positions_raw:
                positions.append({
                    "symbol": pos.get("symbol", ""),
                    "qty": float(pos.get("qty", 0)),
                    "market_value": float(pos.get("market_value", 0)),
                    "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                    "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                    "current_price": float(pos.get("current_price", 0)),
                    "side": pos.get("side", "")
                })
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def execute_order(
        self,
        symbol: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False
    ) -> bool:
        """
        Execute an order through Alpaca.
        
        Args:
            symbol: The ticker symbol
            qty: Number of shares (positive for buy, negative for sell)
            order_type: Type of order (market, limit, stop, stop_limit)
            limit_price: Price for limit orders
            stop_price: Price for stop orders
            time_in_force: Time in force (day, gtc, ioc, fok)
            extended_hours: Whether to allow trading during extended hours
            
        Returns:
            True if order was successfully submitted, False otherwise
        """
        try:
            # Determine side (buy or sell)
            side = "buy" if qty > 0 else "sell"
            abs_qty = abs(qty)
            
            # Prepare order data
            order_data = {
                "symbol": symbol,
                "qty": str(abs_qty),
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
                "extended_hours": extended_hours
            }
            
            # Add limit price if applicable
            if order_type in ["limit", "stop_limit"] and limit_price is not None:
                order_data["limit_price"] = str(limit_price)
                
            # Add stop price if applicable
            if order_type in ["stop", "stop_limit"] and stop_price is not None:
                order_data["stop_price"] = str(stop_price)
                
            # Submit order
            logger.info(f"Submitting order: {order_data}")
            response = self._api_request("POST", "orders", data=order_data)
            
            logger.info(f"Order submitted: {response.get('id')} - {response.get('status')}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order
            
        Returns:
            Status string or None if order not found
        """
        try:
            response = self._api_request("GET", f"orders/{order_id}")
            return response.get("status")
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._api_request("DELETE", "orders")
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._api_request("DELETE", "positions?cancel_orders=true")
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return False
    
    def get_cash(self) -> float:
        """
        Get available cash in the account.
        
        Returns:
            Available cash amount
        """
        try:
            account = self._api_request("GET", "account")
            return float(account.get("cash", 0))
        except Exception as e:
            logger.error(f"Error getting cash: {e}")
            return 0.0
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        try:
            account = self._api_request("GET", "account")
            return float(account.get("portfolio_value", 0))
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position information for a specific symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Position information or None if not held
        """
        try:
            response = self._api_request("GET", f"positions/{symbol}")
            return {
                "symbol": response.get("symbol", ""),
                "qty": float(response.get("qty", 0)),
                "market_value": float(response.get("market_value", 0)),
                "avg_entry_price": float(response.get("avg_entry_price", 0)),
                "unrealized_pl": float(response.get("unrealized_pl", 0)),
                "current_price": float(response.get("current_price", 0)),
                "side": response.get("side", "")
            }
        except Exception as e:
            # If position doesn't exist, Alpaca returns 404
            # which is normal behavior, so don't log as error
            if "404" in str(e):
                return None
                
            logger.error(f"Error getting position for {symbol}: {e}")
            return None 