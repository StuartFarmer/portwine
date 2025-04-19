"""
Alpaca broker implementation.

This module provides a broker implementation that interfaces with the Alpaca trading API.
"""

import logging
from typing import Dict, List, Optional, Any

import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import APIError

from portwine.execution_complex.broker_base import BrokerBase

class AlpacaBroker(BrokerBase):
    """
    Broker implementation for Alpaca trading platform.
    
    This class provides an implementation of the BrokerBase interface for the
    Alpaca trading platform, allowing strategies to execute trades through Alpaca.
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str, paper: bool = True):
        """
        Initialize the Alpaca broker.
        
        Parameters
        ----------
        api_key : str
            Alpaca API key
        api_secret : str
            Alpaca API secret
        base_url : str
            Alpaca API base URL
        paper : bool, default True
            Whether to use paper trading (True) or live trading (False)
        """
        self.logger = logging.getLogger(__name__)
        self.paper = paper
        
        # Initialize Alpaca API client
        self.api = alpaca.REST(api_key, api_secret, base_url)
        self.logger.info(f"Initialized AlpacaBroker with paper={paper}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information from Alpaca.
        
        Returns
        -------
        Dict[str, Any]
            Account information including cash, portfolio value, and positions
        """
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'status': account.status
            }
        except APIError as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def execute_order(self, symbol: str, qty: float, order_type: str = "market") -> bool:
        """
        Execute a trade order through Alpaca.
        
        Parameters
        ----------
        symbol : str
            The ticker symbol of the asset to trade
        qty : float
            The quantity to trade (positive for buy, negative for sell)
        order_type : str, default "market"
            The type of order (market, limit, etc.)
            
        Returns
        -------
        bool
            True if order was executed successfully, False otherwise
        """
        try:
            side = "buy" if qty > 0 else "sell"
            abs_qty = abs(qty)
            
            self.logger.info(f"Executing {side} order for {abs_qty} shares of {symbol}")
            
            self.api.submit_order(
                symbol=symbol,
                qty=abs_qty,
                side=side,
                type=order_type,
                time_in_force="day"
            )
            return True
        except APIError as e:
            self.logger.error(f"Error executing order: {e}")
            return False
    
    def check_market_status(self) -> bool:
        """
        Check if the market is currently open through Alpaca API.
        
        Returns
        -------
        bool
            True if market is open, False otherwise
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except APIError as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """
        Get the status of a specific order.
        
        Parameters
        ----------
        order_id : str
            ID of the order to check
            
        Returns
        -------
        Optional[str]
            Status of the order, or None if the order doesn't exist
        """
        try:
            order = self.api.get_order(order_id)
            return order.status
        except APIError as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders through Alpaca API.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.api.cancel_all_orders()
            self.logger.info("All orders canceled")
            return True
        except APIError as e:
            self.logger.error(f"Error canceling orders: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions through Alpaca API.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.api.close_all_positions()
            self.logger.info("All positions closed")
            return True
        except APIError as e:
            self.logger.error(f"Error closing positions: {e}")
            return False
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from Alpaca API.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of current positions
        """
        try:
            positions = self.api.list_positions()
            result = []
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'current_price': float(position.current_price),
                    'side': position.side
                })
            return result
        except APIError as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def get_cash(self) -> float:
        """
        Get available cash in Alpaca account.
        
        Returns
        -------
        float
            Available cash for trading
        """
        try:
            account = self.api.get_account()
            return float(account.cash)
        except APIError as e:
            self.logger.error(f"Error getting cash: {e}")
            return 0.0
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value from Alpaca.
        
        Returns
        -------
        float
            Total value of the portfolio
        """
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except APIError as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return 0.0 