"""
Alpaca broker implementation.

This module provides a broker implementation that interfaces with the Alpaca trading API.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import APIError

from portwine.execution.brokers.base import BrokerBase, OrderSide, Order, Position, Account, OrderExecutionError


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
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.paper = paper
        
        # Initialize Alpaca API client
        self.api = alpaca.REST(api_key, api_secret, base_url)
        self.logger.info(f"Initialized AlpacaBroker with paper={paper}")
    
    def get_account(self) -> Account:
        """
        Get current account information from Alpaca.
        
        Returns
        -------
        Account
            Account object containing current account information
        """
        try:
            account_data = self.api.get_account()
            return Account(
                balance=float(account_data.cash),
                equity=float(account_data.equity),
                margin=float(account_data.margin_used) if hasattr(account_data, 'margin_used') else 0.0
            )
        except APIError as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions from Alpaca.
        
        Returns
        -------
        Dict[str, Position]
            Dictionary mapping symbol to Position objects
        """
        try:
            alpaca_positions = self.api.list_positions()
            positions = {}
            for pos in alpaca_positions:
                positions[pos.symbol] = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price)
                )
            return positions
        except APIError as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def execute_order(self, symbol: str, quantity: float, side: OrderSide) -> Order:
        """
        Execute a market order through Alpaca.
        
        Parameters
        ----------
        symbol : str
            The ticker symbol of the asset to trade
        quantity : float
            The quantity to trade
        side : OrderSide
            The order side (BUY or SELL)
            
        Returns
        -------
        Order
            Order object representing the executed order
            
        Raises
        ------
        OrderExecutionError
            If the order fails to execute
        """
        try:
            alpaca_side = "buy" if side == OrderSide.BUY else "sell"
            
            self.logger.info(f"Executing {alpaca_side} order for {quantity} shares of {symbol}")
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                type="market",
                time_in_force="day"
            )
            
            return Order(
                order_id=order.id,
                symbol=order.symbol,
                quantity=float(order.qty),
                side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                status=order.status,
                filled_quantity=float(order.filled_qty) if hasattr(order, 'filled_qty') else 0.0,
                average_price=float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') else 0.0,
                created_at=datetime.fromisoformat(order.created_at.replace('Z', '+00:00'))
            )
        except APIError as e:
            self.logger.error(f"Error executing order: {e}")
            raise OrderExecutionError(f"Failed to execute order: {e}")
    
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