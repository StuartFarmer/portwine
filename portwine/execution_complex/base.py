"""
Execution module for the portwine framework.

This module provides the base classes and interfaces for execution_complex modules,
which connect strategy implementations from the backtester to live trading.
"""

from __future__ import annotations

import abc
import logging
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Tuple, TypedDict, Union

import pandas as pd
import numpy as np

from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.execution_complex.broker import BrokerBase, Position, Order, AccountInfo
from portwine.execution_complex.execution_utils import (
    create_bar_dict, 
    calculate_position_changes, 
    generate_orders
)

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution_complex-related errors."""
    pass


class OrderExecutionError(ExecutionError):
    """Exception raised when order execution_complex fails."""
    pass


class DataFetchError(ExecutionError):
    """Exception raised when data fetching fails."""
    pass


class ExecutionBase(abc.ABC):
    """
    Base class for execution_complex implementations.
    
    An execution_complex implementation is responsible for:
    1. Fetching latest market data
    2. Passing data to strategy to get updated weights
    3. Calculating position changes needed
    4. Executing necessary trades using a broker
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_data_loader: MarketDataLoader,
        broker: BrokerBase,
        alternative_data_loader: Optional[MarketDataLoader] = None,
        min_change_pct: float = 0.01,
        min_order_value: float = 1.0,
    ):
        """
        Initialize the execution_complex instance.
        
        Parameters
        ----------
        strategy : StrategyBase
            The strategy implementation to use for generating trading signals
        market_data_loader : MarketDataLoader
            Market data loader for price data
        broker : BrokerBase
            Broker implementation for executing trades
        alternative_data_loader : Optional[MarketDataLoader]
            Additional data loader for alternative data
        min_change_pct : float, default 0.01
            Minimum change percentage required to trigger a trade
        min_order_value : float, default 1.0
            Minimum dollar value required for an order
        """
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.broker = broker
        self.alternative_data_loader = alternative_data_loader
        self.min_change_pct = min_change_pct
        self.min_order_value = min_order_value
        
        # Initialize ticker list from strategy
        self.tickers = strategy.tickers
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.tickers)} tickers")
    
    def fetch_latest_data(self, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Fetch latest market data for the tickers in the strategy.
        
        Parameters
        ----------
        timestamp : Optional[pd.Timestamp]
            Timestamp to get data for, or current time if None
            
        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            Dictionary of latest bar data for each ticker
        
        Raises
        ------
        DataFetchError
            If data cannot be fetched
        """
        try:
            # Use the provided timestamp or current time
            if timestamp is None:
                timestamp = pd.Timestamp.now(tz='UTC')
                
            # Get latest data from market data loader
            data = self.market_data_loader.next(self.tickers, timestamp)
            
            # Also fetch alternative data if available
            if self.alternative_data_loader is not None:
                alt_data = self.alternative_data_loader.next(self.tickers, timestamp)
                # Merge alternative data with market data
                for ticker, ticker_data in alt_data.items():
                    if ticker in data and data[ticker] is not None:
                        data[ticker].update(ticker_data)
            
            return data
        except Exception as e:
            logger.exception(f"Error fetching latest data: {e}")
            raise DataFetchError(f"Failed to fetch latest data: {e}")
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for the specified symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to get prices for
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping symbols to their current prices
        """
        data = self.fetch_latest_data()
        prices = {}
        
        for symbol in symbols:
            if symbol in data and data[symbol] is not None:
                price = data[symbol].get('close')
                if price is not None:
                    prices[symbol] = price
        
        return prices
    
    def step(self, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, bool]:
        """
        Execute a single step of the trading strategy.
        
        This method:
        1. Checks if market is open
        2. Fetches latest data
        3. Gets new signals from the strategy
        4. Calculates position changes
        5. Executes necessary trades
        
        Parameters
        ----------
        timestamp : Optional[pd.Timestamp]
            Timestamp to execute at, or current time if None
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping symbols to trade execution_complex success/failure
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz='UTC')
            
        logger.info(f"Executing step at {timestamp}")
        
        # Check if market is open
        is_open = self.broker.check_market_status()
        if not is_open:
            logger.warning("Market is closed, skipping execution")
            return {}
        
        try:
            # Fetch latest data
            latest_data = self.fetch_latest_data(timestamp)
            
            # Get target allocations from strategy
            self.strategy.step(timestamp, latest_data)
            target_weights = self.strategy.generate_signals()
            
            logger.info(f"Strategy generated target weights: {target_weights}")
            
            # Get current account information
            account_info = self.broker.get_account_info()
            
            # Calculate position changes needed
            current_positions = {symbol: position.get('qty', 0) 
                               for symbol, position in account_info.get('positions', {}).items()}
            
            # Convert weights to target position sizes
            portfolio_value = account_info.get('portfolio_value', 0)
            prices = self.get_current_prices(self.tickers)
            
            target_positions = {}
            for symbol, weight in target_weights.items():
                if symbol in prices and prices[symbol] > 0:
                    target_value = weight * portfolio_value
                    target_positions[symbol] = target_value / prices[symbol]
            
            # Calculate changes needed
            position_changes = calculate_position_changes(target_positions, current_positions)
            
            if not position_changes:
                logger.info("No position changes required")
                return {}
                
            logger.info(f"Position changes: {position_changes}")
            
            # Generate orders
            orders = generate_orders(position_changes)
            
            # Execute orders
            results = {}
            for order in orders:
                symbol = order['symbol']
                qty = order['qty']
                order_type = order.get('order_type', 'market')
                
                logger.info(f"Executing order: {symbol} {qty} shares")
                
                try:
                    # Execute the order using the broker
                    success = self.broker.execute_order(
                        symbol=symbol,
                        qty=qty,
                        order_type=order_type
                    )
                    results[symbol] = success
                    
                    if success:
                        logger.info(f"Successfully executed order for {symbol}")
                    else:
                        logger.error(f"Failed to execute order for {symbol}")
                except Exception as e:
                    logger.exception(f"Error executing order for {symbol}: {e}")
                    results[symbol] = False
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in step execution: {e}")
            return {} 