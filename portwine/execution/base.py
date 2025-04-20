"""
Execution module for the portwine framework.

This module provides the base classes and interfaces for execution modules,
which connect strategy implementations from the backtester to live trading.
"""

from __future__ import annotations

import abc
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Tuple, TypedDict, Union, Any
from functools import wraps

import pandas as pd
import numpy as np
import math

from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.execution.brokers.base import BrokerBase, Position, Order, Account, OrderExecutionError as BrokerOrderExecutionError
from portwine.execution.execution_utils import (
    create_bar_dict, 
    calculate_position_changes, 
    generate_orders
)

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution-related errors."""
    pass


class OrderExecutionError(ExecutionError):
    """Exception raised when order execution fails."""
    pass


class DataFetchError(ExecutionError):
    """Exception raised when data fetching fails."""
    pass


class PortfolioExceededError(ExecutionError):
    """Raised when current portfolio weights exceed 100% of portfolio value."""
    pass


class ExecutionBase(abc.ABC):
    """
    Base class for execution implementations.
    
    An execution implementation is responsible for:
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
        timezone: Optional[datetime.tzinfo] = None,
    ):
        """
        Initialize the execution instance.
        
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
        timezone : Optional[datetime.tzinfo], default None
            Timezone for timestamp conversion
        """
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.broker = broker
        self.alternative_data_loader = alternative_data_loader
        self.min_change_pct = min_change_pct
        self.min_order_value = min_order_value
        
        # Store timezone (tzinfo); default to system local timezone
        self.timezone = timezone if timezone is not None else datetime.now().astimezone().tzinfo
        # Initialize ticker list from strategy
        self.tickers = strategy.tickers
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.tickers)} tickers")
    
    @staticmethod
    def _split_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split full ticker list into regular and alternative tickers.
        Regular tickers have no ':'; alternative contain ':'
        """
        reg: List[str] = []
        alt: List[str] = []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def fetch_latest_data(self, timestamp: Optional[float] = None) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Fetch latest market data for the tickers in the strategy.
        
        Parameters
        ----------
        timestamp : Optional[float]
            UNIX timestamp to get data for, or current time if None
            
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
            # Convert UNIX timestamp to timezone-aware datetime, default to now
            if timestamp is None:
                dt = datetime.now(tz=self.timezone)
            else:
                # timestamp is seconds since epoch
                dt = datetime.fromtimestamp(timestamp, tz=self.timezone)
            # Strip tzinfo for loader to match tz-naive indices
            loader_dt = dt.replace(tzinfo=None)
            # Split tickers into market vs alternative
            reg_tkrs, alt_tkrs = self._split_tickers(self.tickers)
            # Fetch market data only for regular tickers
            data = self.market_data_loader.next(reg_tkrs, loader_dt)
            # Fetch alternative data only for alternative tickers
            if self.alternative_data_loader is not None and alt_tkrs:
                alt_data = self.alternative_data_loader.next(alt_tkrs, loader_dt)
                # Merge alternative entries into result
                data.update(alt_data)
            
            return data
        except Exception as e:
            logger.exception(f"Error fetching latest data: {e}")
            raise DataFetchError(f"Failed to fetch latest data: {e}")
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current closing prices for the specified symbols by querying only market data.

        This method bypasses alternative data and directly uses market_data_loader.next
        with a timezone-naive datetime matching the execution timezone.
        """
        # Build current datetime in execution timezone
        dt = datetime.now(tz=self.timezone)
        # Align to loader timezone (no-op if same) and strip tzinfo
        loader_dt = dt.astimezone(self.timezone).replace(tzinfo=None)
        # Fetch only market data for given symbols
        data = self.market_data_loader.next(symbols, loader_dt)
        prices: Dict[str, float] = {}
        for symbol, bar in data.items():
            if bar is None:
                continue
            price = bar.get('close')
            if price is not None:
                prices[symbol] = price
        return prices
    
    def _get_current_positions(self) -> Tuple[Dict[str, float], float]:
        """
        Get current positions from broker account info.
        
        Returns
        -------
        Tuple[Dict[str, float], float]
            Current position quantities for each ticker and the portfolio value
        """
        positions = self.broker.get_positions()
        account = self.broker.get_account()
        
        current_positions = {symbol: position.quantity for symbol, position in positions.items()}
        portfolio_value = account.equity
        
        return current_positions, portfolio_value

    def _calculate_target_positions(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        fractional: bool = True,
    ) -> Dict[str, float]:
        """
        Convert target weights to absolute position sizes.
        
        Optionally prevent fractional shares by rounding down when `fractional=False`.
        
        Parameters
        ----------
        target_weights : Dict[str, float]
            Target allocation weights for each ticker
        portfolio_value : float
            Current portfolio value
        prices : Dict[str, float]
            Current prices for each ticker
        fractional : bool, default True
            If False, positions are floored to the nearest integer
        
        Returns
        -------
        Dict[str, float]
            Target position quantities for each ticker
        """
        target_positions = {}
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_value = weight * portfolio_value
            raw_qty = target_value / price
            if fractional:
                qty = raw_qty
            else:
                qty = math.floor(raw_qty)
            target_positions[symbol] = qty
        return target_positions

    def _calculate_current_weights(
        self,
        positions: List[Tuple[str, float]],
        portfolio_value: float,
        prices: Dict[str, float],
        raises: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate current weights of positions based on prices and portfolio value.

        Args:
            positions: List of (ticker, quantity) tuples.
            portfolio_value: Total portfolio value.
            prices: Mapping of ticker to current price.
            raises: If True, raise PortfolioExceededError when total weights > 1.

        Returns:
            Dict[ticker, weight] mapping.

        Raises:
            PortfolioExceededError: If raises=True and sum(weights) > 1.
        """
        # Map positions
        pos_map: Dict[str, float] = {t: q for t, q in positions}
        weights: Dict[str, float] = {}
        total: float = 0.0
        for ticker, price in prices.items():
            qty = pos_map.get(ticker, 0.0)
            w = (price * qty) / portfolio_value if portfolio_value else 0.0
            weights[ticker] = w
            total += w
        if raises and total > 1.0:
            raise PortfolioExceededError(
                f"Total weights {total:.2f} exceed 1.0"
            )
        return weights

    def _execute_orders(self, orders: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Execute a list of orders through the broker.
        
        Parameters
        ----------
        orders : List[Dict[str, Any]]
            List of order specifications
            
        Returns
        -------
        Dict[str, bool]
            Execution results by symbol (True for success, False for failure)
            
        Raises
        ------
        OrderExecutionError
            If any order is missing required fields like quantity
        """
        results = {}
        for order in orders:
            symbol = order['symbol']
            qty = order.get('qty')  # Use get() to safely handle missing qty
            
            if qty is None:
                raise OrderExecutionError(f"Missing quantity for order: {symbol}")
            
            logger.info(f"Executing order: {symbol} {qty} shares")
            
            try:
                # Determine the side (BUY for positive qty, SELL for negative)
                side = OrderSide.BUY if qty > 0 else OrderSide.SELL
                
                # Execute the order using the broker
                executed_order = self.broker.submit_order(
                    symbol=symbol,
                    quantity=abs(qty),
                )
                results[symbol] = True
                logger.info(f"Successfully executed order for {symbol}")
            except BrokerOrderExecutionError as e:
                logger.error(f"Failed to execute order for {symbol}: {e}")
                results[symbol] = False
            except Exception as e:
                logger.exception(f"Error executing order for {symbol}: {e}")
                results[symbol] = False
        
        return results
    
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
            Dictionary mapping symbols to trade execution success/failure
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
            target_weights = self.strategy.step(timestamp, latest_data)
            logger.info(f"Strategy generated target weights: {target_weights}")
            
            # Get current positions and portfolio value
            current_positions, portfolio_value = self._get_current_positions()
            
            # Get current prices
            prices = self.get_current_prices(self.tickers)
            
            # Calculate target positions
            target_positions = self._calculate_target_positions(target_weights, portfolio_value, prices)
            
            # Calculate changes needed
            position_changes = calculate_position_changes(target_positions, current_positions)
            
            if not position_changes:
                logger.info("No position changes required")
                return {}
                
            logger.info(f"Position changes: {position_changes}")
            
            # Generate orders
            orders = generate_orders(position_changes)
            
            # Execute orders
            return self._execute_orders(orders)
            
        except Exception as e:
            logger.exception(f"Error in step execution: {e}")
            return {} 