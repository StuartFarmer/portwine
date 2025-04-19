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

from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase

# Configure logging
logger = logging.getLogger(__name__)


class Position(TypedDict):
    """Represents a position in a security."""
    symbol: str
    qty: float  # Quantity of shares/contracts
    market_value: float  # Current market value of position
    avg_entry_price: float  # Average entry price
    unrealized_pl: float  # Unrealized profit/loss


class Order(TypedDict):
    """Represents a trade order."""
    symbol: str
    qty: float  # Positive for buy, negative for sell
    order_type: str  # market, limit, etc.
    time_in_force: str  # day, gtc, etc.
    limit_price: Optional[float]  # For limit orders


class AccountInfo(TypedDict):
    """Represents account information."""
    cash: float  # Available cash
    portfolio_value: float  # Total portfolio value including cash
    positions: Dict[str, Position]  # Current positions


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
    4. Executing necessary trades
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_data_loader: MarketDataLoader,
        alternative_data_loader: Optional[MarketDataLoader] = None,
    ):
        """
        Initialize the execution_complex instance.
        
        Parameters
        ----------
        strategy : StrategyBase
            The strategy implementation to use for generating trading signals
        market_data_loader : MarketDataLoader
            Market data loader for price data
        alternative_data_loader : MarketDataLoader, optional
            Alternative data loader for other data sources
        """
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        self.last_step_time: Optional[pd.Timestamp] = None
    
    @abc.abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information including cash, portfolio value, and positions.
        
        Returns
        -------
        AccountInfo
            Dictionary with account information
        """
        pass
    
    @abc.abstractmethod
    def execute_order(self, order: Order) -> bool:
        """
        Execute a single order.
        
        Parameters
        ----------
        order : Order
            Order to execute
            
        Returns
        -------
        bool
            True if order was successfully submitted, False otherwise
        
        Raises
        ------
        OrderExecutionError
            If order execution_complex fails
        """
        pass
    
    def execute_orders(self, orders: List[Order]) -> Dict[str, bool]:
        """
        Execute multiple orders.
        
        Parameters
        ----------
        orders : List[Order]
            List of orders to execute
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping symbol to execution_complex success
        """
        results = {}
        for order in orders:
            try:
                success = self.execute_order(order)
                results[order["symbol"]] = success
            except OrderExecutionError as e:
                logger.error(f"Failed to execute order for {order['symbol']}: {e}")
                results[order["symbol"]] = False
        return results
    
    def fetch_latest_data(self, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, dict]:
        """
        Fetch latest data for all tickers in the strategy.
        
        Parameters
        ----------
        timestamp : pd.Timestamp, optional
            Timestamp to fetch data for, defaults to now
            
        Returns
        -------
        Dict[str, dict]
            Dictionary of ticker data in the format expected by strategy.step()
            
        Raises
        ------
        DataFetchError
            If data fetching fails
        """
        timestamp = timestamp or pd.Timestamp.now()
        
        # Split tickers into regular and alternative (if format includes ":")
        reg_tickers = [t for t in self.strategy.tickers if ":" not in t]
        alt_tickers = [t for t in self.strategy.tickers if ":" in t]
        
        try:
            # Get regular market data
            if reg_tickers:
                if hasattr(self.market_data_loader, "next"):
                    bar_data = self.market_data_loader.next(reg_tickers, timestamp)
                else:
                    reg_data = self.market_data_loader.fetch_data(reg_tickers)
                    bar_data = self._create_bar_dict(timestamp, reg_data)
            else:
                bar_data = {}
            
            # Get alternative data if available
            if alt_tickers and self.alternative_data_loader:
                if hasattr(self.alternative_data_loader, "next"):
                    alt_data = self.alternative_data_loader.next(alt_tickers, timestamp)
                    bar_data.update(alt_data)
                else:
                    alt_df_dict = self.alternative_data_loader.fetch_data(alt_tickers)
                    alt_bar_data = self._create_bar_dict(timestamp, alt_df_dict)
                    bar_data.update(alt_bar_data)
            
            return bar_data
        
        except Exception as e:
            raise DataFetchError(f"Failed to fetch latest data: {e}")
    
    def _create_bar_dict(self, ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Create a bar dictionary for strategy.step() from dataframes.
        
        Parameters
        ----------
        ts : pd.Timestamp
            Timestamp to get data for
        data : Dict[str, pd.DataFrame]
            Dictionary of ticker dataframes
            
        Returns
        -------
        Dict[str, dict]
            Dictionary of bar data
        """
        out = {}
        for ticker, df in data.items():
            # Find the most recent data point at or before the timestamp
            if df is not None and not df.empty:
                idx = df.index
                pos = idx.searchsorted(ts, side="right") - 1
                if pos >= 0:
                    row = df.iloc[pos]
                    out[ticker] = {
                        "open": float(row.get("open", row.get("close", 0))),
                        "high": float(row.get("high", row.get("close", 0))),
                        "low": float(row.get("low", row.get("close", 0))),
                        "close": float(row.get("close", 0)),
                        "volume": float(row.get("volume", 0)),
                    }
                else:
                    out[ticker] = None
            else:
                out[ticker] = None
        return out
    
    def calculate_position_changes(
        self, target_weights: Dict[str, float], account_info: AccountInfo
    ) -> Dict[str, float]:
        """
        Calculate required position changes based on target weights and current positions.
        
        Parameters
        ----------
        target_weights : Dict[str, float]
            Target portfolio weights for each ticker
        account_info : AccountInfo
            Current account information
            
        Returns
        -------
        Dict[str, float]
            Required position changes in dollars for each ticker
        """
        portfolio_value = account_info["portfolio_value"]
        current_positions = account_info["positions"]
        
        # Calculate current weights
        current_weights = {}
        for symbol, position in current_positions.items():
            current_weights[symbol] = position["market_value"] / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate target dollar values
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = weight * portfolio_value
        
        # Calculate required changes
        changes = {}
        for symbol in set(target_weights.keys()) | set(current_positions.keys()):
            target_value = target_values.get(symbol, 0)
            current_value = current_positions.get(symbol, {}).get("market_value", 0)
            changes[symbol] = target_value - current_value
        
        return changes
    
    def generate_orders(
        self, position_changes: Dict[str, float], prices: Dict[str, float]
    ) -> List[Order]:
        """
        Generate orders from position changes.
        
        Parameters
        ----------
        position_changes : Dict[str, float]
            Required position changes in dollars
        prices : Dict[str, float]
            Current prices for each ticker
            
        Returns
        -------
        List[Order]
            List of orders to execute
        """
        orders = []
        for symbol, dollar_change in position_changes.items():
            if abs(dollar_change) < 1.0:  # Skip very small changes
                continue
                
            price = prices.get(symbol)
            if not price or price <= 0:
                logger.warning(f"No valid price for {symbol}, skipping order")
                continue
                
            # Calculate share quantity
            qty = dollar_change / price
            
            # Round to nearest whole share
            qty = round(qty)
            
            if qty != 0:
                order: Order = {
                    "symbol": symbol,
                    "qty": qty,
                    "order_type": "market",
                    "time_in_force": "day",
                    "limit_price": None,
                }
                orders.append(order)
        
        return orders
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for a list of symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to get prices for
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping symbols to current prices
        """
        data = self.fetch_latest_data()
        prices = {}
        for symbol in symbols:
            if symbol in data and data[symbol] is not None:
                prices[symbol] = data[symbol]["close"]
        return prices
    
    def step(self, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, bool]:
        """
        Execute a single step of the execution_complex process.
        
        Parameters
        ----------
        timestamp : pd.Timestamp, optional
            Timestamp to execute step for, defaults to now
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping symbols to execution_complex success
        """
        timestamp = timestamp or pd.Timestamp.now()
        self.last_step_time = timestamp
        
        try:
            # 1. Fetch latest data
            bar_data = self.fetch_latest_data(timestamp)
            
            # 2. Get signals from strategy
            target_weights = self.strategy.step(timestamp, bar_data)
            
            # 3. Get current account info
            account_info = self.get_account_info()
            
            # 4. Calculate position changes
            position_changes = self.calculate_position_changes(target_weights, account_info)
            
            # 5. Get current prices for all symbols that need changes
            current_prices = self.get_current_prices(list(position_changes.keys()))
            
            # 6. Generate orders
            orders = self.generate_orders(position_changes, current_prices)
            
            # 7. Execute orders
            results = self.execute_orders(orders)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in execution_complex step: {e}")
            raise 