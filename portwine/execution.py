"""
Execution module for the portwine framework.

This module provides the base classes and interfaces for execution modules,
which connect strategy implementations from the backtester to live trading.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional, Tuple, Iterator
import math
import time
from datetime import datetime
import pandas as pd
import pandas_market_calendars as mcal

from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.brokers.base import BrokerBase, Order
from portwine.logger import Logger, log_position_table, log_weight_table, log_order_table
from rich.progress import track, Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
from portwine.scheduler import daily_schedule

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
        fractional: bool = False,
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
        self.fractional = fractional
        # Store timezone (tzinfo); default to system local timezone
        self.timezone = timezone if timezone is not None else datetime.now().astimezone().tzinfo
        # Initialize ticker list from strategy
        self.tickers = strategy.tickers
        # Set up a per-instance rich-enabled logger
        self.logger = Logger.create(self.__class__.__name__, level=logging.INFO)
        self.logger.info(f"Initialized {self.strategy.__class__.__name__} with {len(self.tickers)} tickers")
    
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
        Fetch latest data for all tickers at the given timestamp.
        
        Parameters
        ----------
        timestamp : float, optional
            Unix timestamp in seconds. If None, uses current time.
            
        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            Dictionary mapping tickers to their latest bar data or None
        """
        try:
            # Convert UNIX timestamp to timezone-aware pandas Timestamp
            if timestamp is None:
                dt = pd.Timestamp.now(tz=self.timezone)
            else:
                # timestamp is seconds since epoch
                dt = pd.Timestamp(timestamp, unit='s', tz=self.timezone)
            
            # Split tickers into market vs alternative
            reg_tkrs, alt_tkrs = self._split_tickers(self.tickers)
            # Fetch market data only for regular tickers
            data = self.market_data_loader.next(reg_tkrs, dt)
            # Fetch alternative data only for alternative tickers
            if self.alternative_data_loader is not None and alt_tkrs:
                alt_data = self.alternative_data_loader.next(alt_tkrs, dt)
                # Merge alternative entries into result
                data.update(alt_data)

            self.logger.debug(f"Fetched data keys: {list(data.keys())}")

            return data
        except Exception as e:
            self.logger.exception(f"Error fetching latest data: {e}")
            raise DataFetchError(f"Failed to fetch latest data: {e}")
        
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current closing prices for the specified symbols by querying only market data.

        This method bypasses alternative data and directly uses market_data_loader.next
        with a timezone-aware datetime matching the execution timezone.
        """
        # Build current datetime in execution timezone
        dt = datetime.now(tz=self.timezone)
        # Fetch only market data for given symbols
        data = self.market_data_loader.next(symbols, dt)
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
        
        self.logger.debug(f"Current positions: {current_positions}, portfolio value: {portfolio_value:.2f}")
        return current_positions, portfolio_value

    def _calculate_target_positions(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        fractional: bool = False,
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

    def _target_positions_to_orders(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
    ) -> List[Order]:
        """
        Determine necessary orders given target and current positions, returning Order objects.

        Args:
            target_positions: Mapping ticker -> desired quantity
            current_positions: Mapping ticker -> existing quantity

        Returns:
            List of Order dataclasses for each non-zero change.
        """
        orders: List[Order] = []
        for ticker, target_qty in target_positions.items():
            current_qty = current_positions.get(ticker, 0.0)
            diff = target_qty - current_qty
            if diff == 0:
                continue
            side = 'buy' if diff > 0 else 'sell'
            qty = abs(int(diff))
            # Build an Order dataclass with default/trivial values for non-relevant fields
            order = Order(
                order_id="",
                ticker=ticker,
                side=side,
                quantity=float(qty),
                order_type="market",
                status="new",
                time_in_force="day",
                average_price=0.0,
                remaining_quantity=0.0,
                created_at=0,
                last_updated_at=0,
            )
            orders.append(order)

        log_order_table(self.logger, orders)
        return orders

    def _execute_orders(self, orders: List[Order]) -> List[Order]:
        """
        Execute a list of Order objects through the broker.

        Parameters
        ----------
        orders : List[Order]
            List of Order dataclasses to submit.

        Returns
        -------
        List[Order]
            List of updated Order objects returned by the broker.
        """
        executed_orders: List[Order] = []
        for order in orders:
            # Determine signed quantity: negative for sell, positive for buy
            qty_arg = order.quantity if order.side == 'buy' else -order.quantity
            # Submit each order and collect the updated result
            updated = self.broker.submit_order(
                symbol=order.ticker,
                quantity=qty_arg,
            )
            # Restore expected positive quantity and original side
            updated.quantity = order.quantity
            updated.side = order.side
            executed_orders.append(updated)
            
        self.logger.info(f"Executed {len(executed_orders)} orders")
        return executed_orders
    
    def step(self, timestamp_ms: Optional[int] = None) -> List[Order]:
        """
        Execute a single step of the trading strategy.

        Uses a UNIX timestamp in milliseconds; if None, uses current time.

        Returns a list of updated Order objects.
        """

        # Determine timestamp in ms
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        # Convert ms to datetime in execution timezone
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=self.timezone)

        # Check if market is open
        if not self.broker.market_is_open(dt):
            self.logger.info(f"Market closed at {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return []
        # Log execution start
        local_dt = dt.astimezone()
        self.logger.info(f"Executing step at {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Fetch latest market data
        latest_data = self.fetch_latest_data(dt.timestamp())
        
        # Get target weights from strategy
        target_weights = self.strategy.step(dt, latest_data)
        self.logger.debug(f"Target weights: {target_weights}")

        # Get current positions and portfolio value
        current_positions, portfolio_value = self._get_current_positions()
        
        # Extract prices
        prices = {symbol: bar['close'] for symbol, bar in latest_data.items() if bar and 'close' in bar}
        self.logger.debug(f"Prices: {prices}")

        # Compute target positions and optionally current weights
        target_positions = self._calculate_target_positions(target_weights, portfolio_value, prices, self.fractional)
        current_weights = self._calculate_current_weights(list(current_positions.items()), portfolio_value, prices)
        # Render position changes table
        log_position_table(self.logger, current_positions, target_positions)
        
        # Render weight changes table
        log_weight_table(self.logger, current_weights, target_weights)
        
        # Build and render orders table
        orders = self._target_positions_to_orders(target_positions, current_positions)
        
        # Execute orders and log
        executed = self._execute_orders(orders)
        return executed

    def warmup(self, start_date: str, end_date: str = None, after_open_minutes: int = 0, before_close_minutes: int = 0, interval_seconds: int = None):
        """
        Warm up the strategy by running it over historical data from start_date up to end_date.
        If end_date is None, uses current date.
        """
        tickers = self.tickers
        calendar_name = "NYSE"
        if end_date is None:
            end_date = pd.Timestamp.now(tz=self.timezone).strftime("%Y-%m-%d")
        schedule = daily_schedule(
            after_open_minutes=after_open_minutes,
            before_close_minutes=before_close_minutes,
            interval_seconds=interval_seconds,
            calendar_name=calendar_name,
            start_date=start_date,
            end_date=end_date
        )
        steps = 0
        last_data = {t: None for t in tickers}
        try:
            for ts in schedule:
                dt_aware = pd.to_datetime(ts, unit='ms', utc=True).tz_convert(self.timezone)
                # Fetch data with ffill
                daily_data = self.market_data_loader.next(tickers, dt_aware, ffill=True)
                # Forward-fill missing values
                for t in tickers:
                    if daily_data[t] is None and last_data[t] is not None:
                        daily_data[t] = last_data[t]
                    elif daily_data[t] is not None:
                        last_data[t] = daily_data[t]
                current_signals = self.strategy.step(dt_aware, daily_data)
                self.logger.info(f"Warmup step at {dt_aware}: {current_signals}")
                steps += 1
                if steps % 100 == 0:
                    self.logger.info(f"Warm-up progress: {steps} steps...")
        except StopIteration:
            self.logger.info(f"Warm-up complete after {steps} steps (schedule exhausted).")
            return
        self.logger.info(f"Warm-up complete after {steps} steps (reached now).")

    def run(self, schedule: Iterator[int], warmup_start_date: str = None) -> None:
        """
        Optionally run warmup before main execution loop. If warmup_start_date is provided, run warmup from that date.
        """
        if warmup_start_date is not None:
            # Try to extract warmup params from schedule object
            after_open = getattr(schedule, 'after_open', 0)
            before_close = getattr(schedule, 'before_close', 0)
            interval = getattr(schedule, 'interval', None)
            self.logger.info(
                f"Running warmup from {warmup_start_date} (after_open={after_open}, before_close={before_close}, interval={interval})"
            )
            self.warmup(
                warmup_start_date,
                after_open_minutes=after_open,
                before_close_minutes=before_close,
                interval_seconds=interval
            )
        # allow for missing timezone (e.g. in FakeExec)
        tz = getattr(self, 'timezone', None)
        for timestamp_ms in schedule:
            # Display next execution time
            schedule_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=tz)
            self.logger.info(
                f"Next scheduled execution at {schedule_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            # compute wait using time.time() so it matches patched time in tests
            target_s = timestamp_ms / 1000.0
            now_s = time.time()
            wait = target_s - now_s
            if wait > 0:
                total_seconds = int(wait)
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                )
                with progress:
                    task = progress.add_task("Waiting for next execution", total=total_seconds)
                    for _ in range(total_seconds):
                        time.sleep(1)
                        progress.advance(task)
                rem = wait - total_seconds
                if rem > 0:
                    time.sleep(rem)
            self.logger.info(
                f"Executing step for {schedule_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            self.step(timestamp_ms)
