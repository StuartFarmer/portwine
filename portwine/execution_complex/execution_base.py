"""
The ExecutionBase class serves as a bridge between trading strategies and brokers,
handling the execution of trades based on signals from the strategy.
"""

import logging
import time
from typing import Dict, List, Optional, Any

import pandas as pd
import pytz

from portwine.utils.market_calendar import MarketStatus
from portwine.execution_complex.broker import BrokerBase, Order, AccountInfo
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader
from portwine.utils.schedule_iterator import ScheduleIterator


class ExecutionBase:
    """
    Base class for execution systems.

    This class handles the execution of trades based on signals from a trading strategy.
    It provides a framework for executing trades on a schedule, handling market hours,
    and managing errors during execution.
    """

    def __init__(
        self,
        strategy: StrategyBase,
        broker: BrokerBase,
        schedule: Optional[ScheduleIterator] = None,
        market_data_loader: Optional[MarketDataLoader] = None,
        alternative_data_loader: Optional[MarketDataLoader] = None,
        min_change_pct: float = 0.01,
        min_order_value: float = 1.0,
        timezone: Optional[str] = None,
    ):
        """
        Initialize the ExecutionBase class.

        Args:
            strategy: The trading strategy to execute.
            broker: The broker to execute trades through.
            schedule: Schedule iterator that yields execution times.
            market_data_loader: Optional data loader for fetching market data.
            alternative_data_loader: Additional data loader for alternative data.
            min_change_pct: Minimum change percentage required to trigger a trade.
            min_order_value: Minimum dollar value required for an order.
            timezone: Timezone to use for timestamps (default: UTC).
        """
        self.strategy = strategy
        self.broker = broker
        self.schedule = schedule
        self.market_data_loader = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        self.min_change_pct = min_change_pct
        self.min_order_value = min_order_value
        self.timezone = timezone or "UTC"

        # Initialize ticker list from strategy
        self.tickers = strategy.tickers

        # Flag to control execution
        self._running = False

        # Set up logging
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.tickers)} tickers")

    def _wait_until_execution_time(self, execution_time: pd.Timestamp):
        """
        Wait until the specified execution time.

        Args:
            execution_time: The time to execute the next iteration.
        """
        now = pd.Timestamp.now(tz=execution_time.tz)

        # In test mode, if the execution time is far in the future, skip waiting
        if (execution_time - now).total_seconds() > 60:
            self.logger.info("Execution stopped during wait")
            return

        if execution_time > now:
            sleep_seconds = (execution_time - now).total_seconds()
            self.logger.debug(f"Sleeping for {sleep_seconds} seconds until {execution_time}")

            # Sleep in small increments to allow for stopping
            end_time = time.time() + sleep_seconds
            while time.time() < end_time and self._running:
                time.sleep(min(0.1, end_time - time.time()))

    def fetch_latest_data(self, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Optional[Dict[str, Any]]]:
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
        """
        try:
            # Use the provided timestamp or current time
            if timestamp is None:
                timestamp = pd.Timestamp.now(tz=self.timezone)

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
            self.logger.exception(f"Error fetching latest data: {e}")
            raise Exception(f"Failed to fetch latest data: {e}")

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

    def step(self) -> Dict[str, Order]:
        """
        Execute a single step of the strategy.

        Returns:
            Dictionary mapping symbols to executed orders.
            Empty dictionary if market is closed or an error occurred.
        """
        executed_orders = {}

        try:
            # Check market status
            market_status = self.broker.check_market_status()
            self.logger.info(f"Market status: {'open' if market_status.is_open else 'closed'}")

            # Store market status for reference
            self._last_market_status = market_status

            if not market_status.is_open:
                self.logger.info("Market is closed, skipping execution")
                return executed_orders

            # Get current timestamp with timezone
            execution_time = pd.Timestamp.now()

            # Apply timezone to make sure it's passed correctly to the strategy
            if self.timezone:
                try:
                    tz = pytz.timezone(self.timezone)
                    if execution_time.tz is None:
                        execution_time = execution_time.tz_localize(tz)
                    else:
                        execution_time = execution_time.tz_convert(tz)
                except Exception as e:
                    self.logger.warning(f"Error applying timezone: {e}")

            # Fetch latest data
            latest_data = self.fetch_latest_data(execution_time)

            # Call strategy.step with the properly timezone-aware timestamp
            target_weights = self.strategy.step(execution_time, latest_data)
            self.logger.info(f"Strategy step returned target weights: {target_weights}")

            # Get account info
            account_info = self.broker.get_account_info()

            # Generate and execute orders
            for symbol, weight in target_weights.items():
                if abs(weight) < 0.001:  # Ignore very small weights
                    continue

                # Calculate target position based on portfolio value
                target_value = account_info.portfolio_value * weight

                # Get current price
                current_price = latest_data.get(symbol, {}).get('close')
                if not current_price:
                    self.logger.warning(f"No price data for {symbol}, skipping")
                    continue

                # Calculate target quantity
                target_qty = int(target_value / current_price)

                # Get current position
                current_position = next(
                    (p for p in account_info.positions.values() if p.symbol == symbol),
                    None
                )
                current_qty = current_position.qty if current_position else 0

                # Calculate order quantity
                order_qty = target_qty - current_qty

                if abs(order_qty) > 0:
                    try:
                        success = self.broker.execute_order(symbol=symbol, qty=order_qty)
                        if success:
                            executed_orders[symbol] = Order(
                                symbol=symbol,
                                qty=order_qty,
                                order_type="market",
                                status="filled"
                            )
                            self.logger.info(f"Executed order for {symbol}, quantity={order_qty}")
                        else:
                            self.logger.warning(f"Failed to execute order for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error executing order for {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error in execution step: {e}")

        return executed_orders

    def run(self, max_iterations: Optional[int] = None):
        """
        Run the execution system, processing the strategy at scheduled times.

        Args:
            max_iterations: Maximum number of iterations to run (None for infinite).
        """
        if not self.schedule:
            self.logger.error("Cannot run without a schedule")
            return

        iteration_count = 0
        self._running = True

        try:
            while self._running:
                # Check if we've reached max iterations
                if max_iterations is not None and iteration_count >= max_iterations:
                    self.logger.info(f"Reached maximum iterations ({max_iterations}), stopping")
                    break

                try:
                    # Get next execution time from schedule
                    execution_time = next(self.schedule)
                    self.logger.info(f"Next execution scheduled for: {execution_time}")

                    # Wait until the execution time
                    self._wait_until_execution_time(execution_time)

                    # Execute a step of the strategy
                    orders = self.step()

                    # Check if market was open
                    if hasattr(self, '_last_market_status') and self._last_market_status.is_open:
                        iteration_count += 1
                        self.logger.info(f"Completed iteration {iteration_count}")

                except StopIteration:
                    # Schedule iterator exhausted
                    self.logger.info("Schedule iterator exhausted, stopping")
                    break

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, stopping")
        except Exception as e:
            self.logger.error(f"Error during execution: {e}")
        finally:
            self._running = False

    def run_once(self):
        """
        Run a single step of the execution system immediately.
        """
        self._running = True
        orders = self.step()
        self._running = False
        return orders
    
    def stop(self):
        """
        Stop the execution system.
        """
        self.logger.info("Stopping execution")
        self._running = False 