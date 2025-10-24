"""
Position-based backtesting system.

Interprets strategy outputs as share quantities instead of portfolio weights.
Tracks positions, actions, and portfolio value over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Callable, Optional
from tqdm import tqdm

from portwine.strategies.base import StrategyBase
from portwine.backtester.benchmarks import BenchmarkTypes, STANDARD_BENCHMARKS, get_benchmark_type
from portwine.backtester.core import InvalidBenchmarkError, _split_tickers, DEFAULT_CALENDAR
from portwine.data.interface import DataInterface, MultiDataInterface


class PositionBacktestResult:
    """
    Stores position-based backtest results.

    Similar to BacktestResult but tracks positions/actions instead of weights.
    """

    def __init__(self, datetime_index: pd.DatetimeIndex, tickers: List[str]):
        """
        Initialize result storage.

        Args:
            datetime_index: Trading days
            tickers: List of ticker symbols
        """
        self.datetime_index = datetime_index
        self.tickers = sorted(tickers)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        n_days = len(datetime_index)
        n_tickers = len(self.tickers)

        # Core data arrays
        self.positions_array = np.zeros((n_days, n_tickers), dtype=np.float64)
        self.actions_array = np.zeros((n_days, n_tickers), dtype=np.float64)
        self.prices_array = np.full((n_days, n_tickers), np.nan, dtype=np.float64)

        # Portfolio value over time
        self.portfolio_value = np.zeros(n_days, dtype=np.float64)

    def add_action(self, day_idx: int, ticker: str, quantity: float):
        """
        Record an action (buy/sell) for a ticker on a given day.

        Args:
            day_idx: Index in datetime_index
            ticker: Ticker symbol
            quantity: Number of shares (positive=buy, negative=sell)
        """
        if ticker not in self.ticker_to_idx:
            return  # Skip tickers not in result set

        ticker_idx = self.ticker_to_idx[ticker]
        self.actions_array[day_idx, ticker_idx] = quantity

    def add_price(self, day_idx: int, ticker: str, price: float):
        """
        Record execution price for a ticker on a given day.

        Args:
            day_idx: Index in datetime_index
            ticker: Ticker symbol
            price: Execution price
        """
        if ticker not in self.ticker_to_idx:
            return

        ticker_idx = self.ticker_to_idx[ticker]
        self.prices_array[day_idx, ticker_idx] = price

    def update_positions(self):
        """
        Calculate cumulative positions from actions with signal shifting.

        Signal shifting prevents lookahead bias:
        - Action generated on day t (seeing day t data) executes on day t+1
        - positions[t] = cumsum(actions[t-1])

        This matches the weights backtester behavior where signals are shifted
        forward by 1 day before calculating returns.
        """
        # Shift actions forward by 1 day (action on day t executes on day t+1)
        shifted_actions = np.roll(self.actions_array, 1, axis=0)
        if shifted_actions.shape[0] > 0:
            shifted_actions[0, :] = 0.0  # First day has no previous actions

        # Cumulative sum along time axis
        self.positions_array = np.cumsum(shifted_actions, axis=0)

    def calculate_portfolio_value(self):
        """
        Calculate portfolio value over time.

        portfolio_value[t] = sum(positions[t] × prices[t])
        """
        # Element-wise multiply positions × prices, sum across tickers
        # Handle NaN prices (treat as 0 contribution)
        prices_filled = np.where(np.isnan(self.prices_array), 0.0, self.prices_array)
        self.portfolio_value = np.sum(self.positions_array * prices_filled, axis=1)

    def to_dict(self) -> dict:
        """
        Convert results to dictionary format.

        Returns:
            dict: Results in same format as Backtester output
                - positions_df: DataFrame of positions over time
                - actions_df: DataFrame of actions over time
                - prices_df: DataFrame of execution prices
                - portfolio_value: Series of portfolio value
        """
        return {
            'positions_df': pd.DataFrame(
                self.positions_array,
                index=self.datetime_index,
                columns=self.tickers
            ),
            'actions_df': pd.DataFrame(
                self.actions_array,
                index=self.datetime_index,
                columns=self.tickers
            ),
            'prices_df': pd.DataFrame(
                self.prices_array,
                index=self.datetime_index,
                columns=self.tickers
            ),
            'portfolio_value': pd.Series(
                self.portfolio_value,
                index=self.datetime_index,
                name='portfolio_value'
            )
        }


class PositionBacktester:
    """
    Position-based backtester.

    Interprets strategy output as share quantities instead of portfolio weights.
    Uses same data interfaces and patterns as standard Backtester.
    """

    def __init__(self, data_interface, calendar=DEFAULT_CALENDAR):
        """
        Initialize position backtester.

        Args:
            data_interface: DataInterface or MultiDataInterface
            calendar: DailyMarketCalendar (default: NYSE calendar)
        """
        self.data = data_interface
        self.calendar = calendar

        # Create restricted data interface (same as Backtester)
        if isinstance(data_interface, MultiDataInterface):
            from portwine.data.interface import RestrictedDataInterface
            self.restricted_data = RestrictedDataInterface(data_interface.loaders)
        else:
            from portwine.data.interface import RestrictedDataInterface
            self.restricted_data = RestrictedDataInterface({None: data_interface.data_loader})

    def run_backtest(
        self,
        strategy: StrategyBase,
        start_date: Union[str, None] = None,
        end_date: Union[str, None] = None,
        benchmark: Union[str, Callable, None] = None,
        verbose: bool = False,
        require_all_history: bool = False,
        execution_price: str = 'close'
    ):
        """
        Run position-based backtest.

        Args:
            strategy: StrategyBase instance (interprets output as shares)
            start_date: Start date (auto-detect if None)
            end_date: End date (auto-detect if None)
            benchmark: Benchmark (ticker, function, or None)
            verbose: Show progress bar
            require_all_history: Ensure all tickers have data from start
            execution_price: Which price to use for execution ('close' or 'open')
                - 'close': Execute at next day's close (default, current behavior)
                - 'open': Execute at next day's open (more realistic)

        Returns:
            dict: Position-based results
        """
        # Validate execution_price parameter
        if execution_price not in ('close', 'open'):
            raise ValueError(f"execution_price must be 'close' or 'open', got: {execution_price}")
        # 1. Validate strategy has tickers (same as Backtester)
        if not strategy.universe.all_tickers:
            raise ValueError("Strategy has no tickers. Cannot run backtest with empty universe.")

        regular_tickers, _ = _split_tickers(set(strategy.universe.all_tickers))

        # 2. Determine date range (same as Backtester)
        end_date = self._compute_effective_end_date(end_date, regular_tickers)

        if start_date is None:
            if isinstance(self.data, MultiDataInterface):
                start_date = self.data.earliest_any_date(regular_tickers)
            else:
                start_date = DataInterface(self.data.data_loader).earliest_any_date(regular_tickers)

        # 3. Get datetime index
        datetime_index = self.calendar.get_datetime_index(start_date, end_date)

        # 4. Handle require_all_history
        if require_all_history:
            if isinstance(self.data, MultiDataInterface):
                common_start = self.data.earliest_common_date(regular_tickers)
            else:
                common_start = DataInterface(self.data.data_loader).earliest_common_date(regular_tickers)
            if start_date is None or pd.Timestamp(start_date) < pd.Timestamp(common_start):
                start_date = common_start
            datetime_index = self.calendar.get_datetime_index(start_date, end_date)

        # 5. Initialize result storage
        result = PositionBacktestResult(datetime_index, sorted(regular_tickers))

        # 6. Main backtest loop
        iterator = tqdm(datetime_index, desc="Position Backtest") if verbose else datetime_index

        for i, dt in enumerate(iterator):
            # Update universe
            strategy.universe.set_datetime(dt)
            current_universe_tickers = strategy.universe.get_constituents(dt)

            # Set up restricted data interface
            self.restricted_data.set_current_timestamp(dt)
            regular_tickers_current, _ = _split_tickers(set(current_universe_tickers))
            self.restricted_data.set_restricted_tickers(regular_tickers_current, prefix=None)

            # Call strategy
            dt_datetime = pd.Timestamp(dt).to_pydatetime()
            actions = strategy.step(dt_datetime, self.restricted_data)

            # Normalize actions to dict (same as Backtester._normalize_signals)
            if actions is None:
                actions = {}
            elif isinstance(actions, pd.Series):
                actions = actions.to_dict()
            elif not isinstance(actions, dict):
                raise ValueError(f"Strategy returned invalid type: {type(actions)}")

            # Validate actions
            self.validate_actions(actions, current_universe_tickers)

            # Record actions
            for ticker, quantity in actions.items():
                result.add_action(i, ticker, quantity)

            # Record prices (use specified execution price)
            for ticker in regular_tickers_current:
                try:
                    price_data = self.restricted_data[ticker]
                    price = price_data.get(execution_price)
                    if price is not None:
                        result.add_price(i, ticker, price)
                except (KeyError, ValueError):
                    # Ticker has no data on this day
                    pass

        # 7. Calculate results
        result.update_positions()
        result.calculate_portfolio_value()

        # 8. Return results (no benchmark yet)
        return result.to_dict()

    def _compute_effective_end_date(self, end_date, tickers):
        """Compute effective end date (same logic as Backtester)."""
        if end_date is not None:
            return end_date

        # Find latest date across all tickers
        if isinstance(self.data, MultiDataInterface):
            data_interface = DataInterface(self.data.loaders[None])
        else:
            data_interface = DataInterface(self.data.data_loader)

        latest_dates = []
        for ticker in tickers:
            try:
                latest = data_interface.data_loader.latest(ticker)
                if latest:
                    latest_dates.append(latest)
            except (KeyError, AttributeError):
                continue

        if not latest_dates:
            raise ValueError("No data found for any ticker")

        max_date = max(latest_dates)
        # Convert to string format if it's a Timestamp
        if isinstance(max_date, pd.Timestamp):
            return max_date.strftime('%Y-%m-%d')
        return str(max_date) if hasattr(max_date, 'strftime') else max_date

    def validate_actions(self, actions: Dict[str, float], current_universe_tickers: List[str]):
        """
        Validate position actions.

        Args:
            actions: Dict of ticker → quantity
            current_universe_tickers: Valid tickers for this date

        Raises:
            ValueError: If actions are invalid
        """
        for ticker in actions.keys():
            if ticker not in current_universe_tickers:
                raise ValueError(f"Ticker {ticker} not in current universe")

        for ticker, quantity in actions.items():
            if not isinstance(quantity, (int, float)):
                raise ValueError(f"Action for {ticker} must be numeric, got {type(quantity)}")
            if np.isnan(quantity) or np.isinf(quantity):
                raise ValueError(f"Invalid action for {ticker}: {quantity}")
