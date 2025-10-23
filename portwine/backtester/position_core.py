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
from portwine.backtester.core import InvalidBenchmarkError, _split_tickers, DailyMarketCalendar
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
        Calculate cumulative positions from actions.

        positions[t] = positions[t-1] + actions[t]
        """
        # Cumulative sum along time axis
        self.positions_array = np.cumsum(self.actions_array, axis=0)

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

    Similar to Backtester, but interprets strategy output as share quantities
    rather than portfolio weights.
    """
    pass
