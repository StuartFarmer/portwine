# portwine/backtester.py

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging as _logging
from tqdm import tqdm
from portwine.backtester.benchmarks import STANDARD_BENCHMARKS, BenchmarkTypes, InvalidBenchmarkError, get_benchmark_type
from portwine.logger import Logger

import pandas_market_calendars as mcal
from portwine.data.providers.loader_adapters import MarketDataLoader

from pandas_market_calendars import MarketCalendar
import datetime

from portwine.data.interface import DataInterface, RestrictedDataInterface, MultiDataInterface
from portwine.strategies.base import StrategyBase

# Optional Numba import for JIT compilation
from numba import jit

def validate_dates(start_date: str, end_date: Union[str, None]) -> bool:
    assert isinstance(start_date, str), "Start date is required in string format YYYY-MM-DD."

    # Cast to datetime objects
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    if end_date is None:
        end_date_obj = datetime.datetime.now()
    else:
        end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    assert end_date_obj > start_date_obj, "End date must be after start date."

    return True


class DailyMarketCalendar:
    def __init__(self, calendar_name):
        self.calendar = MarketCalendar.factory(calendar_name)

    def schedule(self, start_date, end_date):
        """Expose the schedule method from the underlying calendar"""
        return self.calendar.schedule(start_date=start_date, end_date=end_date)
    
    def get_datetime_index(self, start_date: str, end_date: Union[str, None]=None):
        validate_dates(start_date, end_date)

        # Use today's date if end_date is None
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        dt_index = schedule['market_open'].index

        dt_localized = dt_index.tz_localize("UTC")
        dt_converted = dt_localized.tz_convert(None)

        datetime_index = dt_converted.to_numpy()

        if len(datetime_index) == 0:
            raise ValueError("No trading days found in the specified date range")

        return datetime_index
    
DEFAULT_CALENDAR = DailyMarketCalendar("NYSE")

def _split_tickers(tickers: set) -> Tuple[List[str], List[str]]:
        """
        Split tickers into regular and alternative data tickers.
        
        Parameters
        ----------
        tickers : set
            Set of ticker symbols
            
        Returns
        -------
        Tuple[List[str], List[str]]
            Tuple of (regular_tickers, alternative_tickers)
        """
        reg, alt = [], []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

class BacktestResult:
    def __init__(self, datetime_index, all_tickers):
        self.datetime_index = datetime_index
        self.all_tickers = all_tickers
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
        
        # Initialize numpy arrays for signals and returns with zeros
        self.sig_array = np.zeros((len(datetime_index), len(all_tickers)), dtype=np.float64)
        self.ret_array = np.zeros((len(datetime_index), len(all_tickers)), dtype=np.float64)
        self.close_array = np.zeros((len(datetime_index), len(all_tickers)), dtype=np.float64)
        self.strategy_returns = np.zeros(len(datetime_index), dtype=np.float64)

    def add_signals(self, i: int, sig: Dict[str, float]):
        """Add signals for a specific time step using vectorized operations."""
        # Vectorized signal assignment - map dictionary values to array positions
        self.sig_array[i, :] = np.array([sig.get(ticker, 0.0) for ticker in self.all_tickers])
    
    def add_close_prices(self, i: int, data_interface):
        """Add close prices for a specific time step using vectorized operations."""
        # Vectorized close price collection - single numpy operation
        # Only access regular market tickers that this BacktestResult is tracking
        close_prices = []
        for ticker in self.all_tickers:
            try:
                ticker_data = data_interface[ticker]
                close_prices.append(ticker_data['close'])
            except (KeyError, ValueError):
                # If ticker data is not available, use NaN
                close_prices.append(np.nan)
        
        self.close_array[i, :] = np.array(close_prices)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_returns(close_array: np.ndarray, ret_array: np.ndarray):
        """Numba-optimized returns calculation."""
        n_days, n_tickers = close_array.shape
        
        for i in range(1, n_days):  # Skip first day (no previous data)
            for j in range(n_tickers):
                prev_close = close_array[i-1, j]
                curr_close = close_array[i, j]
                
                # Handle NaN values
                if np.isnan(prev_close) or np.isnan(curr_close):
                    ret_array[i, j] = np.nan
                elif prev_close > 0:
                    ret_array[i, j] = (curr_close - prev_close) / prev_close
                else:
                    ret_array[i, j] = 0.0
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_strategy_returns(signals: np.ndarray, returns: np.ndarray, strategy_returns: np.ndarray):
        """Numba-optimized strategy returns calculation."""
        n_days, n_tickers = signals.shape
        
        for i in range(n_days):
            daily_return = 0.0
            for j in range(n_tickers):
                daily_return += signals[i, j] * returns[i, j]
            strategy_returns[i] = daily_return
    
    def calculate_results(self):
        self._calculate_returns(self.close_array, self.ret_array)
        # Calculate strategy returns: sum(signals * returns) for each day
        shifted_signals = np.roll(self.sig_array, 1, axis=0)
        if shifted_signals.shape[0] > 0:
            shifted_signals[0, :] = 0.0  # First day has no previous signals
        # Treat NaN returns as 0 for strategy P&L to avoid NaN propagation in aggregates
        returns_no_nan = np.where(np.isnan(self.ret_array), 0.0, self.ret_array)
        self._calculate_strategy_returns(shifted_signals, returns_no_nan, self.strategy_returns)
    
    def get_results(self):
        """Return the final DataFrames."""
        sig_df = pd.DataFrame(self.sig_array, index=self.datetime_index, columns=self.all_tickers)
        sig_df.index.name = None
        
        ret_df = pd.DataFrame(self.ret_array, index=self.datetime_index, columns=self.all_tickers)
        ret_df.index.name = None
        
        strategy_ret_df = pd.DataFrame(self.strategy_returns, index=self.datetime_index, columns=['strategy_returns'])
        strategy_ret_df.index.name = None
        
        return sig_df, ret_df, strategy_ret_df


class Backtester:
    # TODO: streamline data interface initialization. we shouldnt be testing all instances and doing different things for each.
    def __init__(self, data: DataInterface, calendar: DailyMarketCalendar=DEFAULT_CALENDAR):
        self.data = data
        # Pass all loaders to RestrictedDataInterface if data is MultiDataInterface
        if isinstance(data, MultiDataInterface):
            # If caller already provided a RestrictedDataInterface (or subclass), reuse it
            if isinstance(data, RestrictedDataInterface):
                self.restricted_data = data
            else:
                self.restricted_data = RestrictedDataInterface(data.loaders)
        else:
            # DataInterface case
            self.restricted_data = RestrictedDataInterface({None: data.data_loader})
        self.calendar = calendar

    def _get_default_market_store(self):
        """Return the default market data store from the data interface."""
        if isinstance(self.data, MultiDataInterface):
            return self.data.loaders[None]
        # DataInterface case
        return self.data.data_loader

    def _compute_effective_end_date(self, requested_end_date: Optional[str], tickers: List[str]) -> str:
        """
        Determine the effective end_date to use for the backtest.

        - If the caller provided an end_date, return it unchanged
        - Otherwise, query the underlying store for each ticker's latest available
          timestamp and use the most conservative end date: the earliest of those
          latest timestamps so that all tickers have data up to end_date
        """
        if requested_end_date is not None:
            return requested_end_date

        store = self._get_default_market_store()

        latest_dates: List[datetime.datetime] = []
        for symbol in tickers:
            dt_obj = None
            # 1) Preferred: DataStore.latest(symbol)
            try:
                dt_candidate = store.latest(symbol)  # may raise AttributeError on non-DataStore loaders
                if dt_candidate is not None:
                    dt_obj = dt_candidate
            except AttributeError:
                dt_obj = None
            except Exception:
                dt_obj = None

            # 2) Fallback for legacy loaders: fetch_data + index.max()
            if dt_obj is None:
                try:
                    df_map = store.fetch_data([symbol])
                    df = df_map.get(symbol)
                    if df is not None and not df.empty:
                        dt_candidate = df.index.max()
                        if dt_candidate is not None:
                            dt_obj = dt_candidate
                except Exception:
                    dt_obj = None

            if dt_obj is None:
                continue

            # Normalize to datetime for comparison
            if not isinstance(dt_obj, datetime.datetime):
                dt_obj = pd.to_datetime(dt_obj).to_pydatetime()
            latest_dates.append(dt_obj)

        if not latest_dates:
            raise ValueError("Cannot determine end_date: no latest dates available for any ticker in universe")

        # Conservative choice: limit timeline to the earliest of latest dates across tickers
        effective_dt = min(latest_dates)
        return effective_dt.strftime("%Y-%m-%d")
    
    def validate_data(self, tickers: List[str], start_date: str, end_date: str) -> bool:
        for ticker in tickers:
            if not self.data.exists(ticker, start_date, end_date):
                raise ValueError(f"Data for ticker {ticker} does not exist for the given date range.")
        return True
    
    def validate_signals(self, sig: Dict[str, float], dt: pd.Timestamp, current_universe_tickers: List[str]) -> bool:
        # Check for over-allocation: total weights >1
        total_weight = sum(sig.values())
        # Allow for minor floating-point rounding errors
        if total_weight > 1.0 + 1e-8:
            raise ValueError(f"Total allocation {total_weight:.6f} exceeds 1.0 at {dt}")
        
        # Validate that strategy only assigns weights to tickers in the current universe
        invalid_tickers = [t for t in sig.keys() if t not in current_universe_tickers]
        if invalid_tickers:
            raise ValueError(
                f"Strategy assigned weights to tickers not in current universe at {dt}: {invalid_tickers}. "
                f"Current universe: {current_universe_tickers}"
            )

    def run_backtest(self, strategy: StrategyBase, start_date: Union[str, None]=None, end_date: Union[str, None]=None, benchmark: Union[str, Callable, None] = "equal_weight", verbose: bool = False, require_all_history: bool = False):
        # Validate that strategy has tickers
        if not strategy.universe.all_tickers:
            raise ValueError("Strategy has no tickers. Cannot run backtest with empty universe.")

        # Validate that strategy has market tickers (not just alternative data)
        regular_tickers, _ = _split_tickers(set(strategy.universe.all_tickers))
        # Allow strategies with only alternative data for now
        # if not regular_tickers:
        #     raise ValueError("Strategy has no market tickers. Cannot run backtest with only alternative data.")

        # If end_date is None, compute an effective one based on the data store
        # Use only regular market tickers when determining the end_date
        end_date = self._compute_effective_end_date(end_date, regular_tickers)

        # If start_date is None, choose earliest-any date across the requested tickers
        if start_date is None:
            # Earliest date among any tickers
            if isinstance(self.data, MultiDataInterface):
                start_date = self.data.earliest_any_date(regular_tickers)
            else:
                start_date = DataInterface(self.data.data_loader).earliest_any_date(regular_tickers)
        
        # Now get the datetime index with the determined end_date
        datetime_index = self.calendar.get_datetime_index(start_date, end_date)

        # Validate data with the determined date range
        self.validate_data(strategy.universe.all_tickers, start_date, end_date)

        # Handle require_all_history logic correctly: use the latest of each ticker's earliest date
        if require_all_history:
            if isinstance(self.data, MultiDataInterface):
                common_start = self.data.earliest_common_date(regular_tickers)
            else:
                common_start = DataInterface(self.data.data_loader).earliest_common_date(regular_tickers)
            if start_date is None or pd.Timestamp(start_date) < pd.Timestamp(common_start):
                start_date = common_start
            # Recompute index with updated start_date
            datetime_index = self.calendar.get_datetime_index(start_date, end_date)

        # Classify benchmark type
        # Get the default loader from the data interface

        # Determine the loader type using isinstance instead of attribute checks
        if isinstance(self.data, MultiDataInterface):
            # MultiDataInterface case - get the default loader
            data_loader = self.data.loaders[None]
        else:
            # DataInterface case
            data_loader = self.data.data_loader
        bm_type = get_benchmark_type(benchmark, data_loader, self.data)
        
        # Additional validation for ticker benchmarks
        # TODO: the benchmark file has a function to do this. we should use that.
        if bm_type == BenchmarkTypes.TICKER:
            # Verify that the ticker actually exists in the data using the interface wrapper
            try:
                original_timestamp = self.restricted_data.current_timestamp

                # Try multiple timestamps to find one that works
                test_timestamps = [
                    pd.Timestamp('2020-01-01'),
                    pd.Timestamp('2020-01-15'),
                    pd.Timestamp('2020-06-01'),
                    pd.Timestamp('2021-01-01')
                ]

                ticker_found = False
                for test_ts in test_timestamps:
                    try:
                        self.restricted_data.set_current_timestamp(test_ts)
                        _ = self.restricted_data[benchmark]  # Raises KeyError if ticker doesn't exist
                        ticker_found = True
                        break
                    except (KeyError, ValueError):
                        continue

                # Restore original timestamp
                if original_timestamp is not None:
                    self.restricted_data.set_current_timestamp(original_timestamp)

                if not ticker_found:
                    # If ticker doesn't exist, mark as invalid
                    bm_type = BenchmarkTypes.INVALID

            except (KeyError, ValueError):
                # If ticker doesn't exist, mark as invalid
                bm_type = BenchmarkTypes.INVALID
        
        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark.")

        # Initialize BacktestResult to handle data collection
        all_tickers = sorted(strategy.universe.all_tickers)
        # Filter out alternative data tickers from results (only include regular tickers)
        regular_tickers, _ = _split_tickers(set(all_tickers))
        result = BacktestResult(datetime_index, sorted(regular_tickers))
        
        # Add progress bar if verbose is True
        iterator = tqdm(datetime_index, desc="Backtest") if verbose else datetime_index
        
        for i, dt in enumerate(iterator):
            strategy.universe.set_datetime(dt)
            current_universe_tickers = strategy.universe.get_constituents(dt)

            # Create a RestrictedDataInterface for the strategy
            self.restricted_data.set_current_timestamp(dt)
            # Only restrict market data tickers (default loader), not alternative data
            regular_tickers, _ = _split_tickers(set(current_universe_tickers))
            self.restricted_data.set_restricted_tickers(regular_tickers, prefix=None)

            # Convert numpy.datetime64 to Python datetime for strategy compatibility
            dt_datetime = pd.Timestamp(dt).to_pydatetime()
            sig = strategy.step(dt_datetime, self.restricted_data)
            
            self.validate_signals(sig, dt, current_universe_tickers)
            
            # Use BacktestResult to handle signal and close price updates
            result.add_signals(i, sig)
            result.add_close_prices(i, self.restricted_data)
        
        # Calculate returns and strategy returns using BacktestResult
        result.calculate_results()
        
        # Get results from BacktestResult
        sig_df, ret_df, strategy_ret_df = result.get_results()

        # Calculate benchmark returns based on type
        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            benchmark_returns = benchmark(ret_df)
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            benchmark_returns = STANDARD_BENCHMARKS[benchmark](ret_df)
        else:  # TICKER
            # For ticker benchmarks, use the DataInterface to access benchmark data
            benchmark_returns = self._calculate_ticker_benchmark_returns(benchmark, datetime_index, ret_df.index)
        
        # Convert DataFrames to Series for compatibility with analyzers
        strategy_returns_series = strategy_ret_df.iloc[:, 0]  # Extract first column as Series
        benchmark_returns_series = benchmark_returns.iloc[:, 0] if hasattr(benchmark_returns, 'iloc') and benchmark_returns.ndim > 1 else benchmark_returns
        
        # Ensure benchmark returns has the same name as strategy returns for consistency
        if hasattr(strategy_returns_series, 'name'):
            benchmark_returns_series.name = strategy_returns_series.name
        
        # Return results from BacktestResult
        return {
            "signals_df":        sig_df,
            "tickers_returns":   ret_df,
            "strategy_returns":  strategy_returns_series,
            "benchmark_returns": benchmark_returns_series
        }

    def _calculate_ticker_benchmark_returns(self, benchmark_ticker: str, datetime_index, ret_df_index):
        """
        Calculate returns for a ticker benchmark using the DataInterface.
        
        This method loads the benchmark ticker data and calculates its returns
        aligned with the strategy timeline.
        """
        benchmark_returns = []
        
        for dt in datetime_index:
            # Set the current timestamp to get benchmark data via the interface wrapper
            self.restricted_data.set_current_timestamp(dt)
            
            try:
                # Get benchmark data for this timestamp using the DataInterface
                benchmark_data = self.restricted_data[benchmark_ticker]
                if benchmark_data is None:
                    benchmark_returns.append(0.0)
                else:
                    benchmark_returns.append(benchmark_data['close'])
            except (KeyError, ValueError):
                # If benchmark data is not available, use 0 return
                benchmark_returns.append(0.0)
        
        # Convert to pandas Series and calculate returns
        benchmark_prices = pd.Series(benchmark_returns, index=ret_df_index)
        benchmark_returns_series = benchmark_prices.pct_change(fill_method=None).fillna(0.0)
        
        return benchmark_returns_series
