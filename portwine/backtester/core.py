# portwine/backtester.py

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging as _logging
from tqdm import tqdm
from portwine.backtester.benchmarks import STANDARD_BENCHMARKS, BenchmarkTypes, InvalidBenchmarkError, get_benchmark_type
from portwine.logging import Logger

import pandas_market_calendars as mcal
from portwine.loaders.base import MarketDataLoader

from pandas_market_calendars import MarketCalendar
import datetime

from portwine.data.interface import DataInterface, RestrictedDataInterface, MultiDataInterface
from portwine.strategies.base import StrategyBase

# Optional Numba import for JIT compilation
from numba import jit


class DailyMarketCalendar:
    def __init__(self, calendar_name):
        self.calendar = MarketCalendar.factory(calendar_name)

    def schedule(self, start_date, end_date):
        """Expose the schedule method from the underlying calendar"""
        return self.calendar.schedule(start_date=start_date, end_date=end_date)
    
    def validate_dates(self, start_date: str, end_date: Union[str, None]) -> bool:
        assert isinstance(start_date, str), "Start date is required in string format YYYY-MM-DD."

        # Cast to datetime objects
        start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')

        if end_date is None:
            end_date_obj = datetime.datetime.now()
        else:
            end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        assert end_date_obj > start_date_obj, "End date must be after start date."

        return True
    
    def get_datetime_index(self, start_date: str, end_date: Union[str, None]=None):
        self.validate_dates(start_date, end_date)

        # Use today's date if end_date is None
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')

        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        dt_index = schedule['market_open'].index

        dt_localized = dt_index.tz_localize("UTC")
        dt_converted = dt_localized.tz_convert(None)

        return dt_converted.to_numpy()
    
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
        self.close_array[i, :] = np.array([
            data_interface[ticker]['close'] if data_interface[ticker] is not None else 0.0 
            for ticker in self.all_tickers
        ])

    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_returns(close_array: np.ndarray, ret_array: np.ndarray):
        """Numba-optimized returns calculation."""
        n_days, n_tickers = close_array.shape
        
        for i in range(1, n_days):  # Skip first day (no previous data)
            for j in range(n_tickers):
                prev_close = close_array[i-1, j]
                curr_close = close_array[i, j]
                
                if prev_close > 0:
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
        shifted_signals[0, :] = 0.0  # First day has no previous signals
        self._calculate_strategy_returns(shifted_signals, self.ret_array, self.strategy_returns)
    
    def get_results(self):
        """Return the final DataFrames."""
        sig_df = pd.DataFrame(self.sig_array, index=self.datetime_index, columns=self.all_tickers)
        sig_df.index.name = None
        
        ret_df = pd.DataFrame(self.ret_array, index=self.datetime_index, columns=self.all_tickers)
        ret_df.index.name = None
        
        strategy_ret_df = pd.DataFrame(self.strategy_returns, index=self.datetime_index, columns=['strategy_returns'])
        strategy_ret_df.index.name = None
        
        return sig_df, ret_df, strategy_ret_df


class NewBacktester:
    def __init__(self, data: DataInterface, calendar: DailyMarketCalendar):
        self.data = data
        # Pass all loaders to RestrictedDataInterface if data is MultiDataInterface
        if isinstance(data, MultiDataInterface):
            # MultiDataInterface case
            self.restricted_data = RestrictedDataInterface(data.loaders)
        else:
            # DataInterface case
            self.restricted_data = RestrictedDataInterface({None: data.data_loader})
        self.calendar = calendar

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

    def run_backtest(self, strategy: StrategyBase, start_date: Union[str, None]=None, end_date: Union[str, None]=None, benchmark: Union[str, Callable, None] = "equal_weight"):
        datetime_index = self.calendar.get_datetime_index(start_date, end_date)

        if len(datetime_index) == 0:
            raise ValueError("No trading days found in the specified date range")

        # Validate that strategy has tickers
        if not strategy.universe.all_tickers:
            raise ValueError("Strategy has no tickers. Cannot run backtest with empty universe.")

        self.validate_data(strategy.universe.all_tickers, start_date, end_date)

        # Classify benchmark type
        bm_type = get_benchmark_type(benchmark, self.data.data_loader)
        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark.")

        # Initialize BacktestResult to handle data collection
        all_tickers = sorted(strategy.universe.all_tickers)
        result = BacktestResult(datetime_index, all_tickers)
        
        for i, dt in enumerate(datetime_index):
            strategy.universe.set_datetime(dt)
            current_universe_tickers = strategy.universe.get_constituents(dt)

            # Create a RestrictedDataInterface for the strategy
            self.restricted_data.set_current_timestamp(dt)
            self.restricted_data.set_restricted_tickers(current_universe_tickers)

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
            # Set the current timestamp to get benchmark data
            self.data.set_current_timestamp(dt)
            
            try:
                # Get benchmark data for this timestamp using the DataInterface
                benchmark_data = self.data[benchmark_ticker]
                benchmark_returns.append(benchmark_data['close'])
            except (KeyError, ValueError):
                # If benchmark data is not available, use 0 return
                benchmark_returns.append(0.0)
        
        # Convert to pandas Series and calculate returns
        benchmark_prices = pd.Series(benchmark_returns, index=ret_df_index)
        benchmark_returns_series = benchmark_prices.pct_change(fill_method=None).fillna(0.0)
        
        return benchmark_returns_series

class Backtester:
    """
    A step‑driven back‑tester that supports intraday bars and,
    optionally, an exchange trading calendar.
    """

    def __init__(
        self,
        market_data_loader: MarketDataLoader,
        alternative_data_loader=None,
        calendar: Optional[Union[str, DailyMarketCalendar]] = 'NYSE',
        logger: Optional[_logging.Logger] = None,  # pre-configured logger or default
        log: bool = False,  # enable backtester logging if True
    ):
        self.market_data_loader      = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        if isinstance(calendar, str):
            self.calendar = mcal.get_calendar(calendar)
        else:
            self.calendar = calendar
        # --- configure logging for backtester ---
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger.create(__name__, level=_logging.INFO)
            # enable or disable logging based on simple flag
            self.logger.disabled = not log

    def run_backtest(
        self,
        strategy,
        shift_signals: bool = True,
        benchmark: Union[str, Callable, None] = "equal_weight",
        start_date=None,
        end_date=None,
        require_all_history: bool = False,
        require_all_tickers: bool = False,
        verbose: bool = False
    ) -> Optional[Dict[str, pd.DataFrame]]:
        # adjust logging level based on verbosity
        self.logger.setLevel(_logging.DEBUG if verbose else _logging.INFO)
        
        self.logger.info(
            "Starting backtest: tickers=%s, start_date=%s, end_date=%s",
            strategy.universe.all_tickers, start_date, end_date,
        )
        
        # 1) normalize date filters
        sd = pd.Timestamp(start_date) if start_date is not None else None
        ed = pd.Timestamp(end_date)   if end_date   is not None else None
        if sd is not None and ed is not None and sd > ed:
            raise ValueError("start_date must be on or before end_date")

        # 2) split tickers - use all possible tickers from universe
        all_tickers = strategy.universe.all_tickers
        reg_tkrs, alt_tkrs = _split_tickers(all_tickers)
            
        self.logger.debug(
            "Split tickers: %d regular, %d alternative", len(reg_tkrs), len(alt_tkrs)
        )

        # 3) classify benchmark
        bm_type = get_benchmark_type(benchmark, self.market_data_loader)
        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark.")

        # 4) load regular data - load ALL possible tickers for universe filtering
        reg_data = self.market_data_loader.fetch_data(reg_tkrs)
        self.logger.debug(
            "Fetched market data for %d tickers", len(reg_data)
        )
        # identify any tickers for which we got no data
        missing = [t for t in reg_tkrs if t not in reg_data]
        if missing:
            msg = (
                f"Market data loader returned data for {len(reg_data)}/"
                f"{len(reg_tkrs)} requested tickers. Missing: {missing}"
            )
            if require_all_tickers:
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                self.logger.warning(msg)
        # only keep tickers that have data
        reg_tkrs = [t for t in reg_tkrs if t in reg_data]

        # 5) preload benchmark ticker if needed (for require_all_history and later returns)
        if bm_type == BenchmarkTypes.TICKER:
            bm_data = self.market_data_loader.fetch_data([benchmark])

        # 6) build trading dates
        if self.calendar is not None:
            # data span
            first_dt = min(df.index.min() for df in reg_data.values())
            last_dt  = max(df.index.max() for df in reg_data.values())

            # schedule uses dates only
            sched = self.calendar.schedule(
                start_date=first_dt.date(),
                end_date=last_dt.date()
            )
            closes = sched["market_close"]

            # drop tz if present
            if getattr(getattr(closes, "dt", None), "tz", None) is not None:
                closes = closes.dt.tz_convert(None)

            # restrict to actual data
            closes = closes[(closes >= first_dt) & (closes <= last_dt)]

            # require full history across tickers and benchmark if ticker
            if require_all_history:
                # collect earliest available dates
                idx_mins = [df.index.min() for df in reg_data.values()]
                if bm_type == BenchmarkTypes.TICKER and benchmark in bm_data:
                    idx_mins.append(bm_data[benchmark]["close"].index.min())
                if idx_mins:
                    common = max(idx_mins)
                    closes = closes[closes >= common]

            # apply start/end (full timestamp)
            if sd is not None:
                closes = closes[closes >= sd]
            if ed is not None:
                closes = closes[closes <= ed]

            all_ts = list(closes)

        # 5) build trading dates
        if self.calendar is not None:
            # data span
            first_dt = min(df.index.min() for df in reg_data.values())
            last_dt  = max(df.index.max() for df in reg_data.values())

            # schedule uses dates only
            sched = self.calendar.schedule(
                start_date=first_dt.date(),
                end_date=last_dt.date()
            )
            closes = sched["market_close"]

            # drop tz if present
            if getattr(getattr(closes, "dt", None), "tz", None) is not None:
                closes = closes.dt.tz_convert(None)

            # restrict to actual data
            closes = closes[(closes >= first_dt) & (closes <= last_dt)]

            # require history
            if require_all_history and reg_tkrs:
                common = max(df.index.min() for df in reg_data.values())
                closes = closes[closes >= common]

            # apply start/end (full timestamp)
            if sd is not None:
                closes = closes[closes >= sd]
            if ed is not None:
                closes = closes[closes <= ed]

            all_ts = list(closes)

            # **raise** on empty calendar range
            if not all_ts:
                raise ValueError("No trading dates after filtering")

        else:
            # legacy union of data indices
            if hasattr(self.market_data_loader, "get_all_dates"):
                all_ts = self.market_data_loader.get_all_dates(reg_tkrs)
            else:
                all_ts = sorted({ts for df in reg_data.values() for ts in df.index})

            # require full history across tickers and benchmark if ticker
            if require_all_history:
                idx_mins = [df.index.min() for df in reg_data.values()]
                if bm_type == BenchmarkTypes.TICKER and benchmark in bm_data:
                    idx_mins.append(bm_data[benchmark]["close"].index.min())
                if idx_mins:
                    common = max(idx_mins)
                    all_ts = [d for d in all_ts if d >= common]

            # apply start/end
            if sd is not None:
                all_ts = [d for d in all_ts if d >= sd]
            if ed is not None:
                all_ts = [d for d in all_ts if d <= ed]

            if not all_ts:
                raise ValueError("No trading dates after filtering")

        # 7) main loop: signals
        sig_rows = []
        self.logger.debug(
            "Processing %d backtest steps", len(all_ts)
        )
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts

        for ts in iterator:
            # Convert pandas Timestamp to numpy datetime64
            ts_np = np.datetime64(ts)
            # set universe to current timestamp for dynamic tickers
            strategy.universe.set_datetime(ts_np)
            # Get current universe tickers
            current_universe_tickers = strategy.universe.get_constituents(ts_np)
            
            if hasattr(self.market_data_loader, "next"):
                bar = self.market_data_loader.next(current_universe_tickers, ts)
            else:
                # Filter the data to only include current universe tickers
                filtered_reg_data = {t: reg_data[t] for t in current_universe_tickers if t in reg_data}
                bar = self._bar_dict(ts, filtered_reg_data)

            if self.alternative_data_loader:
                alt_ld = self.alternative_data_loader
                if hasattr(alt_ld, "next"):
                    bar.update(alt_ld.next(alt_tkrs, ts))
                else:
                    for t, df in alt_ld.fetch_data(alt_tkrs).items():
                        bar[t] = self._bar_dict(ts, {t: df})[t]

            sig = strategy.step(ts, bar)
            # Check for over-allocation: total weights >1
            total_weight = sum(sig.values())
            # Allow for minor floating-point rounding errors
            if total_weight > 1.0 + 1e-8:
                raise ValueError(f"Total allocation {total_weight:.6f} exceeds 1.0 at {ts}")
            
            # Validate that strategy only assigns weights to tickers in the current universe
            current_universe_tickers = strategy.universe.get_constituents(ts_np)
            invalid_tickers = [t for t in sig.keys() if t not in current_universe_tickers]
            if invalid_tickers:
                raise ValueError(
                    f"Strategy assigned weights to tickers not in current universe at {ts}: {invalid_tickers}. "
                    f"Current universe: {current_universe_tickers}"
                )
            
            row = {"date": ts}
            # for t in strategy.tickers:
            for t in reg_tkrs:
                row[t] = sig.get(t, 0.0)
            sig_rows.append(row)

        # 8) construct signals_df
        sig_df = pd.DataFrame(sig_rows).set_index("date").sort_index()
        sig_df.index.name = None
        sig_reg = ((sig_df.shift(1).ffill() if shift_signals else sig_df)
                   .fillna(0.0)[reg_tkrs])

        # 9) compute returns
        px     = pd.DataFrame({t: reg_data[t]["close"] for t in reg_tkrs})
        px     = px.reindex(sig_reg.index).ffill()
        ret_df = px.pct_change(fill_method=None).fillna(0.0)
        strat_ret = (ret_df * sig_reg).sum(axis=1)

        # 10) benchmark returns
        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            bm_ret = benchmark(ret_df)
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            bm_ret = STANDARD_BENCHMARKS[benchmark](ret_df)
        else:  # TICKER
            ser = bm_data[benchmark]["close"].reindex(sig_reg.index).ffill()
            bm_ret = ser.pct_change(fill_method=None).fillna(0.0)

        # 11) dynamic alternative data update
        if self.alternative_data_loader and hasattr(self.alternative_data_loader, "update"):
            for ts in sig_reg.index:
                raw_sigs = sig_df.loc[ts].to_dict()
                raw_rets = ret_df.loc[ts].to_dict()
                self.alternative_data_loader.update(ts, raw_sigs, raw_rets, float(strat_ret.loc[ts]))

        # log completion
        self.logger.info(
            "Backtest complete: processed %d timestamps", len(all_ts)
        )
        return {
            "signals_df":        sig_reg,
            "tickers_returns":   ret_df,
            "strategy_returns":  strat_ret,
            "benchmark_returns": bm_ret
        }

    @staticmethod
    def _bar_dict(ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict | None]:
        out: Dict[str, dict | None] = {}
        for t, df in data.items():
            if ts in df.index:
                row = df.loc[ts]
                out[t] = {
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                }
            else:
                out[t] = None
        return out
