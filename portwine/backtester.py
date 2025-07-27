# portwine/backtester.py

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging as _logging
from tqdm import tqdm
from portwine.logging import Logger

import pandas_market_calendars as mcal
from portwine.loaders.base import MarketDataLoader

class InvalidBenchmarkError(Exception):
    """Raised when the requested benchmark is neither a standard name nor a valid ticker."""
    pass

# ----------------------------------------------------------------------
# Built‑in benchmark helpers
# ----------------------------------------------------------------------
def benchmark_equal_weight(ret_df: pd.DataFrame, *_, **__) -> pd.Series:
    return ret_df.mean(axis=1)

def benchmark_markowitz(
    ret_df: pd.DataFrame,
    lookback: int = 60,
    shift_signals: bool = True,
    verbose: bool = False,
) -> pd.Series:
    tickers = ret_df.columns
    n = len(tickers)
    iterator = tqdm(ret_df.index, desc="Markowitz") if verbose else ret_df.index
    w_rows: List[np.ndarray] = []
    for ts in iterator:
        win = ret_df.loc[:ts].tail(lookback)
        if len(win) < 2:
            w = np.ones(n) / n
        else:
            cov = win.cov().values
            w_var = cp.Variable(n, nonneg=True)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w_var, cov)), [cp.sum(w_var) == 1])
            try:
                prob.solve()
                w = w_var.value if w_var.value is not None else np.ones(n) / n
            except Exception:
                w = np.ones(n) / n
        w_rows.append(w)
    w_df = pd.DataFrame(w_rows, index=ret_df.index, columns=tickers)
    if shift_signals:
        w_df = w_df.shift(1).ffill().fillna(1.0 / n)
    return (w_df * ret_df).sum(axis=1)

STANDARD_BENCHMARKS: Dict[str, Callable] = {
    "equal_weight": benchmark_equal_weight,
    "markowitz":    benchmark_markowitz,
}

class BenchmarkTypes:
    STANDARD_BENCHMARK = 0
    TICKER             = 1
    CUSTOM_METHOD      = 2
    INVALID            = 3

# ------------------------------------------------------------------------------
# Optimized Backtester
# ------------------------------------------------------------------------------
class Backtester:
    """
    An optimized step‑driven back‑tester that addresses performance bottlenecks
    identified through profiling.
    
    Key optimizations:
    1. Pre-computed data access patterns
    2. Vectorized operations where possible
    3. Reduced pandas indexing overhead
    4. Memory-efficient data structures
    """

    def __init__(
        self,
        market_data_loader: MarketDataLoader,
        alternative_data_loader=None,
        calendar: Optional[Union[str, mcal.ExchangeCalendar]] = None,
        logger: Optional[_logging.Logger] = None,
        log: bool = False,
    ):
        self.market_data_loader      = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        if isinstance(calendar, str):
            self.calendar = mcal.get_calendar(calendar)
        else:
            self.calendar = calendar
        
        # Configure logging
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger.create(__name__, level=_logging.INFO)
            self.logger.disabled = not log
        
        # Performance optimization: cache for data access
        self._data_cache = {}
        self._price_cache = {}
        self._date_index_cache = {}

    def _split_tickers(self, tickers: set) -> Tuple[List[str], List[str]]:
        """Split tickers into regular and alternative data tickers."""
        reg, alt = [], []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def _precompute_data_access(self, reg_data: Dict[str, pd.DataFrame], all_ts: List[pd.Timestamp]) -> Dict:
        """
        Pre-compute data access patterns to avoid repeated pandas indexing.
        This is the key optimization that addresses the major bottleneck.
        """
        self.logger.debug("Pre-computing data access patterns...")
        
        # Create fast lookup structures
        data_cache = {}
        price_cache = {}
        date_index_cache = {}
        
        for ticker, df in reg_data.items():
            # Create fast date-to-index mapping
            date_to_idx = {date: idx for idx, date in enumerate(df.index)}
            date_index_cache[ticker] = date_to_idx
            
            # Pre-extract price data as numpy arrays for faster access
            price_cache[ticker] = {
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'volume': df['volume'].values,
                'dates': df.index.values
            }
            
            # Create fast lookup for each timestamp using numpy arrays (no pandas indexing)
            data_cache[ticker] = {}
            for ts in all_ts:
                if ts in date_to_idx:
                    idx = date_to_idx[ts]
                    # Use numpy arrays directly instead of pandas .iloc[]
                    data_cache[ticker][ts] = {
                        'open': float(price_cache[ticker]['open'][idx]),
                        'high': float(price_cache[ticker]['high'][idx]),
                        'low': float(price_cache[ticker]['low'][idx]),
                        'close': float(price_cache[ticker]['close'][idx]),
                        'volume': float(price_cache[ticker]['volume'][idx]),
                    }
                else:
                    data_cache[ticker][ts] = None
        
        return {
            'data_cache': data_cache,
            'price_cache': price_cache,
            'date_index_cache': date_index_cache
        }

    def _fast_bar_dict(self, ts: pd.Timestamp, data_cache: Dict) -> Dict[str, dict | None]:
        """
        Optimized version of _bar_dict that uses pre-computed data access.
        This eliminates the pandas indexing bottleneck.
        """
        out: Dict[str, dict | None] = {}
        for ticker, ticker_cache in data_cache.items():
            out[ticker] = ticker_cache.get(ts)
        return out

    def get_benchmark_type(self, benchmark) -> int:
        if isinstance(benchmark, str):
            if benchmark in STANDARD_BENCHMARKS:
                return BenchmarkTypes.STANDARD_BENCHMARK
            if self.market_data_loader.fetch_data([benchmark]).get(benchmark) is not None:
                return BenchmarkTypes.TICKER
            return BenchmarkTypes.INVALID
        if callable(benchmark):
            return BenchmarkTypes.CUSTOM_METHOD
        return BenchmarkTypes.INVALID

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
        # Adjust logging level based on verbosity
        self.logger.setLevel(_logging.DEBUG if verbose else _logging.INFO)
        
        self.logger.info(
            "Starting optimized backtest: tickers=%s, start_date=%s, end_date=%s",
            strategy.tickers, start_date, end_date,
        )
        
        # 1) normalize date filters
        sd = pd.Timestamp(start_date) if start_date is not None else None
        ed = pd.Timestamp(end_date)   if end_date   is not None else None
        if sd is not None and ed is not None and sd > ed:
            raise ValueError("start_date must be on or before end_date")

        # 2) split tickers
        all_tickers = strategy.universe.all_tickers
        reg_tkrs, alt_tkrs = self._split_tickers(all_tickers)
            
        self.logger.debug(
            "Split tickers: %d regular, %d alternative", len(reg_tkrs), len(alt_tkrs)
        )

        # 3) classify benchmark
        bm_type = self.get_benchmark_type(benchmark)
        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark.")

        # 4) load regular data
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

        # 5) preload benchmark ticker if needed
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

        # 7) OPTIMIZATION: Pre-compute data access patterns
        access_cache = self._precompute_data_access(reg_data, all_ts)
        data_cache = access_cache['data_cache']
        price_cache = access_cache['price_cache']

        # 8) main loop: signals (optimized)
        # OPTIMIZATION: Pre-allocate arrays instead of building lists
        n_dates = len(all_ts)
        tickers_list = list(strategy.tickers)
        n_tickers = len(tickers_list)
        signals_array = np.zeros((n_dates, n_tickers), dtype=np.float64)
        dates_array = np.empty(n_dates, dtype=object)
        
        # OPTIMIZATION: Cache universe lookups to avoid repeated calls
        universe_cache = {}
        
        self.logger.debug(
            "Processing %d backtest steps", len(all_ts)
        )
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts

        for i, ts in enumerate(iterator):
            # OPTIMIZATION: Cache universe lookups
            if ts not in universe_cache:
                universe_cache[ts] = strategy.universe.get_constituents(ts)
            current_universe_tickers = universe_cache[ts]
            
            # OPTIMIZATION: Use pre-computed data access
            if hasattr(self.market_data_loader, "next"):
                bar = self.market_data_loader.next(current_universe_tickers, ts)
            else:
                # Use optimized bar_dict
                filtered_data_cache = {t: data_cache[t] for t in current_universe_tickers if t in data_cache}
                bar = self._fast_bar_dict(ts, filtered_data_cache)

            if self.alternative_data_loader:
                alt_ld = self.alternative_data_loader
                if hasattr(alt_ld, "next"):
                    bar.update(alt_ld.next(alt_tkrs, ts))
                else:
                    for t, df in alt_ld.fetch_data(alt_tkrs).items():
                        bar[t] = self._bar_dict(ts, {t: df})[t]

            sig = strategy.step(ts, bar)
            
            # Validate that strategy only assigns weights to tickers in the current universe
            # OPTIMIZATION: Reuse cached universe lookup
            invalid_tickers = [t for t in sig.keys() if t not in current_universe_tickers]
            if invalid_tickers:
                raise ValueError(
                    f"Strategy assigned weights to tickers not in current universe at {ts}: {invalid_tickers}. "
                    f"Current universe: {current_universe_tickers}"
                )
            
            # OPTIMIZATION: Direct array assignment instead of building dictionaries
            dates_array[i] = ts
            for j, ticker in enumerate(tickers_list):
                signals_array[i, j] = sig.get(ticker, 0.0)

        # 9) OPTIMIZATION: Pure numpy signal processing (replaces pandas operations)
        # Apply signal shifting using numpy operations
        if shift_signals:
            shifted_signals = np.zeros_like(signals_array)
            if signals_array.shape[0] > 1:
                shifted_signals[1:] = signals_array[:-1]  # Shift down by 1
            signals_processed = shifted_signals
        else:
            signals_processed = signals_array
        
        # Forward fill using numpy (replace .ffill()) - only for NaN values, not zeros
        # Note: 0.0 is a valid signal value meaning "no allocation", so we don't forward fill zeros
        # In our numpy array, we initialized with zeros, so no NaN handling needed here
        # The original pandas .ffill() would only forward fill NaN values
        # Since we pre-allocated with zeros and strategy.step() returns valid values,
        # no forward fill is needed for this step-based backtester
        
        # Select only reg_tkrs columns (replace pandas column selection)
        tickers_list = list(strategy.tickers)
        reg_ticker_indices = [i for i, ticker in enumerate(tickers_list) if ticker in reg_tkrs]
        sig_reg_array = signals_processed[:, reg_ticker_indices]
        
        # Create minimal DataFrame only for API compatibility at the end
        sig_reg = pd.DataFrame(
            sig_reg_array,
            index=pd.DatetimeIndex(dates_array),
            columns=[tickers_list[i] for i in reg_ticker_indices]
        )

        # 10) OPTIMIZATION: Pure numpy price processing and returns calculation
        self.logger.debug("Computing returns using pure numpy operations...")
        
        # Build price matrix using numpy operations
        n_dates = len(dates_array)
        n_reg_tickers = len(reg_tkrs)
        price_matrix = np.full((n_dates, n_reg_tickers), np.nan, dtype=np.float64)
        
        # Create date lookup for alignment
        date_to_idx = {date: i for i, date in enumerate(dates_array)}
        
        # Fill price matrix for each ticker
        for ticker_idx, ticker in enumerate(reg_tkrs):
            if ticker in price_cache:
                ticker_dates = price_cache[ticker]['dates']
                ticker_prices = price_cache[ticker]['close']
                
                # Align prices with our date array
                for date_idx, date in enumerate(ticker_dates):
                    if date in date_to_idx:
                        price_matrix[date_to_idx[date], ticker_idx] = ticker_prices[date_idx]
        
        # Forward fill missing prices using numpy
        for col in range(n_reg_tickers):
            mask = np.isnan(price_matrix[:, col])
            valid_indices = np.where(~mask)[0]
            if len(valid_indices) > 0:
                # Forward fill from first valid value
                for i in range(valid_indices[0] + 1, n_dates):
                    if mask[i]:
                        # Find last valid value
                        last_valid_idx = i - 1
                        while last_valid_idx >= 0 and mask[last_valid_idx]:
                            last_valid_idx -= 1
                        if last_valid_idx >= 0:
                            price_matrix[i, col] = price_matrix[last_valid_idx, col]
        
        # Calculate returns using numpy (replaces pct_change)
        returns_matrix = np.zeros_like(price_matrix)
        returns_matrix[1:] = (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1]
        # Replace NaN/inf with 0.0
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate strategy returns using numpy (replaces pandas matrix multiplication)
        strat_ret_array = np.sum(returns_matrix * sig_reg_array, axis=1)
        
        # Create minimal pandas objects for API compatibility
        ret_df = pd.DataFrame(
            returns_matrix,
            index=pd.DatetimeIndex(dates_array),
            columns=reg_tkrs
        )
        strat_ret = pd.Series(
            strat_ret_array,
            index=pd.DatetimeIndex(dates_array)
        )

        # 11) benchmark returns
        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            bm_ret = benchmark(ret_df)
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            bm_ret = STANDARD_BENCHMARKS[benchmark](ret_df)
        else:  # TICKER
            ser = bm_data[benchmark]["close"].reindex(sig_reg.index).ffill()
            bm_ret = ser.pct_change(fill_method=None).fillna(0.0)

        # 12) dynamic alternative data update
        if self.alternative_data_loader and hasattr(self.alternative_data_loader, "update"):
            for i, ts in enumerate(sig_reg.index):
                # Create raw_sigs dict directly from numpy array
                raw_sigs = {ticker: float(signals_processed[i, j]) 
                           for j, ticker in enumerate(tickers_list)}
                
                # Create raw_rets dict directly from numpy array (ret_df still pandas for now)
                raw_rets = ret_df.loc[ts].to_dict()
                
                # Get strategy return as float (strat_ret still pandas for now)
                self.alternative_data_loader.update(ts, raw_sigs, raw_rets, float(strat_ret.loc[ts]))

        # log completion
        self.logger.info(
            "Optimized backtest complete: processed %d timestamps", len(all_ts)
        )
        return {
            "signals_df":        sig_reg,
            "tickers_returns":   ret_df,
            "strategy_returns":  strat_ret,
            "benchmark_returns": bm_ret
        }

    @staticmethod
    def _bar_dict(ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict | None]:
        """Original _bar_dict method for compatibility."""
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
