"""
Vectorized strategy base class and updated backtester implementation.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from portwine.backtester.benchmarks import STANDARD_BENCHMARKS
from portwine.strategies import StrategyBase
from portwine.backtester import Backtester
from typing import Dict, List, Optional, Tuple, Set, Union
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def calculate_returns(price_array):
    """Calculate returns with Numba acceleration."""
    n_rows, n_cols = price_array.shape
    returns = np.zeros((n_rows-1, n_cols), dtype=np.float32)
    for i in range(1, n_rows):
        for j in range(n_cols):
            returns[i-1, j] = price_array[i, j] / price_array[i-1, j] - 1.0
    return returns

@nb.njit(parallel=True, fastmath=True)
def apply_weights(returns, weights):
    """Calculate weighted returns with Numba acceleration."""
    n_rows, n_cols = returns.shape
    result = np.zeros(n_rows)
    for i in range(n_rows):
        for j in range(n_cols):
            result[i] += returns[i, j] * weights[i, j]
    return result


def create_price_dataframe(market_data_loader, tickers, start_date=None, end_date=None):
    """
    Create a DataFrame of prices from a market data loader.

    Parameters:
    -----------
    market_data_loader : MarketDataLoader
        Market data loader from portwine
    tickers : list[str]
        List of ticker symbols to include
    start_date : datetime or str, optional
        Start date for data extraction
    end_date : datetime or str, optional
        End date for data extraction

    Returns:
    --------
    DataFrame
        DataFrame with dates as index and tickers as float columns
    """
    # 1) fetch raw data
    data_dict = market_data_loader.fetch_data(tickers)

    # 2) collect all dates
    all_dates = set()
    for df in data_dict.values():
        if df is not None and not df.empty:
            all_dates.update(df.index)
    all_dates = sorted(all_dates)

    # 3) apply optional date filters
    # all_dates_array = np.array(all_dates)
    all_dates_array = pd.to_datetime(np.array(all_dates))

    mask = np.ones(len(all_dates_array), dtype=bool)  # Start with all True
    

    # if start_date:
    #     sd = pd.to_datetime(start_date).to_numpy()
    #     mask &= (all_dates_array >= sd)
        
    # if end_date:
    #     ed = pd.to_datetime(end_date).to_numpy()
    #     mask &= (all_dates_array <= ed)

    if start_date:
        sd = pd.to_datetime(start_date)
        mask &= (all_dates_array >= sd)
    if end_date:
        ed = pd.to_datetime(end_date)
        mask &= (all_dates_array <= ed)

    # Apply the combined mask
    all_dates = all_dates_array[mask].tolist()

    # 4) build empty float DataFrame
    df_prices = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)

    # 5) fill in the close prices (alignment by index)
    for ticker, df in data_dict.items():
        if df is not None and not df.empty and ticker in tickers:
            df_prices[ticker] = df['close']

    # 6) forward‐fill across a float‐typed DataFrame → no downcasting warning
    df_prices = df_prices.ffill()

    # 7) drop any dates where we have no data at all
    df_prices = df_prices.dropna(how='all')

    return df_prices


class VectorizedStrategyBase(StrategyBase):
    """
    Base class for vectorized strategies that process the entire dataset at once.
    Subclasses must implement batch() to return a float‐typed weights DataFrame.
    """
    def __init__(self, tickers):
        # Store original tickers list for deterministic ordering
        self._initial_tickers = tickers
        # Initialize static universe via base class
        super().__init__(tickers)
        self.prices_df = None
        self.weights_df = None
    
    @property
    def tickers(self) -> List[str]:
        """Return the original tickers list for this vectorized strategy."""
        return self._initial_tickers

    def batch(self, prices_df):
        raise NotImplementedError("Subclasses must implement batch()")

    def step(self, current_date, daily_data):
        if self.weights_df is None or current_date not in self.weights_df.index:
            # fallback to equal weight
            return {t: 1.0 / len(self.tickers) for t in self.tickers}
        row = self.weights_df.loc[current_date]
        return {t: float(w) for t, w in row.items()}


class VectorizedBacktester:
    """
    A vectorized backtester that processes the entire dataset at once.
    """
    def __init__(self, market_data_loader=None):
        self.market_data_loader = market_data_loader


    def run_backtest(
        self,
        strategy: VectorizedStrategyBase,
        benchmark="equal_weight",
        start_date=None,
        end_date=None,
        shift_signals=True,
        require_all_history: bool = False,
        verbose=False
    ):
        # 0) type check
        if not isinstance(strategy, VectorizedStrategyBase):
            raise TypeError("Strategy must be a VectorizedStrategyBase")

        # 1) load full history of prices (float dtype)
        full_prices = create_price_dataframe(
            self.market_data_loader,
            tickers=strategy.tickers,
            start_date=start_date,
            end_date=end_date
        )

        # 2) compute all weights in one shot
        if verbose:
            print("Computing strategy weights…")
        all_weights = strategy.batch(full_prices)

        # 3) require that all tickers have data from a common start date?
        if require_all_history:
            # find first valid (non-NaN) date for each ticker
            first_dates = [full_prices[t].first_valid_index() for t in strategy.tickers]
            if any(d is None for d in first_dates):
                raise ValueError("Not all tickers have any data in the supplied range")
            common_start = max(first_dates)
            # truncate both prices and weights
            full_prices = full_prices.loc[full_prices.index >= common_start]
            all_weights = all_weights.loc[all_weights.index >= common_start]

        # 4) align dates between prices and weights
        common_idx = full_prices.index.intersection(all_weights.index)
        price_df = full_prices.loc[common_idx]
        weights_df = all_weights.loc[common_idx]

        # 5) shift signals if requested
        if shift_signals:
            weights_df = weights_df.shift(1).fillna(0.0)

        # 6) compute per‐ticker returns
        returns_df = price_df.pct_change(fill_method=None).fillna(0.0)

        # 7) strategy P&L
        strategy_rets = (returns_df * weights_df).sum(axis=1)

        # 8) benchmark
        benchmark_rets = None
        if benchmark is not None:
            if isinstance(benchmark, str) and benchmark in STANDARD_BENCHMARKS:
                benchmark_rets = STANDARD_BENCHMARKS[benchmark](returns_df)
            elif isinstance(benchmark, str) and self.market_data_loader:
                raw = self.market_data_loader.fetch_data([benchmark])
                series = raw.get(benchmark)
                if series is not None:
                    bm = series['close'].reindex(common_idx).ffill()
                    benchmark_rets = bm.pct_change(fill_method=None).fillna(0)
                    benchmark_rets.name = None  # Reset the name
            elif callable(benchmark):
                benchmark_rets = benchmark(returns_df)

        return {
            'signals_df': weights_df,
            'tickers_returns': returns_df,
            'strategy_returns': strategy_rets,
            'benchmark_returns': benchmark_rets,
        }


def benchmark_equal_weight(returns_df: pd.DataFrame) -> pd.Series:
    return returns_df.mean(axis=1)



def load_price_matrix(loader, tickers, start_date=None, end_date=None):
    """
    Optimized price matrix loader that minimizes pandas-numpy conversions.
    """
    # 1) Fetch raw data
    data_dict = loader.fetch_data(tickers)

    # 2) Collect all dates directly as numpy array
    all_dates_set = set()
    for df in data_dict.values():
        if df is not None and not df.empty:
            all_dates_set.update(df.index.values)

    all_dates_array = np.array(sorted(all_dates_set))

    # 3) Apply date filters in numpy
    mask = np.ones(len(all_dates_array), dtype=bool)
    if start_date:
        sd = pd.to_datetime(start_date)
        if len(all_dates_array) > 0:
            mask &= (all_dates_array >= sd)
    if end_date:
        ed = pd.to_datetime(end_date)
        if len(all_dates_array) > 0:
            mask &= (all_dates_array <= ed)

    all_dates_array = all_dates_array[mask]

    # Create a date-to-index mapping for fast lookups
    date_to_idx = {d: i for i, d in enumerate(all_dates_array)}

    # 4) Pre-allocate price matrix directly
    n_dates = len(all_dates_array)
    n_tickers = len(tickers)
    price_matrix = np.full((n_dates, n_tickers), np.nan, dtype=np.float32)

    # 5) Fill matrix directly without pandas intermediates
    for t_idx, ticker in enumerate(tickers):
        df = data_dict.get(ticker)
        if df is not None and not df.empty:
            # Get close prices as numpy array
            prices = df['close'].values

            # Get dates as numpy array
            dates = df.index.values

            # For each date in this ticker's data, find its position in our matrix
            for date_idx, date in enumerate(dates):
                if date in date_to_idx:
                    price_matrix[date_to_idx[date], t_idx] = prices[date_idx]

    # 6) Forward fill using numpy operations
    for col in range(n_tickers):
        mask = np.isnan(price_matrix[:, col])
        # Find first valid index
        valid_indices = np.where(~mask)[0]
        if len(valid_indices) > 0:
            # Forward fill
            for i in range(valid_indices[0], n_dates):
                if mask[i]:
                    # Find the last valid value
                    last_valid = np.where(~mask[:i])[0]
                    if len(last_valid) > 0:
                        price_matrix[i, col] = price_matrix[last_valid[-1], col]

    # 7) Compute returns directly in numpy
    returns_matrix = np.zeros_like(price_matrix[1:])
    returns_matrix = (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1]

    # 8) Create a minimal pandas DataFrame only for reference
    # This is just for API compatibility and doesn't get used in computations
    price_df = pd.DataFrame(price_matrix, index=all_dates_array, columns=tickers)

    return price_matrix, returns_matrix, all_dates_array[1:], price_df
class NumPyVectorizedStrategyBase(StrategyBase):
    """
    Base class for vectorized strategies that process the entire dataset at once
    using NumPy arrays for optimal performance.
    """
    def __init__(self, tickers: List[str]):
        """
        Initialize with the tickers this strategy uses.
    
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols this strategy will use
        """
        # Store original tickers list for deterministic ordering
        self._initial_tickers = tickers
        # Delegate universe setup to base class
        super().__init__(tickers)

    @property
    def tickers(self) -> List[str]:
        """Override to return the original tickers list."""
        return self._initial_tickers
    
    def batch(self, price_matrix: np.ndarray, dates: List[pd.Timestamp], 
              column_indices: List[int]) -> np.ndarray:
        """
        Compute weights for all dates based on price history.
        Must be implemented by subclasses.
        
        Parameters:
        -----------
        price_matrix : np.ndarray
            Price matrix with shape (n_dates, n_tickers)
        dates : List[pd.Timestamp]
            List of dates corresponding to rows in price_matrix
        column_indices : List[int]
            List of column indices in price_matrix that correspond to this strategy's tickers
            
        Returns:
        --------
        np.ndarray
            Weight matrix with shape (n_dates, n_strategy_tickers)
        """
        raise NotImplementedError("Subclasses must implement batch()")
    
    def step(self, current_date, daily_data):
        """
        Compatibility method for use with traditional backtester.
        This should generally not be used directly - prefer batch processing.
        """
        raise NotImplementedError("NumPyVectorizedStrategyBase is designed for batch processing")


class NumpyVectorizedBacktester:
    """
    A highly optimized NumPy-based vectorized backtester that supports
    strategies using subsets of tickers.
    """
    def __init__(self, loader, universe_tickers: List[str], start_date: str, end_date: str):
        """
        Initialize with minimal pandas-numpy conversions.
        """
        price_matrix, returns_matrix, dates_ret, price_df = load_price_matrix(
            loader, universe_tickers, start_date, end_date
        )
        
        # Store everything as numpy arrays
        self.price_matrix = price_matrix
        self.returns_matrix = returns_matrix
        self.dates_array = dates_ret  # store as numpy array
        
        # Create mappings for lookups
        self.universe_tickers = universe_tickers
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(universe_tickers)}
        
        # Keep minimal pandas objects
        self.date_to_i = None  # Don't create this dictionary unless needed
        self.price_df = None   # Don't store pandas objects
        
        # Keep reference to loader for benchmark calculations
        self.loader = loader

    def get_indices_for_tickers(self, tickers: List[str]) -> List[int]:
        """
        Get the column indices in the price/returns matrices for the given tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to get indices for
            
        Returns:
        --------
        List[int]
            List of column indices
        """
        return [self.ticker_to_idx.get(ticker) for ticker in tickers 
                if ticker in self.ticker_to_idx]

    def run_backtest(self, strategy, benchmark="equal_weight", shift_signals=True, verbose=False, **kwargs):
        """
        Run backtest with minimal pandas-numpy conversions.
        """
        # Get strategy info
        strategy_indices = np.array([
            self.ticker_to_idx.get(t, -1) for t in strategy.tickers
        ])
        strategy_indices = strategy_indices[strategy_indices >= 0]
        
        if len(strategy_indices) == 0:
            raise ValueError(f"None of the strategy tickers {strategy.tickers} are in the universe")
        
        # Extract price submatrix - avoid pandas operations
        strategy_price_matrix = self.price_matrix[:, strategy_indices]
        
        # Get weights - working entirely in numpy
        if verbose:
            print(f"Computing weights for {len(strategy_indices)} tickers...")
        
        # Get strategy weights as numpy array
        weights_matrix = strategy.batch(
            strategy_price_matrix, 
            self.dates_array, 
            strategy_indices
        )
        
        # Prepare benchmark weights - all in numpy
        benchmark_weights = None
        if isinstance(benchmark, str) and benchmark == "equal_weight":
            benchmark_weights = np.ones(len(strategy_indices), dtype=np.float32) / len(strategy_indices)
        elif isinstance(benchmark, list):
            # Convert benchmark tickers to indices
            benchmark_indices = np.array([
                self.ticker_to_idx.get(t, -1) for t in benchmark
            ])
            benchmark_indices = benchmark_indices[benchmark_indices >= 0]
            
            if len(benchmark_indices) == 0:
                raise ValueError(f"None of the benchmark tickers {benchmark} are in the universe")
            
            # Create benchmark weights array
            benchmark_weights = np.zeros(len(strategy_indices), dtype=np.float32)
            
            # Map universe indices to strategy indices
            for b_idx in benchmark_indices:
                if b_idx in strategy_indices:
                    s_idx = np.where(strategy_indices == b_idx)[0][0]
                    benchmark_weights[s_idx] = 1.0
            
            # Normalize
            if np.sum(benchmark_weights) > 0:
                benchmark_weights /= np.sum(benchmark_weights)
        elif isinstance(benchmark, np.ndarray):
            if len(benchmark) != len(strategy_indices):
                raise ValueError(
                    f"Benchmark weights has {len(benchmark)} elements, "
                    f"but strategy has {len(strategy_indices)} tickers"
                )
            benchmark_weights = benchmark.astype(np.float32)
        
        # Extract returns submatrix - avoid pandas operations
        strategy_returns_matrix = self.returns_matrix[:, strategy_indices]
        
        # Calculate returns using numpy operations
        result_dict = self.run_backtest_npy(
            returns_matrix=strategy_returns_matrix,
            weights_matrix=weights_matrix,
            benchmark_weights=benchmark_weights,
            shift_signals=shift_signals
        )
        
        # Only convert to pandas at the very end
        strategy_ticker_list = [self.universe_tickers[i] for i in strategy_indices]
        
        # Create minimal pandas output - only at the end
        return {
            'signals_df': pd.DataFrame(
                weights_matrix, 
                index=pd.DatetimeIndex(self.dates_array), 
                columns=strategy_ticker_list
            ),
            'tickers_returns': pd.DataFrame(
                strategy_returns_matrix,
                index=pd.DatetimeIndex(self.dates_array),
                columns=strategy_ticker_list
            ),
            'strategy_returns': pd.Series(
                result_dict["strategy_returns"],
                index=pd.DatetimeIndex(self.dates_array)
            ),
            'benchmark_returns': pd.Series(
                result_dict["benchmark_returns"],
                index=pd.DatetimeIndex(self.dates_array)
            ) if result_dict["benchmark_returns"] is not None else None,
        }
    
    def run_backtest_npy(self, returns_matrix, weights_matrix, benchmark_weights=None, shift_signals=True):
        """
        Pure numpy implementation of backtest calculations.
        """
        # Shift signals if needed - all numpy operations
        if shift_signals:
            W = np.zeros_like(weights_matrix)
            if weights_matrix.shape[0] > 1:
                W[1:] = weights_matrix[:-1]
        else:
            W = weights_matrix
            
        # Calculate strategy returns - treat NaNs in returns as 0 to match Backtester semantics
        if W.shape[1] > 0:
            returns_no_nan = np.where(np.isnan(returns_matrix), 0.0, returns_matrix)
            strat_rets = np.sum(returns_no_nan * W, axis=1)
        else:
            strat_rets = np.zeros(returns_matrix.shape[0], dtype=np.float32)
        
        # Calculate benchmark returns if needed
        if benchmark_weights is not None:
            # Use matrix multiplication for benchmark
            bench_rets = returns_matrix @ benchmark_weights
        else:
            bench_rets = np.zeros_like(strat_rets)
            
        return {
            "strategy_returns": strat_rets,
            "benchmark_returns": bench_rets
        }

class SubsetStrategy(NumPyVectorizedStrategyBase):
    """Strategy that only uses a subset of tickers."""
    def __init__(self, tickers: List[str], weight_type='equal'):
        super().__init__(tickers)
        self.weight_type = weight_type
    
    def batch(self, price_matrix: np.ndarray, dates: List[pd.Timestamp], 
              column_indices: List[int]) -> np.ndarray:
        """Returns equal weights for all tickers in the subset."""
        n_dates, n_tickers = price_matrix.shape
        
        if self.weight_type == 'equal':
            # Equal weight for all tickers
            weights = np.ones((n_dates, n_tickers)) / n_tickers
            # Return weights for dates[1:] to match returns dimension
            return weights[1:]
        elif self.weight_type == 'momentum':
            # Simple momentum strategy
            returns = np.zeros((n_dates, n_tickers))
            lookback = 20  # 20-day momentum
            
            # Calculate returns over lookback period
            for i in range(lookback, n_dates):
                returns[i] = price_matrix[i] / price_matrix[i-lookback] - 1
            
            # Set weights based on positive momentum
            weights = np.zeros((n_dates, n_tickers))
            weights[returns > 0] = 1
            
            # Normalize weights
            row_sums = np.sum(weights, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            weights = weights / row_sums
            
            # Return weights for dates[1:] to match returns dimension
            return weights[1:]