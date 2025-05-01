"""
Vectorized strategy base class and updated backtester implementation.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from portwine.strategies import StrategyBase
from portwine.backtester import Backtester, STANDARD_BENCHMARKS
from typing import Dict, List, Optional, Tuple, Set, Union

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
    if start_date:
        sd = pd.to_datetime(start_date)
        all_dates = [d for d in all_dates if d >= sd]
    if end_date:
        ed = pd.to_datetime(end_date)
        all_dates = [d for d in all_dates if d <= ed]

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
        self.tickers = tickers
        self.prices_df = None
        self.weights_df = None

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

def load_price_matrix(
    loader,
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], pd.DataFrame]:
    """
    Fetches each ticker's 'close' series, aligns on a union of dates,
    forward‐fills, and returns:
      • price_matrix    : ndarray [n_dates × n_tickers]
      • returns_matrix  : ndarray [(n_dates−1) × n_tickers]
      • dates_ret       : list of pd.Timestamp of length n_dates−1
      • price_df        : DataFrame with dates and tickers for reference
    """
    # fetch raw DataFrames
    data = loader.fetch_data(tickers)
    # collect all dates
    all_dates = sorted({d for df in data.values() if df is not None for d in df.index})
    # slice and index
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    all_dates = [d for d in all_dates if sd <= d <= ed]
    # build price matrix
    price_df = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    for t in tickers:
        df = data.get(t)
        if df is None: 
            continue
        s = df["close"].reindex(all_dates).ffill()
        price_df[t] = s
    price_matrix = price_df.values  # shape (n_dates, n_tickers)
    # compute returns
    returns_matrix = np.diff(price_matrix, axis=0) / price_matrix[:-1]
    dates_ret = all_dates[1:]
    return price_matrix, returns_matrix, dates_ret, price_df


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
        self.tickers = tickers
        
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
        Initialize backtester with price and returns matrices for all tickers in the universe.
        
        Parameters:
        -----------
        loader : MarketDataLoader
            Data loader that implements fetch_data(tickers)
        universe_tickers : List[str]
            List of all ticker symbols in the universe
        start_date : str
            Start date in string format
        end_date : str
            End date in string format
        """
        price_matrix, returns_matrix, dates_ret, price_df = load_price_matrix(
            loader, universe_tickers, start_date, end_date
        )
        
        # Store as instance variables
        self.price_matrix = price_matrix
        self.returns_matrix = returns_matrix
        self.dates = dates_ret  # Already correct length for returns
        
        # Create date lookup for efficient indexing
        self.date_to_i = {d: i for i, d in enumerate(self.dates)}
        
        # Store ticker information
        self.universe_tickers = universe_tickers
        self.ticker_to_i = {ticker: i for i, ticker in enumerate(universe_tickers)}
        
        # Store DataFrame for reference
        self.price_df = price_df
        
        # Keep reference to loader
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
        return [self.ticker_to_i.get(ticker) for ticker in tickers 
                if ticker in self.ticker_to_i]

    def run_backtest(
        self,
        strategy: NumPyVectorizedStrategyBase,
        benchmark: Union[str, np.ndarray, List[str]] = "equal_weight",
        shift_signals: bool = True,
        verbose: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Run a vectorized backtest using NumPy arrays for maximum performance.
        
        Parameters:
        -----------
        strategy : NumPyVectorizedStrategyBase
            Strategy that implements the batch method
        benchmark : str or np.ndarray or List[str]
            Benchmark type ("equal_weight"), weights array, or list of tickers for equal weight
        shift_signals : bool
            Whether to shift signals by one day
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with strategy returns, benchmark returns, etc.
        """
        # Get column indices for strategy tickers
        strategy_indices = self.get_indices_for_tickers(strategy.tickers)
        
        if not strategy_indices:
            raise ValueError(f"None of the strategy tickers {strategy.tickers} are in the universe")
            
        if verbose:
            print(f"Computing strategy weights for {len(strategy_indices)} tickers...")
            
        # Get weight matrix from strategy (only for relevant columns)
        # Extract submatrix for just the strategy's tickers
        strategy_price_matrix = self.price_matrix[:, strategy_indices]
        
        # Call batch with the relevant price data and column indices
        # The strategy should return weights for dates[1:] directly
        weights_matrix = strategy.batch(
            strategy_price_matrix, 
            self.dates, 
            strategy_indices
        )
        
        # Check that weights shape matches expectations for columns
        if weights_matrix.shape[1] != len(strategy_indices):
            raise ValueError(
                f"Strategy returned weights with {weights_matrix.shape[1]} columns, "
                f"but expected {len(strategy_indices)}"
            )
        
        # No longer check for row count - strategy should return the correct shape
        
        # Prepare benchmark weights if needed
        benchmark_weights = None
        if isinstance(benchmark, str) and benchmark == "equal_weight":
            # Equal weight across all strategy tickers
            benchmark_weights = np.ones(len(strategy_indices)) / len(strategy_indices)
        elif isinstance(benchmark, list):
            # Equal weight for specified benchmark tickers
            benchmark_indices = self.get_indices_for_tickers(benchmark)
            if not benchmark_indices:
                raise ValueError(f"None of the benchmark tickers {benchmark} are in the universe")
            benchmark_weights = np.zeros(len(strategy_indices))
            # Need to map from universe indices to strategy indices
            strategy_idx_set = set(strategy_indices)
            for idx in benchmark_indices:
                if idx in strategy_idx_set:
                    # Find position in strategy_indices list
                    pos = strategy_indices.index(idx)
                    benchmark_weights[pos] = 1.0
            # Normalize
            if np.sum(benchmark_weights) > 0:
                benchmark_weights /= np.sum(benchmark_weights)
        elif isinstance(benchmark, np.ndarray):
            # Direct weight specification
            if len(benchmark) != len(strategy_indices):
                raise ValueError(
                    f"Benchmark weights has {len(benchmark)} elements, "
                    f"but strategy has {len(strategy_indices)} tickers"
                )
            benchmark_weights = benchmark
        
        # Extract returns just for the strategy's tickers
        strategy_returns_matrix = self.returns_matrix[:, strategy_indices]
        
        # Get raw return arrays
        result_npy = self.run_backtest_npy(
            returns_matrix=strategy_returns_matrix,
            weights_matrix=weights_matrix,
            benchmark_weights=benchmark_weights,
            shift_signals=shift_signals
        )
        
        # Convert to pandas Series for compatibility
        strategy_returns = pd.Series(
            result_npy["strategy_returns"],
            index=self.dates
        )
        
        benchmark_returns = None
        if "benchmark_returns" in result_npy:
            benchmark_returns = pd.Series(
                result_npy["benchmark_returns"],
                index=self.dates
            )
        
        # Create a DataFrame with the strategy tickers for the signals
        strategy_ticker_list = [self.universe_tickers[i] for i in strategy_indices]
        
        # Convert back to pandas DataFrames for output
        weights_df = pd.DataFrame(
            weights_matrix,
            index=self.dates,
            columns=strategy_ticker_list
        )
        
        returns_df = pd.DataFrame(
            strategy_returns_matrix,
            index=self.dates,
            columns=strategy_ticker_list
        )
        
        return {
            'signals_df': weights_df,
            'tickers_returns': returns_df,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
        }
    
    def run_backtest_npy(self,
                        returns_matrix: np.ndarray,
                        weights_matrix: np.ndarray,
                        benchmark_weights: Optional[np.ndarray] = None,
                        shift_signals: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run backtest with pure NumPy arrays for maximum performance.
        
        Parameters:
        -----------
        returns_matrix : np.ndarray
            Returns matrix with shape (n_dates, n_strategy_tickers)
        weights_matrix : np.ndarray
            Weight matrix with shape (n_dates+1, n_strategy_tickers) or (n_dates, n_strategy_tickers)
        benchmark_weights : np.ndarray, optional
            Benchmark weights (1D array of length n_strategy_tickers)
        shift_signals : bool
            Whether to shift signals by one day
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with numpy arrays of returns
        """
        # Fix for array shape mismatch: ensure weights have correct shape for returns
        if weights_matrix.shape[0] > returns_matrix.shape[0]:
            # Trim weights to match returns length if needed
            weights_matrix = weights_matrix[1:]
        
        if shift_signals:
            W = np.vstack([
                np.zeros((1, weights_matrix.shape[1]), dtype=float),
                weights_matrix[:-1]
            ])
        else:
            W = weights_matrix

        strat_rets = np.sum(W * returns_matrix, axis=1)
        
        if benchmark_weights is not None:
            bench_rets = returns_matrix.dot(benchmark_weights)
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