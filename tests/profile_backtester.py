import time
import pandas as pd
import numpy as np
from portwine.vectorized import VectorizedBacktester, OptimizedVectorizedBacktester, VectorizedStrategyBase

import cProfile
import pstats
import io
from pstats import SortKey

class MockMarketDataLoader:
    def __init__(self, n_tickers=100, n_days=1000, include_nans=True):
        """Mock data loader that generates random price data."""
        self.n_tickers = n_tickers
        self.n_days = n_days
        self.include_nans = include_nans
        # Generate ticker symbols
        self.tickers = [f"TICKER{i}" for i in range(n_tickers)]
        # Generate dates
        self.dates = pd.date_range('2020-01-01', periods=n_days)
        # Generate data
        self.data = self._generate_data()
        
    def _generate_data(self):
        """Generate random price data with some structure."""
        data_dict = {}
        for ticker in self.tickers:
            # Generate random starting price between 10 and 1000
            start_price = np.random.uniform(10, 1000)
            # Generate daily returns with some autocorrelation
            daily_returns = np.random.normal(0.0005, 0.015, self.n_days)
            # Introduce some autocorrelation
            for i in range(1, self.n_days):
                daily_returns[i] = 0.7 * daily_returns[i] + 0.3 * daily_returns[i-1]
            
            # Convert returns to prices
            prices = start_price * np.cumprod(1 + daily_returns)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': prices * (1 - np.random.uniform(0, 0.005, self.n_days)),
                'high': prices * (1 + np.random.uniform(0, 0.01, self.n_days)),
                'low': prices * (1 - np.random.uniform(0, 0.01, self.n_days)),
                'close': prices,
                'volume': np.random.randint(1000, 1000000, self.n_days)
            }, index=self.dates)
            
            # Add some NaNs if needed
            if self.include_nans:
                # Randomly mask 2% of values
                mask = np.random.random(self.n_days) < 0.02
                df.loc[mask, 'close'] = np.nan
            
            data_dict[ticker] = df
            
        return data_dict
    
    def fetch_data(self, tickers):
        """Return data for requested tickers."""
        return {ticker: self.data.get(ticker) for ticker in tickers}

class SimpleMovingAverageStrategy(VectorizedStrategyBase):
    """Simple strategy that uses moving average crossover."""
    def __init__(self, tickers, short_window=20, long_window=50):
        super().__init__(tickers)
        self.short_window = short_window
        self.long_window = long_window
        
    def batch(self, prices_df):
        """Compute strategy weights using moving average crossover."""
        # Calculate moving averages
        short_ma = prices_df.rolling(window=self.short_window).mean()
        long_ma = prices_df.rolling(window=self.long_window).mean()
        
        # Generate signals (1 when short MA > long MA, 0 otherwise)
        signals = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
        signals[short_ma > long_ma] = 1
        
        # Normalize weights across all assets
        row_sums = signals.sum(axis=1)
        # Avoid division by zero
        row_sums = row_sums.replace(0, 1)
        normalized_weights = signals.div(row_sums, axis=0)
        
        return normalized_weights

def multi_run_profile(backtester_class, num_runs=5, profile_output_file=None, 
                     n_tickers=100, n_days=1000, same_data=True, same_strategy=True):
    """
    Profile multiple runs of a backtester class to identify caching benefits.
    
    Parameters:
    -----------
    backtester_class : class
        The backtester class to profile
    num_runs : int
        Number of backtest runs to profile
    profile_output_file : str, optional
        File to save the profiling results
    n_tickers : int
        Number of tickers to use in test data
    n_days : int
        Number of days to use in test data
    same_data : bool
        If True, reuse the same data for all runs
    same_strategy : bool
        If True, reuse the same strategy instance for all runs
    """
    class_name = backtester_class.__name__
    print(f"\n*** Multi-run profiling for {class_name} ({num_runs} runs) ***")
    print(f"Settings: tickers={n_tickers}, days={n_days}, same_data={same_data}, same_strategy={same_strategy}")
    
    # Create data loader (once if same_data is True)
    if same_data:
        data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
    
    # Create strategy (once if same_strategy is True)
    if same_strategy and same_data:
        strategy = SimpleMovingAverageStrategy(
            tickers=data_loader.tickers[:n_tickers],
            short_window=20,
            long_window=50
        )
    
    # Create backtester (once)
    if same_data:
        backtester = backtester_class(data_loader)
    
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Track execution times for each run
    run_times = []
    
    # Run multiple backtests
    for i in range(num_runs):
        # Create new data/strategy if not reusing
        if not same_data:
            data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
            backtester = backtester_class(data_loader)
            
        if not same_strategy or not same_data:
            strategy = SimpleMovingAverageStrategy(
                tickers=data_loader.tickers[:n_tickers],
                short_window=20,
                long_window=50
            )
        
        # Measure time for this specific run
        start_time = time.time()
        
        # Run the backtest
        results = backtester.run_backtest(
            strategy=strategy,
            benchmark='equal_weight',
            shift_signals=True,
            verbose=False
        )
        
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time)
        print(f"  Run {i+1}/{num_runs} completed in {run_time:.4f} seconds")
    
    profiler.disable()
    
    # Report run times
    print(f"\nRun times: {[f'{t:.4f}s' for t in run_times]}")
    print(f"First run: {run_times[0]:.4f}s")
    if num_runs > 1:
        avg_subsequent = sum(run_times[1:]) / (num_runs - 1)
        print(f"Average of subsequent runs: {avg_subsequent:.4f}s")
        print(f"Speedup after first run: {run_times[0]/avg_subsequent:.2f}x")
    
    # Print profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    print("\nDetailed Profile Results:")
    print(s.getvalue())
    
    # Save to file if requested
    if profile_output_file:
        profile_name = f"{profile_output_file}_{class_name}_{num_runs}runs.prof"
        ps.dump_stats(profile_name)
        print(f"Profile data saved to {profile_name}")
    
    return profiler, run_times, results

def compare_implementations_with_caching():
    """Compare both implementations with different caching strategies."""
    test_configs = [
        # Original backtester
        {"backtester": VectorizedBacktester, "same_data": True, "same_strategy": True, 
         "desc": "Original - Reusing data & strategy (caching benefits)"},
        {"backtester": VectorizedBacktester, "same_data": False, "same_strategy": False, 
         "desc": "Original - New data & strategy each run (no caching)"},
        
        # Optimized backtester
        {"backtester": OptimizedVectorizedBacktester, "same_data": True, "same_strategy": True, 
         "desc": "Optimized - Reusing data & strategy (caching benefits)"},
        {"backtester": OptimizedVectorizedBacktester, "same_data": False, "same_strategy": False, 
         "desc": "Optimized - New data & strategy each run (no caching)"}
    ]
    
    results = {}
    
    for i, config in enumerate(test_configs):
        print(f"\n==== Test Configuration {i+1}: {config['desc']} ====")
        
        _, run_times, _ = multi_run_profile(
            backtester_class=config["backtester"],
            num_runs=5,
            same_data=config["same_data"],
            same_strategy=config["same_strategy"],
            profile_output_file=f"cache_test_{i+1}"
        )
        
        results[config["desc"]] = run_times
    
    # Print summary comparison
    print("\n==== SUMMARY COMPARISON ====")
    for desc, times in results.items():
        first_run = times[0]
        avg_subsequent = sum(times[1:]) / (len(times) - 1) if len(times) > 1 else 0
        cache_benefit = first_run / avg_subsequent if avg_subsequent > 0 else 0
        
        print(f"\n{desc}:")
        print(f"  First run: {first_run:.4f}s")
        print(f"  Avg subsequent: {avg_subsequent:.4f}s")
        print(f"  Caching benefit: {cache_benefit:.2f}x speedup")
    
    return results

def profile_specific_functions():
    """Profile specific functions to identify bottlenecks."""
    # Create test data
    data_loader = MockMarketDataLoader(n_tickers=100, n_days=1000)
    strategy = SimpleMovingAverageStrategy(
        tickers=data_loader.tickers,
        short_window=20, 
        long_window=50
    )
    
    # Get raw data for more detailed profiling
    full_prices = create_price_dataframe(
        data_loader,
        tickers=strategy.tickers,
        start_date=None,
        end_date=None
    )
    
    # Profile strategy batch method
    print("\n==== Profiling Strategy Batch Method ====")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        all_weights = strategy.batch(full_prices)
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
    
    # Profile returns calculation
    print("\n==== Profiling Returns Calculation ====")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        returns_df = full_prices.pct_change(fill_method=None).fillna(0.0)
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
    
    # Profile numpy returns calculation
    print("\n==== Profiling NumPy Returns Calculation ====")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        prices_array = full_prices.values
        returns_array = np.zeros_like(prices_array)
        returns_array[1:] = (prices_array[1:] / prices_array[:-1] - 1)
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == "__main__":
    # Import missing function if needed
    from portwine.vectorized import create_price_dataframe
    
    # Run multiple tests
    print("COMPARING IMPLEMENTATIONS WITH CACHING EFFECTS")
    compare_implementations_with_caching()
    
    print("\nPROFILING SPECIFIC BOTTLENECK FUNCTIONS")
    profile_specific_functions()
    
    # Optional: Run different sized datasets
    print("\nSCALING TEST: SMALL DATASET")
    multi_run_profile(OptimizedVectorizedBacktester, num_runs=3, n_tickers=20, n_days=500)
    
    print("\nSCALING TEST: LARGE DATASET")
    multi_run_profile(OptimizedVectorizedBacktester, num_runs=3, n_tickers=500, n_days=2000)