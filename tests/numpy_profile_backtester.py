import time
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt

from portwine.vectorized import (
    NumpyVectorizedBacktester, 
    NumPyVectorizedStrategyBase,
    VectorizedBacktester,
    VectorizedStrategyBase
)

class MockMarketDataLoader:
    """Mock data loader that generates random price data."""
    def __init__(self, n_tickers=100, n_days=1000, include_nans=True):
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

# Original Pandas-based Strategy
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
    
class NumPyMovingAverageStrategy(NumPyVectorizedStrategyBase):
    """Simple strategy that uses moving average crossover with NumPy."""
    def __init__(self, tickers, short_window=20, long_window=50):
        super().__init__(tickers)
        self.short_window = short_window
        self.long_window = long_window
        
    def batch(self, price_matrix, dates, column_indices):
        """Compute strategy weights using moving average crossover."""
        n_dates, n_tickers = price_matrix.shape
        
        # Important: Return weights for all dates except the first one
        # to match returns_matrix shape directly
        
        # Initialize arrays
        short_ma = np.zeros_like(price_matrix)
        long_ma = np.zeros_like(price_matrix)
        
        # Calculate moving averages efficiently with cumsum
        cumsum = np.cumsum(np.insert(price_matrix, 0, 0, axis=0), axis=0)
        
        # Short MA using cumsum with vectorized calc
        for i in range(self.short_window, n_dates + 1):
            short_ma[i-1] = (cumsum[i] - cumsum[i - self.short_window]) / self.short_window
        
        # Long MA using cumsum with vectorized calc
        for i in range(self.long_window, n_dates + 1):
            long_ma[i-1] = (cumsum[i] - cumsum[i - self.long_window]) / self.long_window
        
        # Generate signals (1 when short MA > long MA, 0 otherwise)
        signals = np.zeros_like(price_matrix)
        signals[short_ma > long_ma] = 1
        
        # Normalize weights
        row_sums = np.sum(signals, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        normalized_weights = signals / row_sums
        
        # Return weights for dates[1:] to match returns dimension
        return normalized_weights[1:]

def run_performance_test(n_tickers=100, n_days=1000, runs=3, subset_pct=1.0):
    """
    Run performance test comparing original and NumPy-optimized backtester.
    
    Parameters:
    -----------
    n_tickers : int
        Number of tickers in universe
    n_days : int
        Number of days in backtest
    runs : int
        Number of times to run each backtest for averaging
    subset_pct : float
        Percentage of tickers to use in strategy (for testing subset functionality)
    """
    # Create mock data loader
    data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
    
    # Create full universe tickers list
    universe_tickers = data_loader.tickers
    
    # Create strategy tickers (all or subset)
    if subset_pct < 1.0:
        n_strategy_tickers = int(n_tickers * subset_pct)
        strategy_tickers = universe_tickers[:n_strategy_tickers]
        print(f"Using {n_strategy_tickers} tickers in strategy (subset of {n_tickers})")
    else:
        strategy_tickers = universe_tickers
    
    # Create strategies
    original_strategy = SimpleMovingAverageStrategy(
        tickers=strategy_tickers,
        short_window=20,
        long_window=50
    )
    
    numpy_strategy = NumPyMovingAverageStrategy(
        tickers=strategy_tickers,
        short_window=20,
        long_window=50
    )
    
    # Set up backtester for original approach
    original_backtester = VectorizedBacktester(data_loader)
    
    # Set up NumPy-optimized backtester
    numpy_backtester = NumpyVectorizedBacktester(
        loader=data_loader,
        universe_tickers=universe_tickers,
        start_date=str(data_loader.dates[0].date()),
        end_date=str(data_loader.dates[-1].date())
    )
    
    # Run original backtester
    original_times = []
    for i in range(runs):
        start_time = time.time()
        original_results = original_backtester.run_backtest(
            strategy=original_strategy,
            benchmark='equal_weight',
            shift_signals=True
        )
        end_time = time.time()
        run_time = end_time - start_time
        original_times.append(run_time)
        print(f"Original Run {i+1}/{runs}: {run_time:.4f}s")
    
# Run original backtester
    original_times = []
    for i in range(runs):
        start_time = time.time()
        original_results = original_backtester.run_backtest(
            strategy=original_strategy,
            benchmark='equal_weight',
            shift_signals=True
        )
        end_time = time.time()
        run_time = end_time - start_time
        original_times.append(run_time)
        print(f"Original Run {i+1}/{runs}: {run_time:.4f}s")
    
    # Run optimized backtester
    numpy_times = []
    for i in range(runs):
        start_time = time.time()
        numpy_results = numpy_backtester.run_backtest(
            strategy=numpy_strategy,
            benchmark='equal_weight',
            shift_signals=True
        )
        end_time = time.time()
        run_time = end_time - start_time
        numpy_times.append(run_time)
        print(f"NumPy Run {i+1}/{runs}: {run_time:.4f}s")
    
    # Calculate average times
    avg_original = sum(original_times) / len(original_times)
    avg_numpy = sum(numpy_times) / len(numpy_times)
    
    # Verify results are similar - handle different array lengths
    # Original strategy returns might have one more element than NumPy strategy returns
    # Align the indices before comparing
    common_indices = original_results['strategy_returns'].index.intersection(
        numpy_results['strategy_returns'].index
    )
    
    if len(common_indices) > 0:
        original_aligned = original_results['strategy_returns'].loc[common_indices].values
        numpy_aligned = numpy_results['strategy_returns'].loc[common_indices].values
        corr = np.corrcoef(original_aligned, numpy_aligned)[0, 1]
    else:
        # If no common indices, can't calculate correlation
        corr = float('nan')
        print("WARNING: No common indices between original and NumPy strategy returns")
    
    # Print results
    print(f"\nPerformance Test Results (n_tickers={n_tickers}, n_days={n_days}):")
    print(f"  Original Backtester:      {avg_original:.4f} seconds (avg of {runs} runs)")
    print(f"  NumPy-Optimized Backtester: {avg_numpy:.4f} seconds (avg of {runs} runs)")
    print(f"  Speedup:                  {avg_original/avg_numpy:.2f}x")
    print(f"  Results Correlation:      {corr:.4f}")
    print(f"  Original length: {len(original_results['strategy_returns'])}, NumPy length: {len(numpy_results['strategy_returns'])}")
    
    return {
        'original_time': avg_original,
        'numpy_time': avg_numpy,
        'speedup': avg_original/avg_numpy,
        'correlation': corr,
        'original_results': original_results,
        'numpy_results': numpy_results
    }

def profile_with_cprofile():
    """Profile both backtesters with cProfile."""
    import cProfile
    import pstats
    import io
    
    # Set up test data
    n_tickers = 100
    n_days = 1000
    data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
    universe_tickers = data_loader.tickers
    
    # Original approach
    original_strategy = SimpleMovingAverageStrategy(
        tickers=universe_tickers,
        short_window=20,
        long_window=50
    )
    original_backtester = VectorizedBacktester(data_loader)
    
    # NumPy approach
    numpy_strategy = NumPyMovingAverageStrategy(
        tickers=universe_tickers,
        short_window=20,
        long_window=50
    )
    numpy_backtester = NumpyVectorizedBacktester(
        loader=data_loader,
        universe_tickers=universe_tickers,
        start_date=str(data_loader.dates[0].date()),
        end_date=str(data_loader.dates[-1].date())
    )
    
    # Profile original backtester
    print("\nProfiling Original Backtester...")
    original_pr = cProfile.Profile()
    original_pr.enable()
    original_results = original_backtester.run_backtest(
        strategy=original_strategy,
        benchmark='equal_weight',
        shift_signals=True
    )
    original_pr.disable()
    
    # Print original stats
    s = io.StringIO()
    ps = pstats.Stats(original_pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    # Profile NumPy backtester
    print("\nProfiling NumPy-Optimized Backtester...")
    numpy_pr = cProfile.Profile()
    numpy_pr.enable()
    numpy_results = numpy_backtester.run_backtest(
        strategy=numpy_strategy,
        benchmark='equal_weight',
        shift_signals=True
    )
    numpy_pr.disable()
    
    # Print NumPy stats
    s = io.StringIO()
    ps = pstats.Stats(numpy_pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    return original_results, numpy_results

def run_scaling_tests():
    """Run tests with different data sizes to assess scaling behavior."""
    results = {}
    
    # Test configurations
    configs = [
        {"n_tickers": 20, "n_days": 500, "name": "Small Dataset"},
        {"n_tickers": 100, "n_days": 1000, "name": "Medium Dataset"},
        {"n_tickers": 500, "n_days": 2000, "name": "Large Dataset"}
    ]
    
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        results[config['name']] = run_performance_test(
            n_tickers=config['n_tickers'], 
            n_days=config['n_days'], 
            runs=3
        )
    
    # Print comparative summary
    print("\n=== SUMMARY OF SCALING TESTS ===")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Original: {result['original_time']:.4f}s")
        print(f"  NumPy:    {result['numpy_time']:.4f}s")
        print(f"  Speedup:  {result['speedup']:.2f}x")
    
    return results

def run_subset_tests():
    """Run tests with different subset percentages."""
    results = {}
    
    # Test configurations with different subset percentages
    configs = [
        {"subset_pct": 1.0, "name": "Full Ticker Set (100%)"},
        {"subset_pct": 0.5, "name": "Half Ticker Set (50%)"},
        {"subset_pct": 0.1, "name": "Small Ticker Set (10%)"}
    ]
    
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        results[config['name']] = run_performance_test(
            n_tickers=100, 
            n_days=1000, 
            runs=3,
            subset_pct=config['subset_pct']
        )
    
    # Print comparative summary
    print("\n=== SUMMARY OF SUBSET TESTS ===")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Original: {result['original_time']:.4f}s")
        print(f"  NumPy:    {result['numpy_time']:.4f}s")
        print(f"  Speedup:  {result['speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    # Run basic performance test
    print("=== BASIC PERFORMANCE TEST ===")
    basic_results = run_performance_test(n_tickers=100, n_days=1000, runs=3)
    
    # Run cProfile analysis
    print("\n=== DETAILED PROFILING WITH CPROFILE ===")
    profile_results = profile_with_cprofile()
    
    # Run scaling tests
    print("\n=== SCALING TESTS ===")
    scaling_results = run_scaling_tests()
    
    # Run subset tests
    print("\n=== SUBSET TESTS ===")
    subset_results = run_subset_tests()