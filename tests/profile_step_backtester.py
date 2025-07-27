#!/usr/bin/env python3
"""
Performance profiler for the step-driven Backtester's run_backtest method.

This script provides comprehensive profiling capabilities to identify bottlenecks
and optimization opportunities in the step-driven backtesting process.
"""

import time
import cProfile
import pstats
import io
from pstats import SortKey
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase
from portwine.universe import Universe


class MockMarketDataLoader:
    """Mock data loader for profiling tests."""
    
    def __init__(self, n_tickers=100, n_days=1000, include_nans=True):
        """Initialize mock data loader with random price data."""
        self.n_tickers = n_tickers
        self.n_days = n_days
        self.include_nans = include_nans
        
        # Generate ticker symbols
        self.tickers = [f"TICKER{i}" for i in range(n_tickers)]
        
        # Generate dates
        self.dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Generate data
        self.data = self._generate_data()
        
    def _generate_data(self) -> Dict[str, pd.DataFrame]:
        """Generate random price data with realistic structure."""
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
            
            # Create DataFrame with OHLCV data
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
    
    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Return data for requested tickers."""
        return {ticker: self.data.get(ticker) for ticker in tickers}
    
    def get_all_dates(self, tickers: List[str]) -> List[pd.Timestamp]:
        """Get all available dates for the given tickers."""
        return sorted(self.dates)


class SimpleMovingAverageStrategy(StrategyBase):
    """Simple strategy that uses moving average crossover for profiling."""
    
    def __init__(self, tickers, short_window=20, long_window=50):
        super().__init__(tickers)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = {}
        
    def step(self, current_date, daily_data):
        """Compute strategy weights using moving average crossover."""
        signals = {}
        
        for ticker in self.tickers:
            if ticker in daily_data and daily_data[ticker] is not None:
                # Get current price
                current_price = daily_data[ticker]['close']
                
                # Update price history
                if ticker not in self.price_history:
                    self.price_history[ticker] = []
                self.price_history[ticker].append(current_price)
                
                # Calculate moving averages if we have enough data
                if len(self.price_history[ticker]) >= self.long_window:
                    prices = self.price_history[ticker]
                    short_ma = np.mean(prices[-self.short_window:])
                    long_ma = np.mean(prices[-self.long_window:])
                    
                    # Generate signal (1 when short MA > long MA, 0 otherwise)
                    if short_ma > long_ma:
                        signals[ticker] = 1.0
                    else:
                        signals[ticker] = 0.0
                else:
                    signals[ticker] = 0.0
            else:
                signals[ticker] = 0.0
        
        # Normalize weights
        total_weight = sum(signals.values())
        if total_weight > 0:
            for ticker in signals:
                signals[ticker] /= total_weight
        
        return signals


class EqualWeightStrategy(StrategyBase):
    """Simple equal weight strategy for baseline profiling."""
    
    def step(self, current_date, daily_data):
        """Return equal weights for all tickers with data."""
        available_tickers = [t for t in self.tickers 
                           if t in daily_data and daily_data[t] is not None]
        
        if not available_tickers:
            return {t: 0.0 for t in self.tickers}
        
        weight = 1.0 / len(available_tickers)
        signals = {t: 0.0 for t in self.tickers}
        
        for ticker in available_tickers:
            signals[ticker] = weight
            
        return signals


def profile_backtest_method(
    n_tickers: int = 100,
    n_days: int = 1000,
    strategy_type: str = "equal_weight",
    benchmark: str = "equal_weight",
    shift_signals: bool = True,
    require_all_history: bool = False,
    require_all_tickers: bool = False,
    verbose: bool = False,
    profile_output_file: Optional[str] = None,
    num_runs: int = 1
) -> Dict:
    """
    Profile the run_backtest method with detailed timing and bottleneck analysis.
    
    Parameters
    ----------
    n_tickers : int
        Number of tickers to simulate
    n_days : int
        Number of days to simulate
    strategy_type : str
        Type of strategy to use ("equal_weight" or "moving_average")
    benchmark : str
        Benchmark to use for comparison
    shift_signals : bool
        Whether to shift signals by one day
    require_all_history : bool
        Whether to require all tickers to have full history
    require_all_tickers : bool
        Whether to require all tickers to have data
    verbose : bool
        Whether to enable verbose logging
    profile_output_file : str, optional
        File to save profiling results
    num_runs : int
        Number of runs to average timing results
        
    Returns
    -------
    Dict
        Profiling results including timing and bottleneck analysis
    """
    print(f"\n=== Profiling Backtester.run_backtest ===")
    print(f"Configuration: {n_tickers} tickers, {n_days} days, {strategy_type} strategy")
    print(f"Benchmark: {benchmark}, Shift signals: {shift_signals}")
    print(f"Require all history: {require_all_history}, Require all tickers: {require_all_tickers}")
    
    # Create data loader
    data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
    
    # Create strategy
    if strategy_type == "equal_weight":
        strategy = EqualWeightStrategy(data_loader.tickers)
    elif strategy_type == "moving_average":
        strategy = SimpleMovingAverageStrategy(data_loader.tickers, short_window=20, long_window=50)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Create backtester
    backtester = Backtester(
        market_data_loader=data_loader,
        logger=None,
        log=False
    )
    
    # Track execution times
    run_times = []
    results_list = []
    
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple backtests
    for i in range(num_runs):
        start_time = time.time()
        
        results = backtester.run_backtest(
            strategy=strategy,
            shift_signals=shift_signals,
            benchmark=benchmark,
            start_date=None,
            end_date=None,
            require_all_history=require_all_history,
            require_all_tickers=require_all_tickers,
            verbose=verbose
        )
        
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time)
        results_list.append(results)
        
        print(f"  Run {i+1}/{num_runs}: {run_time:.4f} seconds")
    
    profiler.disable()
    
    # Calculate timing statistics
    avg_time = sum(run_times) / len(run_times)
    min_time = min(run_times)
    max_time = max(run_times)
    
    print(f"\nTiming Results:")
    print(f"  Average time: {avg_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"  Standard deviation: {np.std(run_times):.4f} seconds")
    
    # Analyze profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    profile_output = s.getvalue()
    
    print(f"\nTop 30 Functions by Cumulative Time:")
    print(profile_output)
    
    # Save profiling data if requested
    if profile_output_file:
        profile_name = f"{profile_output_file}_backtester_{n_tickers}tickers_{n_days}days.prof"
        ps.dump_stats(profile_name)
        print(f"Profile data saved to {profile_name}")
    
    # Extract key metrics from results
    if results_list:
        last_results = results_list[-1]
        n_signals = len(last_results.get('signals_df', pd.DataFrame()))
        n_returns = len(last_results.get('strategy_returns', pd.Series()))
        
        print(f"\nResults Summary:")
        print(f"  Number of signal dates: {n_signals}")
        print(f"  Number of return dates: {n_returns}")
        print(f"  Processing rate: {n_signals/avg_time:.2f} dates/second")
    
    return {
        'timing': {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': np.std(run_times),
            'run_times': run_times
        },
        'profile_output': profile_output,
        'results': results_list[-1] if results_list else None,
        'profiler': profiler
    }


def profile_specific_sections(
    n_tickers: int = 100,
    n_days: int = 1000,
    strategy_type: str = "equal_weight"
) -> Dict:
    """
    Profile specific sections of the backtest process to identify bottlenecks.
    
    Parameters
    ----------
    n_tickers : int
        Number of tickers to simulate
    n_days : int
        Number of days to simulate
    strategy_type : str
        Type of strategy to use
        
    Returns
    -------
    Dict
        Detailed profiling results for each section
    """
    print(f"\n=== Profiling Specific Sections ===")
    
    # Create data loader
    data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
    
    # Create strategy
    if strategy_type == "equal_weight":
        strategy = EqualWeightStrategy(data_loader.tickers)
    else:
        strategy = SimpleMovingAverageStrategy(data_loader.tickers, short_window=20, long_window=50)
    
    # Create backtester
    backtester = Backtester(
        market_data_loader=data_loader,
        logger=None,
        log=False
    )
    
    results = {}
    
    # Profile data loading
    print("\n--- Profiling Data Loading ---")
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(5):
        reg_data = data_loader.fetch_data(data_loader.tickers)
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(15)
    results['data_loading'] = s.getvalue()
    print(results['data_loading'])
    
    # Profile strategy step method
    print("\n--- Profiling Strategy Step Method ---")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Get some sample data
    sample_date = data_loader.dates[100]
    sample_data = {}
    for ticker in data_loader.tickers:
        if sample_date in data_loader.data[ticker].index:
            row = data_loader.data[ticker].loc[sample_date]
            sample_data[ticker] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
    
    for _ in range(1000):  # Many iterations to get meaningful timing
        strategy.step(sample_date, sample_data)
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(15)
    results['strategy_step'] = s.getvalue()
    print(results['strategy_step'])
    
    # Profile returns calculation
    print("\n--- Profiling Returns Calculation ---")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Create sample price data
    price_data = {}
    for ticker in data_loader.tickers[:10]:  # Use subset for faster profiling
        price_data[ticker] = data_loader.data[ticker]['close']
    
    px_df = pd.DataFrame(price_data)
    
    for _ in range(100):
        ret_df = px_df.pct_change(fill_method=None).fillna(0.0)
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(15)
    results['returns_calculation'] = s.getvalue()
    print(results['returns_calculation'])
    
    return results


def scaling_analysis():
    """Analyze performance scaling with different data sizes."""
    print(f"\n=== Scaling Analysis ===")
    
    # Test configurations
    configs = [
        {"n_tickers": 20, "n_days": 500, "name": "Small Dataset"},
        {"n_tickers": 100, "n_days": 1000, "name": "Medium Dataset"},
        {"n_tickers": 500, "n_days": 2000, "name": "Large Dataset"}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        result = profile_backtest_method(
            n_tickers=config['n_tickers'],
            n_days=config['n_days'],
            strategy_type="equal_weight",
            num_runs=3
        )
        results[config['name']] = result
    
    # Print scaling summary
    print(f"\n=== Scaling Summary ===")
    for name, result in results.items():
        avg_time = result['timing']['average_time']
        print(f"{name}: {avg_time:.4f} seconds")
    
    return results


def memory_usage_analysis(n_tickers: int = 100, n_days: int = 1000):
    """Analyze memory usage during backtest execution."""
    print(f"\n=== Memory Usage Analysis ===")
    
    try:
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Create data loader and run backtest
        data_loader = MockMarketDataLoader(n_tickers=n_tickers, n_days=n_days)
        strategy = EqualWeightStrategy(data_loader.tickers)
        backtester = Backtester(market_data_loader=data_loader, logger=None, log=False)
        
        # Memory after data loading
        after_data_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after data loading: {after_data_memory:.2f} MB")
        print(f"Data loading memory increase: {after_data_memory - initial_memory:.2f} MB")
        
        # Run backtest
        results = backtester.run_backtest(
            strategy=strategy,
            benchmark='equal_weight',
            verbose=False
        )
        
        # Memory after backtest
        after_backtest_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after backtest: {after_backtest_memory:.2f} MB")
        print(f"Backtest memory increase: {after_backtest_memory - after_data_memory:.2f} MB")
        print(f"Total memory increase: {after_backtest_memory - initial_memory:.2f} MB")
        
        return {
            'initial_memory': initial_memory,
            'after_data_memory': after_data_memory,
            'after_backtest_memory': after_backtest_memory
        }
        
    except ImportError:
        print("psutil not available. Skipping memory analysis.")
        return None


def main():
    """Run comprehensive profiling analysis."""
    print("Step-Driven Backtester Performance Profiler")
    print("=" * 50)
    
    # Basic profiling
    print("\n1. Basic Performance Profiling")
    basic_results = profile_backtest_method(
        n_tickers=100,
        n_days=1000,
        strategy_type="equal_weight",
        num_runs=3
    )
    
    # Strategy comparison
    print("\n2. Strategy Performance Comparison")
    equal_weight_results = profile_backtest_method(
        n_tickers=100,
        n_days=1000,
        strategy_type="equal_weight",
        num_runs=2
    )
    
    moving_avg_results = profile_backtest_method(
        n_tickers=100,
        n_days=1000,
        strategy_type="moving_average",
        num_runs=2
    )
    
    print(f"\nStrategy Comparison:")
    print(f"  Equal Weight: {equal_weight_results['timing']['average_time']:.4f}s")
    print(f"  Moving Average: {moving_avg_results['timing']['average_time']:.4f}s")
    print(f"  Moving Average overhead: {moving_avg_results['timing']['average_time']/equal_weight_results['timing']['average_time']:.2f}x")
    
    # Detailed section profiling
    print("\n3. Detailed Section Profiling")
    section_results = profile_specific_sections(
        n_tickers=100,
        n_days=1000,
        strategy_type="equal_weight"
    )
    
    # Scaling analysis
    print("\n4. Scaling Analysis")
    scaling_results = scaling_analysis()
    
    # Memory analysis
    print("\n5. Memory Usage Analysis")
    memory_results = memory_usage_analysis(n_tickers=100, n_days=1000)
    
    # Summary and recommendations
    print(f"\n=== Performance Summary and Recommendations ===")
    print(f"1. Baseline performance: {basic_results['timing']['average_time']:.4f}s for 100 tickers, 1000 days")
    print(f"2. Strategy complexity impact: Moving average adds {moving_avg_results['timing']['average_time']/equal_weight_results['timing']['average_time']:.2f}x overhead")
    print(f"3. Scaling behavior: Check scaling analysis results above")
    print(f"4. Memory usage: {memory_results['after_backtest_memory'] - memory_results['initial_memory']:.2f} MB increase if memory_results else 'N/A'")
    
    print(f"\nKey areas to investigate for optimization:")
    print(f"- Strategy step method performance")
    print(f"- Data loading and caching")
    print(f"- Returns calculation efficiency")
    print(f"- Memory allocation patterns")
    
    return {
        'basic_results': basic_results,
        'strategy_comparison': {
            'equal_weight': equal_weight_results,
            'moving_average': moving_avg_results
        },
        'section_results': section_results,
        'scaling_results': scaling_results,
        'memory_results': memory_results
    }


if __name__ == "__main__":
    results = main() 