#!/usr/bin/env python3
"""
Performance profiling decorator for the Backtester class.

This module provides decorators and utilities for profiling the performance
of the step-driven backtester's run_backtest method.
"""

import time
import cProfile
import pstats
import io
from pstats import SortKey
import functools
import logging
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np


class PerformanceProfiler:
    """A class to handle performance profiling of backtest methods."""
    
    def __init__(self, 
                 enable_profiling: bool = True,
                 save_profiles: bool = False,
                 profile_dir: str = "profiles",
                 log_level: int = logging.INFO):
        """
        Initialize the performance profiler.
        
        Parameters
        ----------
        enable_profiling : bool
            Whether to enable profiling
        save_profiles : bool
            Whether to save profile data to files
        profile_dir : str
            Directory to save profile files
        log_level : int
            Logging level for profiler output
        """
        self.enable_profiling = enable_profiling
        self.save_profiles = save_profiles
        self.profile_dir = profile_dir
        self.logger = self._setup_logger(log_level)
        
        # Create profile directory if needed
        if self.save_profiles:
            import os
            os.makedirs(profile_dir, exist_ok=True)
    
    def _setup_logger(self, level: int) -> logging.Logger:
        """Set up logger for profiling output."""
        logger = logging.getLogger("performance_profiler")
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def profile_method(self, 
                      method_name: str = None,
                      save_profile: bool = None,
                      profile_filename: str = None) -> Callable:
        """
        Decorator to profile a method's performance.
        
        Parameters
        ----------
        method_name : str, optional
            Name for the method in profiling output
        save_profile : bool, optional
            Whether to save profile data (overrides instance setting)
        profile_filename : str, optional
            Custom filename for profile data
            
        Returns
        -------
        Callable
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                # Determine method name
                method_display_name = method_name or func.__name__
                
                # Start timing
                start_time = time.time()
                
                # Start profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Stop profiling
                    profiler.disable()
                    end_time = time.time()
                    
                    # Calculate execution time
                    execution_time = end_time - start_time
                    
                    # Log basic timing
                    self.logger.info(
                        f"{method_display_name} executed in {execution_time:.4f} seconds"
                    )
                    
                    # Analyze profiling results
                    self._analyze_profile(profiler, method_display_name, execution_time)
                    
                    # Save profile if requested
                    should_save = save_profile if save_profile is not None else self.save_profiles
                    if should_save:
                        self._save_profile(profiler, method_display_name, profile_filename)
                    
                    return result
                    
                except Exception as e:
                    # Stop profiling even if exception occurs
                    profiler.disable()
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self.logger.error(
                        f"{method_display_name} failed after {execution_time:.4f} seconds: {e}"
                    )
                    raise
                    
            return wrapper
        return decorator
    
    def _analyze_profile(self, 
                        profiler: cProfile.Profile, 
                        method_name: str, 
                        execution_time: float):
        """Analyze profiling results and log key findings."""
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(10)  # Top 10 functions
        profile_output = s.getvalue()
        
        # Extract key metrics
        stats = profiler.getstats()
        total_calls = sum(stat.callcount for stat in stats)
        total_time = sum(stat.totaltime for stat in stats)
        
        # Log summary
        self.logger.info(
            f"{method_name} profile: {total_calls} total calls, "
            f"{total_time:.4f}s total time, {execution_time:.4f}s wall time"
        )
        
        # Log top bottlenecks
        self.logger.debug(f"Top 10 functions by cumulative time:\n{profile_output}")
        
        # Identify potential bottlenecks
        self._identify_bottlenecks(stats, method_name)
    
    def _identify_bottlenecks(self, stats: List, method_name: str):
        """Identify potential performance bottlenecks."""
        # Sort by cumulative time
        sorted_stats = sorted(stats, key=lambda x: x.totaltime, reverse=True)
        
        bottlenecks = []
        for stat in sorted_stats[:5]:  # Top 5
            if stat.totaltime > 0.001:  # Only report functions taking >1ms
                bottlenecks.append({
                    'function': f"{stat.code.co_name}",
                    'cumulative_time': stat.totaltime,
                    'call_count': stat.callcount,
                    'avg_time_per_call': stat.totaltime / stat.callcount if stat.callcount > 0 else 0
                })
        
        if bottlenecks:
            self.logger.info(f"Potential bottlenecks in {method_name}:")
            for bottleneck in bottlenecks:
                self.logger.info(
                    f"  {bottleneck['function']}: "
                    f"{bottleneck['cumulative_time']:.4f}s total, "
                    f"{bottleneck['call_count']} calls, "
                    f"{bottleneck['avg_time_per_call']:.6f}s per call"
                )
    
    def _save_profile(self, 
                     profiler: cProfile.Profile, 
                     method_name: str, 
                     custom_filename: str = None):
        """Save profile data to file."""
        import os
        from datetime import datetime
        
        # Generate filename
        if custom_filename:
            filename = custom_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{method_name}_{timestamp}.prof"
        
        filepath = os.path.join(self.profile_dir, filename)
        
        # Save profile
        profiler.dump_stats(filepath)
        self.logger.info(f"Profile data saved to {filepath}")


def profile_backtest_method(enable_profiling: bool = True,
                           save_profiles: bool = False,
                           profile_dir: str = "profiles") -> Callable:
    """
    Decorator specifically for profiling the run_backtest method.
    
    Parameters
    ----------
    enable_profiling : bool
        Whether to enable profiling
    save_profiles : bool
        Whether to save profile data to files
    profile_dir : str
        Directory to save profile files
        
    Returns
    -------
    Callable
        Decorated run_backtest method
    """
    profiler = PerformanceProfiler(
        enable_profiling=enable_profiling,
        save_profiles=save_profiles,
        profile_dir=profile_dir
    )
    
    return profiler.profile_method(
        method_name="run_backtest",
        save_profile=save_profiles
    )


class BacktestPerformanceMonitor:
    """A class to monitor and track backtest performance over time."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.performance_history = []
        self.profiler = PerformanceProfiler()
    
    def monitor_backtest(self, 
                        strategy_name: str = None,
                        ticker_count: int = None,
                        day_count: int = None) -> Callable:
        """
        Decorator to monitor backtest performance and track metrics over time.
        
        Parameters
        ----------
        strategy_name : str, optional
            Name of the strategy being tested
        ticker_count : int, optional
            Number of tickers in the test
        day_count : int, optional
            Number of days in the test
            
        Returns
        -------
        Callable
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Extract metadata from arguments if possible
                metadata = self._extract_metadata(args, kwargs, strategy_name, ticker_count, day_count)
                
                # Start timing
                start_time = time.time()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Record performance data
                performance_data = {
                    'timestamp': pd.Timestamp.now(),
                    'execution_time': execution_time,
                    'strategy_name': metadata.get('strategy_name'),
                    'ticker_count': metadata.get('ticker_count'),
                    'day_count': metadata.get('day_count'),
                    'benchmark': metadata.get('benchmark'),
                    'shift_signals': metadata.get('shift_signals'),
                    'require_all_history': metadata.get('require_all_history'),
                    'require_all_tickers': metadata.get('require_all_tickers')
                }
                
                # Add result metrics if available
                if result and isinstance(result, dict):
                    if 'signals_df' in result:
                        performance_data['signal_count'] = len(result['signals_df'])
                    if 'strategy_returns' in result:
                        performance_data['return_count'] = len(result['strategy_returns'])
                
                self.performance_history.append(performance_data)
                
                # Log performance
                self.profiler.logger.info(
                    f"Backtest completed: {execution_time:.4f}s, "
                    f"{metadata.get('ticker_count', 'N/A')} tickers, "
                    f"{metadata.get('day_count', 'N/A')} days"
                )
                
                return result
            
            return wrapper
        return decorator
    
    def _extract_metadata(self, args, kwargs, strategy_name, ticker_count, day_count) -> Dict:
        """Extract metadata from function arguments."""
        metadata = {}
        
        # Try to extract strategy name
        if strategy_name:
            metadata['strategy_name'] = strategy_name
        elif args and hasattr(args[0], '__class__'):
            metadata['strategy_name'] = args[0].__class__.__name__
        
        # Try to extract ticker count
        if ticker_count:
            metadata['ticker_count'] = ticker_count
        elif args and hasattr(args[0], 'tickers'):
            metadata['ticker_count'] = len(args[0].tickers)
        
        # Try to extract day count from kwargs
        if 'start_date' in kwargs and 'end_date' in kwargs:
            try:
                start = pd.Timestamp(kwargs['start_date'])
                end = pd.Timestamp(kwargs['end_date'])
                metadata['day_count'] = (end - start).days
            except:
                pass
        
        # Extract other parameters
        metadata['benchmark'] = kwargs.get('benchmark')
        metadata['shift_signals'] = kwargs.get('shift_signals')
        metadata['require_all_history'] = kwargs.get('require_all_history')
        metadata['require_all_tickers'] = kwargs.get('require_all_tickers')
        
        return metadata
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get a summary of performance history."""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        
        # Calculate summary statistics
        summary = {
            'total_runs': len(df),
            'avg_execution_time': df['execution_time'].mean(),
            'min_execution_time': df['execution_time'].min(),
            'max_execution_time': df['execution_time'].max(),
            'std_execution_time': df['execution_time'].std(),
            'total_execution_time': df['execution_time'].sum()
        }
        
        return df, summary
    
    def plot_performance_trends(self):
        """Plot performance trends over time."""
        try:
            import matplotlib.pyplot as plt
            
            df, summary = self.get_performance_summary()
            if df.empty:
                print("No performance data available for plotting.")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Execution time over time
            axes[0, 0].plot(df['timestamp'], df['execution_time'])
            axes[0, 0].set_title('Execution Time Over Time')
            axes[0, 0].set_ylabel('Execution Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Execution time histogram
            axes[0, 1].hist(df['execution_time'], bins=20, alpha=0.7)
            axes[0, 1].set_title('Execution Time Distribution')
            axes[0, 1].set_xlabel('Execution Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            
            # Performance by ticker count
            if 'ticker_count' in df.columns and df['ticker_count'].notna().any():
                ticker_groups = df.groupby('ticker_count')['execution_time'].mean()
                axes[1, 0].bar(ticker_groups.index, ticker_groups.values)
                axes[1, 0].set_title('Average Execution Time by Ticker Count')
                axes[1, 0].set_xlabel('Number of Tickers')
                axes[1, 0].set_ylabel('Average Execution Time (seconds)')
            
            # Performance by day count
            if 'day_count' in df.columns and df['day_count'].notna().any():
                day_groups = df.groupby('day_count')['execution_time'].mean()
                axes[1, 1].scatter(day_groups.index, day_groups.values)
                axes[1, 1].set_title('Average Execution Time by Day Count')
                axes[1, 1].set_xlabel('Number of Days')
                axes[1, 1].set_ylabel('Average Execution Time (seconds)')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available. Cannot plot performance trends.")


# Convenience functions for easy use
def quick_profile(func: Callable) -> Callable:
    """Quick profiling decorator for any function."""
    profiler = PerformanceProfiler(enable_profiling=True, save_profiles=False)
    return profiler.profile_method()(func)


def detailed_profile(func: Callable) -> Callable:
    """Detailed profiling decorator that saves profile data."""
    profiler = PerformanceProfiler(enable_profiling=True, save_profiles=True)
    return profiler.profile_method()(func)


# Example usage:
if __name__ == "__main__":
    # Example of how to use the profiler decorators
    
    # Simple profiling
    @quick_profile
    def example_function():
        """Example function to profile."""
        import time
        time.sleep(0.1)
        return "done"
    
    # Detailed profiling with file output
    @detailed_profile
    def another_example():
        """Another example function."""
        import time
        time.sleep(0.2)
        return "done"
    
    # Test the profilers
    print("Testing quick profiler:")
    example_function()
    
    print("\nTesting detailed profiler:")
    another_example() 