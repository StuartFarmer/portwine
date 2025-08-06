#!/usr/bin/env python3
"""
Demonstration script for profiling the step-driven Backtester.

This script shows how to use the performance profiler to identify bottlenecks
and optimization opportunities in the run_backtest method.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.profile_step_backtester import (
    profile_backtest_method,
    profile_specific_sections,
    scaling_analysis,
    memory_usage_analysis
)
from tests.performance_decorator import (
    PerformanceProfiler,
    profile_backtest_method as profile_decorator,
    BacktestPerformanceMonitor
)
from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase
from portwine.universe import Universe
import pandas as pd
import numpy as np


class DemoStrategy(StrategyBase):
    """A demo strategy for profiling purposes."""
    
    def __init__(self, tickers, complexity_level="simple"):
        super().__init__(tickers)
        self.complexity_level = complexity_level
        self.price_history = {}
        
    def step(self, current_date, daily_data):
        """Compute strategy weights with configurable complexity."""
        signals = {}
        
        for ticker in self.tickers:
            if ticker in daily_data and daily_data[ticker] is not None:
                current_price = daily_data[ticker]['close']
                
                # Update price history
                if ticker not in self.price_history:
                    self.price_history[ticker] = []
                self.price_history[ticker].append(current_price)
                
                if self.complexity_level == "simple":
                    # Simple equal weight
                    signals[ticker] = 1.0
                elif self.complexity_level == "medium":
                    # Moving average crossover
                    if len(self.price_history[ticker]) >= 50:
                        short_ma = np.mean(self.price_history[ticker][-20:])
                        long_ma = np.mean(self.price_history[ticker][-50:])
                        signals[ticker] = 1.0 if short_ma > long_ma else 0.0
                    else:
                        signals[ticker] = 0.0
                elif self.complexity_level == "complex":
                    # More complex calculations
                    if len(self.price_history[ticker]) >= 100:
                        prices = self.price_history[ticker]
                        returns = np.diff(prices) / prices[:-1]
                        volatility = np.std(returns[-20:])
                        momentum = np.mean(returns[-10:])
                        
                        # Complex signal based on multiple factors
                        if volatility > 0.01 and momentum > 0:
                            signals[ticker] = 1.0
                        elif volatility < 0.005 and momentum < 0:
                            signals[ticker] = -0.5  # Short position
                        else:
                            signals[ticker] = 0.0
                    else:
                        signals[ticker] = 0.0
            else:
                signals[ticker] = 0.0
        
        # Normalize weights
        total_weight = sum(abs(signals.values()))
        if total_weight > 0:
            for ticker in signals:
                signals[ticker] /= total_weight
        
        return signals


def demo_basic_profiling():
    """Demonstrate basic profiling capabilities."""
    print("=== Basic Performance Profiling Demo ===")
    
    # Profile with different strategy complexities
    complexities = ["simple", "medium", "complex"]
    
    for complexity in complexities:
        print(f"\n--- Profiling {complexity} strategy ---")
        results = profile_backtest_method(
            n_tickers=50,
            n_days=500,
            strategy_type="custom",  # We'll use our demo strategy
            num_runs=2
        )
        
        print(f"Average execution time: {results['timing']['average_time']:.4f} seconds")


def demo_decorator_profiling():
    """Demonstrate the decorator-based profiling approach."""
    print("\n=== Decorator-Based Profiling Demo ===")
    
    # Create a profiler instance
    profiler = PerformanceProfiler(
        enable_profiling=True,
        save_profiles=True,
        profile_dir="demo_profiles"
    )
    
    # Create a simple backtester setup
    from tests.profile_step_backtester import MockMarketDataLoader, EqualWeightStrategy
    
    data_loader = MockMarketDataLoader(n_tickers=30, n_days=300)
    strategy = EqualWeightStrategy(data_loader.tickers)
    backtester = Backtester(market_data_loader=data_loader, logger=None, log=False)
    
    # Apply profiling decorator to the run_backtest method
    original_run_backtest = backtester.run_backtest
    backtester.run_backtest = profiler.profile_method(
        method_name="run_backtest_demo",
        save_profile=True
    )(original_run_backtest)
    
    # Run the backtest with profiling
    print("Running backtest with profiling decorator...")
    results = backtester.run_backtest(
        strategy=strategy,
        benchmark='equal_weight',
        verbose=False
    )
    
    print("Profiling completed. Check the demo_profiles directory for detailed results.")


def demo_performance_monitoring():
    """Demonstrate performance monitoring over multiple runs."""
    print("\n=== Performance Monitoring Demo ===")
    
    # Create performance monitor
    monitor = BacktestPerformanceMonitor()
    
    # Create backtester setup
    from tests.profile_step_backtester import MockMarketDataLoader
    
    data_loader = MockMarketDataLoader(n_tickers=40, n_days=400)
    backtester = Backtester(market_data_loader=data_loader, logger=None, log=False)
    
    # Apply monitoring decorator
    original_run_backtest = backtester.run_backtest
    backtester.run_backtest = monitor.monitor_backtest(
        strategy_name="EqualWeight",
        ticker_count=40,
        day_count=400
    )(original_run_backtest)
    
    # Run multiple backtests to build performance history
    strategies = [
        DemoStrategy(data_loader.tickers, "simple"),
        DemoStrategy(data_loader.tickers, "medium"),
        DemoStrategy(data_loader.tickers, "complex")
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"Running backtest {i+1}/3 with {strategy.complexity_level} strategy...")
        results = backtester.run_backtest(
            strategy=strategy,
            benchmark='equal_weight',
            verbose=False
        )
    
    # Get performance summary
    df, summary = monitor.get_performance_summary()
    
    print(f"\nPerformance Summary:")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average execution time: {summary['avg_execution_time']:.4f} seconds")
    print(f"Min execution time: {summary['min_execution_time']:.4f} seconds")
    print(f"Max execution time: {summary['max_execution_time']:.4f} seconds")
    print(f"Standard deviation: {summary['std_execution_time']:.4f} seconds")
    
    # Try to plot performance trends
    try:
        monitor.plot_performance_trends()
    except Exception as e:
        print(f"Could not plot performance trends: {e}")


def demo_bottleneck_identification():
    """Demonstrate bottleneck identification in specific sections."""
    print("\n=== Bottleneck Identification Demo ===")
    
    # Profile specific sections
    section_results = profile_specific_sections(
        n_tickers=100,
        n_days=1000,
        strategy_type="equal_weight"
    )
    
    print("Section profiling completed. Check the output above for bottleneck analysis.")


def demo_scaling_analysis():
    """Demonstrate scaling analysis."""
    print("\n=== Scaling Analysis Demo ===")
    
    # Run scaling analysis with smaller datasets for demo
    scaling_results = scaling_analysis()
    
    print("Scaling analysis completed. Check the output above for scaling behavior.")


def demo_memory_analysis():
    """Demonstrate memory usage analysis."""
    print("\n=== Memory Usage Analysis Demo ===")
    
    memory_results = memory_usage_analysis(n_tickers=50, n_days=500)
    
    if memory_results:
        print("Memory analysis completed. Check the output above for memory usage patterns.")


def demo_optimization_recommendations():
    """Provide optimization recommendations based on profiling results."""
    print("\n=== Optimization Recommendations ===")
    
    print("Based on typical profiling results, here are common optimization opportunities:")
    
    recommendations = [
        {
            "area": "Strategy Step Method",
            "issues": ["High function call overhead", "Repeated calculations", "Inefficient data structures"],
            "optimizations": [
                "Cache frequently accessed data",
                "Use NumPy operations instead of Python loops",
                "Pre-allocate data structures",
                "Minimize object creation in hot paths"
            ]
        },
        {
            "area": "Data Loading",
            "issues": ["Repeated data fetching", "Inefficient data alignment", "Memory overhead"],
            "optimizations": [
                "Implement data caching",
                "Use vectorized operations for data alignment",
                "Consider using NumPy arrays instead of pandas for large datasets",
                "Lazy loading for large datasets"
            ]
        },
        {
            "area": "Returns Calculation",
            "issues": ["Pandas overhead", "Memory allocation", "Type conversions"],
            "optimizations": [
                "Use NumPy for returns calculation",
                "Pre-allocate arrays",
                "Avoid unnecessary type conversions",
                "Consider using numba for acceleration"
            ]
        },
        {
            "area": "Memory Management",
            "issues": ["Memory fragmentation", "Large object creation", "Garbage collection overhead"],
            "optimizations": [
                "Reuse objects where possible",
                "Use object pools for frequently created objects",
                "Profile memory usage with tools like memory_profiler",
                "Consider using __slots__ for data classes"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['area']}:")
        print(f"  Common Issues: {', '.join(rec['issues'])}")
        print(f"  Optimizations: {', '.join(rec['optimizations'])}")


def main():
    """Run all profiling demonstrations."""
    print("Step-Driven Backtester Performance Profiling Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_basic_profiling()
        demo_decorator_profiling()
        demo_performance_monitoring()
        demo_bottleneck_identification()
        demo_scaling_analysis()
        demo_memory_analysis()
        demo_optimization_recommendations()
        
        print("\n" + "=" * 60)
        print("All profiling demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Run the comprehensive profiler: python tests/profile_step_backtester.py")
        print("2. Apply the decorator to your actual backtester")
        print("3. Analyze the profiling results to identify bottlenecks")
        print("4. Implement optimizations based on the recommendations")
        
    except Exception as e:
        print(f"Error during profiling demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 