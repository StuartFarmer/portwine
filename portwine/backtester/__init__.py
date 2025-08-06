"""
Backtester module for portwine.

This module provides backward compatibility for the moved backtester functionality.
All classes and functions are imported from the core module.
"""

from .core import (
    Backtester,
    InvalidBenchmarkError,
    BenchmarkTypes,
    benchmark_equal_weight,
    benchmark_markowitz,
    STANDARD_BENCHMARKS,
)

__all__ = [
    "Backtester",
    "InvalidBenchmarkError", 
    "BenchmarkTypes",
    "benchmark_equal_weight",
    "benchmark_markowitz",
    "STANDARD_BENCHMARKS",
]
