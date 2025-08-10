"""
Legacy import file for backward compatibility.

This file now imports from the stores package to maintain
existing import statements while the code has been reorganized.
"""

# Import from the new stores package
from .stores.base import DataStore
from .stores.parquet import ParquetDataStore
from .stores.noisy import NoisyDataStore

__all__ = ['DataStore', 'ParquetDataStore', 'NoisyDataStore']

