"""
Execution module for portwine.

This module provides the execution interface and implementations for
connecting strategies to real trading platforms.
"""

# Base classes
from portwine.execution.base import (
    ExecutionBase,
    ExecutionError,
    OrderExecutionError,
    DataFetchError,
)

# Utility functions

# Broker implementations
from portwine.execution.brokers.alpaca import AlpacaBroker
from portwine.execution.brokers.mock import MockBroker

__all__ = [
    # Base classes
    'ExecutionBase',
    'ExecutionError',
    'OrderExecutionError',
    'DataFetchError',

    # Broker implementations
    'AlpacaBroker',
    'MockBroker',
]