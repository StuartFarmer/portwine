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
from portwine.execution.execution_utils import (
    create_bar_dict,
    calculate_position_changes,
    generate_orders,
)

# Broker implementations
from portwine.execution.brokers.alpaca import AlpacaBroker
from portwine.execution.brokers.mock import MockBroker

__all__ = [
    # Base classes
    'ExecutionBase',
    'ExecutionError',
    'OrderExecutionError',
    'DataFetchError',
    
    # Broker classes
    'BrokerBase',
    'Position',
    'Order',
    'Account',
    'OrderSide',
    'BrokerOrderExecutionError',
    'OrderNotFoundError',
    'OrderCancelError',
    
    # Utility functions
    'create_bar_dict',
    'calculate_position_changes',
    'generate_orders',
    
    # Broker implementations
    'AlpacaBroker',
    'MockBroker',
]