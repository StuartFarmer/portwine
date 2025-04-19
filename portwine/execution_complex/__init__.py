"""
Execution module for portwine.

This module provides the execution interface and implementations for
connecting strategies to real trading platforms.
"""

# Base classes
from portwine.execution_complex.base import (
    ExecutionBase,
    ExecutionError,
    OrderExecutionError,
    DataFetchError,
)

# Broker classes
from portwine.execution_complex.broker import (
    BrokerBase,
    AccountInfo,
    Position,
    Order,
)

# Utility functions
from portwine.execution_complex.execution_utils import (
    create_bar_dict,
    calculate_position_changes,
    generate_orders,
)

# Broker implementations
from portwine.execution_complex.alpaca_broker import AlpacaBroker
from portwine.execution_complex.mock_broker import MockBroker

__all__ = [
    # Base classes
    'ExecutionBase',
    'ExecutionError',
    'OrderExecutionError',
    'DataFetchError',
    
    # Broker classes
    'BrokerBase',
    'AccountInfo',
    'Position',
    'Order',
    
    # Utility functions
    'create_bar_dict',
    'calculate_position_changes',
    'generate_orders',
    
    # Broker implementations
    'AlpacaBroker',
    'MockBroker',
]