"""
Execution module for the portwine framework.

This module provides the base classes and interfaces for execution_complex modules,
which connect strategy implementations from the backtester to live trading.
"""

from __future__ import annotations

import abc
import logging
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol, Tuple, TypedDict, Union, Any, Type, Callable
import os

import pandas as pd
import numpy as np

# Re-export the ExecutionBase class from the new implementation file
from portwine.execution_complex.execution_base import ExecutionBase

from portwine.loaders.base import MarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.execution_complex.broker import BrokerBase, Position, Order, AccountInfo
from portwine.execution_complex.execution_utils import (
    create_bar_dict, 
    calculate_position_changes, 
    generate_orders
)
from portwine.utils.schedule_iterator import ScheduleIterator
import sys
import traceback
import inspect
import importlib

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution_complex-related errors."""
    pass


class OrderExecutionError(ExecutionError):
    """Exception raised when order execution_complex fails."""
    pass


class DataFetchError(ExecutionError):
    """Exception raised when data fetching fails."""
    pass

# The ExecutionBase class is now imported from execution_base.py 