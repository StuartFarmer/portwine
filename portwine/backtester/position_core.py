"""
Position-based backtesting system.

Interprets strategy outputs as share quantities instead of portfolio weights.
Tracks positions, actions, and portfolio value over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Callable, Optional
from tqdm import tqdm

from portwine.strategies.base import StrategyBase
from portwine.backtester.benchmarks import BenchmarkTypes, STANDARD_BENCHMARKS, get_benchmark_type
from portwine.backtester.core import InvalidBenchmarkError, _split_tickers, DailyMarketCalendar
from portwine.data.interface import DataInterface, MultiDataInterface


class PositionBacktestResult:
    """Stores position-based backtest results."""
    pass


class PositionBacktester:
    """
    Position-based backtester.

    Similar to Backtester, but interprets strategy output as share quantities
    rather than portfolio weights.
    """
    pass
