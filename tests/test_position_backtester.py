"""Tests for position-based backtester."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from portwine.backtester.position_core import PositionBacktester, PositionBacktestResult
from portwine.strategies.base import StrategyBase
from portwine.data.stores.csvstore import CSVDataStore
from portwine.data.interface import DataInterface


class SimpleStrategy(StrategyBase):
    """Test strategy that returns fixed positions."""

    def step(self, current_date, daily_data):
        return {}  # Start with no actions


def test_imports():
    """Test that imports work."""
    assert PositionBacktester is not None
    assert PositionBacktestResult is not None
