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


def test_position_backtest_result_initialization():
    """Test PositionBacktestResult initializes correctly."""
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOG']

    result = PositionBacktestResult(dates, tickers)

    # Check dimensions
    assert result.positions_array.shape == (10, 3)
    assert result.actions_array.shape == (10, 3)
    assert result.prices_array.shape == (10, 3)
    assert result.portfolio_value.shape == (10,)

    # Check initialization
    assert np.all(result.positions_array == 0)
    assert np.all(result.actions_array == 0)
    assert np.all(np.isnan(result.prices_array))
    assert np.all(result.portfolio_value == 0)

    # Check ticker mapping (note: sorted alphabetically)
    assert result.ticker_to_idx['AAPL'] == 0
    assert result.ticker_to_idx['GOOG'] == 1
    assert result.ticker_to_idx['MSFT'] == 2


def test_position_result_add_action():
    """Test adding actions."""
    dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
    tickers = ['AAPL', 'MSFT']

    result = PositionBacktestResult(dates, tickers)

    # Add actions
    result.add_action(0, 'AAPL', 10.0)
    result.add_action(1, 'AAPL', 5.0)
    result.add_action(1, 'MSFT', -3.0)

    # Check actions recorded
    assert result.actions_array[0, 0] == 10.0  # AAPL day 0
    assert result.actions_array[1, 0] == 5.0   # AAPL day 1
    assert result.actions_array[1, 1] == -3.0  # MSFT day 1


def test_position_result_add_price():
    """Test adding prices."""
    dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
    tickers = ['AAPL', 'MSFT']

    result = PositionBacktestResult(dates, tickers)

    # Add prices
    result.add_price(0, 'AAPL', 150.0)
    result.add_price(1, 'AAPL', 155.0)
    result.add_price(1, 'MSFT', 200.0)

    # Check prices recorded
    assert result.prices_array[0, 0] == 150.0
    assert result.prices_array[1, 0] == 155.0
    assert result.prices_array[1, 1] == 200.0
    assert np.isnan(result.prices_array[0, 1])  # MSFT day 0 not set


def test_position_result_update_positions():
    """Test position calculation from actions."""
    dates = pd.date_range('2020-01-01', '2020-01-04', freq='D')
    tickers = ['AAPL']

    result = PositionBacktestResult(dates, tickers)

    # Add cumulative actions
    result.add_action(0, 'AAPL', 10.0)   # Buy 10
    result.add_action(1, 'AAPL', 5.0)    # Buy 5 more
    result.add_action(2, 'AAPL', -3.0)   # Sell 3
    result.add_action(3, 'AAPL', 0.0)    # No action

    result.update_positions()

    # Check cumulative positions
    assert result.positions_array[0, 0] == 10.0   # 0 + 10
    assert result.positions_array[1, 0] == 15.0   # 10 + 5
    assert result.positions_array[2, 0] == 12.0   # 15 - 3
    assert result.positions_array[3, 0] == 12.0   # 12 + 0


def test_position_result_calculate_portfolio_value():
    """Test portfolio value calculation."""
    dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
    tickers = ['AAPL', 'MSFT']

    result = PositionBacktestResult(dates, tickers)

    # Set up positions
    result.add_action(0, 'AAPL', 10.0)
    result.add_action(0, 'MSFT', 5.0)
    result.add_action(1, 'AAPL', 5.0)   # Add 5 more AAPL
    result.update_positions()

    # Set prices
    result.add_price(0, 'AAPL', 100.0)
    result.add_price(0, 'MSFT', 200.0)
    result.add_price(1, 'AAPL', 110.0)
    result.add_price(1, 'MSFT', 210.0)

    result.calculate_portfolio_value()

    # Day 0: 10 AAPL × $100 + 5 MSFT × $200 = $2000
    assert result.portfolio_value[0] == 2000.0

    # Day 1: 15 AAPL × $110 + 5 MSFT × $210 = $2700
    assert result.portfolio_value[1] == 2700.0

    # Day 2: Same positions, no prices → 0 (NaN treated as 0)
    assert result.portfolio_value[2] == 0.0


def test_position_result_to_dict():
    """Test conversion to output dictionary."""
    dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
    tickers = ['AAPL', 'MSFT']

    result = PositionBacktestResult(dates, tickers)

    # Add some data
    result.add_action(0, 'AAPL', 10.0)
    result.add_action(1, 'MSFT', 5.0)
    result.update_positions()

    result.add_price(0, 'AAPL', 100.0)
    result.add_price(1, 'AAPL', 110.0)
    result.add_price(1, 'MSFT', 200.0)
    result.calculate_portfolio_value()

    # Convert to dict
    output = result.to_dict()

    # Check structure
    assert 'positions_df' in output
    assert 'actions_df' in output
    assert 'prices_df' in output
    assert 'portfolio_value' in output

    # Check types
    assert isinstance(output['positions_df'], pd.DataFrame)
    assert isinstance(output['actions_df'], pd.DataFrame)
    assert isinstance(output['prices_df'], pd.DataFrame)
    assert isinstance(output['portfolio_value'], pd.Series)

    # Check shapes
    assert output['positions_df'].shape == (3, 2)
    assert output['actions_df'].shape == (3, 2)
    assert output['prices_df'].shape == (3, 2)
    assert len(output['portfolio_value']) == 3

    # Check indices
    pd.testing.assert_index_equal(output['positions_df'].index, dates)
    pd.testing.assert_index_equal(output['portfolio_value'].index, dates)

    # Check columns
    assert list(output['positions_df'].columns) == ['AAPL', 'MSFT']

    # Check values
    assert output['positions_df'].loc['2020-01-01', 'AAPL'] == 10.0
    assert output['positions_df'].loc['2020-01-02', 'AAPL'] == 10.0  # No new action
    assert output['positions_df'].loc['2020-01-02', 'MSFT'] == 5.0

    assert output['portfolio_value'].loc['2020-01-01'] == 1000.0  # 10 × 100
    assert output['portfolio_value'].loc['2020-01-02'] == 2100.0  # 10 × 110 + 5 × 200
