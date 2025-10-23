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


def test_position_backtester_initialization():
    """Test PositionBacktester initialization."""
    # Create minimal data interface (mock)
    from portwine.data.stores.csvstore import CSVDataStore
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = CSVDataStore(tmp_dir)
        data = DataInterface(store)

        backtester = PositionBacktester(data)

        assert backtester.data is not None
        assert backtester.calendar is not None
        assert backtester.restricted_data is not None


@pytest.fixture
def sample_csv_data(tmp_path):
    """
    Create sample CSV data for testing.

    Returns:
        DataInterface with 3 days of 2 tickers
    """
    from portwine.data.stores.csvstore import CSVDataStore

    # Create temp directory
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Create AAPL data
    aapl_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-01-03', freq='D'),
        'open': [100.0, 102.0, 101.0],
        'high': [105.0, 106.0, 103.0],
        'low': [99.0, 101.0, 100.0],
        'close': [103.0, 104.0, 102.0],
        'volume': [1000000, 1100000, 1050000]
    })
    aapl_data.to_csv(data_dir / "AAPL.csv", index=False)

    # Create MSFT data
    msft_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-01-03', freq='D'),
        'open': [200.0, 205.0, 203.0],
        'high': [210.0, 208.0, 206.0],
        'low': [198.0, 202.0, 201.0],
        'close': [206.0, 207.0, 204.0],
        'volume': [2000000, 2100000, 2050000]
    })
    msft_data.to_csv(data_dir / "MSFT.csv", index=False)

    # Create data interface
    store = CSVDataStore(str(data_dir))
    data = DataInterface(store)

    return data


class BuyAndHoldStrategy(StrategyBase):
    """Test strategy: buy shares on first day, hold forever."""

    def __init__(self, tickers, shares=10):
        super().__init__(tickers)
        self.shares = shares
        self.bought = False

    def step(self, current_date, daily_data):
        if not self.bought:
            self.bought = True
            # Buy shares of all tickers
            return {ticker: self.shares for ticker in self.tickers}
        return {}  # Hold


class DailyTradeStrategy(StrategyBase):
    """Test strategy: buy 5 shares every day."""

    def step(self, current_date, daily_data):
        return {'AAPL': 5}  # Buy 5 AAPL every day


def test_sample_data_fixture(sample_csv_data):
    """Test that sample data fixture works."""
    data = sample_csv_data

    # Check we can access data
    data.set_current_timestamp(pd.Timestamp('2020-01-01'))
    aapl = data['AAPL']

    assert aapl is not None
    assert 'close' in aapl
    assert aapl['close'] == 103.0


def test_buy_and_hold_strategy(sample_csv_data):
    """Test buy-and-hold strategy with position backtester."""
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-01',
        end_date='2020-01-03'
    )

    # Check results structure
    assert 'positions_df' in results
    assert 'actions_df' in results
    assert 'prices_df' in results
    assert 'portfolio_value' in results

    # Check we have data (Jan 1 is New Year's Day, market closed)
    assert len(results['positions_df']) == 2  # 2 trading days (Jan 2, Jan 3)
    assert len(results['positions_df'].columns) == 2  # 2 tickers

    # Check actions on day 1 (first trading day is Jan 2)
    assert results['actions_df'].loc['2020-01-02', 'AAPL'] == 10.0
    assert results['actions_df'].loc['2020-01-02', 'MSFT'] == 10.0

    # Check actions on day 2 (should be zero)
    assert results['actions_df'].loc['2020-01-03', 'AAPL'] == 0.0
    assert results['actions_df'].loc['2020-01-03', 'MSFT'] == 0.0

    # Check cumulative positions
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 10.0
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.0  # Held

    assert results['positions_df'].loc['2020-01-02', 'MSFT'] == 10.0
    assert results['positions_df'].loc['2020-01-03', 'MSFT'] == 10.0

    # Check prices recorded (prices from CSV for Jan 2, Jan 3)
    assert results['prices_df'].loc['2020-01-02', 'AAPL'] == 104.0
    assert results['prices_df'].loc['2020-01-03', 'AAPL'] == 102.0

    # Check portfolio value
    # Day 1 (Jan 2): 10 AAPL @ $104 + 10 MSFT @ $207 = $3110
    expected_day1 = 10 * 104.0 + 10 * 207.0
    assert results['portfolio_value'].loc['2020-01-02'] == expected_day1

    # Day 2 (Jan 3): 10 AAPL @ $102 + 10 MSFT @ $204 = $3060
    expected_day2 = 10 * 102.0 + 10 * 204.0
    assert results['portfolio_value'].loc['2020-01-03'] == expected_day2
