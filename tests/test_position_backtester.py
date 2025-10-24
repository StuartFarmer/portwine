"""Tests for position-based backtester."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from portwine.backtester.position_core import PositionBacktester, PositionBacktestResult
from portwine.strategies.base import StrategyBase
from portwine.data.stores.csvstore import CSVDataStore
from portwine.data.interface import DataInterface
from tests.calendar_utils import TestDailyMarketCalendar


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
    """Test position calculation from actions with signal shifting."""
    dates = pd.date_range('2020-01-01', '2020-01-04', freq='D')
    tickers = ['AAPL']

    result = PositionBacktestResult(dates, tickers)

    # Add cumulative actions
    result.add_action(0, 'AAPL', 10.0)   # Buy 10 (executes day 1)
    result.add_action(1, 'AAPL', 5.0)    # Buy 5 more (executes day 2)
    result.add_action(2, 'AAPL', -3.0)   # Sell 3 (executes day 3)
    result.add_action(3, 'AAPL', 0.0)    # No action

    result.update_positions()

    # Check cumulative positions WITH SIGNAL SHIFTING
    # Day 0: position = 0 (no previous actions)
    assert result.positions_array[0, 0] == 0.0
    # Day 1: position = 10 (day 0's action of +10)
    assert result.positions_array[1, 0] == 10.0
    # Day 2: position = 15 (cumsum of day 0's +10 and day 1's +5)
    assert result.positions_array[2, 0] == 15.0
    # Day 3: position = 12 (cumsum of +10, +5, -3)
    assert result.positions_array[3, 0] == 12.0


def test_position_result_calculate_portfolio_value():
    """Test portfolio value calculation with signal shifting."""
    dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
    tickers = ['AAPL', 'MSFT']

    result = PositionBacktestResult(dates, tickers)

    # Set up actions (with signal shifting, positions lag by 1 day)
    result.add_action(0, 'AAPL', 10.0)  # Executes day 1
    result.add_action(0, 'MSFT', 5.0)   # Executes day 1
    result.add_action(1, 'AAPL', 5.0)   # Add 5 more AAPL, executes day 2
    result.update_positions()

    # Set prices
    result.add_price(0, 'AAPL', 100.0)
    result.add_price(0, 'MSFT', 200.0)
    result.add_price(1, 'AAPL', 110.0)
    result.add_price(1, 'MSFT', 210.0)

    result.calculate_portfolio_value()

    # Day 0: position = 0 (no previous actions), value = $0
    assert result.portfolio_value[0] == 0.0

    # Day 1: position = 10 AAPL + 5 MSFT (from day 0 actions)
    #        value = 10 × $110 + 5 × $210 = $2150
    assert result.portfolio_value[1] == 2150.0

    # Day 2: position = 15 AAPL + 5 MSFT (day 0 + day 1 actions)
    #        no prices → 0 (NaN treated as 0)
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

    # Check values (WITH SIGNAL SHIFTING)
    # Day 0 (2020-01-01): AAPL action=10 but position=0 (executes day 1)
    assert output['positions_df'].loc['2020-01-01', 'AAPL'] == 0.0
    # Day 1 (2020-01-02): AAPL position=10 (day 0's action), MSFT action=5 but position=0
    assert output['positions_df'].loc['2020-01-02', 'AAPL'] == 10.0
    assert output['positions_df'].loc['2020-01-02', 'MSFT'] == 0.0
    # Day 2 (2020-01-03): AAPL position=10, MSFT position=5 (day 1's action)
    assert output['positions_df'].loc['2020-01-03', 'MSFT'] == 5.0

    # Portfolio values (WITH SIGNAL SHIFTING)
    # Day 0: position=0, value=0
    assert output['portfolio_value'].loc['2020-01-01'] == 0.0
    # Day 1: AAPL position=10 @ $110, MSFT position=0, value = 10 × 110 = 1100
    assert output['portfolio_value'].loc['2020-01-02'] == 1100.0
    # Day 2: AAPL position=10, MSFT position=5 @ $200, but no AAPL price on day 2
    # value = 0 × ? + 5 × 200 = 1000 (AAPL has no price on day 2, treated as 0)
    # Actually wait - let me check what prices are set
    # Prices: day 0 AAPL=$100, day 1 AAPL=$110 and MSFT=$200
    # So day 2 has no prices at all, value = 0
    assert output['portfolio_value'].loc['2020-01-03'] == 0.0


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

    # Create AAPL data (Jan 2-3 only, simulating Jan 1 as market holiday)
    aapl_data = pd.DataFrame({
        'date': pd.date_range('2020-01-02', '2020-01-03', freq='D'),
        'open': [102.0, 101.0],
        'high': [106.0, 103.0],
        'low': [101.0, 100.0],
        'close': [104.0, 102.0],
        'volume': [1100000, 1050000]
    })
    aapl_data.to_csv(data_dir / "AAPL.csv", index=False)

    # Create MSFT data (Jan 2-3 only, simulating Jan 1 as market holiday)
    msft_data = pd.DataFrame({
        'date': pd.date_range('2020-01-02', '2020-01-03', freq='D'),
        'open': [205.0, 203.0],
        'high': [208.0, 206.0],
        'low': [202.0, 201.0],
        'close': [207.0, 204.0],
        'volume': [2100000, 2050000]
    })
    msft_data.to_csv(data_dir / "MSFT.csv", index=False)

    # Create data interface
    store = CSVDataStore(str(data_dir))
    data = DataInterface(store)

    return data


@pytest.fixture
def test_calendar():
    """Create a test calendar for 2020-01-02 to 2020-01-03 (simulating Jan 1 as holiday)."""
    return TestDailyMarketCalendar(
        mode="all",
        allowed_year=2020,
        default_start="2020-01-02",
        default_end="2020-01-03"
    )


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


def test_sample_data_fixture(sample_csv_data, test_calendar):
    """Test that sample data fixture works."""
    data = sample_csv_data

    # Check we can access data (Jan 2 is first trading day)
    data.set_current_timestamp(pd.Timestamp('2020-01-02'))
    aapl = data['AAPL']

    assert aapl is not None
    assert 'close' in aapl
    assert aapl['close'] == 104.0


def test_buy_and_hold_strategy(sample_csv_data, test_calendar):
    """Test buy-and-hold strategy with position backtester."""
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03'
    )

    # Check results structure
    assert 'positions_df' in results
    assert 'actions_df' in results
    assert 'prices_df' in results
    assert 'portfolio_value' in results

    # Check we have data
    assert len(results['positions_df']) == 2  # 2 trading days (Jan 2, Jan 3)
    assert len(results['positions_df'].columns) == 2  # 2 tickers

    # Check actions on day 1 (first trading day is Jan 2)
    assert results['actions_df'].loc['2020-01-02', 'AAPL'] == 10.0
    assert results['actions_df'].loc['2020-01-02', 'MSFT'] == 10.0

    # Check actions on day 2 (should be zero - buy and hold)
    assert results['actions_df'].loc['2020-01-03', 'AAPL'] == 0.0
    assert results['actions_df'].loc['2020-01-03', 'MSFT'] == 0.0

    # Check cumulative positions (WITH SIGNAL SHIFTING)
    # Jan 2: Position = 0 (no previous actions, today's action executes tomorrow)
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 0.0
    assert results['positions_df'].loc['2020-01-02', 'MSFT'] == 0.0

    # Jan 3: Position = 10 (yesterday's action of 10 shares now executes)
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.0
    assert results['positions_df'].loc['2020-01-03', 'MSFT'] == 10.0

    # Check prices recorded (prices from CSV for Jan 2, Jan 3)
    assert results['prices_df'].loc['2020-01-02', 'AAPL'] == 104.0
    assert results['prices_df'].loc['2020-01-03', 'AAPL'] == 102.0

    # Check portfolio value (WITH SIGNAL SHIFTING)
    # Jan 2: Position = 0, so portfolio value = $0
    assert results['portfolio_value'].loc['2020-01-02'] == 0.0

    # Jan 3: Position = 10 AAPL @ $102 + 10 MSFT @ $204 = $3060
    expected_day2 = 10 * 102.0 + 10 * 204.0
    assert results['portfolio_value'].loc['2020-01-03'] == expected_day2


def test_daily_trade_strategy_positions(sample_csv_data, test_calendar):
    """Test accumulating positions with daily trades."""
    strategy = DailyTradeStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03'
    )

    positions = results['positions_df']
    actions = results['actions_df']

    # Check actions each day
    assert actions.loc['2020-01-02', 'AAPL'] == 5.0
    assert actions.loc['2020-01-03', 'AAPL'] == 5.0

    # Check cumulative positions (WITH SIGNAL SHIFTING)
    # Jan 2: action=5, but position=0 (today's action executes tomorrow)
    assert positions.loc['2020-01-02', 'AAPL'] == 0.0
    # Jan 3: action=5, position=5 (yesterday's action now executes)
    assert positions.loc['2020-01-03', 'AAPL'] == 5.0


def test_sell_position(sample_csv_data, test_calendar):
    """Test selling positions."""
    class BuySellStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.day = 0

        def step(self, current_date, daily_data):
            self.day += 1
            if self.day == 1:
                return {'AAPL': 20}  # Buy 20
            elif self.day == 2:
                return {'AAPL': -10}  # Sell 10
            return {}

    strategy = BuySellStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    positions = results['positions_df']

    # WITH SIGNAL SHIFTING:
    # Jan 2 (day 1): action=+20, position=0 (today's action executes tomorrow)
    assert positions.loc['2020-01-02', 'AAPL'] == 0.0
    # Jan 3 (day 2): action=-10, position=20 (yesterday's +20 executes today)
    assert positions.loc['2020-01-03', 'AAPL'] == 20.0


def test_short_position(sample_csv_data, test_calendar):
    """Test short positions (negative quantities)."""
    class ShortStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.day = 0

        def step(self, current_date, daily_data):
            self.day += 1
            if self.day == 1:
                return {'AAPL': -10}  # Sell short 10
            return {}

    strategy = ShortStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    positions = results['positions_df']

    # WITH SIGNAL SHIFTING:
    # Jan 2 (day 1): action=-10, position=0 (executes tomorrow)
    assert positions.loc['2020-01-02', 'AAPL'] == 0.0
    # Jan 3 (day 2): action=0, position=-10 (yesterday's -10 executes)
    assert positions.loc['2020-01-03', 'AAPL'] == -10.0


def test_prices_tracked(sample_csv_data, test_calendar):
    """Test that execution prices are tracked."""
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    prices = results['prices_df']

    # Check prices recorded (should be close prices from CSV)
    assert prices.loc['2020-01-02', 'AAPL'] == 104.0
    assert prices.loc['2020-01-03', 'AAPL'] == 102.0

    assert prices.loc['2020-01-02', 'MSFT'] == 207.0
    assert prices.loc['2020-01-03', 'MSFT'] == 204.0


def test_portfolio_value_calculation(sample_csv_data, test_calendar):
    """Test portfolio value calculation with changing positions."""
    class TradeStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.day = 0

        def step(self, current_date, daily_data):
            self.day += 1
            if self.day == 1:
                return {'AAPL': 10}  # Buy 10 AAPL
            elif self.day == 2:
                return {'AAPL': 5, 'MSFT': 5}  # Buy 5 more AAPL, 5 MSFT
            return {}

    strategy = TradeStrategy(['AAPL', 'MSFT'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    portfolio_value = results['portfolio_value']

    # WITH SIGNAL SHIFTING:
    # Jan 2 (day 1): action=10 AAPL, position=0, value=$0
    assert portfolio_value.loc['2020-01-02'] == 0.0

    # Jan 3 (day 2): action=(5 AAPL, 5 MSFT), position=10 AAPL (from day 1)
    #                value = 10 AAPL @ $102 = $1020
    assert portfolio_value.loc['2020-01-03'] == 10 * 102.0


def test_empty_strategy(sample_csv_data, test_calendar):
    """Test strategy that never trades."""
    class EmptyStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {}  # Never trade

    strategy = EmptyStrategy(['AAPL', 'MSFT'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    # All positions should be zero
    assert (results['positions_df'] == 0).all().all()
    assert (results['actions_df'] == 0).all().all()
    assert (results['portfolio_value'] == 0).all()


def test_fractional_shares(sample_csv_data, test_calendar):
    """Test fractional shares are allowed."""
    class FractionalStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.traded = False

        def step(self, current_date, daily_data):
            if not self.traded:
                self.traded = True
                return {'AAPL': 10.5}  # Fractional share
            return {}

    strategy = FractionalStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    # WITH SIGNAL SHIFTING:
    # Jan 2: action=10.5, position=0 (executes tomorrow)
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 0.0
    # Jan 3: position=10.5 (yesterday's action)
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.5

    # Portfolio value should handle fractional shares (WITH SIGNAL SHIFTING)
    # Jan 2: position=0, value=0
    assert results['portfolio_value'].loc['2020-01-02'] == 0.0
    # Jan 3: position=10.5 @ $102 = $1071
    expected = 10.5 * 102.0
    assert results['portfolio_value'].loc['2020-01-03'] == expected


def test_invalid_ticker_raises_error(sample_csv_data, test_calendar):
    """Test that invalid ticker raises error."""
    class InvalidTickerStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {'INVALID_TICKER': 10}

    strategy = InvalidTickerStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    with pytest.raises(ValueError, match="not in current universe"):
        backtester.run_backtest(strategy)


def test_non_numeric_action_raises_error(sample_csv_data, test_calendar):
    """Test that non-numeric action raises error."""
    class BadActionStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {'AAPL': 'not_a_number'}

    strategy = BadActionStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    with pytest.raises(ValueError, match="must be numeric"):
        backtester.run_backtest(strategy)


def test_nan_action_raises_error(sample_csv_data, test_calendar):
    """Test that NaN action raises error."""
    class NaNStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {'AAPL': np.nan}

    strategy = NaNStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    with pytest.raises(ValueError, match="Invalid action"):
        backtester.run_backtest(strategy)


def test_strategy_returns_none(sample_csv_data, test_calendar):
    """Test strategy that returns None is handled gracefully."""
    class NoneStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return None  # Return None instead of {}

    strategy = NoneStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    # Should be treated same as empty dict
    assert (results['positions_df'] == 0).all().all()


def test_strategy_returns_series(sample_csv_data, test_calendar):
    """Test strategy that returns pandas Series."""
    class SeriesStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.traded = False

        def step(self, current_date, daily_data):
            if not self.traded:
                self.traded = True
                return pd.Series({'AAPL': 10, 'MSFT': 5})
            return pd.Series()

    strategy = SeriesStrategy(['AAPL', 'MSFT'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(strategy)

    # WITH SIGNAL SHIFTING:
    # Jan 2: action=(10, 5), position=(0, 0)
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 0.0
    assert results['positions_df'].loc['2020-01-02', 'MSFT'] == 0.0
    # Jan 3: position=(10, 5)
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.0
    assert results['positions_df'].loc['2020-01-03', 'MSFT'] == 5.0


def test_execution_price_open(sample_csv_data, test_calendar):
    """Test execution at open price instead of close."""
    strategy = BuyAndHoldStrategy(['AAPL'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03',
        execution_price='open'  # Use open prices
    )

    # WITH SIGNAL SHIFTING and execution_price='open':
    # Jan 2: action=10, position=0, price=OPEN($102)
    # Jan 3: position=10 (yesterday's action executes at today's OPEN)
    
    # Check prices are open prices
    assert results['prices_df'].loc['2020-01-02', 'AAPL'] == 102.0  # Open price
    assert results['prices_df'].loc['2020-01-03', 'AAPL'] == 101.0  # Open price
    
    # Check positions
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 0.0
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.0
    
    # Check portfolio value (position executes at Jan 3 open = $101)
    assert results['portfolio_value'].loc['2020-01-02'] == 0.0
    assert results['portfolio_value'].loc['2020-01-03'] == 10.0 * 101.0  # 10 shares @ $101 open


def test_execution_price_comparison(sample_csv_data, test_calendar):
    """Compare execution at open vs close."""
    strategy = BuyAndHoldStrategy(['AAPL'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    # Run with close execution
    results_close = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03',
        execution_price='close'
    )

    # Reset strategy
    strategy.bought = False
    
    # Run with open execution
    results_open = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03',
        execution_price='open'
    )

    # Actions should be identical
    pd.testing.assert_frame_equal(
        results_close['actions_df'],
        results_open['actions_df']
    )

    # Positions should be identical  
    pd.testing.assert_frame_equal(
        results_close['positions_df'],
        results_open['positions_df']
    )

    # Prices should be different (open vs close)
    assert results_close['prices_df'].loc['2020-01-03', 'AAPL'] == 102.0  # Close
    assert results_open['prices_df'].loc['2020-01-03', 'AAPL'] == 101.0   # Open

    # Portfolio values should be different
    # Close: 10 shares @ $102 = $1020
    # Open:  10 shares @ $101 = $1010
    assert results_close['portfolio_value'].loc['2020-01-03'] == 1020.0
    assert results_open['portfolio_value'].loc['2020-01-03'] == 1010.0


def test_execution_price_invalid(sample_csv_data, test_calendar):
    """Test that invalid execution_price raises error."""
    strategy = BuyAndHoldStrategy(['AAPL'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    with pytest.raises(ValueError, match="execution_price must be 'close' or 'open'"):
        backtester.run_backtest(
            strategy,
            execution_price='invalid'
        )


# ==============================================================================
# DUPLICATE TESTS WITH execution_price='open' FOR EXTRA SANITY
# ==============================================================================

def test_buy_and_hold_strategy_open_execution(sample_csv_data, test_calendar):
    """
    Duplicate of test_buy_and_hold_strategy but with execution_price='open'.
    
    This tests that:
    - Actions are identical between close/open execution
    - Positions are identical
    - Prices use open instead of close
    - Portfolio values differ based on open vs close prices
    """
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03',
        execution_price='open'  # USE OPEN PRICES
    )

    # Check results structure (same as close execution)
    assert 'positions_df' in results
    assert 'actions_df' in results
    assert 'prices_df' in results
    assert 'portfolio_value' in results

    # Check we have data
    assert len(results['positions_df']) == 2  # 2 trading days (Jan 2, Jan 3)
    assert len(results['positions_df'].columns) == 2  # 2 tickers

    # Check actions on day 1 (first trading day is Jan 2) - SAME AS CLOSE
    assert results['actions_df'].loc['2020-01-02', 'AAPL'] == 10.0
    assert results['actions_df'].loc['2020-01-02', 'MSFT'] == 10.0

    # Check actions on day 2 (should be zero - buy and hold) - SAME AS CLOSE
    assert results['actions_df'].loc['2020-01-03', 'AAPL'] == 0.0
    assert results['actions_df'].loc['2020-01-03', 'MSFT'] == 0.0

    # Check cumulative positions (WITH SIGNAL SHIFTING) - SAME AS CLOSE
    # Jan 2: Position = 0 (no previous actions, today's action executes tomorrow)
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 0.0
    assert results['positions_df'].loc['2020-01-02', 'MSFT'] == 0.0

    # Jan 3: Position = 10 (yesterday's action of 10 shares now executes)
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.0
    assert results['positions_df'].loc['2020-01-03', 'MSFT'] == 10.0

    # Check prices recorded (OPEN PRICES - DIFFERENT FROM CLOSE)
    # CSV data: Jan 2 open=$102, Jan 3 open=$101 for AAPL
    #           Jan 2 open=$205, Jan 3 open=$203 for MSFT
    assert results['prices_df'].loc['2020-01-02', 'AAPL'] == 102.0  # OPEN (was 104 close)
    assert results['prices_df'].loc['2020-01-03', 'AAPL'] == 101.0  # OPEN (was 102 close)
    assert results['prices_df'].loc['2020-01-02', 'MSFT'] == 205.0  # OPEN (was 207 close)
    assert results['prices_df'].loc['2020-01-03', 'MSFT'] == 203.0  # OPEN (was 204 close)

    # Check portfolio value (WITH SIGNAL SHIFTING AND OPEN EXECUTION)
    # Jan 2: Position = 0, so portfolio value = $0
    assert results['portfolio_value'].loc['2020-01-02'] == 0.0

    # Jan 3: Position = 10 AAPL @ OPEN $101 + 10 MSFT @ OPEN $203 = $3040
    # (Close execution would be: 10 × $102 + 10 × $204 = $3060)
    expected_day2 = 10 * 101.0 + 10 * 203.0  # $3040
    assert results['portfolio_value'].loc['2020-01-03'] == expected_day2
    
    # VERIFY DIFFERENCE: Open execution gives $3040 vs close execution $3060
    # Difference = $20 on a $3060 position = 0.65% lower


def test_prices_tracked_open_execution(sample_csv_data, test_calendar):
    """
    Duplicate of test_prices_tracked but with execution_price='open'.
    
    Verifies that open prices are correctly recorded in prices_df.
    """
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03',
        execution_price='open'
    )

    prices = results['prices_df']

    # Check that OPEN prices are tracked (not close)
    # AAPL: Jan 2 open=$102, Jan 3 open=$101
    assert prices.loc['2020-01-02', 'AAPL'] == 102.0  # Open (close was 104.0)
    assert prices.loc['2020-01-03', 'AAPL'] == 101.0  # Open (close was 102.0)

    # MSFT: Jan 2 open=$205, Jan 3 open=$203
    assert prices.loc['2020-01-02', 'MSFT'] == 205.0  # Open (close was 207.0)
    assert prices.loc['2020-01-03', 'MSFT'] == 203.0  # Open (close was 204.0)


def test_portfolio_value_calculation_open_execution(sample_csv_data, test_calendar):
    """
    Duplicate of test_portfolio_value_calculation but with execution_price='open'.
    
    Tests portfolio value calculation with changing positions using open prices.
    """
    class TradeStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.day = 0

        def step(self, current_date, daily_data):
            self.day += 1
            if self.day == 1:
                return {'AAPL': 10}  # Buy 10 AAPL
            elif self.day == 2:
                return {'AAPL': 5, 'MSFT': 5}  # Buy 5 more AAPL, 5 MSFT
            return {}

    strategy = TradeStrategy(['AAPL', 'MSFT'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        execution_price='open'
    )

    portfolio_value = results['portfolio_value']

    # WITH SIGNAL SHIFTING AND OPEN EXECUTION:
    # Jan 2 (day 1): action=10 AAPL, position=0, value=$0
    assert portfolio_value.loc['2020-01-02'] == 0.0

    # Jan 3 (day 2): action=(5 AAPL, 5 MSFT), position=10 AAPL (from day 1)
    #                value = 10 AAPL @ OPEN $101 = $1010
    # (Close execution would be: 10 × $102 = $1020)
    assert portfolio_value.loc['2020-01-03'] == 10 * 101.0  # $1010 (open)
    
    # VERIFY DIFFERENCE: Open gives $1010 vs close gives $1020
    # Difference = $10 on 10-share position = $1/share or 0.98% lower


def test_fractional_shares_open_execution(sample_csv_data, test_calendar):
    """
    Duplicate of test_fractional_shares but with execution_price='open'.
    
    Tests that fractional shares work correctly with open price execution.
    """
    class FractionalStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.traded = False

        def step(self, current_date, daily_data):
            if not self.traded:
                self.traded = True
                return {'AAPL': 10.5}  # Fractional share
            return {}

    strategy = FractionalStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        execution_price='open'
    )

    # WITH SIGNAL SHIFTING AND OPEN EXECUTION:
    # Jan 2: action=10.5, position=0 (executes tomorrow)
    assert results['positions_df'].loc['2020-01-02', 'AAPL'] == 0.0
    # Jan 3: position=10.5 (yesterday's action)
    assert results['positions_df'].loc['2020-01-03', 'AAPL'] == 10.5

    # Portfolio value should handle fractional shares with OPEN prices
    # Jan 2: position=0, value=0
    assert results['portfolio_value'].loc['2020-01-02'] == 0.0
    
    # Jan 3: position=10.5 @ OPEN $101 = $1060.50
    # (Close execution: 10.5 × $102 = $1071)
    expected = 10.5 * 101.0  # $1060.50
    assert results['portfolio_value'].loc['2020-01-03'] == expected
    
    # VERIFY DIFFERENCE: Open gives $1060.50 vs close gives $1071
    # Difference = $10.50 on 10.5 shares = $1/share or 0.98% lower


def test_short_position_open_execution(sample_csv_data, test_calendar):
    """
    New test for short positions with open execution.
    
    Verifies that short positions are correctly valued with open prices.
    """
    class ShortStrategy(StrategyBase):
        def __init__(self, tickers):
            super().__init__(tickers)
            self.day = 0

        def step(self, current_date, daily_data):
            self.day += 1
            if self.day == 1:
                return {'AAPL': -10}  # Sell short 10
            return {}

    strategy = ShortStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        execution_price='open'
    )

    positions = results['positions_df']

    # WITH SIGNAL SHIFTING AND OPEN EXECUTION:
    # Jan 2 (day 1): action=-10, position=0 (executes tomorrow)
    assert positions.loc['2020-01-02', 'AAPL'] == 0.0
    # Jan 3 (day 2): action=0, position=-10 (yesterday's -10 executes)
    assert positions.loc['2020-01-03', 'AAPL'] == -10.0

    # Portfolio value with short position
    # Jan 2: position=0, value=0
    assert results['portfolio_value'].loc['2020-01-02'] == 0.0
    
    # Jan 3: position=-10 @ OPEN $101 = -$1010
    # (Close execution: -10 × $102 = -$1020)
    assert results['portfolio_value'].loc['2020-01-03'] == -10.0 * 101.0  # -$1010
    
    # VERIFY: Short position at open is LESS negative than at close
    # This is favorable for shorts (lower cost to establish position)


def test_daily_trade_strategy_open_execution(sample_csv_data, test_calendar):
    """
    New test for daily trading with open execution.
    
    Tests accumulating positions with daily trades using open prices.
    """
    strategy = DailyTradeStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data, calendar=test_calendar)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-02',
        end_date='2020-01-03',
        execution_price='open'
    )

    positions = results['positions_df']
    actions = results['actions_df']
    portfolio_value = results['portfolio_value']

    # Check actions each day (same as close execution)
    assert actions.loc['2020-01-02', 'AAPL'] == 5.0
    assert actions.loc['2020-01-03', 'AAPL'] == 5.0

    # Check cumulative positions (WITH SIGNAL SHIFTING) - same as close
    assert positions.loc['2020-01-02', 'AAPL'] == 0.0  # Today's action executes tomorrow
    assert positions.loc['2020-01-03', 'AAPL'] == 5.0  # Yesterday's action executes

    # Check portfolio value with OPEN prices
    # Jan 2: position=0, value=0
    assert portfolio_value.loc['2020-01-02'] == 0.0
    
    # Jan 3: position=5 @ OPEN $101 = $505
    # (Close execution: 5 × $102 = $510)
    assert portfolio_value.loc['2020-01-03'] == 5.0 * 101.0  # $505
