# Position-Based Backtester - Implementation Plan

**Approach**: Lean, iterative, test-driven hybrid
**Philosophy**: Build simplest feature → Test it → Build next feature on top → Repeat

---

## Implementation Strategy

Each iteration:
1. **Implements** the simplest version of one feature
2. **Tests** that feature works correctly
3. **Validates** it integrates with previous features
4. **Commits** with green tests before moving forward

---

## Iteration 0: Project Setup

**Goal**: Create file structure, establish foundation

### Tasks
- [ ] Create `portwine/backtester/position_core.py`
- [ ] Create `tests/test_position_backtester.py`
- [ ] Add imports and basic structure

### Files Created

**`portwine/backtester/position_core.py`**:
```python
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
from portwine.backtester.core import InvalidBenchmarkError, _split_tickers
from portwine.data.interface import DataInterface, MultiDataInterface
from portwine.data.calendar import DailyMarketCalendar


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
```

**`tests/test_position_backtester.py`**:
```python
"""Tests for position-based backtester."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from portwine.backtester.position_core import PositionBacktester, PositionBacktestResult
from portwine.strategies.base import StrategyBase
from portwine.data.stores.csvstore import CSVStore
from portwine.data.interface import DataInterface


class SimpleStrategy(StrategyBase):
    """Test strategy that returns fixed positions."""

    def step(self, current_date, daily_data):
        return {}  # Start with no actions


def test_imports():
    """Test that imports work."""
    assert PositionBacktester is not None
    assert PositionBacktestResult is not None
```

### Acceptance Criteria
- [ ] Files created
- [ ] No import errors
- [ ] `test_imports()` passes

### Validation
```bash
python -c "from portwine.backtester.position_core import PositionBacktester"
pytest tests/test_position_backtester.py::test_imports -v
```

---

## Iteration 1: PositionBacktestResult - Data Storage

**Goal**: Create data structure to hold positions, actions, prices

**Builds On**: Iteration 0 (file structure)

### Implementation

**Add to `PositionBacktestResult` class**:
```python
class PositionBacktestResult:
    """
    Stores position-based backtest results.

    Similar to BacktestResult but tracks positions/actions instead of weights.
    """

    def __init__(self, datetime_index: pd.DatetimeIndex, tickers: List[str]):
        """
        Initialize result storage.

        Args:
            datetime_index: Trading days
            tickers: List of ticker symbols
        """
        self.datetime_index = datetime_index
        self.tickers = sorted(tickers)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        n_days = len(datetime_index)
        n_tickers = len(self.tickers)

        # Core data arrays
        self.positions_array = np.zeros((n_days, n_tickers), dtype=np.float64)
        self.actions_array = np.zeros((n_days, n_tickers), dtype=np.float64)
        self.prices_array = np.full((n_days, n_tickers), np.nan, dtype=np.float64)

        # Portfolio value over time
        self.portfolio_value = np.zeros(n_days, dtype=np.float64)
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
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

    # Check ticker mapping
    assert result.ticker_to_idx['AAPL'] == 0
    assert result.ticker_to_idx['MSFT'] == 2  # Sorted: AAPL, GOOG, MSFT
```

### Acceptance Criteria
- [ ] `PositionBacktestResult` stores arrays
- [ ] Arrays have correct dimensions
- [ ] Arrays initialized to correct defaults
- [ ] Ticker mapping works
- [ ] Test passes

### Validation
```bash
pytest tests/test_position_backtester.py::test_position_backtest_result_initialization -v
```

---

## Iteration 2: PositionBacktestResult - Add Data Methods

**Goal**: Add methods to populate result arrays

**Builds On**: Iteration 1 (data storage structure)

### Implementation

**Add to `PositionBacktestResult` class**:
```python
def add_action(self, day_idx: int, ticker: str, quantity: float):
    """
    Record an action (buy/sell) for a ticker on a given day.

    Args:
        day_idx: Index in datetime_index
        ticker: Ticker symbol
        quantity: Number of shares (positive=buy, negative=sell)
    """
    if ticker not in self.ticker_to_idx:
        return  # Skip tickers not in result set

    ticker_idx = self.ticker_to_idx[ticker]
    self.actions_array[day_idx, ticker_idx] = quantity


def add_price(self, day_idx: int, ticker: str, price: float):
    """
    Record execution price for a ticker on a given day.

    Args:
        day_idx: Index in datetime_index
        ticker: Ticker symbol
        price: Execution price
    """
    if ticker not in self.ticker_to_idx:
        return

    ticker_idx = self.ticker_to_idx[ticker]
    self.prices_array[day_idx, ticker_idx] = price


def update_positions(self):
    """
    Calculate cumulative positions from actions.

    positions[t] = positions[t-1] + actions[t]
    """
    # Cumulative sum along time axis
    self.positions_array = np.cumsum(self.actions_array, axis=0)


def calculate_portfolio_value(self):
    """
    Calculate portfolio value over time.

    portfolio_value[t] = sum(positions[t] × prices[t])
    """
    # Element-wise multiply positions × prices, sum across tickers
    # Handle NaN prices (treat as 0 contribution)
    prices_filled = np.where(np.isnan(self.prices_array), 0.0, self.prices_array)
    self.portfolio_value = np.sum(self.positions_array * prices_filled, axis=1)
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
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
```

### Acceptance Criteria
- [ ] Can add actions
- [ ] Can add prices
- [ ] Positions calculated correctly from cumulative actions
- [ ] Portfolio value = sum(position × price)
- [ ] All tests pass

### Validation
```bash
pytest tests/test_position_backtester.py -v -k "test_position_result"
```

---

## Iteration 3: PositionBacktestResult - Output Formatting

**Goal**: Convert arrays to pandas DataFrames/Series for output

**Builds On**: Iteration 2 (data population methods)

### Implementation

**Add to `PositionBacktestResult` class**:
```python
def to_dict(self) -> dict:
    """
    Convert results to dictionary format.

    Returns:
        dict: Results in same format as Backtester output
            - positions_df: DataFrame of positions over time
            - actions_df: DataFrame of actions over time
            - prices_df: DataFrame of execution prices
            - portfolio_value: Series of portfolio value
    """
    return {
        'positions_df': pd.DataFrame(
            self.positions_array,
            index=self.datetime_index,
            columns=self.tickers
        ),
        'actions_df': pd.DataFrame(
            self.actions_array,
            index=self.datetime_index,
            columns=self.tickers
        ),
        'prices_df': pd.DataFrame(
            self.prices_array,
            index=self.datetime_index,
            columns=self.tickers
        ),
        'portfolio_value': pd.Series(
            self.portfolio_value,
            index=self.datetime_index,
            name='portfolio_value'
        )
    }
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
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
```

### Acceptance Criteria
- [ ] `to_dict()` returns correct structure
- [ ] DataFrames have correct indices (dates)
- [ ] DataFrames have correct columns (tickers)
- [ ] Series has correct index
- [ ] Values preserved correctly
- [ ] Test passes

### Validation
```bash
pytest tests/test_position_backtester.py::test_position_result_to_dict -v
```

---

## Iteration 4: PositionBacktester - Basic Structure

**Goal**: Create PositionBacktester class with initialization

**Builds On**: Iteration 3 (complete PositionBacktestResult)

### Implementation

**Add to `PositionBacktester` class**:
```python
class PositionBacktester:
    """
    Position-based backtester.

    Interprets strategy output as share quantities instead of portfolio weights.
    Uses same data interfaces and patterns as standard Backtester.
    """

    def __init__(self, data_interface, calendar=None):
        """
        Initialize position backtester.

        Args:
            data_interface: DataInterface or MultiDataInterface
            calendar: DailyMarketCalendar (default: NYSE calendar)
        """
        self.data = data_interface
        self.calendar = calendar or DailyMarketCalendar()

        # Create restricted data interface (same as Backtester)
        if isinstance(data_interface, MultiDataInterface):
            from portwine.data.interface import RestrictedDataInterface
            self.restricted_data = RestrictedDataInterface(data_interface.loaders)
        else:
            from portwine.data.interface import RestrictedDataInterface
            self.restricted_data = RestrictedDataInterface({None: data_interface.data_loader})

    def run_backtest(
        self,
        strategy: StrategyBase,
        start_date: Union[str, None] = None,
        end_date: Union[str, None] = None,
        benchmark: Union[str, Callable, None] = None,
        verbose: bool = False,
        require_all_history: bool = False
    ):
        """
        Run position-based backtest.

        Args:
            strategy: StrategyBase instance (interprets output as shares)
            start_date: Start date (auto-detect if None)
            end_date: End date (auto-detect if None)
            benchmark: Benchmark (ticker, function, or None)
            verbose: Show progress bar
            require_all_history: Ensure all tickers have data from start

        Returns:
            dict: Position-based results
        """
        # TODO: Implement in next iteration
        raise NotImplementedError("Coming in next iteration")
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
def test_position_backtester_initialization():
    """Test PositionBacktester initialization."""
    # Create minimal data interface (mock)
    from portwine.data.stores.csvstore import CSVStore
    store = CSVStore("./test_data")
    data = DataInterface(store)

    backtester = PositionBacktester(data)

    assert backtester.data is not None
    assert backtester.calendar is not None
    assert backtester.restricted_data is not None


def test_position_backtester_run_not_implemented():
    """Test run_backtest raises NotImplementedError (for now)."""
    from portwine.data.stores.csvstore import CSVStore
    store = CSVStore("./test_data")
    data = DataInterface(store)
    backtester = PositionBacktester(data)

    strategy = SimpleStrategy(['AAPL'])

    with pytest.raises(NotImplementedError):
        backtester.run_backtest(strategy)
```

### Acceptance Criteria
- [ ] PositionBacktester initializes
- [ ] Takes data interface
- [ ] Creates restricted data interface
- [ ] Has run_backtest method signature
- [ ] Tests pass

### Validation
```bash
pytest tests/test_position_backtester.py::test_position_backtester -v
```

---

## Iteration 5: PositionBacktester - Core Loop (No Actions)

**Goal**: Implement basic backtest loop that runs but doesn't process actions yet

**Builds On**: Iteration 4 (PositionBacktester structure)

### Implementation

**Replace `run_backtest` in `PositionBacktester`**:
```python
def run_backtest(
    self,
    strategy: StrategyBase,
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
    benchmark: Union[str, Callable, None] = None,
    verbose: bool = False,
    require_all_history: bool = False
):
    """Run position-based backtest."""

    # 1. Validate strategy has tickers (same as Backtester)
    if not strategy.universe.all_tickers:
        raise ValueError("Strategy has no tickers.")

    regular_tickers, _ = _split_tickers(set(strategy.universe.all_tickers))

    # 2. Determine date range (same as Backtester)
    end_date = self._compute_effective_end_date(end_date, regular_tickers)

    if start_date is None:
        if isinstance(self.data, MultiDataInterface):
            start_date = self.data.earliest_any_date(regular_tickers)
        else:
            start_date = DataInterface(self.data.data_loader).earliest_any_date(regular_tickers)

    # 3. Get datetime index
    datetime_index = self.calendar.get_datetime_index(start_date, end_date)

    # 4. Handle require_all_history
    if require_all_history:
        if isinstance(self.data, MultiDataInterface):
            common_start = self.data.earliest_common_date(regular_tickers)
        else:
            common_start = DataInterface(self.data.data_loader).earliest_common_date(regular_tickers)
        if start_date is None or pd.Timestamp(start_date) < pd.Timestamp(common_start):
            start_date = common_start
        datetime_index = self.calendar.get_datetime_index(start_date, end_date)

    # 5. Initialize result storage
    result = PositionBacktestResult(datetime_index, sorted(regular_tickers))

    # 6. Main backtest loop
    iterator = tqdm(datetime_index, desc="Position Backtest") if verbose else datetime_index

    for i, dt in enumerate(iterator):
        # Update universe
        strategy.universe.set_datetime(dt)
        current_universe_tickers = strategy.universe.get_constituents(dt)

        # Set up restricted data interface
        self.restricted_data.set_current_timestamp(dt)
        regular_tickers_current, _ = _split_tickers(set(current_universe_tickers))
        self.restricted_data.set_restricted_tickers(regular_tickers_current, prefix=None)

        # Call strategy
        dt_datetime = pd.Timestamp(dt).to_pydatetime()
        actions = strategy.step(dt_datetime, self.restricted_data)

        # TODO: Process actions (next iteration)
        # For now, just continue loop

    # 7. Calculate results
    result.update_positions()
    result.calculate_portfolio_value()

    # 8. Return results (no benchmark yet)
    return result.to_dict()


def _compute_effective_end_date(self, end_date, tickers):
    """Compute effective end date (same logic as Backtester)."""
    if end_date is not None:
        return end_date

    # Find latest date across all tickers
    if isinstance(self.data, MultiDataInterface):
        data_interface = DataInterface(self.data.loaders[None])
    else:
        data_interface = DataInterface(self.data.data_loader)

    latest_dates = []
    for ticker in tickers:
        try:
            latest = data_interface.data_loader.latest(ticker)
            if latest:
                latest_dates.append(latest)
        except (KeyError, AttributeError):
            continue

    if not latest_dates:
        raise ValueError("No data found for any ticker")

    return max(latest_dates)
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
@pytest.fixture
def sample_data_interface():
    """Create sample data interface with mock data."""
    # This will need real test data - for now, skip actual test
    # Just test the structure
    pytest.skip("Need real test data - testing in next iteration")


def test_position_backtester_empty_strategy():
    """Test running backtest with strategy that does nothing."""
    # Will implement with real data in next iteration
    pytest.skip("Need test data setup")
```

### Acceptance Criteria
- [ ] Loop runs through all dates
- [ ] Strategy.step() called for each date
- [ ] Universe updates correctly
- [ ] No errors (even though no actions processed)
- [ ] Returns results dict
- [ ] Test structure ready (actual tests need data)

### Validation
```bash
# This iteration mostly sets up structure
# Real validation comes in next iteration with test data
pytest tests/test_position_backtester.py -v
```

---

## Iteration 6: Test Data Setup

**Goal**: Create fixture with real test data for integration tests

**Builds On**: Iteration 5 (backtester loop structure)

### Implementation

**Add to `tests/test_position_backtester.py`**:
```python
@pytest.fixture
def sample_csv_data(tmp_path):
    """
    Create sample CSV data for testing.

    Returns:
        DataInterface with 3 days of 2 tickers
    """
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
    store = CSVStore(str(data_dir))
    data = DataInterface(store)

    return data


class BuyAndHoldStrategy(StrategyBase):
    """Test strategy: buy 10 shares on first day, hold forever."""

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
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
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

    # Check we have data (even if not processed yet)
    assert len(results['positions_df']) == 3  # 3 days
    assert len(results['positions_df'].columns) == 2  # 2 tickers
```

### Acceptance Criteria
- [ ] Fixture creates real CSV data
- [ ] Can create DataInterface from fixture
- [ ] Can access data through interface
- [ ] Simple strategies defined for testing
- [ ] Test runs without errors
- [ ] Can verify test data is accessible

### Validation
```bash
pytest tests/test_position_backtester.py::test_sample_data_fixture -v
pytest tests/test_position_backtester.py::test_buy_and_hold_strategy -v
```

---

## Iteration 7: Process Actions and Update Positions

**Goal**: Actually process strategy actions and update positions

**Builds On**: Iteration 6 (test data)

### Implementation

**Update `run_backtest` in `PositionBacktester`**:
```python
# In the main loop (replace TODO section):

for i, dt in enumerate(iterator):
    # ... (universe setup same as before)

    # Call strategy
    dt_datetime = pd.Timestamp(dt).to_pydatetime()
    actions = strategy.step(dt_datetime, self.restricted_data)

    # Normalize actions to dict (same as Backtester._normalize_signals)
    if actions is None:
        actions = {}
    elif isinstance(actions, pd.Series):
        actions = actions.to_dict()
    elif not isinstance(actions, dict):
        raise ValueError(f"Strategy returned invalid type: {type(actions)}")

    # Validate actions
    self.validate_actions(actions, current_universe_tickers)

    # Record actions
    for ticker, quantity in actions.items():
        result.add_action(i, ticker, quantity)

    # Record prices (use close price)
    for ticker in regular_tickers_current:
        try:
            price_data = self.restricted_data[ticker]
            close_price = price_data.get('close')
            if close_price is not None:
                result.add_price(i, ticker, close_price)
        except (KeyError, ValueError):
            # Ticker has no data on this day
            pass
```

**Add validation method**:
```python
def validate_actions(self, actions: Dict[str, float], current_universe_tickers: List[str]):
    """
    Validate position actions.

    Args:
        actions: Dict of ticker → quantity
        current_universe_tickers: Valid tickers for this date

    Raises:
        ValueError: If actions are invalid
    """
    for ticker in actions.keys():
        if ticker not in current_universe_tickers:
            raise ValueError(f"Ticker {ticker} not in current universe")

    for ticker, quantity in actions.items():
        if not isinstance(quantity, (int, float)):
            raise ValueError(f"Action for {ticker} must be numeric, got {type(quantity)}")
        if np.isnan(quantity) or np.isinf(quantity):
            raise ValueError(f"Invalid action for {ticker}: {quantity}")
```

### Tests

**Update existing test and add new ones**:
```python
def test_buy_and_hold_strategy_positions(sample_csv_data):
    """Test that positions are tracked correctly."""
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-01',
        end_date='2020-01-03'
    )

    positions = results['positions_df']
    actions = results['actions_df']

    # Check actions on day 1
    assert actions.loc['2020-01-01', 'AAPL'] == 10.0
    assert actions.loc['2020-01-01', 'MSFT'] == 10.0

    # Check actions on day 2 and 3 (should be zero)
    assert actions.loc['2020-01-02', 'AAPL'] == 0.0
    assert actions.loc['2020-01-03', 'AAPL'] == 0.0

    # Check cumulative positions
    assert positions.loc['2020-01-01', 'AAPL'] == 10.0
    assert positions.loc['2020-01-02', 'AAPL'] == 10.0  # Held
    assert positions.loc['2020-01-03', 'AAPL'] == 10.0  # Still held

    assert positions.loc['2020-01-01', 'MSFT'] == 10.0
    assert positions.loc['2020-01-02', 'MSFT'] == 10.0
    assert positions.loc['2020-01-03', 'MSFT'] == 10.0


def test_daily_trade_strategy_positions(sample_csv_data):
    """Test accumulating positions with daily trades."""
    strategy = DailyTradeStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-01',
        end_date='2020-01-03'
    )

    positions = results['positions_df']
    actions = results['actions_df']

    # Check actions each day
    assert actions.loc['2020-01-01', 'AAPL'] == 5.0
    assert actions.loc['2020-01-02', 'AAPL'] == 5.0
    assert actions.loc['2020-01-03', 'AAPL'] == 5.0

    # Check cumulative positions
    assert positions.loc['2020-01-01', 'AAPL'] == 5.0
    assert positions.loc['2020-01-02', 'AAPL'] == 10.0  # 5 + 5
    assert positions.loc['2020-01-03', 'AAPL'] == 15.0  # 10 + 5


def test_sell_position(sample_csv_data):
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
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    positions = results['positions_df']

    assert positions.loc['2020-01-01', 'AAPL'] == 20.0
    assert positions.loc['2020-01-02', 'AAPL'] == 10.0  # 20 - 10
    assert positions.loc['2020-01-03', 'AAPL'] == 10.0  # Hold


def test_short_position(sample_csv_data):
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
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    positions = results['positions_df']

    assert positions.loc['2020-01-01', 'AAPL'] == -10.0
    assert positions.loc['2020-01-02', 'AAPL'] == -10.0
    assert positions.loc['2020-01-03', 'AAPL'] == -10.0
```

### Acceptance Criteria
- [ ] Actions recorded correctly
- [ ] Positions calculated from cumulative actions
- [ ] Buy-and-hold test passes
- [ ] Daily trading test passes
- [ ] Sell position test passes
- [ ] Short position test passes
- [ ] Validation catches invalid actions

### Validation
```bash
pytest tests/test_position_backtester.py -v
```

---

## Iteration 8: Track Prices and Calculate Portfolio Value

**Goal**: Ensure prices are tracked and portfolio value is calculated correctly

**Builds On**: Iteration 7 (position tracking)

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
def test_prices_tracked(sample_csv_data):
    """Test that execution prices are tracked."""
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    prices = results['prices_df']

    # Check prices recorded (should be close prices from CSV)
    assert prices.loc['2020-01-01', 'AAPL'] == 103.0
    assert prices.loc['2020-01-02', 'AAPL'] == 104.0
    assert prices.loc['2020-01-03', 'AAPL'] == 102.0

    assert prices.loc['2020-01-01', 'MSFT'] == 206.0
    assert prices.loc['2020-01-02', 'MSFT'] == 207.0
    assert prices.loc['2020-01-03', 'MSFT'] == 204.0


def test_portfolio_value_calculation(sample_csv_data):
    """Test portfolio value calculation."""
    strategy = BuyAndHoldStrategy(['AAPL', 'MSFT'], shares=10)
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    portfolio_value = results['portfolio_value']
    positions = results['positions_df']
    prices = results['prices_df']

    # Day 1: 10 AAPL @ $103 + 10 MSFT @ $206 = $3090
    expected_day1 = 10 * 103.0 + 10 * 206.0
    assert portfolio_value.loc['2020-01-01'] == expected_day1

    # Day 2: 10 AAPL @ $104 + 10 MSFT @ $207 = $3110
    expected_day2 = 10 * 104.0 + 10 * 207.0
    assert portfolio_value.loc['2020-01-02'] == expected_day2

    # Day 3: 10 AAPL @ $102 + 10 MSFT @ $204 = $3060
    expected_day3 = 10 * 102.0 + 10 * 204.0
    assert portfolio_value.loc['2020-01-03'] == expected_day3

    # Verify formula: portfolio_value = sum(positions × prices)
    manual_calc = (positions * prices).sum(axis=1)
    pd.testing.assert_series_equal(portfolio_value, manual_calc)


def test_portfolio_value_with_trades(sample_csv_data):
    """Test portfolio value with changing positions."""
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
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    portfolio_value = results['portfolio_value']

    # Day 1: 10 AAPL @ $103 = $1030
    assert portfolio_value.loc['2020-01-01'] == 1030.0

    # Day 2: 15 AAPL @ $104 + 5 MSFT @ $207 = $2595
    assert portfolio_value.loc['2020-01-02'] == 15 * 104.0 + 5 * 207.0

    # Day 3: 15 AAPL @ $102 + 5 MSFT @ $204 = $2550
    assert portfolio_value.loc['2020-01-03'] == 15 * 102.0 + 5 * 204.0
```

### Acceptance Criteria
- [ ] Prices tracked correctly (close prices)
- [ ] Portfolio value = sum(positions × prices)
- [ ] Tests pass for static positions
- [ ] Tests pass for changing positions
- [ ] Portfolio value updates correctly each day

### Validation
```bash
pytest tests/test_position_backtester.py::test_prices_tracked -v
pytest tests/test_position_backtester.py::test_portfolio_value -v
```

---

## Iteration 9: Edge Cases and Validation

**Goal**: Handle edge cases and validate input

**Builds On**: Iteration 8 (complete basic functionality)

### Implementation

**Add validation and edge case handling**:
```python
# In validate_actions, add more checks:
def validate_actions(self, actions: Dict[str, float], current_universe_tickers: List[str]):
    """Validate position actions with comprehensive checks."""
    if not isinstance(actions, dict):
        raise ValueError(f"Actions must be dict, got {type(actions)}")

    for ticker in actions.keys():
        if not isinstance(ticker, str):
            raise ValueError(f"Ticker must be string, got {type(ticker)}")
        if ticker not in current_universe_tickers:
            raise ValueError(
                f"Ticker {ticker} not in current universe. "
                f"Available: {current_universe_tickers}"
            )

    for ticker, quantity in actions.items():
        if not isinstance(quantity, (int, float)):
            raise ValueError(
                f"Action for {ticker} must be numeric (int or float), "
                f"got {type(quantity)}"
            )
        if np.isnan(quantity):
            raise ValueError(f"Action for {ticker} is NaN")
        if np.isinf(quantity):
            raise ValueError(f"Action for {ticker} is infinite")
```

### Tests

**Add to `tests/test_position_backtester.py`**:
```python
def test_invalid_ticker_raises_error(sample_csv_data):
    """Test that invalid ticker raises error."""
    class InvalidTickerStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {'INVALID_TICKER': 10}

    strategy = InvalidTickerStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data)

    with pytest.raises(ValueError, match="not in current universe"):
        backtester.run_backtest(strategy)


def test_non_numeric_action_raises_error(sample_csv_data):
    """Test that non-numeric action raises error."""
    class BadActionStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {'AAPL': 'not_a_number'}

    strategy = BadActionStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data)

    with pytest.raises(ValueError, match="must be numeric"):
        backtester.run_backtest(strategy)


def test_nan_action_raises_error(sample_csv_data):
    """Test that NaN action raises error."""
    class NaNStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {'AAPL': np.nan}

    strategy = NaNStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data)

    with pytest.raises(ValueError, match="NaN"):
        backtester.run_backtest(strategy)


def test_empty_strategy(sample_csv_data):
    """Test strategy that never trades."""
    class EmptyStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return {}  # Never trade

    strategy = EmptyStrategy(['AAPL', 'MSFT'])
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    # All positions should be zero
    assert (results['positions_df'] == 0).all().all()
    assert (results['actions_df'] == 0).all().all()
    assert (results['portfolio_value'] == 0).all()


def test_fractional_shares(sample_csv_data):
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
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    assert results['positions_df'].loc['2020-01-01', 'AAPL'] == 10.5

    # Portfolio value should handle fractional shares
    expected = 10.5 * 103.0
    assert results['portfolio_value'].loc['2020-01-01'] == expected


def test_strategy_returns_none(sample_csv_data):
    """Test strategy that returns None is handled gracefully."""
    class NoneStrategy(StrategyBase):
        def step(self, current_date, daily_data):
            return None  # Return None instead of {}

    strategy = NoneStrategy(['AAPL'])
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    # Should be treated same as empty dict
    assert (results['positions_df'] == 0).all().all()


def test_strategy_returns_series(sample_csv_data):
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
    backtester = PositionBacktester(sample_csv_data)

    results = backtester.run_backtest(strategy)

    assert results['positions_df'].loc['2020-01-01', 'AAPL'] == 10.0
    assert results['positions_df'].loc['2020-01-01', 'MSFT'] == 5.0
```

### Acceptance Criteria
- [ ] Invalid tickers caught
- [ ] Non-numeric actions caught
- [ ] NaN/inf actions caught
- [ ] Empty strategy works (all zeros)
- [ ] Fractional shares work
- [ ] None return handled
- [ ] Series return handled
- [ ] All tests pass

### Validation
```bash
pytest tests/test_position_backtester.py -v
```

---

## Iteration 10: Benchmark Support (Optional)

**Goal**: Add basic benchmark support (defer complex comparisons)

**Builds On**: Iteration 9 (complete core functionality)

### Implementation

**Update `run_backtest` to handle benchmarks**:
```python
def run_backtest(
    self,
    strategy: StrategyBase,
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
    benchmark: Union[str, Callable, None] = None,
    verbose: bool = False,
    require_all_history: bool = False
):
    """Run position-based backtest."""

    # ... (previous code for main loop)

    # After main loop, before return:

    # Handle benchmark if provided
    benchmark_returns = None
    if benchmark is not None:
        # Use same benchmark logic as regular Backtester
        # Get benchmark type
        if isinstance(self.data, MultiDataInterface):
            data_loader = self.data.loaders[None]
        else:
            data_loader = self.data.data_loader

        bm_type = get_benchmark_type(benchmark, data_loader)

        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark")

        # Calculate benchmark
        # For simplicity, calculate % returns on portfolio value
        portfolio_value_series = result.to_dict()['portfolio_value']
        portfolio_returns = portfolio_value_series.pct_change().fillna(0)

        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            # Custom benchmark gets returns DataFrame (not relevant for positions)
            # For now, skip custom benchmarks
            pass
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            # Standard benchmarks expect returns DataFrame
            # Create dummy returns DataFrame for now
            # This is a placeholder - real implementation would need thought
            pass
        else:  # TICKER
            # Calculate ticker benchmark returns
            benchmark_returns = self._calculate_ticker_benchmark_returns(
                benchmark, datetime_index, datetime_index
            )

    # Return results
    output = result.to_dict()
    if benchmark_returns is not None:
        output['benchmark_returns'] = benchmark_returns

    return output


def _calculate_ticker_benchmark_returns(self, ticker, datetime_index, result_index):
    """Calculate benchmark returns for a ticker (copy from Backtester)."""
    # Copy implementation from regular Backtester
    # This is straightforward ticker price → returns calculation
    pass
```

### Decision Point

**Question**: Do we want benchmark support in MVP?

**Option A**: Skip benchmarks for MVP (simpler)
- Defer to later iteration
- Get core working first
- Add benchmarks after validation

**Option B**: Add basic ticker benchmark (moderate)
- Support ticker-based benchmarks only
- Use percentage returns
- Skip standard/custom benchmarks

**Option C**: Full benchmark support (complex)
- All benchmark types
- Requires thought on position→weight conversion

**Recommendation**: **Option A** for MVP - skip benchmarks, add in Iteration 11

### Tests

If implementing benchmarks, add:
```python
def test_benchmark_ticker(sample_csv_data):
    """Test ticker-based benchmark."""
    # Skip for MVP
    pytest.skip("Benchmarks deferred to later iteration")
```

### Acceptance Criteria
- [ ] Decision made on benchmark approach
- [ ] If implementing: ticker benchmarks work
- [ ] If skipping: documented for later

---

## Iteration 11: Integration with Existing System

**Goal**: Ensure PositionBacktester integrates with existing portwine code

**Builds On**: Iteration 10 (complete MVP)

### Tasks

**1. Update `portwine/backtester/__init__.py`**:
```python
"""Backtesting module."""

from portwine.backtester.core import Backtester, InvalidBenchmarkError
from portwine.backtester.position_core import PositionBacktester

__all__ = ['Backtester', 'PositionBacktester', 'InvalidBenchmarkError']
```

**2. Create example script**:

**`examples/position_backtest_example.py`**:
```python
"""
Example of using PositionBacktester.

Demonstrates:
- Creating a position-based strategy
- Running position backtest
- Analyzing results
"""

from portwine.backtester import PositionBacktester
from portwine.strategies.base import StrategyBase
from portwine.data.stores.csvstore import CSVStore
from portwine.data.interface import DataInterface


class SimplePositionStrategy(StrategyBase):
    """Buy 100 shares of each ticker on first day, hold."""

    def __init__(self, tickers, shares=100):
        super().__init__(tickers)
        self.shares = shares
        self.initialized = False

    def step(self, current_date, daily_data):
        if not self.initialized:
            self.initialized = True
            return {ticker: self.shares for ticker in self.tickers}
        return {}


def main():
    # Setup data
    store = CSVStore("./data")
    data = DataInterface(store)

    # Create strategy (returns share quantities, not weights!)
    strategy = SimplePositionStrategy(['AAPL', 'MSFT', 'GOOG'], shares=100)

    # Run position backtest
    backtester = PositionBacktester(data)
    results = backtester.run_backtest(
        strategy,
        start_date='2020-01-01',
        end_date='2023-12-31',
        verbose=True
    )

    # Print results
    print("\n=== Position Backtest Results ===\n")

    print("Final Positions:")
    print(results['positions_df'].iloc[-1])

    print("\nPortfolio Value Over Time:")
    print(results['portfolio_value'].describe())

    print(f"\nFinal Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total P&L: ${results['portfolio_value'].iloc[-1]:,.2f}")

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value
    results['portfolio_value'].plot(ax=axes[0], title='Portfolio Value')
    axes[0].set_ylabel('Value ($)')

    # Positions
    results['positions_df'].plot(ax=axes[1], title='Positions Over Time')
    axes[1].set_ylabel('Shares')
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```

**3. Test integration**:
```python
def test_import_from_backtester_module():
    """Test that PositionBacktester can be imported from main module."""
    from portwine.backtester import PositionBacktester
    assert PositionBacktester is not None


def test_position_backtester_compatible_with_weights_backtester():
    """Test that both backtesters can coexist."""
    from portwine.backtester import Backtester, PositionBacktester

    # Both should be importable
    assert Backtester is not None
    assert PositionBacktester is not None

    # Both should use same StrategyBase
    from portwine.strategies.base import StrategyBase

    # This shows they're compatible at the interface level
    assert True  # If we got here, imports work
```

### Acceptance Criteria
- [ ] Can import PositionBacktester from `portwine.backtester`
- [ ] Example script works
- [ ] Documentation exists
- [ ] Both backtesters coexist without conflicts

---

## Iteration 12: Documentation and Polish

**Goal**: Document the new system and clean up code

**Builds On**: Iteration 11 (integration)

### Tasks

**1. Add docstrings**:
- Complete all docstrings in `position_core.py`
- Add examples to key methods
- Document return formats

**2. Create user guide**:

**`docs/user-guide/position-backtesting.md`**:
```markdown
# Position-Based Backtesting

## Overview

The PositionBacktester allows you to backtest strategies that trade explicit share quantities rather than portfolio weights.

## Key Differences from Standard Backtester

| Aspect | Standard Backtester | PositionBacktester |
|--------|--------------------|--------------------|
| Strategy output | Weights (0.0-1.0) | Share quantities |
| Returns | Percentage-based | Dollar-based |
| Equity curve | Normalized (starts at 1.0) | Absolute dollars |
| Use case | Portfolio allocation | Explicit trading |

## Usage

[... example code ...]

## Output Format

[... describe results dict ...]

## Comparison with Weights Backtester

[... when to use which ...]
```

**3. Add type hints**:
- Complete type hints for all methods
- Add type hints for return values

**4. Code cleanup**:
- Remove any TODOs
- Clean up comments
- Consistent naming

### Acceptance Criteria
- [ ] All docstrings complete
- [ ] User guide created
- [ ] Type hints added
- [ ] Code clean and consistent
- [ ] All tests still pass

---

## Iteration 13: Performance Testing and Optimization

**Goal**: Ensure performance is acceptable, add Numba if needed

**Builds On**: Iteration 12 (documented MVP)

### Tasks

**1. Create performance test**:
```python
def test_performance_large_backtest(benchmark_data):
    """Test performance with large dataset."""
    import time

    # 5 years, 50 tickers
    tickers = [f'TICK{i}' for i in range(50)]
    strategy = BuyAndHoldStrategy(tickers, shares=100)
    backtester = PositionBacktester(benchmark_data)

    start = time.time()
    results = backtester.run_backtest(strategy, start_date='2018-01-01', end_date='2023-12-31')
    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed:.2f}s")
    assert elapsed < 10.0  # Should complete in under 10 seconds
```

**2. Profile if needed**:
```bash
python -m cProfile -o profile.stats examples/position_backtest_example.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

**3. Add Numba optimization if bottlenecks found**:
- Identify hot loops
- Add `@numba.jit` decorators
- Verify speedup
- Keep pure Python version for debugging

### Acceptance Criteria
- [ ] Performance test runs
- [ ] Performance is acceptable (<5s for 5 years, 50 tickers)
- [ ] If slow, optimization applied
- [ ] Tests still pass after optimization

---

## Summary: Complete Implementation Flow

### Phase 1: Foundation (Iterations 0-3)
Build data structures and result formatting

**Deliverables**:
- `PositionBacktestResult` class
- Array storage for positions, actions, prices
- Methods to populate arrays
- Output formatting to DataFrames

**Tests**: Unit tests for result class

---

### Phase 2: Backtester Shell (Iterations 4-5)
Create backtester structure and loop

**Deliverables**:
- `PositionBacktester` class
- Initialization
- Main backtest loop (no processing yet)

**Tests**: Structure tests

---

### Phase 3: Core Functionality (Iterations 6-8)
Process actions and track positions

**Deliverables**:
- Test data fixtures
- Action processing
- Position tracking
- Price tracking
- Portfolio value calculation

**Tests**: Integration tests with real data

---

### Phase 4: Robustness (Iteration 9)
Handle edge cases

**Deliverables**:
- Input validation
- Edge case handling
- Error messages

**Tests**: Edge case tests

---

### Phase 5: Integration (Iterations 10-12)
Integrate with existing system

**Deliverables**:
- Module exports
- Example scripts
- Documentation
- User guide

**Tests**: Integration tests

---

### Phase 6: Performance (Iteration 13)
Optimize if needed

**Deliverables**:
- Performance tests
- Optimization (if needed)

**Tests**: Performance benchmarks

---

## Next Steps After MVP

Once MVP is complete and tested:

1. **Analyzers** (Phase 7):
   - `PositionEquityAnalyzer`
   - `PositionTurnoverAnalyzer`
   - `PositionTradeAnalyzer`
   - `PositionToWeightsAdapter`

2. **Advanced Features** (Phase 8):
   - Cost basis tracking (in analyzer)
   - Realized/unrealized P&L split
   - Benchmark comparison strategies
   - Capital constraints (optional mode)

3. **Polish** (Phase 9):
   - More example strategies
   - Comprehensive documentation
   - Tutorial notebook

---

## Testing Philosophy

Each iteration follows:

1. **Write simplest implementation**
2. **Write test that exercises it**
3. **Run test** - should pass
4. **Add next feature on top**
5. **Write test for new feature**
6. **Run all tests** - all should pass
7. **Commit with green tests**

**No moving forward until tests pass!**

---

## Estimated Timeline

**Total**: 2-3 weeks (part-time)

- **Week 1**: Iterations 0-8 (core functionality)
  - Mon-Tue: Result class (0-3)
  - Wed-Thu: Backtester structure (4-5)
  - Fri-Sun: Integration and testing (6-8)

- **Week 2**: Iterations 9-13 (robustness and polish)
  - Mon-Tue: Edge cases (9)
  - Wed: Integration (10-11)
  - Thu: Documentation (12)
  - Fri: Performance (13)

- **Week 3** (optional): Analyzers and advanced features

---

## Success Criteria

**MVP is complete when**:
- [ ] All tests pass
- [ ] Can run position backtest end-to-end
- [ ] Output format matches design
- [ ] Example script works
- [ ] Documentation exists
- [ ] Performance acceptable
- [ ] No known bugs

**Ready for production when**:
- [ ] Core analyzers implemented
- [ ] User guide complete
- [ ] Example strategies work
- [ ] Integrated with main codebase
- [ ] Reviewed by user
