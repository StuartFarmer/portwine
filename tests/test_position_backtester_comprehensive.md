# Comprehensive Test Plan for Position Backtester (100% Coverage)

## Analysis of Current Coverage

### PositionBacktestResult Class

**Current Tests:**
- ✅ `__init__` - basic initialization
- ✅ `add_action` - adding actions
- ✅ `add_price` - adding prices
- ✅ `update_positions` - cumulative position calculation
- ✅ `calculate_portfolio_value` - portfolio value calculation
- ✅ `to_dict` - conversion to output format

**Missing Coverage:**

#### PositionBacktestResult.__init__
- ❌ Empty ticker list
- ❌ Single ticker
- ❌ Unsorted ticker list (verify sorting)
- ❌ Duplicate tickers in list
- ❌ Empty datetime index
- ❌ Single date datetime index
- ❌ Large number of tickers/dates (performance)

#### add_action
- ❌ Ticker not in result set (early return path)
- ❌ Negative quantities
- ❌ Zero quantity
- ❌ Very large quantities
- ❌ Adding action for same ticker multiple times on same day (overwrite)
- ❌ Out of bounds day_idx (edge case)

#### add_price
- ❌ Ticker not in result set (early return path)
- ❌ Zero price
- ❌ Negative price (technically invalid but not validated)
- ❌ Very large price
- ❌ Adding price for same ticker multiple times on same day (overwrite)

#### update_positions
- ❌ All zero actions
- ❌ Mix of positive and negative actions
- ❌ Multiple calls to update_positions (idempotent?)

#### calculate_portfolio_value
- ❌ All NaN prices
- ❌ Mix of NaN and valid prices
- ❌ All zero positions
- ❌ Negative positions (shorts)
- ❌ Mix of long and short positions
- ❌ Multiple calls (idempotent?)

#### to_dict
- ❌ Empty results (no actions/prices)
- ❌ Single ticker single day
- ❌ Verify DataFrame index names
- ❌ Verify Series name

---

### PositionBacktester Class

**Current Tests:**
- ✅ `__init__` - basic initialization with DataInterface
- ✅ `run_backtest` - full end-to-end with various strategies
- ✅ `validate_actions` - invalid ticker, non-numeric, NaN

**Missing Coverage:**

#### __init__
- ❌ Initialization with MultiDataInterface
- ❌ Initialization with custom calendar
- ❌ Verify restricted_data creation for both interface types

#### run_backtest - Strategy Validation
- ❌ Strategy with no tickers (empty universe) - should raise ValueError
- ❌ Strategy with only alternative data tickers (no regular tickers)

#### run_backtest - Date Range Detection
- ❌ Both start_date and end_date provided
- ❌ Only start_date provided
- ❌ Only end_date provided
- ❌ Neither start_date nor end_date provided (auto-detect both)
- ❌ start_date with DataInterface (not MultiDataInterface)
- ❌ start_date with MultiDataInterface

#### run_backtest - require_all_history
- ❌ require_all_history=True with DataInterface
- ❌ require_all_history=True with MultiDataInterface
- ❌ require_all_history=True adjusts start_date
- ❌ require_all_history=False (default path)

#### run_backtest - Verbose
- ❌ verbose=True (creates tqdm progress bar)
- ❌ verbose=False (default, no progress bar)

#### run_backtest - Action Normalization
- ✅ None → {}
- ✅ pd.Series → dict
- ❌ dict (pass through)
- ✅ Invalid type → ValueError

#### run_backtest - Universe Handling
- ❌ Static universe
- ❌ Dynamic universe (constituents change over time)
- ❌ Mixed regular and alternative data tickers

#### run_backtest - Price Recording
- ❌ Missing price data (KeyError)
- ❌ Missing price data (ValueError)
- ❌ close price is None
- ❌ close price is valid

#### run_backtest - Benchmark (currently not implemented)
- ❌ benchmark=None (default)
- ❌ benchmark parameter ignored (future test when implemented)

#### _compute_effective_end_date
- ✅ end_date provided → return it
- ❌ end_date None, DataInterface, single ticker
- ❌ end_date None, DataInterface, multiple tickers
- ❌ end_date None, MultiDataInterface, single ticker
- ❌ end_date None, MultiDataInterface, multiple tickers
- ❌ No tickers have data → ValueError
- ❌ Some tickers have no data (KeyError)
- ❌ Some tickers have no data (AttributeError)
- ❌ Returned date is pd.Timestamp → convert to string
- ❌ Returned date has strftime → convert to string
- ❌ Returned date has no strftime → return as-is

#### validate_actions
- ✅ Invalid ticker → ValueError
- ✅ Non-numeric quantity → ValueError
- ✅ NaN quantity → ValueError
- ❌ Inf quantity → ValueError
- ❌ -Inf quantity → ValueError
- ❌ Empty actions dict (valid)
- ❌ Valid actions (no error)
- ❌ Integer quantity (valid)
- ❌ Float quantity (valid)
- ❌ Multiple invalid tickers
- ❌ Mix of valid and invalid tickers

---

## Test Stubs to Add

Below are the test stubs needed for 100% coverage:

```python
# ============================================================================
# PositionBacktestResult Tests
# ============================================================================

def test_result_init_empty_tickers():
    """Test initialization with empty ticker list."""
    # STUB: dates = pd.date_range(...); result = PositionBacktestResult(dates, [])
    # Assert: len(result.tickers) == 0, arrays have shape (n_days, 0)
    pass


def test_result_init_single_ticker():
    """Test initialization with single ticker."""
    # STUB: result = PositionBacktestResult(dates, ['AAPL'])
    # Assert: len(result.tickers) == 1, ticker_to_idx correct
    pass


def test_result_init_unsorted_tickers():
    """Test that tickers are sorted on initialization."""
    # STUB: result = PositionBacktestResult(dates, ['ZZZ', 'AAA', 'MMM'])
    # Assert: result.tickers == ['AAA', 'MMM', 'ZZZ']
    pass


def test_result_init_duplicate_tickers():
    """Test initialization with duplicate tickers."""
    # STUB: result = PositionBacktestResult(dates, ['AAPL', 'AAPL', 'MSFT'])
    # Note: sorted() doesn't remove duplicates, so this creates duplicates in array
    # Decide if this is valid or should be validated
    pass


def test_result_init_empty_datetime_index():
    """Test initialization with empty datetime index."""
    # STUB: result = PositionBacktestResult(pd.DatetimeIndex([]), ['AAPL'])
    # Assert: arrays have shape (0, 1)
    pass


def test_result_init_single_date():
    """Test initialization with single date."""
    # STUB: result = PositionBacktestResult(pd.DatetimeIndex(['2020-01-01']), ['AAPL'])
    # Assert: arrays have shape (1, 1)
    pass


def test_result_add_action_ticker_not_in_set():
    """Test add_action with ticker not in result set (early return)."""
    # STUB: result = PositionBacktestResult(dates, ['AAPL'])
    # result.add_action(0, 'MSFT', 10)  # MSFT not in tickers
    # Assert: actions_array[0, 0] == 0 (unchanged)
    pass


def test_result_add_action_negative_quantity():
    """Test add_action with negative quantity (sell/short)."""
    # STUB: result.add_action(0, 'AAPL', -10)
    # Assert: actions_array[0, 0] == -10
    pass


def test_result_add_action_zero_quantity():
    """Test add_action with zero quantity."""
    # STUB: result.add_action(0, 'AAPL', 0)
    # Assert: actions_array[0, 0] == 0
    pass


def test_result_add_action_large_quantity():
    """Test add_action with very large quantity."""
    # STUB: result.add_action(0, 'AAPL', 1e10)
    # Assert: actions_array[0, 0] == 1e10
    pass


def test_result_add_action_overwrite():
    """Test add_action overwrites previous value for same day/ticker."""
    # STUB: result.add_action(0, 'AAPL', 10)
    # result.add_action(0, 'AAPL', 20)
    # Assert: actions_array[0, 0] == 20
    pass


def test_result_add_price_ticker_not_in_set():
    """Test add_price with ticker not in result set (early return)."""
    # STUB: result = PositionBacktestResult(dates, ['AAPL'])
    # result.add_price(0, 'MSFT', 100)
    # Assert: prices_array unchanged (all NaN)
    pass


def test_result_add_price_zero():
    """Test add_price with zero price."""
    # STUB: result.add_price(0, 'AAPL', 0.0)
    # Assert: prices_array[0, 0] == 0.0
    pass


def test_result_add_price_negative():
    """Test add_price with negative price (technically invalid but not validated)."""
    # STUB: result.add_price(0, 'AAPL', -100.0)
    # Assert: prices_array[0, 0] == -100.0
    # Note: May want to add validation for negative prices
    pass


def test_result_add_price_large():
    """Test add_price with very large price."""
    # STUB: result.add_price(0, 'AAPL', 1e10)
    # Assert: prices_array[0, 0] == 1e10
    pass


def test_result_add_price_overwrite():
    """Test add_price overwrites previous value."""
    # STUB: result.add_price(0, 'AAPL', 100)
    # result.add_price(0, 'AAPL', 200)
    # Assert: prices_array[0, 0] == 200
    pass


def test_result_update_positions_all_zeros():
    """Test update_positions with all zero actions."""
    # STUB: result with no actions added
    # result.update_positions()
    # Assert: positions_array all zeros
    pass


def test_result_update_positions_mixed_signs():
    """Test update_positions with mix of buys and sells."""
    # STUB: result.add_action(0, 'AAPL', 10)
    # result.add_action(1, 'AAPL', -5)
    # result.add_action(2, 'AAPL', 3)
    # result.update_positions()
    # Assert: positions[0]=10, positions[1]=5, positions[2]=8
    pass


def test_result_update_positions_idempotent():
    """Test that update_positions can be called multiple times."""
    # STUB: result.add_action(0, 'AAPL', 10)
    # result.update_positions()
    # first_positions = result.positions_array.copy()
    # result.update_positions()  # Call again
    # Assert: positions_array == first_positions (same result)
    pass


def test_result_calculate_portfolio_value_all_nan_prices():
    """Test calculate_portfolio_value with all NaN prices."""
    # STUB: result with positions but no prices
    # result.update_positions()
    # result.calculate_portfolio_value()
    # Assert: portfolio_value all zeros
    pass


def test_result_calculate_portfolio_value_mixed_nan():
    """Test calculate_portfolio_value with mix of NaN and valid prices."""
    # STUB: result.add_action(0, 'AAPL', 10)
    # result.add_action(0, 'MSFT', 5)
    # result.add_price(0, 'AAPL', 100)  # MSFT price stays NaN
    # result.update_positions()
    # result.calculate_portfolio_value()
    # Assert: portfolio_value[0] == 1000 (only AAPL counted)
    pass


def test_result_calculate_portfolio_value_all_zero_positions():
    """Test calculate_portfolio_value with all zero positions."""
    # STUB: result with prices but no positions
    # result.calculate_portfolio_value()
    # Assert: portfolio_value all zeros
    pass


def test_result_calculate_portfolio_value_negative_positions():
    """Test calculate_portfolio_value with short positions."""
    # STUB: result.add_action(0, 'AAPL', -10)  # Short
    # result.add_price(0, 'AAPL', 100)
    # result.update_positions()
    # result.calculate_portfolio_value()
    # Assert: portfolio_value[0] == -1000
    pass


def test_result_calculate_portfolio_value_mixed_long_short():
    """Test calculate_portfolio_value with mix of long and short."""
    # STUB: result.add_action(0, 'AAPL', 10)
    # result.add_action(0, 'MSFT', -5)
    # result.add_price(0, 'AAPL', 100)
    # result.add_price(0, 'MSFT', 200)
    # result.update_positions()
    # result.calculate_portfolio_value()
    # Assert: portfolio_value[0] == 1000 + (-1000) == 0
    pass


def test_result_calculate_portfolio_value_idempotent():
    """Test that calculate_portfolio_value can be called multiple times."""
    # STUB: Set up result with positions and prices
    # result.calculate_portfolio_value()
    # first_pv = result.portfolio_value.copy()
    # result.calculate_portfolio_value()
    # Assert: portfolio_value == first_pv
    pass


def test_result_to_dict_empty():
    """Test to_dict with no actions or prices."""
    # STUB: result = PositionBacktestResult(dates, tickers)
    # output = result.to_dict()
    # Assert: All DataFrames/Series have correct shape but all zeros/NaN
    pass


def test_result_to_dict_single_ticker_single_day():
    """Test to_dict with minimal data."""
    # STUB: result = PositionBacktestResult(single_date_index, ['AAPL'])
    # output = result.to_dict()
    # Assert: shapes are (1, 1) and (1,)
    pass


def test_result_to_dict_index_names():
    """Test to_dict preserves index structure."""
    # STUB: output = result.to_dict()
    # Assert: output['positions_df'].index.name correct (if any)
    # Assert: output['portfolio_value'].name == 'portfolio_value'
    pass


# ============================================================================
# PositionBacktester Tests
# ============================================================================

def test_backtester_init_multi_data_interface():
    """Test initialization with MultiDataInterface."""
    # STUB: from portwine.data.interface import MultiDataInterface
    # multi_data = MultiDataInterface({None: store1, 'ECON': store2})
    # backtester = PositionBacktester(multi_data)
    # Assert: backtester.restricted_data created correctly
    pass


def test_backtester_init_custom_calendar():
    """Test initialization with custom calendar."""
    # STUB: custom_cal = DailyMarketCalendar('NASDAQ')
    # backtester = PositionBacktester(data, calendar=custom_cal)
    # Assert: backtester.calendar == custom_cal
    pass


def test_backtester_run_strategy_no_tickers():
    """Test run_backtest with strategy that has no tickers."""
    # STUB: class EmptyUniverse(StrategyBase):
    #     def __init__(self): super().__init__([])  # No tickers
    # strategy = EmptyUniverse()
    # with pytest.raises(ValueError, match="Strategy has no tickers"):
    #     backtester.run_backtest(strategy)
    pass


def test_backtester_run_only_alternative_tickers():
    """Test with strategy that has only alternative data tickers."""
    # STUB: strategy with tickers like ['ECON:GDP', 'INDEX:VIX']
    # After _split_tickers, regular_tickers is empty
    # Behavior: may fail or work depending on implementation
    # Need to decide: is this valid?
    pass


def test_backtester_run_both_dates_provided():
    """Test run_backtest with both start_date and end_date provided."""
    # STUB: results = backtester.run_backtest(strategy, '2020-01-01', '2020-01-31')
    # Assert: Uses provided dates
    pass


def test_backtester_run_only_start_date():
    """Test run_backtest with only start_date provided."""
    # STUB: results = backtester.run_backtest(strategy, start_date='2020-01-01')
    # Assert: end_date computed from data
    pass


def test_backtester_run_only_end_date():
    """Test run_backtest with only end_date provided."""
    # STUB: results = backtester.run_backtest(strategy, end_date='2020-01-31')
    # Assert: start_date computed from data
    pass


def test_backtester_run_auto_detect_dates():
    """Test run_backtest with no dates (auto-detect both)."""
    # STUB: results = backtester.run_backtest(strategy)
    # Assert: Both dates detected from available data
    pass


def test_backtester_run_start_date_data_interface():
    """Test start_date detection with DataInterface (not Multi)."""
    # STUB: backtester with DataInterface
    # results = backtester.run_backtest(strategy, start_date=None)
    # Assert: Calls DataInterface.earliest_any_date
    pass


def test_backtester_run_start_date_multi_interface():
    """Test start_date detection with MultiDataInterface."""
    # STUB: backtester with MultiDataInterface
    # results = backtester.run_backtest(strategy, start_date=None)
    # Assert: Calls MultiDataInterface.earliest_any_date
    pass


def test_backtester_run_require_all_history_true_data_interface():
    """Test require_all_history=True with DataInterface."""
    # STUB: results = backtester.run_backtest(strategy, require_all_history=True)
    # Assert: Calls DataInterface.earliest_common_date
    # Assert: start_date adjusted if needed
    pass


def test_backtester_run_require_all_history_true_multi():
    """Test require_all_history=True with MultiDataInterface."""
    # STUB: backtester with MultiDataInterface
    # results = backtester.run_backtest(strategy, require_all_history=True)
    # Assert: Calls MultiDataInterface.earliest_common_date
    pass


def test_backtester_run_require_all_history_adjusts_start():
    """Test that require_all_history adjusts start_date correctly."""
    # STUB: Set up data where earliest_common > earliest_any
    # results = backtester.run_backtest(strategy, require_all_history=True)
    # Assert: start_date is earliest_common, not earliest_any
    pass


def test_backtester_run_verbose_true():
    """Test run_backtest with verbose=True creates progress bar."""
    # STUB: results = backtester.run_backtest(strategy, verbose=True)
    # Assert: No error (tqdm created)
    # Note: Hard to test tqdm output without mocking
    pass


def test_backtester_run_verbose_false():
    """Test run_backtest with verbose=False (default)."""
    # STUB: results = backtester.run_backtest(strategy, verbose=False)
    # Assert: No tqdm, normal iteration
    pass


def test_backtester_run_action_normalization_dict():
    """Test that dict actions pass through unchanged."""
    # STUB: Strategy returns {'AAPL': 10}
    # Assert: Actions recorded correctly
    pass


def test_backtester_run_dynamic_universe():
    """Test with dynamic universe (constituents change)."""
    # STUB: Create universe that has different tickers on different dates
    # strategy = Strategy(dynamic_universe)
    # results = backtester.run_backtest(strategy)
    # Assert: Handles changing constituents correctly
    pass


def test_backtester_run_mixed_regular_alternative_tickers():
    """Test with mix of regular and alternative data tickers."""
    # STUB: strategy.universe.all_tickers = ['AAPL', 'ECON:GDP']
    # After _split_tickers, only AAPL in regular_tickers
    # Assert: Only regular tickers in results
    pass


def test_backtester_run_price_missing_keyerror():
    """Test price recording when ticker data missing (KeyError)."""
    # STUB: Set up restricted_data to raise KeyError for a ticker
    # Assert: Exception caught, price remains NaN
    pass


def test_backtester_run_price_missing_valueerror():
    """Test price recording when ticker data missing (ValueError)."""
    # STUB: Set up restricted_data to raise ValueError for a ticker
    # Assert: Exception caught, price remains NaN
    pass


def test_backtester_run_price_close_none():
    """Test price recording when close price is None."""
    # STUB: price_data.get('close') returns None
    # Assert: Price not recorded (remains NaN)
    pass


def test_backtester_run_price_close_valid():
    """Test price recording when close price is valid."""
    # Already tested, but explicit:
    # STUB: price_data.get('close') returns 123.45
    # Assert: Price recorded correctly
    pass


def test_backtester_compute_end_date_with_datainterface():
    """Test _compute_effective_end_date with DataInterface (single ticker)."""
    # STUB: backtester with DataInterface
    # end_date = backtester._compute_effective_end_date(None, ['AAPL'])
    # Assert: Returns latest date for AAPL as string
    pass


def test_backtester_compute_end_date_with_datainterface_multiple():
    """Test _compute_effective_end_date with DataInterface (multiple tickers)."""
    # STUB: end_date = backtester._compute_effective_end_date(None, ['AAPL', 'MSFT'])
    # Assert: Returns max of latest dates as string
    pass


def test_backtester_compute_end_date_with_multi_interface():
    """Test _compute_effective_end_date with MultiDataInterface."""
    # STUB: backtester with MultiDataInterface
    # end_date = backtester._compute_effective_end_date(None, ['AAPL'])
    # Assert: Uses loaders[None] correctly
    pass


def test_backtester_compute_end_date_no_data():
    """Test _compute_effective_end_date when no tickers have data."""
    # STUB: All ticker.latest() calls raise exceptions or return None
    # with pytest.raises(ValueError, match="No data found"):
    #     backtester._compute_effective_end_date(None, ['FAKE'])
    pass


def test_backtester_compute_end_date_some_missing_keyerror():
    """Test _compute_effective_end_date when some tickers missing (KeyError)."""
    # STUB: AAPL has data, FAKE raises KeyError
    # end_date = backtester._compute_effective_end_date(None, ['AAPL', 'FAKE'])
    # Assert: Returns AAPL's latest date
    pass


def test_backtester_compute_end_date_some_missing_attributeerror():
    """Test _compute_effective_end_date when some tickers missing (AttributeError)."""
    # STUB: AAPL has data, FAKE raises AttributeError
    # end_date = backtester._compute_effective_end_date(None, ['AAPL', 'FAKE'])
    # Assert: Returns AAPL's latest date
    pass


def test_backtester_compute_end_date_timestamp_conversion():
    """Test _compute_effective_end_date converts pd.Timestamp to string."""
    # STUB: latest() returns pd.Timestamp('2020-12-31')
    # end_date = backtester._compute_effective_end_date(None, ['AAPL'])
    # Assert: end_date == '2020-12-31' (string)
    pass


def test_backtester_compute_end_date_strftime_conversion():
    """Test _compute_effective_end_date converts date with strftime."""
    # STUB: latest() returns datetime.date(2020, 12, 31)
    # end_date = backtester._compute_effective_end_date(None, ['AAPL'])
    # Assert: Converts to string using strftime
    pass


def test_backtester_compute_end_date_no_strftime():
    """Test _compute_effective_end_date with date that has no strftime."""
    # STUB: latest() returns some object without strftime
    # end_date = backtester._compute_effective_end_date(None, ['AAPL'])
    # Assert: Returns as-is (else branch)
    pass


def test_backtester_validate_actions_empty():
    """Test validate_actions with empty dict (valid)."""
    # STUB: backtester.validate_actions({}, ['AAPL', 'MSFT'])
    # Assert: No exception
    pass


def test_backtester_validate_actions_valid():
    """Test validate_actions with valid actions."""
    # STUB: backtester.validate_actions({'AAPL': 10, 'MSFT': -5}, ['AAPL', 'MSFT'])
    # Assert: No exception
    pass


def test_backtester_validate_actions_integer():
    """Test validate_actions accepts integer quantities."""
    # STUB: backtester.validate_actions({'AAPL': 10}, ['AAPL'])  # int not float
    # Assert: No exception (isinstance checks int and float)
    pass


def test_backtester_validate_actions_float():
    """Test validate_actions accepts float quantities."""
    # STUB: backtester.validate_actions({'AAPL': 10.5}, ['AAPL'])
    # Assert: No exception
    pass


def test_backtester_validate_actions_inf():
    """Test validate_actions rejects inf."""
    # STUB: with pytest.raises(ValueError, match="Invalid action"):
    #     backtester.validate_actions({'AAPL': np.inf}, ['AAPL'])
    pass


def test_backtester_validate_actions_neg_inf():
    """Test validate_actions rejects negative inf."""
    # STUB: with pytest.raises(ValueError, match="Invalid action"):
    #     backtester.validate_actions({'AAPL': -np.inf}, ['AAPL'])
    pass


def test_backtester_validate_actions_multiple_invalid_tickers():
    """Test validate_actions with multiple invalid tickers."""
    # STUB: actions = {'FAKE1': 10, 'FAKE2': 5, 'AAPL': 10}
    # with pytest.raises(ValueError) as exc:
    #     backtester.validate_actions(actions, ['AAPL'])
    # Assert: First invalid ticker caught
    pass


def test_backtester_validate_actions_mixed_valid_invalid():
    """Test validate_actions catches first invalid ticker."""
    # STUB: actions = {'AAPL': 10, 'FAKE': 5}
    # with pytest.raises(ValueError, match="FAKE.*not in current universe"):
    #     backtester.validate_actions(actions, ['AAPL'])
    pass


# ============================================================================
# Integration Tests (Additional Coverage)
# ============================================================================

def test_integration_full_backtest_no_trades():
    """Test full backtest with strategy that never trades."""
    # Already covered by test_empty_strategy
    pass


def test_integration_full_backtest_single_ticker():
    """Test full backtest with single ticker strategy."""
    # STUB: strategy = BuyAndHoldStrategy(['AAPL'], shares=10)
    # results = backtester.run_backtest(strategy)
    # Assert: positions_df has only AAPL column
    pass


def test_integration_full_backtest_many_tickers():
    """Test full backtest with many tickers (performance)."""
    # STUB: tickers = [f'TICK{i}' for i in range(100)]
    # strategy = BuyAndHoldStrategy(tickers, shares=1)
    # results = backtester.run_backtest(strategy)
    # Assert: All tickers in results
    pass


def test_integration_position_reversal():
    """Test going from long to short (position reversal)."""
    # STUB: Day 1: buy 10, Day 2: sell 20 (net -10)
    # Assert: positions go from 10 to -10
    pass


def test_integration_partial_fill_simulation():
    """Test fractional shares with incremental buying."""
    # STUB: Buy 0.1 shares per day for 10 days
    # Assert: Final position is 1.0 share
    pass


def test_integration_high_frequency_trades():
    """Test many actions on same ticker."""
    # STUB: Strategy that trades every day
    # Assert: All actions recorded, positions accumulate correctly
    pass


def test_integration_zero_volume_day():
    """Test day with no trades (zero volume)."""
    # STUB: Strategy returns {} for a day
    # Assert: Positions unchanged, portfolio value calculated
    pass


def test_integration_ticker_added_mid_backtest():
    """Test dynamic universe adds ticker mid-backtest."""
    # STUB: Universe starts with AAPL, adds MSFT on day 5
    # Assert: MSFT shows up in actions/positions after day 5
    pass


def test_integration_ticker_removed_mid_backtest():
    """Test dynamic universe removes ticker mid-backtest."""
    # STUB: Universe starts with AAPL+MSFT, removes MSFT on day 5
    # Assert: Can't trade MSFT after removal
    # Note: What happens to existing MSFT position? Test this behavior
    pass
```

---

## Summary

**Total Additional Tests Needed: ~80-90**

### Breakdown:
- **PositionBacktestResult**: ~30 tests
- **PositionBacktester.__init__**: ~3 tests
- **PositionBacktester.run_backtest**: ~30 tests
- **PositionBacktester._compute_effective_end_date**: ~10 tests
- **PositionBacktester.validate_actions**: ~8 tests
- **Integration tests**: ~10 tests

### Priority:
1. **High**: Edge cases in core logic (empty inputs, None handling, type conversions)
2. **Medium**: Branch coverage (if/else paths in _compute_effective_end_date)
3. **Low**: Performance tests, large data sets

### Next Steps:
1. Implement high-priority tests first
2. Run coverage report to verify 100%
3. Add missing tests iteratively
4. Consider property-based testing for numeric edge cases
