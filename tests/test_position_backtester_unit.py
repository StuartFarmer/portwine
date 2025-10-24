"""
Comprehensive unit tests for position backtester (100% coverage).

These tests cover all branches, edge cases, and error conditions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime

from portwine.backtester.position_core import PositionBacktester, PositionBacktestResult
from portwine.strategies.base import StrategyBase
from portwine.data.stores.csvstore import CSVDataStore
from portwine.data.interface import DataInterface, RestrictedDataInterface, MultiDataInterface


# ============================================================================
# PositionBacktestResult Unit Tests
# ============================================================================

class TestPositionBacktestResultInit:
    """Tests for PositionBacktestResult initialization."""

    def test_init_empty_tickers(self):
        """Test initialization with empty ticker list."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, [])

        assert len(result.tickers) == 0
        assert result.positions_array.shape == (3, 0)
        assert result.actions_array.shape == (3, 0)
        assert result.prices_array.shape == (3, 0)
        assert result.portfolio_value.shape == (3,)

    def test_init_single_ticker(self):
        """Test initialization with single ticker."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        assert len(result.tickers) == 1
        assert result.tickers == ['AAPL']
        assert result.ticker_to_idx == {'AAPL': 0}
        assert result.positions_array.shape == (3, 1)

    def test_init_unsorted_tickers(self):
        """Test that tickers are sorted on initialization."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['ZZZ', 'AAA', 'MMM'])

        assert result.tickers == ['AAA', 'MMM', 'ZZZ']
        assert result.ticker_to_idx == {'AAA': 0, 'MMM': 1, 'ZZZ': 2}

    def test_init_empty_datetime_index(self):
        """Test initialization with empty datetime index."""
        result = PositionBacktestResult(pd.DatetimeIndex([]), ['AAPL'])

        assert result.positions_array.shape == (0, 1)
        assert result.actions_array.shape == (0, 1)
        assert result.prices_array.shape == (0, 1)
        assert result.portfolio_value.shape == (0,)

    def test_init_single_date(self):
        """Test initialization with single date."""
        result = PositionBacktestResult(pd.DatetimeIndex(['2020-01-01']), ['AAPL'])

        assert result.positions_array.shape == (1, 1)
        assert result.actions_array.shape == (1, 1)
        assert result.prices_array.shape == (1, 1)
        assert result.portfolio_value.shape == (1,)


class TestPositionBacktestResultAddAction:
    """Tests for add_action method."""

    def test_add_action_ticker_not_in_set(self):
        """Test add_action with ticker not in result set (early return)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'MSFT', 10)  # MSFT not in tickers

        # Actions array should remain all zeros
        assert np.all(result.actions_array == 0)

    def test_add_action_negative_quantity(self):
        """Test add_action with negative quantity (sell/short)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', -10)

        assert result.actions_array[0, 0] == -10

    def test_add_action_zero_quantity(self):
        """Test add_action with zero quantity."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 0)

        assert result.actions_array[0, 0] == 0

    def test_add_action_large_quantity(self):
        """Test add_action with very large quantity."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 1e10)

        assert result.actions_array[0, 0] == 1e10

    def test_add_action_overwrite(self):
        """Test add_action overwrites previous value for same day/ticker."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 10)
        result.add_action(0, 'AAPL', 20)

        assert result.actions_array[0, 0] == 20


class TestPositionBacktestResultAddPrice:
    """Tests for add_price method."""

    def test_add_price_ticker_not_in_set(self):
        """Test add_price with ticker not in result set (early return)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_price(0, 'MSFT', 100)

        # Prices array should remain all NaN
        assert np.all(np.isnan(result.prices_array))

    def test_add_price_zero(self):
        """Test add_price with zero price."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_price(0, 'AAPL', 0.0)

        assert result.prices_array[0, 0] == 0.0

    def test_add_price_negative(self):
        """Test add_price with negative price (not validated)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_price(0, 'AAPL', -100.0)

        assert result.prices_array[0, 0] == -100.0

    def test_add_price_large(self):
        """Test add_price with very large price."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_price(0, 'AAPL', 1e10)

        assert result.prices_array[0, 0] == 1e10

    def test_add_price_overwrite(self):
        """Test add_price overwrites previous value."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_price(0, 'AAPL', 100)
        result.add_price(0, 'AAPL', 200)

        assert result.prices_array[0, 0] == 200


class TestPositionBacktestResultUpdatePositions:
    """Tests for update_positions method."""

    def test_update_positions_all_zeros(self):
        """Test update_positions with all zero actions."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.update_positions()

        assert np.all(result.positions_array == 0)

    def test_update_positions_mixed_signs(self):
        """Test update_positions with mix of buys and sells (with signal shifting)."""
        dates = pd.date_range('2020-01-01', '2020-01-04', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 10)
        result.add_action(1, 'AAPL', -5)
        result.add_action(2, 'AAPL', 3)
        result.update_positions()

        # WITH SIGNAL SHIFTING: positions lag actions by 1 day
        assert result.positions_array[0, 0] == 0   # No previous action
        assert result.positions_array[1, 0] == 10  # Day 0's +10
        assert result.positions_array[2, 0] == 5   # Day 0's +10, day 1's -5
        assert result.positions_array[3, 0] == 8   # Cumsum: +10, -5, +3

    def test_update_positions_idempotent(self):
        """Test that update_positions can be called multiple times."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 10)
        result.update_positions()
        first_positions = result.positions_array.copy()

        result.update_positions()

        np.testing.assert_array_equal(result.positions_array, first_positions)


class TestPositionBacktestResultCalculatePortfolioValue:
    """Tests for calculate_portfolio_value method."""

    def test_calculate_portfolio_value_all_nan_prices(self):
        """Test calculate_portfolio_value with all NaN prices."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 10)
        result.update_positions()
        result.calculate_portfolio_value()

        assert np.all(result.portfolio_value == 0)

    def test_calculate_portfolio_value_mixed_nan(self):
        """Test calculate_portfolio_value with mix of NaN and valid prices (with signal shifting)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL', 'MSFT'])

        result.add_action(0, 'AAPL', 10)
        result.add_action(0, 'MSFT', 5)
        result.add_price(0, 'AAPL', 100)  # MSFT price stays NaN
        result.add_price(1, 'AAPL', 110)  # Add price for day 1
        result.update_positions()
        result.calculate_portfolio_value()

        # WITH SIGNAL SHIFTING:
        # Day 0: position=0, value=0
        assert result.portfolio_value[0] == 0
        # Day 1: AAPL position=10 @ $110, MSFT position=5 @ NaN (treated as 0)
        assert result.portfolio_value[1] == 1100

    def test_calculate_portfolio_value_all_zero_positions(self):
        """Test calculate_portfolio_value with all zero positions."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_price(0, 'AAPL', 100)
        result.calculate_portfolio_value()

        assert np.all(result.portfolio_value == 0)

    def test_calculate_portfolio_value_negative_positions(self):
        """Test calculate_portfolio_value with short positions (with signal shifting)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', -10)
        result.add_price(0, 'AAPL', 100)
        result.add_price(1, 'AAPL', 110)
        result.update_positions()
        result.calculate_portfolio_value()

        # WITH SIGNAL SHIFTING:
        # Day 0: position=0, value=0
        assert result.portfolio_value[0] == 0
        # Day 1: position=-10 @ $110 = -1100
        assert result.portfolio_value[1] == -1100

    def test_calculate_portfolio_value_mixed_long_short(self):
        """Test calculate_portfolio_value with mix of long and short."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL', 'MSFT'])

        result.add_action(0, 'AAPL', 10)
        result.add_action(0, 'MSFT', -5)
        result.add_price(0, 'AAPL', 100)
        result.add_price(0, 'MSFT', 200)
        result.update_positions()
        result.calculate_portfolio_value()

        assert result.portfolio_value[0] == 0  # 1000 + (-1000)

    def test_calculate_portfolio_value_idempotent(self):
        """Test that calculate_portfolio_value can be called multiple times."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 10)
        result.add_price(0, 'AAPL', 100)
        result.update_positions()
        result.calculate_portfolio_value()
        first_pv = result.portfolio_value.copy()

        result.calculate_portfolio_value()

        np.testing.assert_array_equal(result.portfolio_value, first_pv)


class TestPositionBacktestResultToDict:
    """Tests for to_dict method."""

    def test_to_dict_empty(self):
        """Test to_dict with no actions or prices."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        output = result.to_dict()

        assert 'positions_df' in output
        assert 'actions_df' in output
        assert 'prices_df' in output
        assert 'portfolio_value' in output
        assert output['positions_df'].shape == (3, 1)
        assert np.all(output['positions_df'] == 0)
        assert np.all(np.isnan(output['prices_df']))

    def test_to_dict_single_ticker_single_day(self):
        """Test to_dict with minimal data."""
        result = PositionBacktestResult(pd.DatetimeIndex(['2020-01-01']), ['AAPL'])

        output = result.to_dict()

        assert output['positions_df'].shape == (1, 1)
        assert output['actions_df'].shape == (1, 1)
        assert output['prices_df'].shape == (1, 1)
        assert len(output['portfolio_value']) == 1

    def test_to_dict_series_name(self):
        """Test to_dict preserves series name."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        output = result.to_dict()

        assert output['portfolio_value'].name == 'portfolio_value'


# ============================================================================
# PositionBacktester Unit Tests
# ============================================================================

class TestPositionBacktesterInit:
    """Tests for PositionBacktester initialization."""

    def test_init_with_multi_data_interface(self):
        """Test initialization with MultiDataInterface."""
        from portwine.data.interface import MultiDataInterface
        from portwine.data.stores.csvstore import CSVDataStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            # Create minimal CSV data
            df = pd.DataFrame({
                'date': ['2020-01-01'],
                'close': [100.0]
            })
            df.to_csv(data_dir / "TEST.csv", index=False)

            store = CSVDataStore(str(data_dir))
            multi_data = MultiDataInterface({None: store})

            backtester = PositionBacktester(multi_data)

            assert backtester.data == multi_data
            assert isinstance(backtester.restricted_data, RestrictedDataInterface)

    def test_init_with_custom_calendar(self):
        """Test initialization with custom calendar."""
        from portwine.backtester.core import DailyMarketCalendar

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            custom_cal = DailyMarketCalendar('NYSE')

            backtester = PositionBacktester(data, calendar=custom_cal)

            assert backtester.calendar == custom_cal


class TestPositionBacktesterValidateActions:
    """Tests for validate_actions method."""

    def test_validate_actions_empty(self):
        """Test validate_actions with empty dict (valid)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            # Should not raise
            backtester.validate_actions({}, ['AAPL', 'MSFT'])

    def test_validate_actions_valid(self):
        """Test validate_actions with valid actions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            # Should not raise
            backtester.validate_actions({'AAPL': 10, 'MSFT': -5}, ['AAPL', 'MSFT'])

    def test_validate_actions_integer(self):
        """Test validate_actions accepts integer quantities."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            # Should not raise (isinstance checks both int and float)
            backtester.validate_actions({'AAPL': 10}, ['AAPL'])

    def test_validate_actions_float(self):
        """Test validate_actions accepts float quantities."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            # Should not raise
            backtester.validate_actions({'AAPL': 10.5}, ['AAPL'])

    def test_validate_actions_inf(self):
        """Test validate_actions rejects inf."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            with pytest.raises(ValueError, match="Invalid action"):
                backtester.validate_actions({'AAPL': np.inf}, ['AAPL'])

    def test_validate_actions_neg_inf(self):
        """Test validate_actions rejects negative inf."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            with pytest.raises(ValueError, match="Invalid action"):
                backtester.validate_actions({'AAPL': -np.inf}, ['AAPL'])

    def test_validate_actions_multiple_invalid_tickers(self):
        """Test validate_actions with multiple invalid tickers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            actions = {'FAKE1': 10, 'FAKE2': 5, 'AAPL': 10}

            with pytest.raises(ValueError, match="not in current universe"):
                backtester.validate_actions(actions, ['AAPL'])


class TestPositionBacktesterComputeEndDate:
    """Tests for _compute_effective_end_date method."""

    def test_compute_end_date_provided(self):
        """Test _compute_effective_end_date returns provided end_date."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            result = backtester._compute_effective_end_date('2020-12-31', ['AAPL'])

            assert result == '2020-12-31'

    def test_compute_end_date_no_data(self):
        """Test _compute_effective_end_date when no tickers have data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CSVDataStore(tmp_dir)
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            with pytest.raises(ValueError, match="No data found"):
                backtester._compute_effective_end_date(None, ['FAKE'])


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_result_with_large_arrays(self):
        """Test PositionBacktestResult with large data."""
        # 1000 days, 100 tickers
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        tickers = [f'TICK{i:03d}' for i in range(100)]

        result = PositionBacktestResult(dates, tickers)

        assert result.positions_array.shape == (1000, 100)
        assert len(result.tickers) == 100

    def test_result_position_reversal(self):
        """Test going from long to short (position reversal with signal shifting)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 10)  # Long 10 (executes day 1)
        result.add_action(1, 'AAPL', -20)  # Sell 20 (executes day 2), net short 10
        result.update_positions()

        # WITH SIGNAL SHIFTING:
        assert result.positions_array[0, 0] == 0   # No previous action
        assert result.positions_array[1, 0] == 10  # Day 0's +10
        assert result.positions_array[2, 0] == -10 # Day 0's +10, day 1's -20

    def test_result_very_small_fractional(self):
        """Test very small fractional shares (with signal shifting)."""
        dates = pd.date_range('2020-01-01', '2020-01-03', freq='D')
        result = PositionBacktestResult(dates, ['AAPL'])

        result.add_action(0, 'AAPL', 0.0001)
        result.add_price(0, 'AAPL', 10000.0)
        result.add_price(1, 'AAPL', 10000.0)
        result.update_positions()
        result.calculate_portfolio_value()

        # WITH SIGNAL SHIFTING:
        # Day 0: position=0, value=0
        assert result.portfolio_value[0] == 0.0
        # Day 1: position=0.0001 @ $10000 = $1.0
        assert result.portfolio_value[1] == pytest.approx(1.0, rel=1e-6)


# ============================================================================
# Additional Tests for 100% Coverage
# ============================================================================

class TestBacktesterUncoveredBranches:
    """Tests for uncovered code branches."""

    def test_run_backtest_no_tickers_raises_error(self):
        """Test that running backtest with strategy that has no tickers raises error."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            df = pd.DataFrame({
                'date': ['2020-01-02'],
                'close': [100.0]
            })
            df.to_csv(data_dir / "TEST.csv", index=False)

            store = CSVDataStore(str(data_dir))
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            # Strategy with no tickers
            strategy = StrategyBase([])

            with pytest.raises(ValueError, match="Strategy has no tickers"):
                backtester.run_backtest(strategy)

    def test_validate_actions_non_numeric_raises_error(self):
        """Test that non-numeric action values raise ValueError."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            df = pd.DataFrame({
                'date': ['2020-01-02'],
                'close': [100.0]
            })
            df.to_csv(data_dir / "AAPL.csv", index=False)

            store = CSVDataStore(str(data_dir))
            data = DataInterface(store)
            backtester = PositionBacktester(data)

            # Action with string value (non-numeric)
            actions = {'AAPL': 'ten'}
            universe_tickers = ['AAPL']

            with pytest.raises(ValueError, match="must be numeric"):
                backtester.validate_actions(actions, universe_tickers)

    def test_strategy_returns_invalid_type_raises_error(self):
        """Test that strategy returning invalid type raises ValueError."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            df = pd.DataFrame({
                'date': ['2020-01-02', '2020-01-03'],
                'close': [100.0, 101.0]
            })
            df.to_csv(data_dir / "AAPL.csv", index=False)

            store = CSVDataStore(str(data_dir))
            data = DataInterface(store)

            class BadStrategy(StrategyBase):
                def step(self, current_date, daily_data):
                    return "invalid"  # Should return dict, Series, or None

            strategy = BadStrategy(['AAPL'])
            backtester = PositionBacktester(data)

            with pytest.raises(ValueError, match="invalid type"):
                backtester.run_backtest(strategy)

    def test_run_backtest_require_all_history(self):
        """Test require_all_history parameter."""
        import tempfile
        from pathlib import Path
        from tests.calendar_utils import TestDailyMarketCalendar

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            # Create data with different start dates for different tickers
            aapl_df = pd.DataFrame({
                'date': ['2020-01-02', '2020-01-03', '2020-01-04'],
                'close': [100.0, 101.0, 102.0]
            })
            aapl_df.to_csv(data_dir / "AAPL.csv", index=False)

            msft_df = pd.DataFrame({
                'date': ['2020-01-03', '2020-01-04'],  # Starts later
                'close': [200.0, 201.0]
            })
            msft_df.to_csv(data_dir / "MSFT.csv", index=False)

            store = CSVDataStore(str(data_dir))
            data = DataInterface(store)

            class SimpleStrategy(StrategyBase):
                def step(self, current_date, daily_data):
                    return {}

            strategy = SimpleStrategy(['AAPL', 'MSFT'])
            calendar = TestDailyMarketCalendar(allowed_year=2020)
            backtester = PositionBacktester(data, calendar=calendar)

            # With require_all_history=True, should start from latest earliest date
            results = backtester.run_backtest(strategy, require_all_history=True)

            # Should start from 2020-01-03 (when both have data)
            assert len(results['positions_df']) == 2  # Jan 3-4

    def test_compute_effective_end_date_with_multi_data_interface(self):
        """Test _compute_effective_end_date with MultiDataInterface."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            df = pd.DataFrame({
                'date': ['2020-01-02', '2020-01-03'],
                'close': [100.0, 101.0]
            })
            df.to_csv(data_dir / "AAPL.csv", index=False)

            store = CSVDataStore(str(data_dir))
            multi_data = MultiDataInterface({None: store})
            backtester = PositionBacktester(multi_data)

            # Test with None end_date (should compute from data)
            end_date = backtester._compute_effective_end_date(None, ['AAPL'])

            assert end_date is not None
            # Should be the latest date in the data
            assert '2020-01-03' in str(end_date)
