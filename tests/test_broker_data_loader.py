import unittest
import pandas as pd


from datetime import datetime, timezone

from portwine.execution import ExecutionBase
from portwine.brokers.mock import MockBroker
from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.data.providers.loader_adapters import BrokerDataLoader
from portwine.backtester.core import Backtester
from portwine.strategies.base import StrategyBase
from portwine.data.interface import DataInterface
from tests.helpers import MockDataStore
from unittest.mock import Mock

class MockDataInterface(DataInterface):
    """DataInterface backed by MockDataStore for testing"""
    def __init__(self, mock_data=None):
        store = MockDataStore()
        if mock_data:
            store.load_bulk(mock_data)
        super().__init__(store)
        self.current_timestamp = None

    def set_current_timestamp(self, dt):
        self.current_timestamp = dt
        super().set_current_timestamp(dt)

    def __getitem__(self, ticker):
        return super().__getitem__(ticker)

    def exists(self, ticker, start_date, end_date):
        return self.data_loader.exists(ticker, start_date, end_date)
    
    def get(self, ticker, default=None):
        """Implement get method for compatibility with RestrictedDataInterface"""
        try:
            return self.__getitem__(ticker)
        except (KeyError, ValueError):
            return default
    
    def keys(self):
        """Return available tickers"""
        return list(self.data_loader.identifiers())
    
    def __contains__(self, ticker):
        """Check if ticker exists"""
        return ticker in self.data_loader.identifiers()




class MockDailyMarketCalendar:
    """Test-specific DailyMarketCalendar that mimics data-driven behavior"""
    def __init__(self, calendar_name):
        self.calendar_name = calendar_name
        # For testing, we'll use all calendar days to match original behavior
        
    def schedule(self, start_date, end_date):
        """Return all calendar days to match original data-driven behavior"""
        days = pd.date_range(start_date, end_date, freq="D")
        # Set market close to match the data timestamps (00:00:00)
        closes = [pd.Timestamp(d.date()) for d in days]
        return pd.DataFrame({"market_close": closes}, index=days)
    
    def get_datetime_index(self, start_date, end_date):
        """Return datetime index for the given date range"""
        if start_date is None:
            # Align test default window with the 3-day dataset in this test
            start_date = '2025-01-01'
        if end_date is None:
            end_date = '2025-01-03'
        
        # Return all calendar days
        days = pd.date_range(start_date, end_date, freq="D")
        return days.to_numpy()


class SimpleMarketLoader(MarketDataLoader):
    """Market loader that returns the same OHLCV DataFrame for any ticker."""
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        return self._df.copy()


class BrokerIntegrationStrategy(StrategyBase):
    """Strategy that always allocates fully to a single regular ticker and logs data."""
    def __init__(self):
        # Two tickers: regular and broker alt-data
        super().__init__(tickers=["FAKE", "BROKER:ACCOUNT"])
        self.step_calls = []  # list of (timestamp, data) tuples

    def step(self, current_date, data):
        # Record the passed-in data for inspection
        self.step_calls.append((current_date, data.copy()))
        # Fully invest in the regular ticker 'FAKE'
        return {"FAKE": 1.0}


class TestBacktesterBrokerLoaderIntegration(unittest.TestCase):
    def test_offline_broker_loader_integration(self):
        """Test that broker data loader integrates properly with backtester"""
        # Prepare a 3-day series with known returns: 100 -> 200 -> 50
        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
        prices = [100.0, 200.0, 50.0]
        df = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low':  prices,
            'close':prices,
            'volume':[0, 0, 0]
        }, index=dates)

        # Test the broker loader directly
        initial_equity = 1000.0
        broker_loader = BrokerDataLoader(initial_equity=initial_equity)

        # Test that the loader returns the expected equity values
        for i, dt in enumerate(dates):
            # Test next() method
            result = broker_loader.next(['BROKER:ACCOUNT'], dt)
            self.assertIn('BROKER:ACCOUNT', result)
            
            # Check initial equity on first iteration
            if i == 0:
                self.assertEqual(result['BROKER:ACCOUNT']['equity'], initial_equity)
            else:
                # For subsequent iterations, check that equity has evolved from previous updates
                if i == 1:
                    expected_equity = initial_equity * (1 + 0.1)  # +10% from first update
                else:  # i == 2
                    expected_equity = initial_equity * (1 + 0.1) * (1 - 0.05)  # +10% then -5%
                self.assertAlmostEqual(result['BROKER:ACCOUNT']['equity'], expected_equity)

            # Test update() method to evolve equity for next iteration
            if i < len(dates) - 1:  # Don't update on the last iteration
                # Apply a simple return: +10% on first update, -5% on second
                strat_ret = 0.1 if i == 0 else -0.05
                broker_loader.update(dt, raw_sigs={}, raw_rets={}, strat_ret=strat_ret)

        # Test that non-broker tickers return None
        result = broker_loader.next(['AAPL', 'BROKER:ACCOUNT'], dates[0])
        self.assertIsNone(result['AAPL'])
        self.assertIn('BROKER:ACCOUNT', result)
        # Equity should be the final value after all updates
        final_expected_equity = initial_equity * (1 + 0.1) * (1 - 0.05)
        self.assertAlmostEqual(result['BROKER:ACCOUNT']['equity'], final_expected_equity)

class ExecutorIntegrationStrategy(StrategyBase):
    """Strategy that always allocates fully to a single regular ticker and logs input data."""
    def __init__(self):
        super().__init__(tickers=["FAKE", "BROKER:ACCOUNT"])
        self.step_calls = []

    def step(self, current_date, data):
        # Record the timestamp and data dict
        self.step_calls.append((current_date, data.copy()))
        # Fully allocate to FAKE
        return {"FAKE": 1.0}


class TestBrokerDataLoader(unittest.TestCase):
    def test_init_requires_args(self):
        # Must provide either broker or initial_equity
        with self.assertRaises(ValueError):
            BrokerDataLoader()

    def test_offline_mode_next_returns_initial_equity(self):
        loader = BrokerDataLoader(initial_equity=123.45)
        ts = pd.Timestamp('2025-01-01')
        out = loader.next(['BROKER:ACCOUNT'], ts)
        # Should return equity field equal to initial_equity
        self.assertIn('BROKER:ACCOUNT', out)
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 123.45)

    def test_offline_mode_next_handles_unknown_ticker(self):
        loader = BrokerDataLoader(initial_equity=100.0)
        ts = pd.Timestamp('2025-01-01')
        out = loader.next(['AAPL', 'BROKER:ACCOUNT'], ts)
        # Unknown ticker should map to None
        self.assertIsNone(out['AAPL'])
        # Known broker ticker returns equity
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 100.0)

    def test_offline_update_changes_equity(self):
        loader = BrokerDataLoader(initial_equity=100.0)
        ts = pd.Timestamp('2025-01-02')
        loader.update(ts, raw_sigs={}, raw_rets={}, strat_ret=0.1)
        out = loader.next(['BROKER:ACCOUNT'], ts)
        # Equity should grow by 10%
        self.assertAlmostEqual(out['BROKER:ACCOUNT']['equity'], 110.0)

    def test_offline_multiple_updates(self):
        loader = BrokerDataLoader(initial_equity=100.0)
        # +10% -> 110
        loader.update(pd.Timestamp('2025-01-01'), raw_sigs={}, raw_rets={}, strat_ret=0.1)
        # -50% -> 55
        loader.update(pd.Timestamp('2025-01-02'), raw_sigs={}, raw_rets={}, strat_ret=-0.5)
        out = loader.next(['BROKER:ACCOUNT'], pd.Timestamp('2025-01-02'))
        self.assertAlmostEqual(out['BROKER:ACCOUNT']['equity'], 55.0)

    def test_live_mode_next_returns_broker_equity(self):
        # Use MockBroker to simulate live broker equity
        broker = MockBroker(initial_equity=500.0)
        loader = BrokerDataLoader(broker=broker)
        ts = pd.Timestamp('2025-01-01')
        out1 = loader.next(['BROKER:ACCOUNT'], ts)
        self.assertEqual(out1['BROKER:ACCOUNT']['equity'], 500.0)
        # Change broker equity and verify next() reflects it
        broker._equity = 600.0
        out2 = loader.next(['BROKER:ACCOUNT'], ts)
        self.assertEqual(out2['BROKER:ACCOUNT']['equity'], 600.0)

    def test_live_update_does_not_affect_broker_equity(self):
        broker = MockBroker(initial_equity=200.0)
        loader = BrokerDataLoader(broker=broker)
        # Calling update in live mode should not change broker equity
        loader.update(pd.Timestamp('2025-01-02'), raw_sigs={}, raw_rets={}, strat_ret=0.5)
        out = loader.next(['BROKER:ACCOUNT'], pd.Timestamp('2025-01-02'))
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 200.0)

    def test_source_identifier_constant(self):
        """Test that SOURCE_IDENTIFIER is correctly set"""
        loader = BrokerDataLoader(initial_equity=100.0)
        self.assertEqual(loader.SOURCE_IDENTIFIER, "BROKER")

    def test_non_broker_tickers_return_none(self):
        """Test that non-BROKER prefixed tickers return None"""
        loader = BrokerDataLoader(initial_equity=100.0)
        ts = pd.Timestamp('2025-01-01')
        out = loader.next(['AAPL', 'MSFT', 'BROKER:ACCOUNT'], ts)
        # Non-broker tickers should be None
        self.assertIsNone(out['AAPL'])
        self.assertIsNone(out['MSFT'])
        # Broker ticker should return equity
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 100.0)

    def test_malformed_ticker_handling(self):
        """Test handling of malformed ticker strings"""
        loader = BrokerDataLoader(initial_equity=100.0)
        ts = pd.Timestamp('2025-01-01')
        # Test tickers without colons
        out = loader.next(['BROKERACCOUNT', 'AAPL'], ts)
        self.assertIsNone(out['BROKERACCOUNT'])
        self.assertIsNone(out['AAPL'])

    def test_zero_equity_handling(self):
        """Test handling of zero equity"""
        loader = BrokerDataLoader(initial_equity=0.0)
        ts = pd.Timestamp('2025-01-01')
        out = loader.next(['BROKER:ACCOUNT'], ts)
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 0.0)

    def test_negative_equity_handling(self):
        """Test handling of negative equity"""
        loader = BrokerDataLoader(initial_equity=-100.0)
        ts = pd.Timestamp('2025-01-01')
        out = loader.next(['BROKER:ACCOUNT'], ts)
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], -100.0)

    def test_update_with_none_strat_ret(self):
        """Test that update with None strat_ret doesn't change equity"""
        loader = BrokerDataLoader(initial_equity=100.0)
        ts = pd.Timestamp('2025-01-01')
        loader.update(ts, raw_sigs={}, raw_rets={}, strat_ret=None)
        out = loader.next(['BROKER:ACCOUNT'], ts)
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 100.0)

    def test_update_with_zero_strat_ret(self):
        """Test that update with zero strat_ret doesn't change equity"""
        loader = BrokerDataLoader(initial_equity=100.0)
        ts = pd.Timestamp('2025-01-01')
        loader.update(ts, raw_sigs={}, raw_rets={}, strat_ret=0.0)
        out = loader.next(['BROKER:ACCOUNT'], ts)
        self.assertEqual(out['BROKER:ACCOUNT']['equity'], 100.0)


class TestExecutionBrokerLoaderIntegration(unittest.TestCase):
    def test_executor_with_broker_loader(self):
        # Create a single timestamp at market open time (UTC)
        dt = datetime(2025, 4, 21, 12, 0, 0, tzinfo=timezone.utc)
        # Build a one-row OHLCV DataFrame with naive index matching loader's expectation
        dt_naive = dt.replace(tzinfo=None)
        df = pd.DataFrame({
            'open':  [100.0],
            'high':  [100.0],
            'low':   [100.0],
            'close': [100.0],
            'volume':[0]
        }, index=[dt_naive])

        # Set up loaders and broker
        market_loader = SimpleMarketLoader(df)
        initial_equity = 1000.0
        broker = MockBroker(initial_equity=initial_equity)
        broker_loader = BrokerDataLoader(broker=broker)

        # Strategy and executor (force UTC timezone)
        strat = ExecutorIntegrationStrategy()
        exe = ExecutionBase(
            strategy=strat,
            market_data_loader=market_loader,
            broker=broker,
            alternative_data_loader=broker_loader,
            timezone=timezone.utc
        )

        # Convert dt to milliseconds since epoch
        ts_ms = int(dt.timestamp() * 1000)
        # Execute one step
        orders = exe.step(ts_ms)

        # -- Verify strategy was called once with correct date and data --
        self.assertEqual(len(strat.step_calls), 1)
        call_time, call_data = strat.step_calls[0]
        # The strategy sees the UTC-aware datetime
        self.assertEqual(call_time, dt)
        # Data dict should include market ticker and broker ticker
        self.assertIn('FAKE', call_data)
        self.assertIn('BROKER:ACCOUNT', call_data)
        # Check market data close price
        self.assertEqual(call_data['FAKE']['close'], 100.0)
        # Check broker data equity value
        self.assertEqual(call_data['BROKER:ACCOUNT']['equity'], initial_equity)

        # -- Verify orders from executor --
        # One buy order of 10 shares (1000 equity / 100 price)
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.ticker, 'FAKE')
        self.assertEqual(order.side, 'buy')
        self.assertAlmostEqual(order.quantity, 10.0)
        self.assertEqual(order.status, 'filled')

        # -- Verify broker positions updated accordingly --
        positions = broker.get_positions()
        self.assertIn('FAKE', positions)
        pos = positions['FAKE']
        self.assertAlmostEqual(pos.quantity, 10.0)