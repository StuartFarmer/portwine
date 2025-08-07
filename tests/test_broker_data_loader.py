import unittest
import pandas as pd


from datetime import datetime, timezone

from portwine.execution import ExecutionBase
from portwine.brokers.mock import MockBroker
from portwine.loaders.base import MarketDataLoader
from portwine.loaders.broker import BrokerDataLoader
from portwine.backtester.core import NewBacktester
from portwine.strategies.base import StrategyBase
from portwine.data.interface import DataInterface
from unittest.mock import Mock

class MockDataInterface(DataInterface):
    """Mock DataInterface for testing"""
    def __init__(self, mock_data=None):
        self.mock_data = mock_data or {}
        self.current_timestamp = None
        
        # Create a proper mock data loader
        self.data_loader = Mock()
        
        # Configure the mock data loader to return proper data
        def mock_next(tickers, timestamp):
            result = {}
            for ticker in tickers:
                if ticker in self.mock_data:
                    data = self.mock_data[ticker]
                    if self.current_timestamp is not None:
                        # Return data for the current timestamp
                        dt_python = pd.Timestamp(self.current_timestamp)
                        if hasattr(data, 'index'):
                            try:
                                idx = data.index.get_loc(dt_python)
                                result[ticker] = {
                                    'close': data['close'].iloc[idx],
                                    'open': data['open'].iloc[idx],
                                    'high': data['high'].iloc[idx],
                                    'low': data['low'].iloc[idx],
                                    'volume': data['volume'].iloc[idx]
                                }
                            except (KeyError, IndexError):
                                result[ticker] = None
                        else:
                            result[ticker] = data
                    else:
                        result[ticker] = data
                else:
                    result[ticker] = None
            return result
        
        self.data_loader.next = mock_next
        
    def set_current_timestamp(self, dt):
        self.current_timestamp = dt
        super().set_current_timestamp(dt)
        
    def __getitem__(self, ticker):
        if ticker in self.mock_data:
            data = self.mock_data[ticker]
            if self.current_timestamp is not None:
                # Return data for the current timestamp
                dt_python = pd.Timestamp(self.current_timestamp)
                if hasattr(data, 'index'):
                    try:
                        idx = data.index.get_loc(dt_python)
                        return {
                            'close': float(data['close'].iloc[idx]),
                            'open': float(data['open'].iloc[idx]),
                            'high': float(data['high'].iloc[idx]),
                            'low': float(data['low'].iloc[idx]),
                            'volume': float(data['volume'].iloc[idx])
                        }
                    except (KeyError, IndexError):
                        return None
                else:
                    return data
            return data
        return None
        
    def exists(self, ticker, start_date, end_date):
        return ticker in self.mock_data

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
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2020-01-10'
        
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

        # Market and broker loaders
        market_loader = SimpleMarketLoader(df)
        initial_equity = 1000.0
        broker_loader = BrokerDataLoader(initial_equity=initial_equity)

        # Strategy and backtester
        strat = BrokerIntegrationStrategy()
        
        # Convert data to the format expected by NewBacktester
        mock_data = {"FAKE": df, "BROKER:ACCOUNT": {"equity": initial_equity}}
        data_interface = MockDataInterface(mock_data)
        calendar = MockDailyMarketCalendar("NYSE")
        
        bt = NewBacktester(
            data=data_interface,
            calendar=calendar
        )

        # Run backtest without shifting signals (direct mapping)
        results = bt.run_backtest(
            strategy=strat
        )

        # 1) Strategy step was called once per date
        self.assertEqual(len(strat.step_calls), len(dates))

        # 2) In each call, broker loader's equity field is the initial value
        for ts, data in strat.step_calls:
            self.assertIn("BROKER:ACCOUNT", data)
            self.assertEqual(data["BROKER:ACCOUNT"]["equity"], initial_equity)

        # 3) After backtest, broker_loader.equity has been updated by successive returns
        # Returns: [0.0, 1.0, -0.75] => equity factor = 1 * 2 * 0.25 = 0.5
        expected_equity = initial_equity * 0.5
        self.assertAlmostEqual(broker_loader.equity, expected_equity)

        # 4) The backtest results include strategy_returns matching price pct changes
        # Price pct change: [NaN->0.0, 1.0, -0.75]
        expected_returns = pd.Series([0.0, 1.0, -0.75], index=dates)
        pd.testing.assert_series_equal(
            results['strategy_returns'], expected_returns
        )

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