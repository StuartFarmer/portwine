import unittest
import pandas as pd


from datetime import datetime, timezone

from portwine.execution import ExecutionBase
from portwine.brokers.mock import MockBroker
from portwine.loaders.base import MarketDataLoader
from portwine.loaders.broker import BrokerDataLoader
from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase


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
        bt = Backtester(
            market_data_loader=market_loader,
            alternative_data_loader=broker_loader
        )

        # Run backtest without shifting signals (direct mapping)
        results = bt.run_backtest(
            strategy=strat,
            shift_signals=False
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