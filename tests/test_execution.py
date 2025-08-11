import unittest
import os
import time
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch

from portwine.execution import ExecutionBase, DataFetchError, PortfolioExceededError
from portwine.brokers.base import Order, Position, Account, OrderNotFoundError
from portwine.strategies.base import StrategyBase
from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.data.providers.loader_adapters import EODHDMarketDataLoader

# ---- Shared Dummy classes ----

class DummyStrategy:
    """For fetch‐data tests only; holds tickers list."""
    def __init__(self, tickers):
        self.tickers = tickers

class DummyLoader(MarketDataLoader):
    """
    Used for both fetch_latest_data and ExecutionBase.step tests.
    - If initialized with `prices`, returns only {'close': price}.
    - Otherwise returns full OHLCV from `return_value` or default for 'T1'.
    """
    def __init__(self, prices=None, return_value=None):
        self.prices = prices
        if prices is None:
            if return_value is None:
                self.return_value = {
                    'T1': {'open':1.0,'high':2.0,'low':0.5,'close':1.5,'volume':100}
                }
            else:
                self.return_value = return_value
        self.calls = []

    def next(self, tickers, dt):
        self.calls.append(dt)
        if self.prices is not None:
            return {t:{'close': self.prices.get(t,0.0)} for t in tickers}
        return {t:self.return_value.get(t) for t in tickers}

class DummyAltLoader(DummyLoader):
    """Alternative loader for fetch tests."""
    def __init__(self, return_value=None):
        super().__init__(prices=None, return_value=return_value or {'T1':{'extra':999}})

class ErrorLoader(DummyLoader):
    """Loader that always fails, for fetch tests."""
    def next(self, tickers, dt):
        raise RuntimeError('loader failure')

class FakeDateTime:
    """Fake datetime for patching `datetime.now()` in ExecutionBase."""
    current_dt = None

    @classmethod
    def now(cls, tz=None):
        return cls.current_dt

# ---- ExecutionBase.fetch_latest_data tests ----

class TestExecutionBaseFetchLatestData(unittest.TestCase):
    def setUp(self):
        self.strategy = DummyStrategy(tickers=['T1'])
        self.broker = object()  # not used in fetch_latest_data

    def test_no_alt_no_timestamp(self):
        loader = DummyLoader()
        exec_base = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=loader,
            broker=self.broker,
            alternative_data_loader=None,
            timezone=timezone.utc,
        )
        data = exec_base.fetch_latest_data()
        # Should return loader data unchanged
        self.assertIn('T1', data)
        self.assertIsInstance(data['T1'], dict)
        self.assertEqual(data['T1']['close'], 1.5)
        # next called once with datetime
        self.assertEqual(len(loader.calls), 1)
        self.assertIsInstance(loader.calls[0], datetime)
        # Should be timezone-aware datetime
        self.assertEqual(loader.calls[0].tzinfo, timezone.utc)

    def test_with_timestamp_float(self):
        loader = DummyLoader()
        exec_base = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=loader,
            broker=self.broker,
            alternative_data_loader=None,
            timezone=timezone.utc,
        )
        ts = 1600000000.0  # UNIX seconds
        data = exec_base.fetch_latest_data(ts)
        # Should map timestamp to timezone-aware datetime matching UNIX seconds
        self.assertEqual(len(loader.calls), 1)
        dt_passed = loader.calls[0]
        self.assertEqual(dt_passed.tzinfo, timezone.utc)
        expected_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        self.assertEqual(dt_passed, expected_dt)
        self.assertEqual(data['T1']['open'], 1.0)

    def test_loader_exception_raises_data_fetch_error(self):
        loader = ErrorLoader()
        exec_base = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=loader,
            broker=self.broker,
            alternative_data_loader=None,
            timezone=timezone.utc,
        )
        with self.assertRaises(DataFetchError):
            exec_base.fetch_latest_data()

class TestExecutionBaseWithRealData(unittest.TestCase):
    """Fetch real EODHD data via ExecutionBase.fetch_latest_data"""
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        loader = EODHDMarketDataLoader(data_path=data_dir)
        strategy = StrategyBase(['AAPL', 'MSFT'])
        self.exec_base = ExecutionBase(
            strategy=strategy,
            market_data_loader=loader,
            broker=object(),
            alternative_data_loader=None,
            timezone=timezone.utc
        )
        self.loader = loader

    def test_fetch_latest_data_real(self):
        # load DataFrames directly
        df_aapl = self.loader.load_ticker('AAPL')
        df_msft = self.loader.load_ticker('MSFT')
        # choose a row existing in AAPL
        self.assertTrue(len(df_aapl) > 1, "AAPL test data must have at least 2 rows")
        test_date = df_aapl.index[1]
        ts = test_date.timestamp()
        data = self.exec_base.fetch_latest_data(ts)
        # keys from strategy tickers
        self.assertIn('AAPL', data)
        self.assertIn('MSFT', data)
        # match AAPL close price exactly
        self.assertEqual(
            data['AAPL']['close'],
            float(df_aapl.loc[test_date]['close'])
        )
        # expected MSFT row at or before test_date
        pos = df_msft.index.searchsorted(test_date, side='right') - 1
        expected_msft = df_msft.iloc[pos]
        self.assertEqual(
            data['MSFT']['close'],
            float(expected_msft['close'])
        )
        # ensure other fields present
        for field in ['open', 'high', 'low', 'volume']:
            self.assertIn(field, data['AAPL'])
            self.assertIn(field, data['MSFT'])

class TestExecutionBaseSplitLoaders(unittest.TestCase):
    def test_fetch_latest_data_splits_loaders(self):
        # Prepare strategy with regular and alternative tickers
        strategy = DummyStrategy(tickers=['REG1', 'SRC:ALT1', 'REG2', 'SRC:ALT2'])

        class MockMarketLoader(MarketDataLoader):
            def __init__(self):
                super().__init__()
                self.calls = []
            def next(self, tickers, dt):
                self.calls.append(('market', list(tickers), dt))
                return {t:{'close':10.0} for t in tickers}

        class MockAltLoader(MarketDataLoader):
            def __init__(self):
                super().__init__()
                self.calls = []
            def next(self, tickers, dt):
                self.calls.append(('alt', list(tickers), dt))
                return {t:{'alt_field':f"alt_{t}"} for t in tickers}

        market_loader = MockMarketLoader()
        alt_loader = MockAltLoader()
        exec_base = ExecutionBase(
            strategy=strategy,
            market_data_loader=market_loader,
            broker=object(),
            alternative_data_loader=alt_loader,
            timezone=timezone.utc
        )
        data = exec_base.fetch_latest_data()
        # Market loader should be called once with only regular tickers
        self.assertEqual(len(market_loader.calls), 1)
        _, market_tks, _ = market_loader.calls[0]
        self.assertCountEqual(market_tks, ['REG1', 'REG2'])
        # Alt loader should be called once with only alternative tickers
        self.assertEqual(len(alt_loader.calls), 1)
        _, alt_tks, _ = alt_loader.calls[0]
        self.assertCountEqual(alt_tks, ['SRC:ALT1', 'SRC:ALT2'])
        # Combined data should include keys from both loaders
        self.assertCountEqual(list(data.keys()), ['REG1','REG2','SRC:ALT1','SRC:ALT2'])
        # Check that market data and alt data merged correctly
        self.assertEqual(data['REG1']['close'], 10.0)
        self.assertEqual(data['SRC:ALT1']['alt_field'], 'alt_SRC:ALT1')

class TestGetCurrentPricesRealData(unittest.TestCase):
    """Unit tests for get_current_prices over real EODHD data with controlled dates"""
    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cls.loader = EODHDMarketDataLoader(data_path=data_dir)
        cls.exec_base = ExecutionBase(
            strategy=DummyStrategy([]),
            market_data_loader=cls.loader,
            broker=object(),
            alternative_data_loader=None,
            timezone=timezone.utc
        )
        cls.df_aapl = cls.loader.load_ticker('AAPL')
        cls.df_msft = cls.loader.load_ticker('MSFT')
        cls.df_nflx = cls.loader.load_ticker('NFLX')

    @patch('portwine.execution.datetime', new=FakeDateTime)
    def test_one_ticker(self):
        test_date = self.df_aapl.index[1]
        dt = datetime(test_date.year, test_date.month, test_date.day, tzinfo=timezone.utc)
        FakeDateTime.current_dt = dt
        prices = self.exec_base.get_current_prices(['AAPL'])
        self.assertEqual(len(prices), 1)
        expected = float(self.df_aapl.loc[test_date, 'close'])
        self.assertAlmostEqual(prices['AAPL'], expected)

    @patch('portwine.execution.datetime', new=FakeDateTime)
    def test_two_tickers(self):
        test_date = self.df_aapl.index[0]
        dt = datetime(test_date.year, test_date.month, test_date.day, tzinfo=timezone.utc)
        FakeDateTime.current_dt = dt
        prices = self.exec_base.get_current_prices(['AAPL', 'MSFT'])
        self.assertCountEqual(list(prices.keys()), ['AAPL', 'MSFT'])
        self.assertAlmostEqual(prices['AAPL'], float(self.df_aapl.loc[test_date, 'close']))
        if test_date in self.df_msft.index:
            expected_msft = float(self.df_msft.loc[test_date, 'close'])
        else:
            pos = self.df_msft.index.searchsorted(test_date, side='right') - 1
            expected_msft = float(self.df_msft.iloc[pos]['close'])
        self.assertAlmostEqual(prices['MSFT'], expected_msft)

    @patch('portwine.execution.datetime', new=FakeDateTime)
    def test_three_tickers_with_missing_nflx(self):
        test_date = pd.Timestamp('2024-02-09')
        dt = datetime(test_date.year, test_date.month, test_date.day, tzinfo=timezone.utc)
        FakeDateTime.current_dt = dt
        prices = self.exec_base.get_current_prices(['AAPL', 'MSFT', 'NFLX'])
        self.assertCountEqual(list(prices.keys()), ['AAPL', 'MSFT', 'NFLX'])
        self.assertAlmostEqual(prices['AAPL'], float(self.df_aapl.loc[test_date, 'close']))
        if test_date in self.df_msft.index:
            exp_msft = float(self.df_msft.loc[test_date, 'close'])
        else:
            pos = self.df_msft.index.searchsorted(test_date, side='right') - 1
            exp_msft = float(self.df_msft.iloc[pos]['close'])
        self.assertAlmostEqual(prices['MSFT'], exp_msft)
        pos_nflx = self.df_nflx.index.searchsorted(test_date, side='right') - 1
        exp_nflx = float(self.df_nflx.iloc[pos_nflx]['close'])
        self.assertAlmostEqual(prices['NFLX'], exp_nflx)

# ---- ExecutionBase.step tests ----

class DummyBrokerForStep:
    """Fake broker for ExecutionBase.step() tests."""
    def __init__(self, positions: dict, equity: float):
        self._positions = positions.copy()
        self._account = Account(equity=equity, last_updated_at=0)
        self.calls = []

    def get_account(self) -> Account:
        return self._account

    def get_positions(self) -> dict:
        return {t: Position(ticker=t, quantity=q, last_updated_at=0)
                for t, q in self._positions.items()}

    def market_is_open(self, dt) -> bool:
        return True

    def submit_order(self, symbol: str, quantity: float) -> Order:
        self.calls.append((symbol, quantity))
        prev = self._positions.get(symbol, 0.0)
        self._positions[symbol] = prev + quantity
        side = 'buy' if quantity > 0 else 'sell'
        return Order(
            order_id=f"id-{symbol}",
            ticker=symbol,
            side=side,
            quantity=abs(quantity),
            order_type='market',
            status='filled',
            time_in_force='day',
            average_price=0.0,
            remaining_quantity=0.0,
            created_at=0,
            last_updated_at=0,
        )

class MultiStepLoader(MarketDataLoader):
    """Returns price data each step, for multi‐step tests."""
    def __init__(self, prices):
        self.prices = prices
    def next(self, tickers, dt):
        return {t: {'close': self.prices.get(t, 0.0)} for t in tickers}

class MultiStepBroker:
    """Broker that tracks its own positions across steps."""
    def __init__(self, equity: float):
        self._positions = {}
        self._account = Account(equity=equity, last_updated_at=0)
        self.calls = []
    def get_account(self) -> Account:
        return self._account
    def get_positions(self) -> dict:
        return {t: Position(ticker=t, quantity=q, last_updated_at=0)
                for t, q in self._positions.items()}
    def market_is_open(self, dt) -> bool:
        return True
    def submit_order(self, symbol: str, quantity: float) -> Order:
        self.calls.append((symbol, quantity))
        prev = self._positions.get(symbol, 0.0)
        self._positions[symbol] = prev + quantity
        side = 'buy' if quantity > 0 else 'sell'
        return Order(
            order_id=f"id-{symbol}{len(self.calls)}",
            ticker=symbol,
            side=side,
            quantity=abs(quantity),
            order_type='market',
            status='filled',
            time_in_force='day',
            average_price=0.0,
            remaining_quantity=0.0,
            created_at=0,
            last_updated_at=0,
        )

def make_exec_base(tickers, loader, broker):
    return ExecutionBase(
        strategy=StrategyBase(tickers),
        market_data_loader=loader,
        broker=broker,
        alternative_data_loader=None,
        timezone=timezone.utc
    )

class TestExecutionStepSingle(unittest.TestCase):
    """Single‐step ExecutionBase.step() tests."""
    def test_initial_buy_single(self):
        prices = {'AAPL': 100.0, 'MSFT': 200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':0.0,'MSFT':0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':1.0,'MSFT':0.0}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        self.assertEqual(len(executed), 1)
        order = executed[0]
        self.assertIsInstance(order, Order)
        self.assertEqual(order.ticker, 'AAPL')
        self.assertEqual(order.quantity, 1000.0)
        self.assertEqual(order.side, 'buy')
        self.assertEqual(broker.calls, [('AAPL', 1000.0)])

    def test_initial_buy_multiple(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':0.0,'MSFT':0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':0.5,'MSFT':0.5}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        expected = [('AAPL',500.0,'buy'),('MSFT',250.0,'buy')]
        self.assertEqual(len(executed),2)
        for (t,q,s), o in zip(expected, executed):
            self.assertEqual(o.ticker, t)
            self.assertEqual(o.quantity, q)
            self.assertEqual(o.side, s)
        self.assertEqual(broker.calls, [(e[0],e[1]) for e in expected])

    def test_add_on_buy_single(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':500.0,'MSFT':0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':1.0,'MSFT':0.0}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        self.assertEqual(len(executed),1)
        order = executed[0]
        self.assertEqual(order.ticker,'AAPL')
        self.assertEqual(order.quantity,500.0)
        self.assertEqual(order.side,'buy')

    def test_add_on_buy_multiple(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':250.0,'MSFT':100.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':0.5,'MSFT':0.5}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        expected = [('AAPL',250.0,'buy'),('MSFT',150.0,'buy')]
        self.assertEqual(len(executed),2)
        for (t,q,s), o in zip(expected, executed):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_reduce_single(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':1000.0,'MSFT':0.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':0.5,'MSFT':0.0}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        self.assertEqual(len(executed),1)
        order = executed[0]
        self.assertEqual(order.ticker,'AAPL')
        self.assertEqual(order.quantity,500.0)
        self.assertEqual(order.side,'sell')

    def test_reduce_multiple(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':500.0,'MSFT':250.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':0.25,'MSFT':0.25}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        expected = [('AAPL',250.0,'sell'),('MSFT',125.0,'sell')]
        self.assertEqual(len(executed),2)
        for (t,q,s), o in zip(expected, executed):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_mixed_add_and_reduce(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':500.0,'MSFT':250.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':0.25,'MSFT':0.75}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        expected = [('AAPL',250.0,'sell'),('MSFT',125.0,'buy')]
        self.assertEqual(len(executed),2)
        for (t,q,s), o in zip(expected, executed):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_initial_buy_and_add_on(self):
        prices = {'AAPL':100.0,'MSFT':100.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':0.0,'MSFT':10.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':1.0,'MSFT':0.1}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        expected = [('AAPL',1000.0,'buy'),('MSFT',90.0,'buy')]
        self.assertEqual(len(executed),2)
        for (t,q,s), o in zip(expected, executed):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_initial_buy_and_sell(self):
        prices = {'AAPL':100.0,'MSFT':100.0}
        loader = DummyLoader(prices=prices)
        broker = DummyBrokerForStep({'AAPL':0.0,'MSFT':100.0}, equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
            def step(self, dt, data):
                return {'AAPL':1.0,'MSFT':0.0}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        executed = ex.step(0)
        expected = [('AAPL',1000.0,'buy'),('MSFT',100.0,'sell')]
        self.assertEqual(len(executed),2)
        for (t,q,s), o in zip(expected, executed):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

class TestExecutionStepMulti(unittest.TestCase):
    """Multi‐step ExecutionBase.step() tests."""
    def test_initial_buy_single(self):
        prices = {'AAPL':100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':1.0}
        ex = make_exec_base(['AAPL'], loader, broker)
        ex.strategy = Strat()
        o1 = ex.step(0)
        self.assertEqual(len(o1),1)
        o2 = ex.step(0)
        self.assertEqual(o2,[])

    def test_initial_buy_multiple(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':0.5,'MSFT':0.5}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        o1 = ex.step(0)
        expected1 = [('AAPL',500.0,'buy'),('MSFT',250.0,'buy')]
        self.assertEqual(len(o1),2)
        for (t,q,s), o in zip(expected1, o1):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)
        o2 = ex.step(0)
        self.assertEqual(o2,[])

    def test_add_on_buy_single(self):
        prices = {'AAPL':100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':0.9} if self.day==1 else {'AAPL':1.0}
        ex = make_exec_base(['AAPL'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        self.assertEqual(len(o2),1)
        self.assertEqual(o2[0].ticker,'AAPL')
        self.assertEqual(o2[0].quantity,100.0)
        self.assertEqual(o2[0].side,'buy')

    def test_add_on_buy_multiple(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                if self.day==1:
                    return {'AAPL':0.9,'MSFT':0.02}
                return {'AAPL':1.0,'MSFT':0.05}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        expected2 = [('AAPL',100.0,'buy'),('MSFT',15.0,'buy')]
        self.assertEqual(len(o2),2)
        for (t,q,s), o in zip(expected2, o2):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_reduce_single(self):
        prices = {'AAPL':100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':1.0} if self.day==1 else {'AAPL':0.4}
        ex = make_exec_base(['AAPL'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        self.assertEqual(len(o2),1)
        self.assertEqual(o2[0].ticker,'AAPL')
        self.assertEqual(o2[0].quantity,600.0)
        self.assertEqual(o2[0].side,'sell')

    def test_reduce_multiple(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':0.9,'MSFT':0.1} if self.day==1 else {'AAPL':0.4,'MSFT':0.02}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        expected2 = [('AAPL',500.0,'sell'),('MSFT',40.0,'sell')]
        self.assertEqual(len(o2),2)
        for (t,q,s), o in zip(expected2, o2):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_mixed_add_and_reduce(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                if self.day==1:
                    return {'AAPL':0.5,'MSFT':0.1}
                return {'AAPL':1.0,'MSFT':0.05}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        expected2 = [('AAPL',500.0,'buy'),('MSFT',25.0,'sell')]
        self.assertEqual(len(o2),2)
        for (t,q,s), o in zip(expected2, o2):
            self.assertEqual(o.ticker,t)
            self.assertEqual(o.quantity,q)
            self.assertEqual(o.side,s)

    def test_initial_buy_and_add_on(self):
        prices = {'AAPL':100.0,'MSFT':100.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':1.0,'MSFT':0.1} if self.day==1 else {'AAPL':1.0,'MSFT':0.25}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        self.assertEqual(len(o2),1)
        self.assertEqual(o2[0].ticker,'MSFT')
        self.assertEqual(o2[0].quantity,150.0)
        self.assertEqual(o2[0].side,'buy')

    def test_initial_buy_and_sell(self):
        prices = {'AAPL':100.0,'MSFT':200.0}
        loader = MultiStepLoader(prices)
        broker = MultiStepBroker(equity=100_000)
        class Strat(StrategyBase):
            def __init__(self):
                super().__init__(['AAPL','MSFT'])
                self.day=0
            def step(self, dt, data):
                self.day+=1
                return {'AAPL':1.0,'MSFT':0.5} if self.day==1 else {'AAPL':1.0,'MSFT':0.0}
        ex = make_exec_base(['AAPL','MSFT'], loader, broker)
        ex.strategy = Strat()
        ex.step(0)
        o2 = ex.step(0)
        self.assertEqual(len(o2),1)
        self.assertEqual(o2[0].ticker,'MSFT')
        self.assertEqual(o2[0].quantity,250.0)
        self.assertEqual(o2[0].side,'sell')

# ---- Internal helper tests ----

class TestCalculateTargetPositions(unittest.TestCase):
    def setUp(self):
        self.exec_base = make_exec_base(['AAPL','MSFT'], loader=None, broker=None)
        self.portfolio_value = 100_000.0
        self.prices = {'AAPL':100.0,'MSFT':200.0,'X':50.0}

    def test_all_in_one_ticker(self):
        target_weights = {"AAPL":1.0,"MSFT":0.0}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, self.prices)
        self.assertEqual(positions["AAPL"],1000)
        self.assertEqual(positions["MSFT"],0)

    def test_mixed_tickers(self):
        target_weights = {"AAPL":0.5,"MSFT":0.5}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, self.prices)
        self.assertEqual(positions["AAPL"],500)
        self.assertEqual(positions["MSFT"],250)

    def test_no_fractional_rounds_down(self):
        prices = {'AAPL':100.0,'MSFT':200.1}
        target_weights = {"AAPL":0.7,"MSFT":0.3}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=False)
        self.assertEqual(positions["AAPL"],700)
        self.assertEqual(positions["MSFT"],149)

    def test_fractional_keeps_value(self):
        prices = {'AAPL':100.0,'MSFT':200.1}
        target_weights = {"AAPL":0.7,"MSFT":0.3}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices, fractional=True)
        self.assertEqual(positions["AAPL"],700)
        self.assertAlmostEqual(positions["MSFT"],30000.0/200.1,places=8)

    def test_symbols_not_in_prices_skipped(self):
        target_weights = {"AAPL":1.0,"UNKNOWN":0.5}
        prices = {"AAPL":100.0}
        positions = self.exec_base._calculate_target_positions(
            target_weights, self.portfolio_value, prices)
        self.assertIn("AAPL",positions)
        self.assertNotIn("UNKNOWN",positions)

class TestCalculateTargetPositionsWithRealData(unittest.TestCase):
    """Tests _calculate_target_positions using real EODHD test data"""
    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__),'test_data')
        loader = EODHDMarketDataLoader(data_path=data_dir)
        cls.df_aapl = loader.load_ticker('AAPL')
        cls.df_msft = loader.load_ticker('MSFT')
        cls.exec_base = make_exec_base(['AAPL','MSFT'], loader=None, broker=None)
        cls.portfolio_value = 100_000.0

    def test_real_all_in_one_ticker(self):
        dt = self.df_aapl.index[0]
        price_aapl = float(self.df_aapl.loc[dt,'close'])
        pos = self.df_msft.index.searchsorted(dt, side='right')-1
        price_msft = float(self.df_msft.iloc[pos]['close'])
        positions = self.exec_base._calculate_target_positions(
            {'AAPL':1.0,'MSFT':0.0}, self.portfolio_value,
            {'AAPL':price_aapl,'MSFT':price_msft}, fractional=False
        )
        expected_shares = int(self.portfolio_value/price_aapl)
        self.assertEqual(positions['AAPL'], expected_shares)
        self.assertEqual(positions['MSFT'], 0)

    def test_real_mixed_tickers(self):
        dt = self.df_aapl.index[0]
        price_aapl = float(self.df_aapl.loc[dt,'close'])
        pos = self.df_msft.index.searchsorted(dt, side='right')-1
        price_msft = float(self.df_msft.iloc[pos]['close'])
        positions = self.exec_base._calculate_target_positions(
            {'AAPL':0.5,'MSFT':0.5}, self.portfolio_value,
            {'AAPL':price_aapl,'MSFT':price_msft}, fractional=False
        )
        self.assertEqual(positions['AAPL'], int((self.portfolio_value*0.5)/price_aapl))
        self.assertEqual(positions['MSFT'], int((self.portfolio_value*0.5)/price_msft))

    def test_real_no_fractional_rounds_down(self):
        dt = self.df_aapl.index[1]
        price_aapl = float(self.df_aapl.loc[dt,'close'])
        pos = self.df_msft.index.searchsorted(dt, side='right')-1
        price_msft = float(self.df_msft.iloc[pos]['close'])
        positions = self.exec_base._calculate_target_positions(
            {'AAPL':0.7,'MSFT':0.3}, self.portfolio_value,
            {'AAPL':price_aapl,'MSFT':price_msft}, fractional=False
        )
        raw_msft = (self.portfolio_value*0.3)/price_msft
        import math
        self.assertEqual(positions['MSFT'], math.floor(raw_msft))

    def test_real_fractional_keeps(self):
        dt = self.df_aapl.index[1]
        price_aapl = float(self.df_aapl.loc[dt,'close'])
        pos = self.df_msft.index.searchsorted(dt, side='right')-1
        price_msft = float(self.df_msft.iloc[pos]['close'])
        positions = self.exec_base._calculate_target_positions(
            {'AAPL':0.7,'MSFT':0.3}, self.portfolio_value,
            {'AAPL':price_aapl,'MSFT':price_msft}, fractional=True
        )
        self.assertAlmostEqual(positions['AAPL'], (self.portfolio_value*0.7)/price_aapl)
        self.assertAlmostEqual(positions['MSFT'], (self.portfolio_value*0.3)/price_msft)

class TestCalculateCurrentWeights(unittest.TestCase):
    def test_single_ticker(self):
        exec_base = make_exec_base(['AAPL'], loader=None, broker=None)
        weights = exec_base._calculate_current_weights(
            [('AAPL',20.0)], 100_000.0, {'AAPL':200.0})
        self.assertAlmostEqual(weights['AAPL'], 0.04)

    def test_includes_ticker_with_no_position(self):
        exec_base = make_exec_base(['AAPL','MSFT'], loader=None, broker=None)
        weights = exec_base._calculate_current_weights(
            [('AAPL',10.0)], 100_000.0, {'AAPL':100.0,'MSFT':200.0})
        self.assertAlmostEqual(weights['AAPL'], 0.01)
        self.assertAlmostEqual(weights['MSFT'], 0.0)

    def test_multiple_tickers(self):
        exec_base = make_exec_base(['AAPL','MSFT'], loader=None, broker=None)
        weights = exec_base._calculate_current_weights(
            [('AAPL',10.0),('MSFT',20.0)], 100_000.0, {'AAPL':100.0,'MSFT':200.0})
        self.assertAlmostEqual(weights['AAPL'], 0.01)
        self.assertAlmostEqual(weights['MSFT'], 0.04)

    def test_exceeds_portfolio_raises(self):
        exec_base = make_exec_base(['AAPL','MSFT'], loader=None, broker=None)
        with self.assertRaises(PortfolioExceededError):
            exec_base._calculate_current_weights(
                [('AAPL',1000.0),('MSFT',20.0)], 100_000.0,
                {'AAPL':100.0,'MSFT':200.0}, raises=True)

    def test_exceeds_portfolio_returns_when_not_raising(self):
        exec_base = make_exec_base(['AAPL','MSFT'], loader=None, broker=None)
        weights = exec_base._calculate_current_weights(
            [('AAPL',1000.0),('MSFT',20.0)], 100_000.0,
            {'AAPL':100.0,'MSFT':200.0}, raises=False)
        self.assertAlmostEqual(weights['AAPL'], 1.0)
        self.assertAlmostEqual(weights['MSFT'], 0.04)

class TestTargetPositionsToOrders(unittest.TestCase):
    def setUp(self):
        self.exec_base = make_exec_base([], loader=None, broker=None)

    def test_initial_buy_single(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':0}, {'AAPL':0,'MSFT':0})
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertIsInstance(o, Order)
        self.assertEqual(o.ticker, 'AAPL')
        self.assertEqual(o.quantity, 1000.0)
        self.assertEqual(o.side, 'buy')

    def test_initial_buy_multiple(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':25}, {'AAPL':0,'MSFT':0})
        self.assertEqual(len(orders), 2)
        expected = [('AAPL',1000.0,'buy'),('MSFT',25.0,'buy')]
        for exp, o in zip(expected, orders):
            self.assertEqual(o.ticker, exp[0])
            self.assertEqual(o.quantity, exp[1])
            self.assertEqual(o.side, exp[2])

    def test_add_on_buy_single(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':0}, {'AAPL':900,'MSFT':0})
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertEqual(o.ticker, 'AAPL')
        self.assertEqual(o.quantity, 100.0)
        self.assertEqual(o.side, 'buy')

    def test_add_on_buy_multiple(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':25}, {'AAPL':900,'MSFT':10})
        self.assertEqual(len(orders), 2)
        expected = [('AAPL',100.0,'buy'),('MSFT',15.0,'buy')]
        for exp, o in zip(expected, orders):
            self.assertEqual(o.ticker, exp[0])
            self.assertEqual(o.quantity, exp[1])
            self.assertEqual(o.side, exp[2])

    def test_reduce_single(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':400,'MSFT':0}, {'AAPL':1000,'MSFT':0})
        self.assertEqual(len(orders),1)
        o = orders[0]
        self.assertEqual(o.ticker,'AAPL')
        self.assertEqual(o.quantity,600.0)
        self.assertEqual(o.side,'sell')

    def test_reduce_multiple(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':400,'MSFT':10}, {'AAPL':900,'MSFT':50})
        self.assertEqual(len(orders),2)
        expected = [('AAPL',500.0,'sell'),('MSFT',40.0,'sell')]
        for exp, o in zip(expected, orders):
            self.assertEqual(o.ticker, exp[0])
            self.assertEqual(o.quantity, exp[1])
            self.assertEqual(o.side, exp[2])

    def test_mixed_add_and_reduce(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':25}, {'AAPL':500,'MSFT':50})
        self.assertEqual(len(orders),2)
        expected = [('AAPL',500.0,'buy'),('MSFT',25.0,'sell')]
        for exp, o in zip(expected, orders):
            self.assertEqual(o.ticker, exp[0])
            self.assertEqual(o.quantity, exp[1])
            self.assertEqual(o.side, exp[2])

    def test_initial_buy_and_add_on(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':25}, {'AAPL':0,'MSFT':10})
        self.assertEqual(len(orders),2)
        expected = [('AAPL',1000.0,'buy'),('MSFT',15.0,'buy')]
        for exp, o in zip(expected, orders):
            self.assertEqual(o.ticker, exp[0])
            self.assertEqual(o.quantity, exp[1])
            self.assertEqual(o.side, exp[2])

    def test_initial_buy_and_sell(self):
        orders = self.exec_base._target_positions_to_orders(
            {'AAPL':1000,'MSFT':25}, {'AAPL':0,'MSFT':50})
        self.assertEqual(len(orders),2)
        expected = [('AAPL',1000.0,'buy'),('MSFT',25.0,'sell')]
        for exp, o in zip(expected, orders):
            self.assertEqual(o.ticker, exp[0])
            self.assertEqual(o.quantity, exp[1])
            self.assertEqual(o.side, exp[2])

# ---- ExecutionBase._execute_orders tests ----

import unittest
from datetime import timezone

from portwine.execution import ExecutionBase
from portwine.brokers.base import Order, OrderExecutionError
from portwine.strategies.base import StrategyBase


class DummyBroker:
    """A fake broker that returns updated Order objects based on inputs."""
    def __init__(self):
        self.calls = []

    def submit_order(self, symbol: str, quantity: float) -> Order:
        # Record the call
        self.calls.append((symbol, quantity))
        # Return an Order with filled status and dummy metadata
        return Order(
            order_id=f"id-{symbol}",
            ticker=symbol,
            side="buy" if quantity > 0 else "sell",
            quantity=quantity,
            order_type="market",
            status="filled",
            time_in_force="day",
            average_price=123.45,
            remaining_quantity=0.0,
            created_at=1610000000000,
            last_updated_at=1610000001000,
        )


class ErrorBroker:
    """A fake broker that always raises an OrderExecutionError."""
    def submit_order(self, symbol: str, quantity: float) -> Order:
        raise OrderExecutionError("Broker failed to execute order")

class TestExecuteOrders(unittest.TestCase):
    def test_execute_orders_success(self):
        broker = DummyBroker()
        exec_base = ExecutionBase(
            strategy=StrategyBase([]),
            market_data_loader=None,
            broker=broker,
            alternative_data_loader=None,
            timezone=timezone.utc
        )
        # Create two dummy orders to execute
        orders = [
            Order(order_id="", ticker="AAPL", side="buy", quantity=10.0,
                  order_type="", status="", time_in_force="", average_price=0.0,
                  remaining_quantity=0.0, created_at=0, last_updated_at=0),
            Order(order_id="", ticker="MSFT", side="sell", quantity=5.0,
                  order_type="", status="", time_in_force="", average_price=0.0,
                  remaining_quantity=0.0, created_at=0, last_updated_at=0),
        ]
        executed = exec_base._execute_orders(orders)
        # Should return a list of updated Order objects
        self.assertEqual(len(executed), 2)
        # Broker should have been called with each symbol and signed quantity
        self.assertEqual(broker.calls, [("AAPL", 10.0), ("MSFT", -5.0)])

        # Validate returned Order fields
        for updated, original in zip(executed, orders):
            self.assertIsInstance(updated, Order)
            # Order ID should be populated by broker
            self.assertTrue(updated.order_id.startswith("id-"))
            # Ticker and quantity should match original
            self.assertEqual(updated.ticker, original.ticker)
            self.assertEqual(updated.quantity, original.quantity)
            # Status and metadata from DummyBroker
            self.assertEqual(updated.status, "filled")
            self.assertEqual(updated.average_price, 123.45)
            self.assertEqual(updated.remaining_quantity, 0.0)
            self.assertEqual(updated.time_in_force, "day")
            self.assertEqual(updated.created_at, 1610000000000)
            self.assertEqual(updated.last_updated_at, 1610000001000)

    def test_execute_orders_error_propagates(self):
        broker = ErrorBroker()
        exec_base = ExecutionBase(
            strategy=StrategyBase([]),
            market_data_loader=None,
            broker=broker,
            alternative_data_loader=None,
            timezone=timezone.utc
        )
        orders = [
            Order(order_id="", ticker="AAPL", side="buy", quantity=10.0,
                  order_type="", status="", time_in_force="", average_price=0.0,
                  remaining_quantity=0.0, created_at=0, last_updated_at=0)
        ]
        with self.assertRaises(OrderExecutionError):
            exec_base._execute_orders(orders)

# ---- ExecutionBase.run tests ----

class FakeExec(ExecutionBase):
    """Override step for testing run()."""
    def __init__(self):
        # Do not call super().__init__; run() only uses step(), but we need timezone and logger
        self.calls = []
        # Provide a dummy logger
        import logging
        self.logger = logging.getLogger('FakeExec')
        # Provide a timezone (None => naive timestamps)
        self.timezone = None

    def step(self, timestamp_ms=None):
        # Record invocation
        self.calls.append(timestamp_ms)
        # Return empty list to satisfy signature
        return []


def schedule_generator(base_time, intervals):
    """
    Yield timestamps at base_time + each interval.
    intervals are in ms.
    """
    for offset in intervals:
        yield base_time + offset


class TestExecutionRun(unittest.TestCase):
    def test_run_calls_step_for_each_timestamp(self):
        base_time_s = 1.0
        # The run() computes now_ms = int(time.time() * 1000)
        # For base_time_s=1.0, now_ms=1000
        # schedule yields [1100, 1200, 1300]
        intervals = [100, 200, 300]
        schedule = schedule_generator(1000, intervals)

        fake_exec = FakeExec()
        # Patch time.time and time.sleep so no real waiting
        with patch('time.time', return_value=base_time_s), \
             patch('time.sleep') as mock_sleep:
            fake_exec.run(schedule)

        # Ensure step() was called with each scheduled timestamp
        expected = [1000 + i for i in intervals]
        self.assertEqual(fake_exec.calls, expected)
        # Sleep should have been called for each timestamp (3 calls)
        self.assertEqual(mock_sleep.call_count, len(intervals))
        # Check that sleep was called with the correct durations
        # Durations in seconds: (timestamp - now_ms) / 1000 => [0.1, 0.2, 0.3]
        calls = [call.args[0] for call in mock_sleep.call_args_list]
        for i in range(len(intervals)):
            self.assertAlmostEqual(calls[i], intervals[i] / 1000.0)


if __name__ == '__main__':
    unittest.main()
