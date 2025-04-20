import unittest
from datetime import datetime, timezone
import os
from unittest.mock import patch
import pandas as pd

from portwine.execution import ExecutionBase, DataFetchError
from portwine.loaders.eodhd import EODHDMarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader


class DummyStrategy:
    def __init__(self, tickers):
        self.tickers = tickers


class DummyLoader:
    def __init__(self, return_value=None):
        # return_value should be a dict of ticker -> bar dict
        self.calls = []
        self.return_value = return_value or {ticker: {'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'volume': 100} for ticker in ['T1']}

    def next(self, tickers, dt):
        # record the datetime passed
        self.calls.append(dt)
        # return only keys for requested tickers
        return {t: self.return_value.get(t) for t in tickers}


class DummyAltLoader(DummyLoader):
    def __init__(self, return_value=None):
        super().__init__(return_value or {'T1': {'extra': 999}})


class ErrorLoader(DummyLoader):
    def next(self, tickers, dt):
        raise RuntimeError('loader failure')


class AlternatingStrategy(StrategyBase):
    """
    Strategy that alternates allocation fully between two tickers each day.
    Day index even -> first ticker, odd -> second.
    """
    def __init__(self, tickers):
        super().__init__(tickers)
        if len(tickers) != 2:
            raise ValueError("AlternatingStrategy requires exactly two tickers")

    def step(self, current_date, daily_data):
        # alternate based on date ordinal (day component)
        # use date day-of-month parity
        day = current_date.day
        first, second = self.tickers
        if day % 2 == 0:
            return {first: 1.0, second: 0.0}
        else:
            return {first: 0.0, second: 1.0}


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
        # tzinfo stripped, should be naive datetime
        self.assertIsNone(loader.calls[0].tzinfo)

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
        # Should map timestamp to naive datetime matching UNIX seconds
        self.assertEqual(len(loader.calls), 1)
        dt_passed = loader.calls[0]
        # tzinfo should be None
        self.assertIsNone(dt_passed.tzinfo)
        # should equal naive UTC datetime
        expected_dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
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
        # path to test_data folder
        data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        # real loader
        loader = EODHDMarketDataLoader(data_path=data_dir)
        strategy = AlternatingStrategy(tickers=['AAPL', 'MSFT'])
        # broker not used in fetch_latest_data
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


if __name__ == '__main__':
    unittest.main()

# ---- New tests for splitting regular vs alternative tickers ----
class MockMarketLoader(MarketDataLoader):
    def __init__(self):
        super().__init__()
        self.calls = []

    def next(self, tickers, dt):
        # record call type and tickers
        self.calls.append(('market', list(tickers), dt))
        # return dummy OHLCV for each ticker
        return {t: {'open': 10.0, 'high': 20.0, 'low': 5.0, 'close': 15.0, 'volume': 1000} for t in tickers}


class MockAltLoader(MarketDataLoader):
    def __init__(self):
        super().__init__()
        self.calls = []

    def next(self, tickers, dt):
        self.calls.append(('alt', list(tickers), dt))
        # return dummy alternative data for each ticker
        return {t: {'alt_field': f"alt_{t}"} for t in tickers}


class TestExecutionBaseSplitLoaders(unittest.TestCase):
    def test_fetch_latest_data_splits_loaders(self):
        # Prepare strategy with regular and alternative tickers
        strategy = DummyStrategy(tickers=['REG1', 'SRC:ALT1', 'REG2', 'SRC:ALT2'])
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
        call_type, call_tks, dt = market_loader.calls[0]
        self.assertEqual(call_type, 'market')
        self.assertCountEqual(call_tks, ['REG1', 'REG2'])
        # Alt loader should be called once with only alternative tickers
        self.assertEqual(len(alt_loader.calls), 1)
        call_type2, call_tks2, dt2 = alt_loader.calls[0]
        self.assertEqual(call_type2, 'alt')
        self.assertCountEqual(call_tks2, ['SRC:ALT1', 'SRC:ALT2'])
        # Combined data should include keys from both loaders
        self.assertCountEqual(list(data.keys()), ['REG1', 'REG2', 'SRC:ALT1', 'SRC:ALT2'])
        # Check that market data and alt data merged correctly
        self.assertEqual(data['REG1']['close'], 15.0)
        self.assertEqual(data['SRC:ALT1']['alt_field'], 'alt_SRC:ALT1')
        # Ensure no overlap or missing entries
        for t in ['REG2', 'SRC:ALT2']:
            if ':' in t:
                self.assertIn('alt_field', data[t])
            else:
                self.assertIn('close', data[t])


class FakeDateTime:
    """Fake datetime class to override now() in ExecutionBase."""
    current_dt = None
    @classmethod
    def now(cls, tz=None):
        return cls.current_dt


class TestGetCurrentPricesRealData(unittest.TestCase):
    """Unit tests for get_current_prices over real EODHD data with controlled dates"""
    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cls.loader = EODHDMarketDataLoader(data_path=data_dir)
        # dummy strategy tickers list isn't used here
        cls.exec_base = ExecutionBase(
            strategy=DummyStrategy(tickers=[]),
            market_data_loader=cls.loader,
            broker=object(),
            alternative_data_loader=None,
            timezone=timezone.utc
        )
        # load data frames
        cls.df_aapl = cls.loader.load_ticker('AAPL')
        cls.df_msft = cls.loader.load_ticker('MSFT')
        cls.df_nflx = cls.loader.load_ticker('NFLX')

    @patch('portwine.execution.datetime', new=FakeDateTime)
    def test_one_ticker(self):
        # Test a single ticker on a date where data exists
        # choose second row of AAPL
        test_date = self.df_aapl.index[1]
        # simulate now() at UTC midnight of that date
        dt = datetime(test_date.year, test_date.month, test_date.day, tzinfo=timezone.utc)
        FakeDateTime.current_dt = dt
        prices = self.exec_base.get_current_prices(['AAPL'])
        # should return exactly one entry
        self.assertEqual(len(prices), 1)
        expected = float(self.df_aapl.loc[test_date, 'close'])
        self.assertAlmostEqual(prices['AAPL'], expected)

    @patch('portwine.execution.datetime', new=FakeDateTime)
    def test_two_tickers(self):
        # Test two tickers with different start dates on a date where both have data
        # choose first date AAPL has data
        test_date = self.df_aapl.index[0]
        dt = datetime(test_date.year, test_date.month, test_date.day, tzinfo=timezone.utc)
        FakeDateTime.current_dt = dt
        prices = self.exec_base.get_current_prices(['AAPL', 'MSFT'])
        # both tickers should be present
        self.assertCountEqual(list(prices.keys()), ['AAPL', 'MSFT'])
        self.assertAlmostEqual(prices['AAPL'], float(self.df_aapl.loc[test_date, 'close']))
        # MSFT index may start earlier or same, should find same date
        if test_date in self.df_msft.index:
            expected_msft = float(self.df_msft.loc[test_date, 'close'])
        else:
            # use previous available bar
            pos = self.df_msft.index.searchsorted(test_date, side='right') - 1
            expected_msft = float(self.df_msft.iloc[pos]['close'])
        self.assertAlmostEqual(prices['MSFT'], expected_msft)

    @patch('portwine.execution.datetime', new=FakeDateTime)
    def test_three_tickers_with_missing_nflx(self):
        # Test three tickers, NFLX missing on specific date; use 2024-02-09 where NFLX is missing
        test_date = pd.Timestamp('2024-02-09')
        dt = datetime(test_date.year, test_date.month, test_date.day, tzinfo=timezone.utc)
        FakeDateTime.current_dt = dt
        prices = self.exec_base.get_current_prices(['AAPL', 'MSFT', 'NFLX'])
        # All three should be present using last available for NFLX
        self.assertCountEqual(list(prices.keys()), ['AAPL', 'MSFT', 'NFLX'])
        # AAPL exact
        self.assertAlmostEqual(prices['AAPL'], float(self.df_aapl.loc[test_date, 'close']))
        # MSFT exact or prev
        if test_date in self.df_msft.index:
            exp_msft = float(self.df_msft.loc[test_date, 'close'])
        else:
            pos = self.df_msft.index.searchsorted(test_date, side='right') - 1
            exp_msft = float(self.df_msft.iloc[pos]['close'])
        self.assertAlmostEqual(prices['MSFT'], exp_msft)
        # NFLX missing that day -> prev bar is 2024-02-08
        pos_nflx = self.df_nflx.index.searchsorted(test_date, side='right') - 1
        exp_nflx = float(self.df_nflx.iloc[pos_nflx]['close'])
        self.assertAlmostEqual(prices['NFLX'], exp_nflx) 