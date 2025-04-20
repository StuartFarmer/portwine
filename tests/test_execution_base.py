import unittest
from datetime import datetime, timezone

from portwine.execution.base import ExecutionBase, DataFetchError


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
        # timezone-aware, equals UTC tzinfo
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
        # Should map timestamp to datetime with UTC tz
        self.assertEqual(len(loader.calls), 1)
        dt_passed = loader.calls[0]
        self.assertEqual(dt_passed, datetime.fromtimestamp(ts, tz=timezone.utc))
        self.assertEqual(data['T1']['open'], 1.0)

    def test_with_alternative_data(self):
        loader = DummyLoader()
        alt_loader = DummyAltLoader()
        exec_base = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=loader,
            broker=self.broker,
            alternative_data_loader=alt_loader,
            timezone=timezone.utc,
        )
        ts = 1600000000.0
        data = exec_base.fetch_latest_data(ts)
        # Should call both loaders
        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(len(alt_loader.calls), 1)
        # Should merge extra field
        self.assertIn('extra', data['T1'])
        self.assertEqual(data['T1']['extra'], 999)
        # Original fields still present
        self.assertEqual(data['T1']['close'], 1.5)

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

    def test_alt_loader_exception_raises_data_fetch_error(self):
        loader = DummyLoader()
        alt_loader = ErrorLoader()
        exec_base = ExecutionBase(
            strategy=self.strategy,
            market_data_loader=loader,
            broker=self.broker,
            alternative_data_loader=alt_loader,
            timezone=timezone.utc,
        )
        with self.assertRaises(DataFetchError):
            exec_base.fetch_latest_data()


if __name__ == '__main__':
    unittest.main() 