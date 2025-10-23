import unittest
import pandas as pd
from datetime import datetime

from portwine.data.providers.loader_adapters import MarketDataLoader

class DummyLoader(MarketDataLoader):
    """
    A dummy loader that returns pre‐supplied DataFrames for certain tickers
    and counts how many times load_ticker is called.
    """
    def __init__(self, data_map):
        super().__init__()
        self.data_map = data_map
        self.load_calls = []

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        self.load_calls.append(ticker)
        df = self.data_map.get(ticker)
        return df.copy() if df is not None else None
    
    def _get_provider(self, ticker: str):
        """Dummy implementation to avoid NotImplementedError."""
        return None


class MarketDataLoaderTests(unittest.TestCase):
    def setUp(self):
        # Create two sample DataFrames with different date indices
        idx1 = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-05'])
        self.df1 = pd.DataFrame({
            'open':   [1, 2, 3],
            'high':   [1, 2, 3],
            'low':    [1, 2, 3],
            'close':  [1, 2, 3],
            'volume': [10, 20, 30],
        }, index=idx1)

        idx2 = pd.to_datetime(['2025-01-02', '2025-01-03'])
        self.df2 = pd.DataFrame({
            'open':   [5, 6],
            'high':   [5, 6],
            'low':    [5, 6],
            'close':  [5, 6],
            'volume': [50, 60],
        }, index=idx2)

    def test_load_ticker_not_implemented(self):
        """Base class load_ticker should raise if not overridden."""
        loader = MarketDataLoader()
        with self.assertRaises(NotImplementedError):
            loader.load_ticker('ANY')

    def test_fetch_data_with_unknown_tickers(self):
        """
        fetch_data returns only the tickers whose load_ticker returns a DataFrame,
        and skips those that return None.
        """
        dummy = DummyLoader({'A': self.df1})
        out = dummy.fetch_data(['A', 'B'])
        self.assertIn('A', out)
        self.assertNotIn('B', out)
        pd.testing.assert_frame_equal(out['A'], self.df1)

    def test_fetch_data_caching(self):
        """
        On first fetch_data, load_ticker is called. On subsequent fetch_data
        for the same ticker, load_ticker is not called again.
        """
        dummy = DummyLoader({'X': self.df1})
        _ = dummy.fetch_data(['X'])
        self.assertEqual(dummy.load_calls, ['X'])
        _ = dummy.fetch_data(['X'])
        # still only one call
        self.assertEqual(dummy.load_calls, ['X'])

    def test_get_all_dates_union_and_sort(self):
        """
        get_all_dates returns the sorted union of all timestamps across tickers.
        """
        dummy = DummyLoader({'A': self.df1, 'B': self.df2})
        dates = dummy.get_all_dates(['A', 'B'])
        expected = sorted(set(self.df1.index).union(self.df2.index))
        self.assertEqual(dates, expected)

    def test_get_all_dates_empty_list(self):
        """Calling get_all_dates with no tickers yields an empty list."""
        dummy = DummyLoader({})
        dates = dummy.get_all_dates([])
        self.assertEqual(dates, [])

    def test__get_bar_at_or_before_exact(self):
        """_get_bar_at_or_before returns the exact row if ts matches an index entry."""
        dummy = DummyLoader({'A': self.df1})
        row = dummy._get_bar_at_or_before(self.df1, pd.Timestamp('2025-01-02'))
        pd.testing.assert_series_equal(row, self.df1.loc['2025-01-02'])

    def test__get_bar_at_or_before_between(self):
        """If ts lies between two bars, returns the earlier one."""
        dummy = DummyLoader({'A': self.df1})
        # 2025-01-03 isn’t in df1, so we get 2025-01-02
        row = dummy._get_bar_at_or_before(self.df1, pd.Timestamp('2025-01-03'))
        pd.testing.assert_series_equal(row, self.df1.loc['2025-01-02'])

    def test__get_bar_at_or_before_before_all(self):
        """If ts is before the first index, returns None."""
        dummy = DummyLoader({'A': self.df1})
        row = dummy._get_bar_at_or_before(self.df1, pd.Timestamp('2024-12-31'))
        self.assertIsNone(row)

    def test__get_bar_at_or_before_empty_df(self):
        """With an empty DataFrame, always returns None."""
        empty = pd.DataFrame(columns=['open','high','low','close','volume'])
        dummy = DummyLoader({'A': empty})
        row = dummy._get_bar_at_or_before(empty, pd.Timestamp('2025-01-01'))
        self.assertIsNone(row)

    def test_next_exact_and_closest(self):
        """
        next(...) returns dicts for each ticker giving the bar at-or-before ts.
        Tests both exact-match and closest-past behavior.
        """
        dummy = DummyLoader({'A': self.df1, 'B': self.df2})

        # Exact timestamp
        bar = dummy.next(['A', 'B'], pd.Timestamp('2025-01-02'))
        self.assertEqual(bar['A']['close'], 2.0)
        self.assertEqual(bar['B']['close'], 5.0)

        # Between bars
        bar = dummy.next(['A', 'B'], pd.Timestamp('2025-01-04'))
        self.assertEqual(bar['A']['close'], 2.0)  # from 2025-01-02
        self.assertEqual(bar['B']['close'], 6.0)  # from 2025-01-03

    def test_next_before_all(self):
        """If ts is before any data, next(...) returns None for that ticker."""
        dummy = DummyLoader({'A': self.df1})
        bar = dummy.next(['A'], pd.Timestamp('2024-12-31'))
        self.assertIsNone(bar['A'])

    def test_next_unknown_ticker(self):
        """
        If next(...) is asked for a ticker not in fetch_data,
        it simply omits that key from the returned dict.
        """
        dummy = DummyLoader({'A': self.df1})
        bar = dummy.next(['B'], pd.Timestamp('2025-01-02'))
        self.assertNotIn('B', bar)

    def test_next_malformed_dataframe(self):
        """
        If the DataFrame has missing columns (e.g. no 'close'),
        next(...) raises a KeyError that alerts the developer.
        """
        bad = pd.DataFrame({
            'open': [1],
            'high': [1],
            'low': [1],
            'volume': [1]
        }, index=[pd.Timestamp('2025-01-01')])
        dummy = DummyLoader({'A': bad})
        with self.assertRaises(KeyError):
            dummy.next(['A'], pd.Timestamp('2025-01-01'))


if __name__ == "__main__":
    unittest.main()
