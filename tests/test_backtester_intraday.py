import unittest
from datetime import time
import pandas as pd

from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase




class OvernightIntradayStrategy(StrategyBase):
    """
    Goes long only on the 16:00 bar; flat at all other times.
    """
    def __init__(self, tickers):
        super().__init__(tickers)

    def step(self, current_date, bar_data):
        if current_date.time() == time(16, 0):
            return {t: 1.0 for t in self.tickers}
        return {t: 0.0 for t in self.tickers}


class IntradayOvernightStrategy(StrategyBase):
    """
    Goes long only on the 09:30 bar; flat at all other times.
    """
    def __init__(self, tickers):
        super().__init__(tickers)

    def step(self, current_date, bar_data):
        if current_date.time() == time(9, 30):
            return {t: 1.0 for t in self.tickers}
        return {t: 0.0 for t in self.tickers}


class MockIntradayLoader:
    """
    Synthetic intraday OHLCV for ticker 'TEST':
      2025-04-14 09:30, 16:00
      2025-04-15 09:30, 16:00
    """
    def __init__(self):
        dates = pd.to_datetime([
            '2025-04-14 09:30', '2025-04-14 16:00',
            '2025-04-15 09:30', '2025-04-15 16:00',
        ])
        df = pd.DataFrame({
            'open':   [1, 1, 1, 1],
            'high':   [1, 1, 1, 1],
            'low':    [1, 1, 1, 1],
            'close':  [1, 1, 1, 1],
            'volume': [100, 100, 100, 100],
        }, index=dates)
        self.data = {'TEST': df}

    def fetch_data(self, tickers):
        return {t: self.data[t] for t in tickers}


class CustomIntradayLoader:
    """
    Synthetic intraday OHLCV for ticker 'TEST' with varying close prices:
      2025‑04‑14 09:30 -> 100
      2025‑04‑14 16:00 -> 104
      2025‑04‑15 09:30 -> 108
      2025‑04‑15 16:00 -> 102
    """
    def __init__(self):
        dates = pd.to_datetime([
            '2025-04-14 09:30', '2025-04-14 16:00',
            '2025-04-15 09:30', '2025-04-15 16:00',
        ])
        self.df = pd.DataFrame({
            'open':   [100,   104,   108,   102],
            'high':   [101,   105,   109,   103],
            'low':    [ 99,   103,   107,   101],
            'close':  [100,   104,   108,   102],
            'volume': [1000,  1000,  1000,  1000],
        }, index=dates)

    def fetch_data(self, tickers):
        return {t: self.df for t in tickers}


class TestIntradayBacktester(unittest.TestCase):
    def setUp(self):
        self.loader = MockIntradayLoader()
        self.bt = Backtester(self.loader, calendar=None)

    def test_overnight_intraday_signals_raw(self):
        strat = OvernightIntradayStrategy(['TEST'])
        res = self.bt.run_backtest(strat, shift_signals=False)
        sig = res['signals_df']

        # 09:30 -> flat
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 09:30'), 'TEST'], 0.0)
        # 16:00 -> long
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 16:00'), 'TEST'], 1.0)
        # Next day same pattern
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 09:30'), 'TEST'], 0.0)
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 16:00'), 'TEST'], 1.0)

    def test_intraday_overnight_signals_raw(self):
        strat = IntradayOvernightStrategy(['TEST'])
        res = self.bt.run_backtest(strat, shift_signals=False)
        sig = res['signals_df']

        # 09:30 -> long
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 09:30'), 'TEST'], 1.0)
        # 16:00 -> flat
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-14 16:00'), 'TEST'], 0.0)
        # Next day same pattern
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 09:30'), 'TEST'], 1.0)
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 16:00'), 'TEST'], 0.0)

    def test_signals_shifted(self):
        strat = OvernightIntradayStrategy(['TEST'])
        # default shift_signals=True
        res = self.bt.run_backtest(strat)
        sig = res['signals_df']

        # First bar (2025-04-14 09:30) => no prior signal => 0
        self.assertEqual(sig.iloc[0]['TEST'], 0.0)
        # Raw at 2025-04-14 16:00 = 1 => after shift appears at 2025-04-15 09:30
        self.assertEqual(sig.loc[pd.Timestamp('2025-04-15 09:30'), 'TEST'], 1.0)

    def test_start_end_date_filtering(self):
        strat = OvernightIntradayStrategy(['TEST'])
        res = self.bt.run_backtest(
            strat,
            shift_signals=False,
            start_date='2025-04-14 16:00',
            end_date='2025-04-14 16:00'
        )
        sig = res['signals_df']

        self.assertListEqual(
            list(sig.index),
            [pd.Timestamp('2025-04-14 16:00')]
        )
        self.assertEqual(sig.iloc[0]['TEST'], 1.0)

    def test_union_ts_merges_and_sorts(self):
        strat = OvernightIntradayStrategy(['TEST'])
        res = self.bt.run_backtest(strat, shift_signals=False)
        sig = res['signals_df']

        expected = pd.to_datetime([
            '2025-04-14 09:30',
            '2025-04-14 16:00',
            '2025-04-15 09:30',
            '2025-04-15 16:00'
        ])
        pd.testing.assert_index_equal(sig.index, expected, check_names=False)


class TestIntradayReturnCalculations(unittest.TestCase):
    def setUp(self):
        self.bt = Backtester(CustomIntradayLoader(), calendar=None)

        # precompute the four percent returns:
        # first bar: 09:30 -> no prior bar -> pct_change = NaN -> filled to 0
        # 16:00: (104/100 -1) = 0.04
        # next 09:30: (108/104 -1) ≈ 0.038461538
        # next 16:00: (102/108 -1) ≈ -0.055555556
        self.expected_ret = {
            '2025-04-14 09:30': 0.0,
            '2025-04-14 16:00':  0.04,
            '2025-04-15 09:30':  (108/104) - 1,
            '2025-04-15 16:00':  (102/108) - 1,
        }

    def test_tickers_returns(self):
        """tickers_returns matches the true overnight/intraday pct_change."""
        strat = OvernightIntradayStrategy(['TEST'])
        res = self.bt.run_backtest(strat, shift_signals=False)
        ret_df = res['tickers_returns']

        for ts_str, exp in self.expected_ret.items():
            ts = pd.Timestamp(ts_str)
            self.assertAlmostEqual(
                ret_df.loc[ts, 'TEST'],
                exp,
                places=8,
                msg=f"ret_df at {ts} should be {exp}"
            )

    def test_strategy_returns_overnight_intraday(self):
        """
        OvernightIntradayStrategy only captures intraday bars (16:00),
        so its strategy_returns at 16:00 should equal the intraday pct_change,
        and zero at the opens.
        """
        strat = OvernightIntradayStrategy(['TEST'])
        res = self.bt.run_backtest(strat, shift_signals=False)
        sr = res['strategy_returns']

        # 09:30 bars => no position => 0
        self.assertEqual(sr[pd.Timestamp('2025-04-14 09:30')], 0.0)
        self.assertEqual(sr[pd.Timestamp('2025-04-15 09:30')], 0.0)

        # 16:00 bars => position => intraday pct_change
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-14 16:00')],
            self.expected_ret['2025-04-14 16:00'],
            places=8
        )
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-15 16:00')],
            self.expected_ret['2025-04-15 16:00'],
            places=8
        )

    def test_strategy_returns_intraday_overnight(self):
        """
        IntradayOvernightStrategy only captures overnight bars (09:30),
        so its strategy_returns at 09:30 should equal the overnight pct_change,
        and zero at the closes.
        """
        strat = IntradayOvernightStrategy(['TEST'])
        res = self.bt.run_backtest(strat, shift_signals=False)
        sr = res['strategy_returns']

        # 16:00 bars => no position => 0
        self.assertEqual(sr[pd.Timestamp('2025-04-14 16:00')], 0.0)
        self.assertEqual(sr[pd.Timestamp('2025-04-15 16:00')], 0.0)

        # 09:30 bars => position => overnight pct_change
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-14 09:30')],
            self.expected_ret['2025-04-14 09:30'],
            places=8
        )
        self.assertAlmostEqual(
            sr[pd.Timestamp('2025-04-15 09:30')],
            self.expected_ret['2025-04-15 09:30'],
            places=8
        )


if __name__ == '__main__':
    unittest.main()
