import unittest
import pandas as pd
from datetime import datetime
from portwine.strategies.base import StrategyBase
from portwine.backtester import Backtester

# Fake market data loader for integration testing
class FakeLoader:
    def __init__(self):
        # 5 days of dummy data
        self.dates = pd.date_range('2025-01-01', '2025-01-05', freq='D')
    def fetch_data(self, tickers):
        dfs = {}
        for t in tickers:
            # create a DataFrame with constant prices
            dfs[t] = pd.DataFrame({
                'open':   1.0,
                'high':   1.0,
                'low':    1.0,
                'close':  1.0,
                'volume': 100
            }, index=self.dates)
        return dfs

class TestStrategyBase(unittest.TestCase):
    def test_dedup_tickers(self):
        # duplicates should be removed, preserving order
        s = StrategyBase(['A', 'B', 'A', 'C', 'B'])
        self.assertEqual(s.tickers, ['A', 'B', 'C'])

class TestBacktesterIntegration(unittest.TestCase):
    def test_backtest_runs_and_respects_dedup(self):
        loader = FakeLoader()
        bt = Backtester(loader)
        # Initialize strategy with duplicate tickers
        s = StrategyBase(['X', 'X', 'Y'])
        # After init, duplicates must be removed
        self.assertEqual(s.tickers, ['X', 'Y'])
        # Run backtest; should not error
        res = bt.run_backtest(s, verbose=False)
        # Should return a dict including 'strategy_returns'
        self.assertIsInstance(res, dict)
        self.assertIn('strategy_returns', res)
        # Verify the returns series has entries for the 5 data days (1st day may be NaN if pct_change)
        sr = res['strategy_returns']
        self.assertGreaterEqual(len(sr), 4)  # at least 4 valid return entries

if __name__ == "__main__":
    unittest.main()
