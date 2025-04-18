import unittest
import pandas as pd
from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader

class SimplePriceLoader(MarketDataLoader):
    """
    Provides daily OHLCV for two tickers A and B over 10 days.
    A: prices 1→10, B: prices 10→1.
    """
    def __init__(self):
        super().__init__()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        self.price_A = pd.DataFrame({
            "open":   range(1, 11),
            "high":   range(1, 11),
            "low":    range(1, 11),
            "close":  range(1, 11),
            "volume": [100] * 10
        }, index=dates)
        self.price_B = pd.DataFrame({
            "open":   list(range(10, 0, -1)),
            "high":   list(range(10, 0, -1)),
            "low":    list(range(10, 0, -1)),
            "close":  list(range(10, 0, -1)),
            "volume": [100] * 10
        }, index=dates)

    def load_ticker(self, ticker: str):
        if ticker == "A":
            return self.price_A.copy()
        if ticker == "B":
            return self.price_B.copy()
        return None

class ZeroStrategy(StrategyBase):
    """Always flat (zero weights)."""
    def __init__(self, tickers):
        super().__init__(tickers)
    def step(self, ts, bar_data):
        return {t: 0.0 for t in self.tickers}

class TestRequireAllHistory(unittest.TestCase):
    def setUp(self):
        self.loader = SimplePriceLoader()
        self.bt = Backtester(self.loader)
        self.strat = ZeroStrategy(["A", "B"])

    def test_require_all_history_false_keeps_full_length(self):
        res = self.bt.run_backtest(self.strat, require_all_history=False)
        self.assertEqual(len(res["signals_df"]), 10)

    def test_require_all_history_true_trims_to_common_start(self):
        res = self.bt.run_backtest(self.strat, require_all_history=True)
        # Both tickers start on 2020-01-01, so still 10
        self.assertEqual(len(res["signals_df"]), 10)

if __name__ == "__main__":
    unittest.main()
