import unittest
from portwine.backtester import Backtester
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader
import pandas as pd

class SimplePriceLoader(MarketDataLoader):
    """Same as above."""
    def __init__(self):
        super().__init__()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        self.price_A = pd.DataFrame({
            "open":   range(1, 6),
            "high":   range(1, 6),
            "low":    range(1, 6),
            "close":  range(1, 6),
            "volume": [100] * 5
        }, index=dates)

    def load_ticker(self, ticker: str):
        if ticker == "A":
            return self.price_A.copy()
        return None

class ZeroStrategy(StrategyBase):
    """Always flat."""
    def __init__(self, tickers):
        super().__init__(tickers)
    def step(self, ts, bar_data):
        return {t: 0.0 for t in self.tickers}

class TestOutputsExcludeAltTickers(unittest.TestCase):
    def setUp(self):
        self.loader = SimplePriceLoader()
        self.bt = Backtester(self.loader)
        # Include one alt ticker that should be ignored
        self.strat = ZeroStrategy(["A", "ALT:X"])

    def test_alt_tickers_not_in_outputs(self):
        res = self.bt.run_backtest(self.strat)
        # Regular outputs only include 'A'
        self.assertListEqual(list(res["signals_df"].columns), ["A"])
        self.assertListEqual(list(res["tickers_returns"].columns), ["A"])
        # strategy_returns is a Series indexed by dates
        self.assertEqual(res["strategy_returns"].name, None)

if __name__ == "__main__":
    unittest.main()
