import unittest
import pandas as pd
from portwine.backtester import Backtester, InvalidBenchmarkError
from portwine.strategies.base import StrategyBase
from portwine.loaders.base import MarketDataLoader

class SimplePriceLoader(MarketDataLoader):
    """Same two‑ticker loader as above."""
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
    """Always flat."""
    def __init__(self, tickers):
        super().__init__(tickers)
    def step(self, ts, bar_data):
        return {t: 0.0 for t in self.tickers}

class EqualStrategy(StrategyBase):
    """Always equal weight."""
    def __init__(self, tickers):
        super().__init__(tickers)
    def step(self, ts, bar_data):
        w = 1.0 / len(self.tickers)
        return {t: w for t in self.tickers}

class TestBenchmarkDefaultAndInvalid(unittest.TestCase):
    def setUp(self):
        self.loader = SimplePriceLoader()
        self.bt = Backtester(self.loader)

    def test_default_benchmark_equal_weight(self):
        strat = EqualStrategy(["A", "B"])
        res = self.bt.run_backtest(strat)  # no benchmark specified
        bm = res["benchmark_returns"]
        ret = res["tickers_returns"]
        expected = (ret["A"] + ret["B"]) / 2
        pd.testing.assert_series_equal(bm, expected)

    def test_invalid_benchmark_raises(self):
        strat = ZeroStrategy(["A"])
        with self.assertRaises(InvalidBenchmarkError):
            self.bt.run_backtest(strat, benchmark="NONEXISTENT")

if __name__ == "__main__":
    unittest.main()
