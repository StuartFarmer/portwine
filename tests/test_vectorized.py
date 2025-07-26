# tests/test_vectorized_misc.py

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from portwine.vectorized import (
    create_price_dataframe,
    VectorizedStrategyBase,
    VectorizedBacktester
)


class DummyLoader:
    """A dummy market data loader for testing."""
    def __init__(self, data_dict):
        # data_dict: {ticker: DataFrame or None}
        self._data = data_dict

    def fetch_data(self, tickers):
        # Return only the keys requested
        return {t: self._data.get(t) for t in tickers}


class TestCreatePriceDataFrame(unittest.TestCase):
    def setUp(self):
        # Prepare sample data for two tickers
        idx = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
        self.df_a = pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=idx)
        self.df_b = pd.DataFrame({"close": [200.0, 210.0, 220.0]}, index=idx)

    def test_basic(self):
        loader = DummyLoader({"A": self.df_a, "B": self.df_b})
        df = create_price_dataframe(loader, ["A", "B"])
        # Should forward-fill identical closes
        expected = pd.DataFrame({
            "A": [100.0, 110.0, 120.0],
            "B": [200.0, 210.0, 220.0]
        }, index=self.df_a.index)
        pd.testing.assert_frame_equal(df, expected)

    def test_date_filters(self):
        loader = DummyLoader({"A": self.df_a, "B": self.df_b})
        # Only include 2021-01-02 and 2021-01-03
        df = create_price_dataframe(
            loader, ["A", "B"],
            start_date="2021-01-02", end_date="2021-01-03"
        )
        expected = pd.DataFrame({
            "A": [110.0, 120.0],
            "B": [210.0, 220.0]
        }, index=pd.to_datetime(["2021-01-02", "2021-01-03"]))
        pd.testing.assert_frame_equal(df, expected)

    def test_invalid_ticker(self):
        loader = DummyLoader({"A": self.df_a})
        # Request a ticker "B" that the loader doesn't know
        df = create_price_dataframe(loader, ["A", "B"])
        # Column "B" should exist but be all NaN
        self.assertIn("B", df.columns)
        # "A" should match known values
        np.testing.assert_allclose(df["A"].values, [100.0, 110.0, 120.0])
        # "B" must be NaN throughout
        self.assertTrue(df["B"].isna().all())

    def test_no_data(self):
        # Loader returns empty DataFrames
        empty = pd.DataFrame(columns=["close"])
        loader = DummyLoader({"A": empty, "B": empty})
        df = create_price_dataframe(loader, ["A", "B"])
        # No rows should remain
        self.assertEqual(len(df), 0)
        # Columns must still be present
        self.assertListEqual(list(df.columns), ["A", "B"])

    def test_loader_error(self):
        class BadLoader:
            def fetch_data(self, tickers):
                raise RuntimeError("loader failure")
        with self.assertRaises(RuntimeError):
            create_price_dataframe(BadLoader(), ["A", "B"])


class ConstantWeightStrategy(VectorizedStrategyBase):
    """A simple vectorized strategy that returns constant weights."""
    def __init__(self, tickers, weight):
        super().__init__(tickers)
        self.weight = weight

    def batch(self, prices_df):
        # Return same weight for every date
        idx = prices_df.index
        data = {t: [self.weight] * len(idx) for t in self.tickers}
        return pd.DataFrame(data, index=idx)


class TestVectorizedBacktester(unittest.TestCase):
    def setUp(self):
        # Price data for two dates, two tickers
        idx = pd.to_datetime(["2021-01-01", "2021-01-02"])
        self.prices = {
            "X": pd.DataFrame({"close": [100.0, 110.0]}, index=idx),
            "Y": pd.DataFrame({"close": [200.0, 220.0]}, index=idx),
            # For fallback benchmark tests
            "BM": pd.DataFrame({"close": [50.0, 55.0]}, index=idx),
        }
        self.loader = DummyLoader(self.prices)
        self.bt = VectorizedBacktester(market_data_loader=self.loader)

    def test_invalid_strategy_type(self):
        # Passing a non-VectorizedStrategyBase should error
        with self.assertRaises(TypeError):
            self.bt.run_backtest(strategy=object())

    def test_basic_backtest_equal_weight(self):
        strat = ConstantWeightStrategy(["X", "Y"], weight=0.5)
        out = self.bt.run_backtest(strat, benchmark="equal_weight")
        # Compute returns_df: X -> [0.0,0.10], Y -> [0.0,0.10]
        # Equal weights => strategy_returns = 0.5*X + 0.5*Y = [0.0,0.10]
        expected = pd.Series([0.0, 0.10], index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        pd.testing.assert_series_equal(out["strategy_returns"], expected)
        # Benchmark equal_weight => mean of X and Y returns => same as strategy here
        pd.testing.assert_series_equal(out["benchmark_returns"], expected)

    def test_shift_signals_true(self):
        # Use a strategy that sets weight=1 only on first date
        class OneDayStrategy(VectorizedStrategyBase):
            def __init__(self):
                super().__init__(["X"])
            def batch(self, prices_df):
                idx = prices_df.index
                # weight 1 on first day, 0 next
                weights = [1.0] + [0.0] * (len(idx) - 1)
                return pd.DataFrame({"X": weights}, index=idx)
        strat = OneDayStrategy()
        out = self.bt.run_backtest(strat, shift_signals=True)
        signals = out["signals_df"]
        # After shift: first row all zeros, second row weight=1
        self.assertEqual(signals.iloc[0]["X"], 0.0)
        self.assertEqual(signals.iloc[1]["X"], 1.0)

    def test_shift_signals_false(self):
        strat = ConstantWeightStrategy(["X", "Y"], weight=1.0)
        out = self.bt.run_backtest(strat, shift_signals=False)
        signals = out["signals_df"]
        # With no shift, all weights remain 1
        self.assertTrue((signals == 1.0).all().all())

    def test_custom_callable_benchmark(self):
        strat = ConstantWeightStrategy(["X", "Y"], weight=0.5)
        # Define a benchmark that picks the max return each day
        benchmark_fn = lambda rets: rets.max(axis=1)
        out = self.bt.run_backtest(strat, benchmark=benchmark_fn)
        # X and Y returns both [0.0,0.10] => max = [0.0,0.10]
        expected = pd.Series([0.0, 0.10], index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        pd.testing.assert_series_equal(out["benchmark_returns"], expected)

    def test_string_benchmark_loader_fallback(self):
        # Use the fallback code path for string benchmark not in STANDARD_BENCHMARKS
        strat = ConstantWeightStrategy(["X"], weight=1.0)
        out = self.bt.run_backtest(strat, benchmark="BM")
        # BM price [50.0,55.0] => returns [0.0,0.10]
        expected = pd.Series([0.0, 0.10], index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
        pd.testing.assert_series_equal(out["benchmark_returns"], expected)

    def test_unknown_string_benchmark(self):
        # If benchmark string isn't found and no loader data, returns None
        strat = ConstantWeightStrategy(["X"], weight=1.0)
        out = self.bt.run_backtest(strat, benchmark="UNKNOWN")
        self.assertIsNone(out["benchmark_returns"])


if __name__ == "__main__":
    unittest.main()
