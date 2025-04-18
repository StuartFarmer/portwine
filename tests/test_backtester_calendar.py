import unittest
import pandas as pd
from datetime import timedelta
from portwine.backtester import Backtester, InvalidBenchmarkError
from portwine.loaders.base import MarketDataLoader

# A simple strategy that does nothing (zero weights)
class ZeroStrategy:
    def __init__(self, tickers):
        self.tickers = tickers
    def step(self, ts, bar_data):
        return {t: 0.0 for t in self.tickers}

# A fake calendar that only returns odd‐numbered days at 16:00
class FakeCalendar:
    tz = "UTC"
    def schedule(self, start_date, end_date):
        # build business days between start and end
        days = pd.date_range(start_date, end_date, freq="D")
        # pick only those with odd day-of-month
        sel = [d for d in days if d.day % 2 == 1]
        closes = [pd.Timestamp(d.date()) + pd.Timedelta(hours=16) for d in sel]
        df = pd.DataFrame({"market_close": closes}, index=sel)
        return df

class SimpleDateLoader(MarketDataLoader):
    """
    Loads a DataFrame with business‐daily closes from 2020‑01‑01 to 2020‑01‑10.
    Prices and volume are dummy.
    """
    def __init__(self):
        super().__init__()
        dates = pd.date_range("2020-01-13", "2020-01-18", freq="D")
        df = pd.DataFrame({
            "open":   range(len(dates)),
            "high":   range(len(dates)),
            "low":    range(len(dates)),
            "close":  range(len(dates)),
            "volume": [1.0] * len(dates)
        }, index=dates)
        self._data_cache["X"] = df

    def load_ticker(self, ticker: str):
        return None  # unused because we pre‐cached

class TestBacktesterWithCalendar(unittest.TestCase):
    def setUp(self):
        # loader that has all dates 1–10 Jan 2020
        self.loader = SimpleDateLoader()
        # use our FakeCalendar to only trade on odd days
        self.bt = Backtester(
            market_data_loader=self.loader,
            calendar=FakeCalendar()
        )
        self.strategy = ZeroStrategy(["X"])

        # the calendar schedule: odd days 1,3,5,7,9 at 16:00
        base_days = pd.date_range("2020-01-13","2020-01-18",freq="D")
        sel = [d for d in base_days if d.day % 2 == 1]
        self.calendar_ts = pd.DatetimeIndex(
            [pd.Timestamp(d.date()) + pd.Timedelta(hours=16) for d in sel]
        )

    def test_calendar_overrides_data_dates(self):
        res = self.bt.run_backtest(self.strategy, shift_signals=False)
        # signals_df.index must equal our calendar_ts
        pd.testing.assert_index_equal(res["signals_df"].index, self.calendar_ts)

    def test_start_end_filters_with_calendar(self):
        """
        Calendar start/end filtering should include exactly
        Jan 13, Jan 15 and Jan 17 of 2025 (all valid NYSE closes).
        """
        # --- start_date only: from 2025-01-13 16:00 onward ---
        start = pd.Timestamp("2020-01-13 16:00")
        res = self.bt.run_backtest(
            self.strategy,
            start_date=start
        )
        expected_after_start = pd.DatetimeIndex([
            pd.Timestamp("2020-01-13 16:00"),
            pd.Timestamp("2020-01-15 16:00"),
            pd.Timestamp("2020-01-17 16:00"),
        ])

        pd.testing.assert_index_equal(
            res["signals_df"].index,
            expected_after_start
        )

        # --- end_date only: up to 2025-01-17 16:00 inclusive ---
        end = pd.Timestamp("2020-01-17 16:00")
        res = self.bt.run_backtest(
            self.strategy,
            shift_signals=False,
            end_date=end
        )
        expected_before_end = pd.DatetimeIndex([
            pd.Timestamp("2020-01-13 16:00"),
            pd.Timestamp("2020-01-15 16:00"),
            pd.Timestamp("2020-01-17 16:00"),
        ])
        pd.testing.assert_index_equal(
            res["signals_df"].index,
            expected_before_end
        )


    def test_invalid_date_range_raises(self):
        # start > end must raise
        with self.assertRaises(ValueError):
            self.bt.run_backtest(
                self.strategy,
                shift_signals=False,
                start_date="2020-01-10",
                end_date="2020-01-01"
            )

    def test_non_overlapping_date_range_raises_value_error(self):
        # future window with no overlap => None
        with self.assertRaises(ValueError):
            res = self.bt.run_backtest(
                self.strategy,
                shift_signals=False,
                start_date="2030-01-01",
                end_date="2030-01-05"
            )

    def test_require_all_history_with_calendar(self):
        # require_all_history should not further trim (calendar already within data span)
        res1 = self.bt.run_backtest(self.strategy, require_all_history=False)
        res2 = self.bt.run_backtest(self.strategy, require_all_history=True)
        pd.testing.assert_index_equal(res1["signals_df"].index,
                                      res2["signals_df"].index)

    def test_benchmark_equal_weight_with_calendar(self):
        # single‐ticker equal‐weight => benchmark_returns == strategy_returns == zero
        res = self.bt.run_backtest(self.strategy)
        pd.testing.assert_series_equal(
            res["benchmark_returns"],
            res["strategy_returns"]
        )

    def test_invalid_benchmark_raises(self):
        with self.assertRaises(InvalidBenchmarkError):
            self.bt.run_backtest(self.strategy, benchmark="NONEXISTENT")

if __name__ == "__main__":
    unittest.main()
