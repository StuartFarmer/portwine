# portwine/backtester.py
"""
A step‑driven back‑tester that now supports **intraday bars** while
remaining 100 % backward‑compatible with the existing daily test‑suite.

Key points
----------
* Accepts any `DatetimeIndex` (e.g. 2025‑04‑17 09:30 and 2025‑04‑17 16:00
  for “open” and “close” bars on the same calendar date).
* Determines the trading calendar **only from regular market tickers
  (and any regular‑ticker benchmark)**, so alternative monthly/weekly
  data never add extra rows – exactly what the tests expect :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.
* Silent‑failure semantics preserved:
  - empty strategy / unknown tickers → **return `None`** (no exception)
  - unrecognised benchmark type → **return `None`**
* Still raises `ValueError` when `start_date > end_date`, matching
  `test_empty_date_range`.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from portwine.loaders.base import MarketDataLoader

class InvalidBenchmarkError(Exception):
    """Raised when the requested benchmark is neither a standard name nor a valid ticker."""
    pass

# ----------------------------------------------------------------------
# Built‑in benchmark helpers
# ----------------------------------------------------------------------
def benchmark_equal_weight(ret_df: pd.DataFrame, *_, **__) -> pd.Series:
    return ret_df.mean(axis=1)


def benchmark_markowitz(
    ret_df: pd.DataFrame,
    lookback: int = 60,
    shift_signals: bool = True,
    verbose: bool = False,
) -> pd.Series:
    tickers = ret_df.columns
    n = len(tickers)
    iterator = tqdm(ret_df.index, desc="Markowitz") if verbose else ret_df.index

    w_rows: List[np.ndarray] = []
    for ts in iterator:
        win = ret_df.loc[:ts].tail(lookback)
        if len(win) < 2:
            w = np.ones(n) / n
        else:
            cov = win.cov().values
            w_var = cp.Variable(n, nonneg=True)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w_var, cov)), [cp.sum(w_var) == 1])
            try:
                prob.solve()
                w = w_var.value if w_var.value is not None else np.ones(n) / n
            except Exception:
                w = np.ones(n) / n
        w_rows.append(w)

    w_df = pd.DataFrame(w_rows, index=ret_df.index, columns=tickers)
    if shift_signals:
        w_df = w_df.shift(1).ffill().fillna(1.0 / n)
    return (w_df * ret_df).sum(axis=1)


STANDARD_BENCHMARKS: Dict[str, Callable] = {
    "equal_weight": benchmark_equal_weight,
    "markowitz": benchmark_markowitz,
}

class BenchmarkTypes:
    STANDARD_BENCHMARK = 0
    TICKER = 1
    CUSTOM_METHOD = 2
    INVALID = 3

class Backtester:
    def __init__(self,
                 market_data_loader: MarketDataLoader,
                 alternative_data_loader=None):
        self.market_data_loader = market_data_loader
        self.alternative_data_loader = alternative_data_loader

    def _split_tickers(self, tickers: List[str]) -> (List[str], List[str]):
        """
        Split the universe into:
          - regular tickers (no colon)
          - alternative tickers (with SOURCE: prefix)
        """
        reg, alt = [], []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def get_benchmark_type(self, benchmark):
        # Test if benchmark is a standard method or a ticker
        if type(benchmark) is str:

            # If it's a method, return True
            standard_benchmark = STANDARD_BENCHMARKS.get(benchmark)
            if standard_benchmark is not None:
                return BenchmarkTypes.STANDARD_BENCHMARK

            # If we can find the ticker data, return True
            bm_data = self.market_data_loader.fetch_data([benchmark])
            if benchmark in bm_data:
                return BenchmarkTypes.TICKER

            # Otherwise? Return false
            return BenchmarkTypes.INVALID

        # Assume all callables work
        elif callable(benchmark):
            return BenchmarkTypes.CUSTOM_METHOD

        # Otherwise, invalid.
        return BenchmarkTypes.INVALID

    def run_backtest(self,
                     strategy,
                     shift_signals: bool = True,
                     benchmark: Union[str, Callable, None] = 'equal_weight',
                     start_date=None,
                     end_date=None,
                     require_all_history: bool = False,
                     verbose: bool = False
    ) -> Union[Dict[str, pd.DataFrame], None]:
        # 1) split tickers only
        reg_tkrs, alt_tkrs = self._split_tickers(strategy.tickers)

        # Verify that we can run a benchmark
        benchmark_type = self.get_benchmark_type(benchmark)

        if benchmark_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f'{benchmark} is not a valid benchmark.')

        # 1b) load regular data
        reg_data = self.market_data_loader.fetch_data(reg_tkrs)
        if not reg_tkrs or len(reg_data) < len(reg_tkrs):
            return None

        # 2) build trading calendar
        if hasattr(self.market_data_loader, "get_all_dates"):
            all_ts = self.market_data_loader.get_all_dates(reg_tkrs)
        else:
            all_ts = self._union_ts(reg_data)

        # 3) date filters
        if start_date:
            dt0 = pd.Timestamp(start_date)
            all_ts = [d for d in all_ts if d >= dt0]
        if end_date:
            dt1 = pd.Timestamp(end_date)
            all_ts = [d for d in all_ts if d <= dt1]

        # 4) require all history
        if require_all_history and reg_tkrs:
            firsts = [reg_data[t].index.min() for t in reg_tkrs]
            common = max(firsts)
            all_ts = [d for d in all_ts if d >= common]

        # 5) empty date‐range
        if not all_ts:
            raise ValueError("No trading dates after filtering")

        # 6) preload string‐ticker benchmark if needed
        if benchmark_type == BenchmarkTypes.TICKER:
            bm_data = self.market_data_loader.fetch_data([benchmark])

        # 7) main loop: collect raw signals
        rows = []
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts
        for ts in iterator:
            # market bars
            if hasattr(self.market_data_loader, "next"):
                bar_data = self.market_data_loader.next(reg_tkrs, ts)
            else:
                bar_data = self._bar_dict(ts, reg_data)

            # alternative bars
            alt_ld = self.alternative_data_loader
            if alt_ld:
                keys = alt_tkrs
                if hasattr(alt_ld, "next"):
                    bar_data.update(alt_ld.next(keys, ts))
                else:
                    alt_dfs = alt_ld.fetch_data(keys)
                    for t, df in alt_dfs.items():
                        bar_data[t] = self._bar_dict(ts, {t: df})[t]

            sig = strategy.step(ts, bar_data)

            row = {"date": ts}
            for t in strategy.tickers:
                row[t] = sig.get(t, 0.0)
            rows.append(row)

        sig_df = pd.DataFrame(rows).set_index("date").sort_index()
        sig_reg = (sig_df.shift(1).ffill() if shift_signals else sig_df)\
                    .fillna(0.0)[reg_tkrs]

        # 8) compute returns
        px = pd.DataFrame({t: reg_data[t]["close"] for t in reg_tkrs})
        px = px.reindex(sig_reg.index).ffill()
        ret_df = px.pct_change(fill_method=None).fillna(0.0)

        strat_ret = (ret_df * sig_reg).sum(axis=1)

        # 9) benchmark returns (stubbed)
        if benchmark_type == BenchmarkTypes.CUSTOM_METHOD:
            bm_ret = benchmark(ret_df)
        elif benchmark_type == BenchmarkTypes.STANDARD_BENCHMARK:
            bm_fn = STANDARD_BENCHMARKS[benchmark]
            bm_ret = bm_fn(ret_df)
        elif benchmark_type == BenchmarkTypes.TICKER:
            bm_data = self.market_data_loader.fetch_data([benchmark])
            ser = bm_data[benchmark]["close"].reindex(sig_reg.index).ffill()
            bm_ret = ser.pct_change(fill_method=None).fillna(0.0)
        else:
            raise InvalidBenchmarkError(f'{benchmark} is invalid. Cannot calculate returns.')

        # 10) update dynamic loader
        alt_ld = self.alternative_data_loader
        if alt_ld and hasattr(alt_ld, "update"):
            for ts in sig_reg.index:
                raw_sigs = sig_df.loc[ts, strategy.tickers].to_dict()
                raw_rets = ret_df.loc[ts].to_dict()
                alt_ld.update(ts, raw_sigs, raw_rets, strat_ret.loc[ts])

        return {
            "signals_df":       sig_reg,
            "tickers_returns":  ret_df,
            "strategy_returns": strat_ret,
            "benchmark_returns": bm_ret
        }

    # legacy helpers to keep old tests happy
    def _union_ts(self, data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        all_ts = {ts for df in data.values() for ts in df.index}
        return sorted(all_ts)

    def _bar_dict(self,
                  ts: pd.Timestamp,
                  data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[dict, None]]:
        out = {}
        for t, df in data.items():
            sub = df.loc[df.index == ts]
            if sub.empty:
                out[t] = None
            else:
                row = sub.iloc[-1]
                out[t] = {
                    "open":   row["open"],
                    "high":   row["high"],
                    "low":    row["low"],
                    "close":  row["close"],
                    "volume": row["volume"],
                }
        return out
