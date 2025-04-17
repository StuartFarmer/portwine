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

# ----------------------------------------------------------------------
# Back‑tester
# ----------------------------------------------------------------------
class Backtester:
    """
    A flexible back‑tester for daily **and intraday** OHLCV bars.

    Only “regular” market tickers (those **without** a ``SOURCE:`` prefix)
    define the trading calendar; alternative data never create extra rows.
    """

    # ---- construction -------------------------------------------------
    def __init__(self, market_data_loader, alternative_data_loader=None):
        self.market_data_loader = market_data_loader
        self.alternative_data_loader = alternative_data_loader

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _union_ts(data_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        out = pd.Index([])
        for df in data_dict.values():
            out = out.union(pd.DatetimeIndex(df.index))
        return out.sort_values()

    @staticmethod
    def _bar_dict(ts: pd.Timestamp, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Optional[dict]]:
        out: Dict[str, Optional[dict]] = {}
        for tkr, df in data_dict.items():
            if ts in df.index:
                row = df.loc[ts]
                if isinstance(row, pd.DataFrame):  # duplicate timestamps
                    row = row.iloc[-1]
                out[tkr] = row.to_dict()
            else:
                out[tkr] = None
        return out

    @staticmethod
    def _split_tickers(tickers: Sequence[str]) -> Tuple[List[str], List[str]]:
        regular, alt = [], []
        for t in tickers:
            (alt if ":" in t else regular).append(t)
        return regular, alt

    # ---- main entry ---------------------------------------------------
    def run_backtest(
        self,
        strategy,
        shift_signals: bool = True,
        benchmark: Optional[Union[str, Callable]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date:   Optional[Union[str, datetime]] = None,
        require_all_history: bool = False,
        verbose: bool = False,
    ):
        # ------------------------------------------------------------------
        # 0) trivial early‑outs (honours legacy tests)
        # ------------------------------------------------------------------
        if not strategy.tickers:
            return None

        # ------------------------------------------------------------------
        # 1) classify tickers & pull data
        # ------------------------------------------------------------------
        reg_tkrs, alt_tkrs = self._split_tickers(strategy.tickers)

        reg_data = (
            self.market_data_loader.fetch_data(reg_tkrs)
            if reg_tkrs else {}
        )
        alt_data = {}
        if alt_tkrs and self.alternative_data_loader is not None:
            alt_data = self.alternative_data_loader.fetch_data(alt_tkrs)

        strategy_data = {**reg_data, **alt_data}
        if not strategy_data:
            return None  # unknown tickers – expected by tests

        # ------------------------------------------------------------------
        # 2) benchmark parsing & loading
        # ------------------------------------------------------------------
        single_bm: Optional[str] = None
        bm_func: Optional[Callable] = None

        if benchmark is None:
            pass
        elif isinstance(benchmark, str):
            if benchmark in STANDARD_BENCHMARKS:
                bm_func = STANDARD_BENCHMARKS[benchmark]
            else:
                single_bm = benchmark
        elif callable(benchmark):
            bm_func = benchmark
        else:                       # unrecognised type – legacy tests expect silent failure
            return None

        bm_data = {}
        if single_bm:
            loader = (
                self.alternative_data_loader if ":" in single_bm else self.market_data_loader
            )
            bm_data = loader.fetch_data([single_bm]) if loader else {}
            strategy_data.update(bm_data)

        # ------------------------------------------------------------------
        # 3) determine the **trading calendar** (regular market data only)
        # ------------------------------------------------------------------
        market_only_data: Dict[str, pd.DataFrame] = {}
        for t in reg_tkrs:
            if t in reg_data:
                market_only_data[t] = reg_data[t]
        # add regular‑ticker benchmark if present
        if single_bm and ":" not in single_bm and single_bm in bm_data:
            market_only_data[single_bm] = bm_data[single_bm]

        if not market_only_data:         # “alt‑only” strategies – tests expect None
            return None

        all_ts = self._union_ts(market_only_data)

        # ------------------------------------------------------------------
        # 4) user start/end filtering & require_all_history
        # ------------------------------------------------------------------
        if start_date and end_date and pd.to_datetime(start_date) > pd.to_datetime(end_date):
            raise ValueError("start_date cannot be after end_date")

        if start_date:
            all_ts = all_ts[all_ts >= pd.to_datetime(start_date)]
        if end_date:
            all_ts = all_ts[all_ts <= pd.to_datetime(end_date)]

        if require_all_history:
            first_dates = [df.index.min() for df in market_only_data.values()]
            common_start = max(first_dates)
            all_ts = all_ts[all_ts >= common_start]

        if len(all_ts) == 0:
            return None

        # ------------------------------------------------------------------
        # 5) iterate through bars
        # ------------------------------------------------------------------
        sig_rows: List[Dict[str, Union[pd.Timestamp, float]]] = []
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts

        for ts in iterator:
            bar_data = self._bar_dict(ts, strategy_data)
            sig = strategy.step(ts, bar_data)
            row = {"date": ts}
            for t in strategy.tickers:
                row[t] = sig.get(t, 0.0)
            sig_rows.append(row)

        sig_df = pd.DataFrame(sig_rows).set_index("date").sort_index()

        # ------------------------------------------------------------------
        # 6) look‑ahead shift & regular‑ticker masking
        # ------------------------------------------------------------------
        sig_df = (sig_df.shift(1).ffill() if shift_signals else sig_df).fillna(0.0)
        sig_reg = sig_df[reg_tkrs].copy() if reg_tkrs else pd.DataFrame(index=sig_df.index)

        # ------------------------------------------------------------------
        # 7) price/return matrices for **regular** tickers
        # ------------------------------------------------------------------
        px_df = pd.DataFrame(index=sig_df.index)
        for t in reg_tkrs:
            px_df[t] = reg_data[t]["close"].reindex(sig_df.index).ffill()
        ret_df = px_df.pct_change(fill_method=None).fillna(0.0)

        strat_ret = (ret_df * sig_reg).sum(axis=1)

        # ------------------------------------------------------------------
        # 8) benchmark returns
        # ------------------------------------------------------------------
        bm_ret = None
        if single_bm and single_bm in bm_data:
            bm_px = bm_data[single_bm]["close"].reindex(sig_df.index).ffill()
            bm_ret = bm_px.pct_change(fill_method=None).fillna(0.0)
        elif bm_func is not None:
            bm_ret = bm_func(ret_df, verbose=verbose)

        # ------------------------------------------------------------------
        # 9) output
        # ------------------------------------------------------------------
        return {
            "signals_df": sig_reg,          # regular tickers only
            "tickers_returns": ret_df,      # regular tickers only
            "strategy_returns": strat_ret,
            "benchmark_returns": bm_ret,
        }
