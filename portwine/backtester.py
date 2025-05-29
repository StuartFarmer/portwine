# portwine/backtester.py

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging as _logging
from tqdm import tqdm
from portwine.logging import Logger

import pandas_market_calendars as mcal
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
    "markowitz":    benchmark_markowitz,
}

class BenchmarkTypes:
    STANDARD_BENCHMARK = 0
    TICKER             = 1
    CUSTOM_METHOD      = 2
    INVALID            = 3

# ------------------------------------------------------------------------------
# Backtester
# ------------------------------------------------------------------------------
class Backtester:
    """
    A step‑driven back‑tester that supports intraday bars and,
    optionally, an exchange trading calendar.
    """

    def __init__(
        self,
        market_data_loader: MarketDataLoader,
        alternative_data_loader=None,
        calendar: Optional[Union[str, mcal.ExchangeCalendar]] = None,
        logger: Optional[_logging.Logger] = None,  # pre-configured logger or default
        log: bool = False,  # enable backtester logging if True
    ):
        self.market_data_loader      = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        if isinstance(calendar, str):
            self.calendar = mcal.get_calendar(calendar)
        else:
            self.calendar = calendar
        # --- configure logging for backtester ---
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger.create(__name__, level=_logging.INFO)
            # enable or disable logging based on simple flag
            self.logger.disabled = not log

    def _split_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        reg, alt = [], []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def get_benchmark_type(self, benchmark) -> int:
        if isinstance(benchmark, str):
            if benchmark in STANDARD_BENCHMARKS:
                return BenchmarkTypes.STANDARD_BENCHMARK
            if self.market_data_loader.fetch_data([benchmark]).get(benchmark) is not None:
                return BenchmarkTypes.TICKER
            return BenchmarkTypes.INVALID
        if callable(benchmark):
            return BenchmarkTypes.CUSTOM_METHOD
        return BenchmarkTypes.INVALID

    def run_backtest(
        self,
        strategy,
        shift_signals: bool = True,
        benchmark: Union[str, Callable, None] = "equal_weight",
        start_date=None,
        end_date=None,
        require_all_history: bool = False,
        require_all_tickers: bool = False,
        verbose: bool = False
    ) -> Optional[Dict[str, pd.DataFrame]]:
        # adjust logging level based on verbosity
        self.logger.setLevel(_logging.DEBUG if verbose else _logging.INFO)
        self.logger.info(
            "Starting backtest: tickers=%s, start_date=%s, end_date=%s",
            strategy.tickers, start_date, end_date,
        )
        # 1) normalize date filters
        sd = pd.Timestamp(start_date) if start_date is not None else None
        ed = pd.Timestamp(end_date)   if end_date   is not None else None
        if sd is not None and ed is not None and sd > ed:
            raise ValueError("start_date must be on or before end_date")

        # 2) split tickers
        reg_tkrs, alt_tkrs = self._split_tickers(strategy.tickers)
        self.logger.debug(
            "Split tickers: %d regular, %d alternative", len(reg_tkrs), len(alt_tkrs)
        )

        # 3) classify benchmark
        bm_type = self.get_benchmark_type(benchmark)
        if bm_type == BenchmarkTypes.INVALID:
            raise InvalidBenchmarkError(f"{benchmark} is not a valid benchmark.")

        # 4) load regular data
        reg_data = self.market_data_loader.fetch_data(reg_tkrs)
        self.logger.debug(
            "Fetched market data for %d tickers", len(reg_data)
        )
        # identify any tickers for which we got no data
        missing = [t for t in reg_tkrs if t not in reg_data]
        if missing:
            msg = (
                f"Market data loader returned data for {len(reg_data)}/"
                f"{len(reg_tkrs)} requested tickers. Missing: {missing}"
            )
            if require_all_tickers:
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                self.logger.warning(msg)
        # only keep tickers that have data
        reg_tkrs = [t for t in reg_tkrs if t in reg_data]

        # 5) preload benchmark ticker if needed (for require_all_history and later returns)
        if bm_type == BenchmarkTypes.TICKER:
            bm_data = self.market_data_loader.fetch_data([benchmark])

        # 6) build trading dates
        if self.calendar is not None:
            # data span
            first_dt = min(df.index.min() for df in reg_data.values())
            last_dt  = max(df.index.max() for df in reg_data.values())

            # schedule uses dates only
            sched = self.calendar.schedule(
                start_date=first_dt.date(),
                end_date=last_dt.date()
            )
            closes = sched["market_close"]

            # drop tz if present
            if getattr(getattr(closes, "dt", None), "tz", None) is not None:
                closes = closes.dt.tz_convert(None)

            # restrict to actual data
            closes = closes[(closes >= first_dt) & (closes <= last_dt)]

            # require full history across tickers and benchmark if ticker
            if require_all_history:
                # collect earliest available dates
                idx_mins = [df.index.min() for df in reg_data.values()]
                if bm_type == BenchmarkTypes.TICKER and benchmark in bm_data:
                    idx_mins.append(bm_data[benchmark]["close"].index.min())
                if idx_mins:
                    common = max(idx_mins)
                    closes = closes[closes >= common]

            # apply start/end (full timestamp)
            if sd is not None:
                closes = closes[closes >= sd]
            if ed is not None:
                closes = closes[closes <= ed]

            all_ts = list(closes)

        # 5) build trading dates
        if self.calendar is not None:
            # data span
            first_dt = min(df.index.min() for df in reg_data.values())
            last_dt  = max(df.index.max() for df in reg_data.values())

            # schedule uses dates only
            sched = self.calendar.schedule(
                start_date=first_dt.date(),
                end_date=last_dt.date()
            )
            closes = sched["market_close"]

            # drop tz if present
            if getattr(getattr(closes, "dt", None), "tz", None) is not None:
                closes = closes.dt.tz_convert(None)

            # restrict to actual data
            closes = closes[(closes >= first_dt) & (closes <= last_dt)]

            # require history
            if require_all_history and reg_tkrs:
                common = max(df.index.min() for df in reg_data.values())
                closes = closes[closes >= common]

            # apply start/end (full timestamp)
            if sd is not None:
                closes = closes[closes >= sd]
            if ed is not None:
                closes = closes[closes <= ed]

            all_ts = list(closes)

            # **raise** on empty calendar range
            if not all_ts:
                raise ValueError("No trading dates after filtering")

        else:
            # legacy union of data indices
            if hasattr(self.market_data_loader, "get_all_dates"):
                all_ts = self.market_data_loader.get_all_dates(reg_tkrs)
            else:
                all_ts = sorted({ts for df in reg_data.values() for ts in df.index})

            # require full history across tickers and benchmark if ticker
            if require_all_history:
                idx_mins = [df.index.min() for df in reg_data.values()]
                if bm_type == BenchmarkTypes.TICKER and benchmark in bm_data:
                    idx_mins.append(bm_data[benchmark]["close"].index.min())
                if idx_mins:
                    common = max(idx_mins)
                    all_ts = [d for d in all_ts if d >= common]

            # apply start/end
            if sd is not None:
                all_ts = [d for d in all_ts if d >= sd]
            if ed is not None:
                all_ts = [d for d in all_ts if d <= ed]

            if not all_ts:
                raise ValueError("No trading dates after filtering")

        # 7) main loop: signals
        sig_rows = []
        self.logger.debug(
            "Processing %d backtest steps", len(all_ts)
        )
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts

        for ts in iterator:
            if hasattr(self.market_data_loader, "next"):
                bar = self.market_data_loader.next(reg_tkrs, ts)
            else:
                bar = self._bar_dict(ts, reg_data)

            if self.alternative_data_loader:
                alt_ld = self.alternative_data_loader
                if hasattr(alt_ld, "next"):
                    bar.update(alt_ld.next(alt_tkrs, ts))
                else:
                    for t, df in alt_ld.fetch_data(alt_tkrs).items():
                        bar[t] = self._bar_dict(ts, {t: df})[t]

            sig = strategy.step(ts, bar)
            row = {"date": ts}
            for t in strategy.tickers:
                row[t] = sig.get(t, 0.0)
            sig_rows.append(row)

        # 8) construct signals_df
        sig_df = pd.DataFrame(sig_rows).set_index("date").sort_index()
        sig_df.index.name = None
        sig_reg = ((sig_df.shift(1).ffill() if shift_signals else sig_df)
                   .fillna(0.0)[reg_tkrs])

        # 9) compute returns
        px     = pd.DataFrame({t: reg_data[t]["close"] for t in reg_tkrs})
        px     = px.reindex(sig_reg.index).ffill()
        ret_df = px.pct_change(fill_method=None).fillna(0.0)
        strat_ret = (ret_df * sig_reg).sum(axis=1)

        # 10) benchmark returns
        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            bm_ret = benchmark(ret_df)
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            bm_ret = STANDARD_BENCHMARKS[benchmark](ret_df)
        else:  # TICKER
            ser = bm_data[benchmark]["close"].reindex(sig_reg.index).ffill()
            bm_ret = ser.pct_change(fill_method=None).fillna(0.0)

        # 11) dynamic alternative data update
        if self.alternative_data_loader and hasattr(self.alternative_data_loader, "update"):
            for ts in sig_reg.index:
                raw_sigs = sig_df.loc[ts, strategy.tickers].to_dict()
                raw_rets = ret_df.loc[ts].to_dict()
                self.alternative_data_loader.update(ts, raw_sigs, raw_rets, float(strat_ret.loc[ts]))

        # log completion
        self.logger.info(
            "Backtest complete: processed %d timestamps", len(all_ts)
        )
        return {
            "signals_df":        sig_reg,
            "tickers_returns":   ret_df,
            "strategy_returns":  strat_ret,
            "benchmark_returns": bm_ret
        }

    @staticmethod
    def _bar_dict(ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict | None]:
        out: Dict[str, dict | None] = {}
        for t, df in data.items():
            if ts in df.index:
                row = df.loc[ts]
                out[t] = {
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                }
            else:
                out[t] = None
        return out
