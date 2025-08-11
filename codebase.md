# .gitignore

```
.ipynb_checkpoints
dist/
private/
poetry.lock
.idea/
codebase.md

test_data/

# Python bytecode files
*.pyc
__pycache__/
*.pyo
*.pyd
.Python

.DS_Store
```

# mkdocs.yml

```yml
site_name: Portwine Documentation
site_description: A clean, elegant portfolio backtester
site_author: Portwine Team
site_url: https://github.com/StuartFarmer/portwine

repo_name: portwine
repo_url: https://github.com/StuartFarmer/portwine
edit_uri: edit/main/docs/

theme:
  name: readthedocs
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.suggest
    - search.highlight
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/toggle-switch
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/toggle-switch-off-outline
        name: Switch to light mode

markdown_extensions:
  - admonition
  - codehilite
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:mermaid2.fence_mermaid
  - pymdownx.tabbed
  - pymdownx.details
  - pymdownx.emoji
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.smartsymbols
  - pymdownx.snippets:
      check_paths: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.saneheaders

plugins:
  - search
  - mkdocstrings
  - mermaid2:
      arguments:
        theme: default
        flowchart:
          useMaxWidth: false
          htmlLabels: true
          curve: basis
          rankSpacing: 30
          nodeSpacing: 20
          width: 800
          height: 600

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quick-start.md
  - User Guide:
    - Strategies: user-guide/strategies.md
    - Backtesting: user-guide/backtesting.md
    - Data Management: user-guide/data-management.md
    - Analysis: user-guide/analysis.md
  # - API Reference:
  #   - Strategies: api/strategies.md
  #   - Backtester: api/backtester.md
  #   - Analyzers: api/analyzers.md
  #   - Data Loaders: api/data-loaders.md
  # - Examples:
  #   - Basic Strategies: examples/basic-strategies.md
  #   - Advanced Strategies: examples/advanced-strategies.md
  - Contributing: contributing.md

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/StuartFarmer/portwine 
```

# portwine/__init__.py

```py
from portwine.strategies.base import StrategyBase
from portwine.backtester import Backtester, BenchmarkTypes, benchmark_equal_weight, benchmark_markowitz
from portwine.universe import Universe

__all__ = ['StrategyBase', 'Backtester', 'BenchmarkTypes', 'benchmark_equal_weight', 'benchmark_markowitz', 'Universe']
```

# portwine/backtester.py

```py
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
from portwine.data.providers.loader_adapters import MarketDataLoader

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
# Optimized Backtester
# ------------------------------------------------------------------------------
class Backtester:
    """
    An optimized step‑driven back‑tester that addresses performance bottlenecks
    identified through profiling.
    
    Key optimizations:
    1. Pre-computed data access patterns
    2. Vectorized operations where possible
    3. Reduced pandas indexing overhead
    4. Memory-efficient data structures
    """

    def __init__(
        self,
        market_data_loader: MarketDataLoader,
        alternative_data_loader=None,
        calendar: Optional[Union[str, mcal.ExchangeCalendar]] = None,
        logger: Optional[_logging.Logger] = None,
        log: bool = False,
    ):
        self.market_data_loader      = market_data_loader
        self.alternative_data_loader = alternative_data_loader
        if isinstance(calendar, str):
            self.calendar = mcal.get_calendar(calendar)
        else:
            self.calendar = calendar
        
        # Configure logging
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger.create(__name__, level=_logging.INFO)
            self.logger.disabled = not log
        
        # Performance optimization: cache for data access
        self._data_cache = {}
        self._price_cache = {}
        self._date_index_cache = {}

    def _split_tickers(self, tickers: set) -> Tuple[List[str], List[str]]:
        """Split tickers into regular and alternative data tickers."""
        reg, alt = [], []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def _precompute_data_access(self, reg_data: Dict[str, pd.DataFrame], all_ts: List[pd.Timestamp]) -> Dict:
        """
        Pre-compute data access patterns to avoid repeated pandas indexing.
        This is the key optimization that addresses the major bottleneck.
        """
        self.logger.debug("Pre-computing data access patterns...")
        
        # Create fast lookup structures
        data_cache = {}
        price_cache = {}
        date_index_cache = {}
        
        for ticker, df in reg_data.items():
            # Create fast date-to-index mapping
            date_to_idx = {date: idx for idx, date in enumerate(df.index)}
            date_index_cache[ticker] = date_to_idx
            
            # Pre-extract price data as numpy arrays for faster access
            price_cache[ticker] = {
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'volume': df['volume'].values,
                'dates': df.index.values
            }
            
            # Create fast lookup for each timestamp using numpy arrays (no pandas indexing)
            data_cache[ticker] = {}
            for ts in all_ts:
                if ts in date_to_idx:
                    idx = date_to_idx[ts]
                    # Use numpy arrays directly instead of pandas .iloc[]
                    data_cache[ticker][ts] = {
                        'open': float(price_cache[ticker]['open'][idx]),
                        'high': float(price_cache[ticker]['high'][idx]),
                        'low': float(price_cache[ticker]['low'][idx]),
                        'close': float(price_cache[ticker]['close'][idx]),
                        'volume': float(price_cache[ticker]['volume'][idx]),
                    }
                else:
                    data_cache[ticker][ts] = None
        
        return {
            'data_cache': data_cache,
            'price_cache': price_cache,
            'date_index_cache': date_index_cache
        }

    def _fast_bar_dict(self, ts: pd.Timestamp, data_cache: Dict) -> Dict[str, dict | None]:
        """
        Optimized version of _bar_dict that uses pre-computed data access.
        This eliminates the pandas indexing bottleneck.
        """
        out: Dict[str, dict | None] = {}
        for ticker, ticker_cache in data_cache.items():
            out[ticker] = ticker_cache.get(ts)
        return out

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
        # Adjust logging level based on verbosity
        self.logger.setLevel(_logging.DEBUG if verbose else _logging.INFO)
        
        self.logger.info(
            "Starting optimized backtest: tickers=%s, start_date=%s, end_date=%s",
            strategy.tickers, start_date, end_date,
        )
        
        # 1) normalize date filters
        sd = pd.Timestamp(start_date) if start_date is not None else None
        ed = pd.Timestamp(end_date)   if end_date   is not None else None
        if sd is not None and ed is not None and sd > ed:
            raise ValueError("start_date must be on or before end_date")

        # 2) split tickers
        all_tickers = strategy.universe.all_tickers
        reg_tkrs, alt_tkrs = self._split_tickers(all_tickers)
            
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

        # 5) preload benchmark ticker if needed
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

        # 7) OPTIMIZATION: Pre-compute data access patterns
        access_cache = self._precompute_data_access(reg_data, all_ts)
        data_cache = access_cache['data_cache']
        price_cache = access_cache['price_cache']

        # 8) main loop: signals (optimized)
        # OPTIMIZATION: Pre-allocate arrays instead of building lists
        n_dates = len(all_ts)
        tickers_list = list(strategy.tickers)
        n_tickers = len(tickers_list)
        signals_array = np.zeros((n_dates, n_tickers), dtype=np.float64)
        dates_array = np.empty(n_dates, dtype=object)
        
        # OPTIMIZATION: Cache universe lookups to avoid repeated calls
        universe_cache = {}
        
        self.logger.debug(
            "Processing %d backtest steps", len(all_ts)
        )
        iterator = tqdm(all_ts, desc="Backtest") if verbose else all_ts

        for i, ts in enumerate(iterator):
            # OPTIMIZATION: Cache universe lookups
            if ts not in universe_cache:
                universe_cache[ts] = strategy.universe.get_constituents(ts)
            current_universe_tickers = universe_cache[ts]
            
            # OPTIMIZATION: Use pre-computed data access
            if hasattr(self.market_data_loader, "next"):
                bar = self.market_data_loader.next(current_universe_tickers, ts)
            else:
                # Use optimized bar_dict
                filtered_data_cache = {t: data_cache[t] for t in current_universe_tickers if t in data_cache}
                bar = self._fast_bar_dict(ts, filtered_data_cache)


            if self.alternative_data_loader:
                alt_ld = self.alternative_data_loader
                if hasattr(alt_ld, "next"):
                    bar.update(alt_ld.next(alt_tkrs, ts))
                else:
                    alt_data = alt_ld.fetch_data(alt_tkrs)
                    for t, df in alt_data.items():
                        bar_result = self._bar_dict(ts, {t: df})[t]
                        bar[t] = bar_result

            sig = strategy.step(ts, bar)
            
            # Validate that strategy only assigns weights to tickers in the current universe
            # OPTIMIZATION: Reuse cached universe lookup
            invalid_tickers = [t for t in sig.keys() if t not in current_universe_tickers]
            if invalid_tickers:
                raise ValueError(
                    f"Strategy assigned weights to tickers not in current universe at {ts}: {invalid_tickers}. "
                    f"Current universe: {current_universe_tickers}"
                )
            
            # OPTIMIZATION: Direct array assignment instead of building dictionaries
            dates_array[i] = ts
            for j, ticker in enumerate(tickers_list):
                signals_array[i, j] = sig.get(ticker, 0.0)

        # 9) OPTIMIZATION: Pure numpy signal processing (replaces pandas operations)
        # Apply signal shifting using numpy operations
        if shift_signals:
            shifted_signals = np.zeros_like(signals_array)
            if signals_array.shape[0] > 1:
                shifted_signals[1:] = signals_array[:-1]  # Shift down by 1
            signals_processed = shifted_signals
        else:
            signals_processed = signals_array
        
        # Forward fill using numpy (replace .ffill()) - only for NaN values, not zeros
        # Note: 0.0 is a valid signal value meaning "no allocation", so we don't forward fill zeros
        # In our numpy array, we initialized with zeros, so no NaN handling needed here
        # The original pandas .ffill() would only forward fill NaN values
        # Since we pre-allocated with zeros and strategy.step() returns valid values,
        # no forward fill is needed for this step-based backtester
        
        # Select only reg_tkrs columns (replace pandas column selection)
        tickers_list = list(strategy.tickers)
        reg_ticker_indices = [i for i, ticker in enumerate(tickers_list) if ticker in reg_tkrs]
        sig_reg_array = signals_processed[:, reg_ticker_indices]
        
        # Create minimal DataFrame only for API compatibility at the end
        sig_reg = pd.DataFrame(
            sig_reg_array,
            index=pd.DatetimeIndex(dates_array),
            columns=[tickers_list[i] for i in reg_ticker_indices]
        )

        # 10) OPTIMIZATION: Pure numpy price processing and returns calculation
        self.logger.debug("Computing returns using pure numpy operations...")
        
        # Build price matrix using numpy operations
        n_dates = len(dates_array)
        n_reg_tickers = len(reg_tkrs)
        price_matrix = np.full((n_dates, n_reg_tickers), np.nan, dtype=np.float64)
        
        # Create date lookup for alignment
        date_to_idx = {date: i for i, date in enumerate(dates_array)}
        
        # Fill price matrix for each ticker
        for ticker_idx, ticker in enumerate(reg_tkrs):
            if ticker in price_cache:
                ticker_dates = price_cache[ticker]['dates']
                ticker_prices = price_cache[ticker]['close']
                
                # Align prices with our date array
                for date_idx, date in enumerate(ticker_dates):
                    if date in date_to_idx:
                        price_matrix[date_to_idx[date], ticker_idx] = ticker_prices[date_idx]
        
        # Forward fill missing prices using numpy
        for col in range(n_reg_tickers):
            mask = np.isnan(price_matrix[:, col])
            valid_indices = np.where(~mask)[0]
            if len(valid_indices) > 0:
                # Forward fill from first valid value
                for i in range(valid_indices[0] + 1, n_dates):
                    if mask[i]:
                        # Find last valid value
                        last_valid_idx = i - 1
                        while last_valid_idx >= 0 and mask[last_valid_idx]:
                            last_valid_idx -= 1
                        if last_valid_idx >= 0:
                            price_matrix[i, col] = price_matrix[last_valid_idx, col]
        
        # Calculate returns using numpy (replaces pct_change)
        returns_matrix = np.zeros_like(price_matrix)
        returns_matrix[1:] = (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1]
        # Replace NaN/inf with 0.0
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate strategy returns using numpy (replaces pandas matrix multiplication)
        strat_ret_array = np.sum(returns_matrix * sig_reg_array, axis=1)
        
        # Create minimal pandas objects for API compatibility
        ret_df = pd.DataFrame(
            returns_matrix,
            index=pd.DatetimeIndex(dates_array),
            columns=reg_tkrs
        )
        strat_ret = pd.Series(
            strat_ret_array,
            index=pd.DatetimeIndex(dates_array)
        )

        # 11) benchmark returns
        if bm_type == BenchmarkTypes.CUSTOM_METHOD:
            bm_ret = benchmark(ret_df)
        elif bm_type == BenchmarkTypes.STANDARD_BENCHMARK:
            bm_ret = STANDARD_BENCHMARKS[benchmark](ret_df)
        else:  # TICKER
            # OPTIMIZATION: Replace pandas operations with numpy for benchmark calculation
            bm_df = bm_data[benchmark]
            bm_prices = bm_df["close"].values
            bm_dates = bm_df.index.values
            
            # Align benchmark data with strategy dates using numpy
            strategy_dates = pd.DatetimeIndex(dates_array).values
            aligned_prices = np.full(len(strategy_dates), np.nan)
            
            # Fast date alignment using searchsorted
            for i, target_date in enumerate(strategy_dates):
                pos = np.searchsorted(bm_dates, target_date, side="right") - 1
                if pos >= 0:
                    aligned_prices[i] = bm_prices[pos]
            
            # Forward fill using numpy (replace .ffill())
            last_valid_price = np.nan
            for i in range(len(aligned_prices)):
                if not np.isnan(aligned_prices[i]):
                    last_valid_price = aligned_prices[i]
                elif not np.isnan(last_valid_price):
                    aligned_prices[i] = last_valid_price
            
            # Calculate returns using numpy (replace .pct_change().fillna())
            bm_ret_array = np.zeros(len(aligned_prices))
            bm_ret_array[1:] = (aligned_prices[1:] - aligned_prices[:-1]) / aligned_prices[:-1]
            bm_ret_array = np.nan_to_num(bm_ret_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create minimal pandas series for API compatibility
            bm_ret = pd.Series(bm_ret_array, index=pd.DatetimeIndex(dates_array))

        # 12) dynamic alternative data update
        if self.alternative_data_loader and hasattr(self.alternative_data_loader, "update"):
            for i, ts in enumerate(sig_reg.index):
                # Create raw_sigs dict directly from numpy array
                raw_sigs = {ticker: float(signals_processed[i, j]) 
                           for j, ticker in enumerate(tickers_list)}
                
                # Create raw_rets dict directly from numpy array (ret_df still pandas for now)
                raw_rets = ret_df.loc[ts].to_dict()
                
                # Get strategy return as float (strat_ret still pandas for now)
                self.alternative_data_loader.update(ts, raw_sigs, raw_rets, float(strat_ret.loc[ts]))

        # log completion
        self.logger.info(
            "Optimized backtest complete: processed %d timestamps", len(all_ts)
        )
        return {
            "signals_df":        sig_reg,
            "tickers_returns":   ret_df,
            "strategy_returns":  strat_ret,
            "benchmark_returns": bm_ret
        }

    @staticmethod
    def _bar_dict(ts: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, dict | None]:
        """
        OPTIMIZED: _bar_dict method with 'at or before' logic for compatibility.
        Uses the same logic as the loader optimization but for direct DataFrame access.
        """
        out: Dict[str, dict | None] = {}
        for t, df in data.items():
            if df.empty:
                out[t] = None
                continue
                
            # Use 'at or before' logic like the optimized loader
            idx = df.index
            pos = idx.searchsorted(ts, side="right") - 1
            if pos < 0:
                out[t] = None
            else:
                row = df.iloc[pos]
                out[t] = {
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                }
        return out 

```

# portwine/brokers/__init__.py

```py

```

# portwine/brokers/alpaca.py

```py
import requests
import time
from datetime import datetime
from typing import Dict, List

from portwine.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderExecutionError,
    OrderNotFoundError,
    OrderCancelError,
)


def _parse_datetime(dt_str: str) -> datetime:
    """
    Parse an ISO‑8601 timestamp from Alpaca, handling the trailing 'Z' and trimming fractional seconds to microseconds.
    """
    if dt_str is None:
        return None
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    # Trim fractional seconds to microsecond precision
    t_idx = dt_str.find("T")
    plus_idx = dt_str.rfind("+")
    minus_idx = dt_str.rfind("-")
    idx = None
    if plus_idx > t_idx:
        idx = plus_idx
    elif minus_idx > t_idx:
        idx = minus_idx
    if idx is not None:
        dt_main = dt_str[:idx]
        tz = dt_str[idx:]
    else:
        dt_main = dt_str
        tz = ""
    if "." in dt_main:
        date_part, frac = dt_main.split(".", 1)
        frac = frac[:6]
        dt_main = f"{date_part}.{frac}"
    dt_str = dt_main + tz
    return datetime.fromisoformat(dt_str)


class AlpacaBroker(BrokerBase):
    """
    Alpaca REST API implementation of BrokerBase using the `requests` library.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Args:
            api_key: Your Alpaca API key ID.
            api_secret: Your Alpaca secret key.
            base_url: Alpaca REST endpoint (paper or live).
        """
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json",
        })

    def get_account(self) -> Account:
        url = f"{self._base_url}/v2/account"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Account fetch failed: {resp.text}")
        data = resp.json()
        return Account(equity=float(data["equity"]), last_updated_at=int(time.time() * 1000))

    def get_positions(self) -> Dict[str, Position]:
        url = f"{self._base_url}/v2/positions"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Positions fetch failed: {resp.text}")
        positions = {}
        for p in resp.json():
            positions[p["symbol"]] = Position(
                ticker=p["symbol"],
                quantity=float(p["qty"]),
                last_updated_at=int(time.time() * 1000)
            )
        return positions

    def get_position(self, ticker: str) -> Position:
        url = f"{self._base_url}/v2/positions/{ticker}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            return Position(ticker=ticker, quantity=0.0, last_updated_at=int(time.time() * 1000))
        if not resp.ok:
            raise OrderExecutionError(f"Position fetch failed: {resp.text}")
        p = resp.json()
        return Position(ticker=p["symbol"], quantity=float(p["qty"]), last_updated_at=int(time.time() * 1000))

    def get_order(self, order_id: str) -> Order:
        url = f"{self._base_url}/v2/orders/{order_id}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            raise OrderNotFoundError(f"Order {order_id} not found")
        if not resp.ok:
            raise OrderExecutionError(f"Order fetch failed: {resp.text}")
        o = resp.json()
        return Order(
            order_id=o["id"],
            ticker=o["symbol"],
            side=o["side"],
            quantity=float(o["qty"]),
            order_type=o["type"],
            status=o["status"],
            time_in_force=o["time_in_force"],
            average_price=float(o["filled_avg_price"] or 0.0),
            remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
            created_at=_parse_datetime(o["created_at"]),
            last_updated_at=_parse_datetime(o["updated_at"]),
        )

    def get_orders(self) -> List[Order]:
        url = f"{self._base_url}/v2/orders"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Orders fetch failed: {resp.text}")
        orders = []
        for o in resp.json():
            orders.append(Order(
                order_id=o["id"],
                ticker=o["symbol"],
                side=o["side"],
                quantity=float(o["qty"]),
                order_type=o["type"],
                status=o["status"],
                time_in_force=o["time_in_force"],
                average_price=float(o["filled_avg_price"] or 0.0),
                remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
                created_at=_parse_datetime(o["created_at"]),
                last_updated_at=_parse_datetime(o["updated_at"]),
            ))
        return orders

    def submit_order(self, symbol: str, quantity: float) -> Order:
        """
        Execute a market order on Alpaca.
        Positive quantity → buy, negative → sell.
        """
        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        url = f"{self._base_url}/v2/orders"
        resp = self._session.post(url, json=payload)
        if not resp.ok:
            raise OrderExecutionError(f"Order submission failed: {resp.text}")
        o = resp.json()
        return Order(
            order_id=o["id"],
            ticker=o["symbol"],
            side=o["side"],
            quantity=float(o["qty"]),
            order_type=o["type"],
            status=o["status"],
            time_in_force=o["time_in_force"],
            average_price=float(o["filled_avg_price"] or 0.0),
            remaining_quantity=float(o["qty"]) - float(o["filled_qty"]),
            created_at=_parse_datetime(o["created_at"]),
            last_updated_at=_parse_datetime(o["updated_at"]),
        )

    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the time of the request using the Alpaca clock endpoint.
        """
        url = f"{self._base_url}/v2/clock"
        resp = self._session.get(url)
        if not resp.ok:
            raise OrderExecutionError(f"Clock fetch failed: {resp.text}")
        data = resp.json()
        return bool(data.get("is_open", False))

```

# portwine/brokers/base.py

```py
"""
Broker base class for trading interfaces.

This module provides the base class for all broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class Position:
    """Represents a trading position."""
    ticker: str
    quantity: float
    last_updated_at: int # UNIX timestamp in second for last time the data was updated

class TimeInForce(Enum):
    DAY = "day"
    GOOD_TILL_CANCELLED = "gtc"
    IMMEDIATE_OR_CANCEL = "ioc"
    FILL_OR_KILL = "fok"
    MARKET_ON_OPEN = "opg"
    MARKET_ON_CLOSE = "cls"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    
class OrderStatus(Enum):
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    PENDING = "pending"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

'''
    Order dataclass. All brokers must adhere to this standard. 
'''
@dataclass
class Order:
    order_id: str               # unique identifier for the order
    ticker: str                 # asset ticker
    side: str                   # buy or sell
    quantity: float             # amount / size of order
    order_type: str             # market, limit, etc
    status: str                 # submitted, rejected, filled, etc
    time_in_force: str          # gtc, fok, etc
    average_price: float        # average fill price of order
    remaining_quantity: float   # how much of the order still needs to be filled
    created_at: int        # when the order was created (UNIX timestamp milliseconds)
    last_updated_at: int   # when the data on this order was last updated with the broker

@dataclass
class Account:
    equity: float         # amount of money available to purchase securities (can include margin)
    last_updated_at: int
    # net_liquidation_value     liquid value, total_equity... all are names for the same thing, equity...
    # buying power includes margin, equity does not


class OrderExecutionError(Exception):
    """Raised when an order fails to execute."""
    pass


class OrderNotFoundError(Exception):
    """Raised when an order cannot be found."""
    pass


class OrderCancelError(Exception):
    """Raised when an order fails to cancel."""
    pass


class BrokerBase(ABC):
    """Base class for all broker implementations."""

    @abstractmethod
    def get_account(self) -> Account:
        """
        Get the current account information.
        
        Returns:
            Account object containing current account state
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbol to Position objects
        """
        pass

    @abstractmethod
    def get_position(self, ticker) -> Position:
        # Returns a position object for a given ticker
        # if there is no position, then an empty position is returned
        # with quantity 0
        pass

    @abstractmethod
    def get_order(self, order_id) -> Order:
        pass

    @abstractmethod
    def get_orders(self) -> List[Order]:
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        quantity: float,
    ) -> Order:
        """
        Execute a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            quantity: Order quantity
        
        Returns:
            Order object representing the executed order
        
        Raises:
            ValueError: If the order parameters are invalid
            OrderExecutionError: If the order fails to execute
        """
        pass

    @abstractmethod
    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the given datetime.
        """
        pass

```

# portwine/brokers/mock.py

```py
import threading
import time
from datetime import datetime
from typing import Dict, List

from portwine.brokers.base import (
    BrokerBase,
    Account,
    Position,
    Order,
    OrderNotFoundError,
)


class MockBroker(BrokerBase):
    """
    A simple in‑memory mock broker for testing.
    Fills all market orders immediately at a fixed price and
    tracks positions and orders in dictionaries.
    """

    def __init__(self, initial_equity: float = 1_000_000.0, fill_price: float = 100.0):
        self._equity = initial_equity
        self._fill_price = fill_price
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._lock = threading.Lock()

    def get_account(self) -> Account:
        """
        Return a snapshot of the account with a unix‐timestamp last_updated_at (ms).
        """
        ts = int(time.time() * 1_000)
        return Account(
            equity=self._equity,
            last_updated_at=ts
        )

    def get_positions(self) -> Dict[str, Position]:
        """
        Return all current positions, preserving each position's last_updated_at.
        """
        return {
            symbol: Position(
                ticker=pos.ticker,
                quantity=pos.quantity,
                last_updated_at=pos.last_updated_at
            )
            for symbol, pos in self._positions.items()
        }

    def get_position(self, ticker: str) -> Position:
        """
        Return position for a single ticker (zero if not held),
        with last_updated_at from the stored position or now if none.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            ts = int(time.time() * 1_000)
            return Position(ticker=ticker, quantity=0.0, last_updated_at=ts)
        return Position(
            ticker=pos.ticker,
            quantity=pos.quantity,
            last_updated_at=pos.last_updated_at
        )

    def get_order(self, order_id: str) -> Order:
        """
        Retrieve a single order by ID; raise if not found.
        """
        try:
            return self._orders[order_id]
        except KeyError:
            raise OrderNotFoundError(f"Order {order_id} not found")

    def get_orders(self) -> List[Order]:
        """
        Return a list of all orders submitted so far.
        """
        return list(self._orders.values())

    def submit_order(self, symbol: str, quantity: float) -> Order:
        """
        Simulate a market order fill:
          - Immediately 'fills' at self._fill_price
          - Updates in‑memory positions with a unix‐timestamp last_updated_at (ms)
          - Records the order with status 'filled' and last_updated_at as unix timestamp (ms)
        """
        with self._lock:
            self._order_counter += 1
            oid = str(self._order_counter)

        side = "buy" if quantity > 0 else "sell"
        qty = abs(quantity)
        now_ts = int(time.time() * 1_000)

        # Update position
        prev = self._positions.get(symbol)
        prev_qty = prev.quantity if prev is not None else 0.0
        new_qty = prev_qty + quantity

        if new_qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                ticker=symbol,
                quantity=new_qty,
                last_updated_at=now_ts
            )

        order = Order(
            order_id=oid,
            ticker=symbol,
            side=side,
            quantity=qty,
            order_type="market",
            status="filled",
            time_in_force="day",
            average_price=self._fill_price,
            remaining_quantity=0.0,
            created_at=now_ts,
            last_updated_at=now_ts
        )

        self._orders[oid] = order
        return order

    def market_is_open(self, timestamp: datetime) -> bool:
        """
        Stub implementation of market hours:
        - Returns True if it's a weekday (Monday=0 … Friday=4)
          between 09:30 and 16:00 in the local system time.
        """
        # Weekday check
        if timestamp.weekday() >= 5:
            return False

        # Hour/minute check: between 9:30 and 16:00
        hm = timestamp.hour * 60 + timestamp.minute
        open_time = 9 * 60 + 30   # 9:30 = 570 minutes
        close_time = 16 * 60      # 16:00 = 960 minutes
        return open_time <= hm < close_time

```

# portwine/data/interface.py

```py
from datetime import datetime

class DataInterface:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.current_timestamp = None

    def set_current_timestamp(self, timestamp: datetime):
        self.current_timestamp = timestamp
    
    def __getitem__(self, ticker: str):
        """
        Access data for a ticker using bracket notation: interface['AAPL']
        
        Returns the latest OHLCV data for the ticker at the current timestamp.
        This enables lazy loading and caching without passing large dictionaries to strategies.
        
        Args:
            ticker: The ticker symbol to retrieve data for
            
        Returns:
            dict: OHLCV data dictionary with keys ['open', 'high', 'low', 'close', 'volume']
            
        Raises:
            ValueError: If current_timestamp is not set
            KeyError: If the ticker is not found or has no data
        """
        if self.current_timestamp is None:
            raise ValueError("Current timestamp not set. Call set_current_timestamp() first.")
        
        # Get data for this ticker at the current timestamp
        data = self.data_loader.next([ticker], self.current_timestamp)
        
        if ticker not in data or data[ticker] is None:
            raise KeyError(f"No data found for ticker: {ticker}")
        
        return data[ticker]
```

# portwine/data/provider.py

```py
import abc
from datetime import datetime
from typing import Union
import httpx
import asyncio

class DataProvider(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    def get_data(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        # gets for a given identifier, start_date, and end_date
        # data can be ANY format, OHLCV, fundamental data, etc.
        # this is just the interface for the data provider
        ...

    async def get_data_async(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        # gets for a given identifier, start_date, and end_date
        # data can be ANY format, OHLCV, fundamental data, etc.
        # this is just the interface for the data provider
        ...

'''
EODHD Historical Data Provider

This provider is used to get historical data from EODHD.

https://eodhd.com/

import requests

url = f'https://eodhd.com/api/eod/MCD.US?api_token=67740bda7e4247.39007920&fmt=json'
data = requests.get(url).json()

print(data)
'''
class EODHDHistoricalDataProvider(DataProvider):
    def __init__(self, api_key: str, exchange_code: str):
        self.api_key = api_key
        self.exchange_code = exchange_code

    def _get_url(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        url = f'https://eodhd.com/api/eod/{identifier}.{self.exchange_code}?api_token={self.api_key}&fmt=json'
        if end_date is not None:
            url += f'&to={end_date.strftime("%Y-%m-%d")}'
        if start_date is not None:
            url += f'&from={start_date.strftime("%Y-%m-%d")}'
        return url

    def get_data(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        url = self._get_url(identifier, start_date, end_date)
        data = httpx.get(url).json()
        
        return data
    
    async def get_data_async(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        url = self._get_url(identifier, start_date, end_date)
        
        async with httpx.AsyncClient() as client:
            data = await client.get(url)
            data = data.json()

        return data
```

# portwine/data/store.py

```py
'''
DataStore is a class that stores data fetched from a data provider.

It can store it in flat files, in a database, or in memory; whatever the developer wants.
'''

from datetime import datetime
from typing import Union, OrderedDict
from collections import OrderedDict as OrderedDictType
import abc
import os
import pandas as pd
from pathlib import Path

'''

Data format:

identifier: {
    datetime_str: {
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    },
    datetime_str: {
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    },
    ...
}

Could also be fundamental data, etc. like:

identifier: {
    datetime_str: {
        gdp: float,
        inflation: float,
        unemployment: float,
        interest_rate: float,
        etc.
    },
    datetime_str: {
        gdp: float,
        inflation: float,
        unemployment: float,
        interest_rate: float,
        etc.
    },
    ...
}

'''

class DataStore(abc.ABC):
    def __init__(self, *args, **kwargs):
        ...

    '''
    Adds data to the store. Assumes that data is immutable, and that if the data already exists for the given times, 
    it is skipped.
    '''
    def add(self, identifier: str, data: dict):
        ...

    '''
    Gets data from the store in chronological order (earliest to latest).
    If the data is not found, it returns None.
    '''
    def get(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        ...

    '''
    Gets the latest data point for the identifier.
    If the data is not found, it returns None.
    '''
    def get_latest(self, identifier: str) -> Union[dict, None]:
        ...

    '''
    Gets the latest date for a given identifier from the store.
    If the data is not found, it returns None.
    '''
    def latest(self, identifier: str) -> Union[datetime, None]:
        ...

    '''
    Checks if data exists for a given identifier, start_date, and end_date.

    If start_date is None, it assumes the earliest date for that piece of data.
    If end_date is None, it assumes the latest date for that piece of data.

    If start_date is not None and end_date is not None, it checks if the data exists
    for the given start_date until the end of the data.
    '''
    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        ...
    
    '''
    Gets all identifiers from the store.
    '''
    def identifiers(self):
        ...


class ParquetDataStore(DataStore):
    """
    A DataStore implementation that stores data in parquet files.
    
    File structure:
    data_dir/
    ├── <identifier_1>.pqt
    ├── <identifier_2>.pqt
    ├── <identifier_3>.pqt
    └── <identifier_4>.pqt
    
    Each parquet file contains a DataFrame with:
    - Index: datetime (chronologically ordered)
    - Columns: data fields (open, high, low, close, volume, etc.)
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the ParquetDataStore.
        
        Args:
            data_dir: Directory where parquet files are stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, identifier: str) -> Path:
        """Get the parquet file path for an identifier."""
        return self.data_dir / f"{identifier}.pqt"
    
    def _load_dataframe(self, identifier: str) -> pd.DataFrame:
        """Load DataFrame from parquet file, return empty DataFrame if file doesn't exist."""
        file_path = self._get_file_path(identifier)
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                return df.sort_index()  # Ensure chronological order
            except Exception as e:
                print(f"Error loading parquet file for {identifier}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save_dataframe(self, identifier: str, df: pd.DataFrame):
        """Save DataFrame to parquet file."""
        file_path = self._get_file_path(identifier)
        try:
            # Ensure chronological order before saving
            df_sorted = df.sort_index()
            df_sorted.to_parquet(file_path, index=True)
        except Exception as e:
            print(f"Error saving parquet file for {identifier}: {e}")
    
    def add(self, identifier: str, data: dict, overwrite: bool = False):
        """
        Adds data to the store.
        
        Args:
            identifier: The identifier for the data
            data: Dictionary with datetime keys and data dictionaries as values
            overwrite: If True, overwrite existing data for the same dates
        """
        if not data:
            return
        
        # Load existing data
        df_existing = self._load_dataframe(identifier)
        
        # Convert new data to DataFrame
        new_data = []
        for dt, values in data.items():
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            row_data = {'date': dt, **values}
            new_data.append(row_data)
        
        if not new_data:
            return
        
        df_new = pd.DataFrame(new_data)
        df_new.set_index('date', inplace=True)
        
        if df_existing.empty:
            # No existing data, just save new data
            self._save_dataframe(identifier, df_new)
        else:
            if overwrite:
                # Remove existing rows for the same dates, then concatenate
                df_existing = df_existing.drop(df_new.index, errors='ignore')
                df_combined = pd.concat([df_existing, df_new])
            else:
                # For non-overwrite mode, filter out dates that already exist
                existing_dates = df_existing.index
                df_new_filtered = df_new[~df_new.index.isin(existing_dates)]
                
                if df_new_filtered.empty:
                    # No new data to add
                    return
                
                # Concatenate existing data with only new dates
                df_combined = pd.concat([df_existing, df_new_filtered])
            
            self._save_dataframe(identifier, df_combined)
    
    def get(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        """
        Gets data from the store in chronological order (earliest to latest).
        If the data is not found, it returns None.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        
        # Filter by date range
        if end_date is None:
            end_date = df.index.max()
        
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df[mask]
        
        if df_filtered.empty:
            return None
        
        # Convert to OrderedDict with datetime keys
        result = OrderedDict()
        for dt, row in df_filtered.iterrows():
            result[dt] = row.to_dict()
        
        return result
    
    def get_latest(self, identifier: str) -> Union[dict, None]:
        """
        Gets the latest data point for the identifier.
        If the data is not found, it returns None.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        
        latest_row = df.iloc[-1]
        return latest_row.to_dict()
    
    def latest(self, identifier: str) -> Union[datetime, None]:
        """
        Gets the latest date for a given identifier from the store.
        If the data is not found, it returns None.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return None
        
        return df.index.max()
    
    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        """
        Checks if data exists for a given identifier, start_date, and end_date.
        """
        df = self._load_dataframe(identifier)
        if df.empty:
            return False
        
        if start_date is None:
            start_date = df.index.min()
        if end_date is None:
            end_date = df.index.max()
        
        # Check if any data exists in the specified range
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df[mask].shape[0] > 0
    
    def identifiers(self):
        """
        Gets all identifiers from the store.
        """
        identifiers = []
        for file_path in self.data_dir.glob("*.pqt"):
            # Extract identifier from filename (remove .pqt extension)
            identifier = file_path.stem
            identifiers.append(identifier)
        return identifiers


```

# portwine/execution.py

```py
"""
Execution module for the portwine framework.

This module provides the base classes and interfaces for execution modules,
which connect strategy implementations from the backtester to live trading.
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional, Tuple, Iterator
import math
import time
from datetime import datetime
import pandas as pd
import pandas_market_calendars as mcal

from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.strategies.base import StrategyBase
from portwine.brokers.base import BrokerBase, Order
from portwine.logging import Logger, log_position_table, log_weight_table, log_order_table
from rich.progress import track, Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
from portwine.scheduler import daily_schedule

class ExecutionError(Exception):
    """Base exception for execution-related errors."""
    pass


class OrderExecutionError(ExecutionError):
    """Exception raised when order execution fails."""
    pass


class DataFetchError(ExecutionError):
    """Exception raised when data fetching fails."""
    pass


class PortfolioExceededError(ExecutionError):
    """Raised when current portfolio weights exceed 100% of portfolio value."""
    pass


class ExecutionBase(abc.ABC):
    """
    Base class for execution implementations.
    
    An execution implementation is responsible for:
    1. Fetching latest market data
    2. Passing data to strategy to get updated weights
    3. Calculating position changes needed
    4. Executing necessary trades using a broker
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_data_loader: MarketDataLoader,
        broker: BrokerBase,
        alternative_data_loader: Optional[MarketDataLoader] = None,
        min_change_pct: float = 0.01,
        min_order_value: float = 1.0,
        fractional: bool = False,
        timezone: Optional[datetime.tzinfo] = None,
    ):
        """
        Initialize the execution instance.
        
        Parameters
        ----------
        strategy : StrategyBase
            The strategy implementation to use for generating trading signals
        market_data_loader : MarketDataLoader
            Market data loader for price data
        broker : BrokerBase
            Broker implementation for executing trades
        alternative_data_loader : Optional[MarketDataLoader]
            Additional data loader for alternative data
        min_change_pct : float, default 0.01
            Minimum change percentage required to trigger a trade
        min_order_value : float, default 1.0
            Minimum dollar value required for an order
        timezone : Optional[datetime.tzinfo], default None
            Timezone for timestamp conversion
        """
        self.strategy = strategy
        self.market_data_loader = market_data_loader
        self.broker = broker
        self.alternative_data_loader = alternative_data_loader
        self.min_change_pct = min_change_pct
        self.min_order_value = min_order_value
        self.fractional = fractional
        # Store timezone (tzinfo); default to system local timezone
        self.timezone = timezone if timezone is not None else datetime.now().astimezone().tzinfo
        # Initialize ticker list from strategy
        self.tickers = strategy.tickers
        # Set up a per-instance rich-enabled logger
        self.logger = Logger.create(self.__class__.__name__, level=logging.INFO)
        self.logger.info(f"Initialized {self.strategy.__class__.__name__} with {len(self.tickers)} tickers")
    
    @staticmethod
    def _split_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split full ticker list into regular and alternative tickers.
        Regular tickers have no ':'; alternative contain ':'
        """
        reg: List[str] = []
        alt: List[str] = []
        for t in tickers:
            if isinstance(t, str) and ":" in t:
                alt.append(t)
            else:
                reg.append(t)
        return reg, alt

    def fetch_latest_data(self, timestamp: Optional[float] = None) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Fetch latest data for all tickers at the given timestamp.
        
        Parameters
        ----------
        timestamp : float, optional
            Unix timestamp in seconds. If None, uses current time.
            
        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            Dictionary mapping tickers to their latest bar data or None
        """
        try:
            # Convert UNIX timestamp to timezone-aware pandas Timestamp
            if timestamp is None:
                dt = pd.Timestamp.now(tz=self.timezone)
            else:
                # timestamp is seconds since epoch
                dt = pd.Timestamp(timestamp, unit='s', tz=self.timezone)
            
            # Split tickers into market vs alternative
            reg_tkrs, alt_tkrs = self._split_tickers(self.tickers)
            # Fetch market data only for regular tickers
            data = self.market_data_loader.next(reg_tkrs, dt)
            # Fetch alternative data only for alternative tickers
            if self.alternative_data_loader is not None and alt_tkrs:
                alt_data = self.alternative_data_loader.next(alt_tkrs, dt)
                # Merge alternative entries into result
                data.update(alt_data)

            self.logger.debug(f"Fetched data keys: {list(data.keys())}")

            return data
        except Exception as e:
            self.logger.exception(f"Error fetching latest data: {e}")
            raise DataFetchError(f"Failed to fetch latest data: {e}")
        
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current closing prices for the specified symbols by querying only market data.

        This method bypasses alternative data and directly uses market_data_loader.next
        with a timezone-aware datetime matching the execution timezone.
        """
        # Build current datetime in execution timezone
        dt = datetime.now(tz=self.timezone)
        # Fetch only market data for given symbols
        data = self.market_data_loader.next(symbols, dt)
        prices: Dict[str, float] = {}
        for symbol, bar in data.items():
            if bar is None:
                continue
            price = bar.get('close')
            if price is not None:
                prices[symbol] = price
        return prices
    
    def _get_current_positions(self) -> Tuple[Dict[str, float], float]:
        """
        Get current positions from broker account info.
        
        Returns
        -------
        Tuple[Dict[str, float], float]
            Current position quantities for each ticker and the portfolio value
        """
        positions = self.broker.get_positions()
        account = self.broker.get_account()
        
        current_positions = {symbol: position.quantity for symbol, position in positions.items()}
        portfolio_value = account.equity
        
        self.logger.debug(f"Current positions: {current_positions}, portfolio value: {portfolio_value:.2f}")
        return current_positions, portfolio_value

    def _calculate_target_positions(
        self,
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        fractional: bool = False,
    ) -> Dict[str, float]:
        """
        Convert target weights to absolute position sizes.
        
        Optionally prevent fractional shares by rounding down when `fractional=False`.
        
        Parameters
        ----------
        target_weights : Dict[str, float]
            Target allocation weights for each ticker
        portfolio_value : float
            Current portfolio value
        prices : Dict[str, float]
            Current prices for each ticker
        fractional : bool, default True
            If False, positions are floored to the nearest integer
        
        Returns
        -------
        Dict[str, float]
            Target position quantities for each ticker
        """
        target_positions = {}
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_value = weight * portfolio_value
            raw_qty = target_value / price
            if fractional:
                qty = raw_qty
            else:
                qty = math.floor(raw_qty)
            target_positions[symbol] = qty
            
        return target_positions

    def _calculate_current_weights(
        self,
        positions: List[Tuple[str, float]],
        portfolio_value: float,
        prices: Dict[str, float],
        raises: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate current weights of positions based on prices and portfolio value.

        Args:
            positions: List of (ticker, quantity) tuples.
            portfolio_value: Total portfolio value.
            prices: Mapping of ticker to current price.
            raises: If True, raise PortfolioExceededError when total weights > 1.

        Returns:
            Dict[ticker, weight] mapping.

        Raises:
            PortfolioExceededError: If raises=True and sum(weights) > 1.
        """
        # Map positions
        pos_map: Dict[str, float] = {t: q for t, q in positions}
        weights: Dict[str, float] = {}
        total: float = 0.0
        for ticker, price in prices.items():
            qty = pos_map.get(ticker, 0.0)
            w = (price * qty) / portfolio_value if portfolio_value else 0.0
            weights[ticker] = w
            total += w
        if raises and total > 1.0:
            raise PortfolioExceededError(
                f"Total weights {total:.2f} exceed 1.0"
            )
        return weights

    def _target_positions_to_orders(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
    ) -> List[Order]:
        """
        Determine necessary orders given target and current positions, returning Order objects.

        Args:
            target_positions: Mapping ticker -> desired quantity
            current_positions: Mapping ticker -> existing quantity

        Returns:
            List of Order dataclasses for each non-zero change.
        """
        orders: List[Order] = []
        for ticker, target_qty in target_positions.items():
            current_qty = current_positions.get(ticker, 0.0)
            diff = target_qty - current_qty
            if diff == 0:
                continue
            side = 'buy' if diff > 0 else 'sell'
            qty = abs(int(diff))
            # Build an Order dataclass with default/trivial values for non-relevant fields
            order = Order(
                order_id="",
                ticker=ticker,
                side=side,
                quantity=float(qty),
                order_type="market",
                status="new",
                time_in_force="day",
                average_price=0.0,
                remaining_quantity=0.0,
                created_at=0,
                last_updated_at=0,
            )
            orders.append(order)

        log_order_table(self.logger, orders)
        return orders

    def _execute_orders(self, orders: List[Order]) -> List[Order]:
        """
        Execute a list of Order objects through the broker.

        Parameters
        ----------
        orders : List[Order]
            List of Order dataclasses to submit.

        Returns
        -------
        List[Order]
            List of updated Order objects returned by the broker.
        """
        executed_orders: List[Order] = []
        for order in orders:
            # Determine signed quantity: negative for sell, positive for buy
            qty_arg = order.quantity if order.side == 'buy' else -order.quantity
            # Submit each order and collect the updated result
            updated = self.broker.submit_order(
                symbol=order.ticker,
                quantity=qty_arg,
            )
            # Restore expected positive quantity and original side
            updated.quantity = order.quantity
            updated.side = order.side
            executed_orders.append(updated)
            
        self.logger.info(f"Executed {len(executed_orders)} orders")
        return executed_orders
    
    def step(self, timestamp_ms: Optional[int] = None) -> List[Order]:
        """
        Execute a single step of the trading strategy.

        Uses a UNIX timestamp in milliseconds; if None, uses current time.

        Returns a list of updated Order objects.
        """

        # Determine timestamp in ms
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        # Convert ms to datetime in execution timezone
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=self.timezone)

        # Check if market is open
        if not self.broker.market_is_open(dt):
            self.logger.info(f"Market closed at {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return []
        # Log execution start
        local_dt = dt.astimezone()
        self.logger.info(f"Executing step at {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Fetch latest market data
        latest_data = self.fetch_latest_data(dt.timestamp())
        
        # Get target weights from strategy
        target_weights = self.strategy.step(dt, latest_data)
        self.logger.debug(f"Target weights: {target_weights}")

        # Get current positions and portfolio value
        current_positions, portfolio_value = self._get_current_positions()
        
        # Extract prices
        prices = {symbol: bar['close'] for symbol, bar in latest_data.items() if bar and 'close' in bar}
        self.logger.debug(f"Prices: {prices}")

        # Compute target positions and optionally current weights
        target_positions = self._calculate_target_positions(target_weights, portfolio_value, prices, self.fractional)
        current_weights = self._calculate_current_weights(list(current_positions.items()), portfolio_value, prices)
        # Render position changes table
        log_position_table(self.logger, current_positions, target_positions)
        
        # Render weight changes table
        log_weight_table(self.logger, current_weights, target_weights)
        
        # Build and render orders table
        orders = self._target_positions_to_orders(target_positions, current_positions)
        
        # Execute orders and log
        executed = self._execute_orders(orders)
        return executed

    def warmup(self, start_date: str, end_date: str = None, after_open_minutes: int = 0, before_close_minutes: int = 0, interval_seconds: int = None):
        """
        Warm up the strategy by running it over historical data from start_date up to end_date.
        If end_date is None, uses current date.
        """
        tickers = self.tickers
        calendar_name = "NYSE"
        if end_date is None:
            end_date = pd.Timestamp.now(tz=self.timezone).strftime("%Y-%m-%d")
        schedule = daily_schedule(
            after_open_minutes=after_open_minutes,
            before_close_minutes=before_close_minutes,
            interval_seconds=interval_seconds,
            calendar_name=calendar_name,
            start_date=start_date,
            end_date=end_date
        )
        steps = 0
        last_data = {t: None for t in tickers}
        try:
            for ts in schedule:
                dt_aware = pd.to_datetime(ts, unit='ms', utc=True).tz_convert(self.timezone)
                # Fetch data with ffill
                daily_data = self.market_data_loader.next(tickers, dt_aware, ffill=True)
                # Forward-fill missing values
                for t in tickers:
                    if daily_data[t] is None and last_data[t] is not None:
                        daily_data[t] = last_data[t]
                    elif daily_data[t] is not None:
                        last_data[t] = daily_data[t]
                current_signals = self.strategy.step(dt_aware, daily_data)
                self.logger.info(f"Warmup step at {dt_aware}: {current_signals}")
                steps += 1
                if steps % 100 == 0:
                    self.logger.info(f"Warm-up progress: {steps} steps...")
        except StopIteration:
            self.logger.info(f"Warm-up complete after {steps} steps (schedule exhausted).")
            return
        self.logger.info(f"Warm-up complete after {steps} steps (reached now).")

    def run(self, schedule: Iterator[int], warmup_start_date: str = None) -> None:
        """
        Optionally run warmup before main execution loop. If warmup_start_date is provided, run warmup from that date.
        """
        if warmup_start_date is not None:
            # Try to extract warmup params from schedule object
            after_open = getattr(schedule, 'after_open', 0)
            before_close = getattr(schedule, 'before_close', 0)
            interval = getattr(schedule, 'interval', None)
            self.logger.info(
                f"Running warmup from {warmup_start_date} (after_open={after_open}, before_close={before_close}, interval={interval})"
            )
            self.warmup(
                warmup_start_date,
                after_open_minutes=after_open,
                before_close_minutes=before_close,
                interval_seconds=interval
            )
        # allow for missing timezone (e.g. in FakeExec)
        tz = getattr(self, 'timezone', None)
        for timestamp_ms in schedule:
            # Display next execution time
            schedule_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=tz)
            self.logger.info(
                f"Next scheduled execution at {schedule_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            # compute wait using time.time() so it matches patched time in tests
            target_s = timestamp_ms / 1000.0
            now_s = time.time()
            wait = target_s - now_s
            if wait > 0:
                total_seconds = int(wait)
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                )
                with progress:
                    task = progress.add_task("Waiting for next execution", total=total_seconds)
                    for _ in range(total_seconds):
                        time.sleep(1)
                        progress.advance(task)
                rem = wait - total_seconds
                if rem > 0:
                    time.sleep(rem)
            self.logger.info(
                f"Executing step for {schedule_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            self.step(timestamp_ms)

```

# portwine/indexes.py

```py
### Indexes and standard baskets of equities. Not guarenteed to be up to date.

# 100 Oldest Companies (not standardized)
# Taken at May 27, 2025 from: https://stockanalysis.com/list/oldest-companies/
oldest_100 = [
    'DBD', 'BCO', 'NTB', 'FITB', 'FHB', 'EML', 'SR', 'SAN', 'MWA', 'BBVA', 'ACNB', 'SNN', 'NBTB', 'MTB', 'TD', 'MGEE',
    'CXT', 'CR', 'WNEB', 'TRV', 'OTIS', 'NVRI', 'LEVI', 'CNA', 'BLCO', 'WFC', 'THG', 'SWBI', 'PNC', 'BHRB', 'WU', 'TRI',
    'NYT', 'GLW', 'DOLE', 'BHP', 'AROW', 'MATW', 'CCU', 'AXP', 'PFE', 'DCO', 'CMA', 'UNM', 'PUK', 'LAZ', 'FNF', 'CLF',
    'SGI', 'CHD', 'BHLB', 'BC', 'PSO', 'TRC', 'SWK', 'RYI', 'CNH', 'DNB', 'PFS', 'BRK.B', 'PG', 'DE', 'TMP', 'ONB',
    'HIFS', 'MCK', 'IFF', 'CHMG', 'ROG', 'BNS', 'M', 'WTW', 'CFG', 'KEY', 'MO', 'EBC', 'BG', 'FISI', 'BMO', 'YORW', 'C',
    'HIG', 'WLY', 'CL', 'WASH', 'JPM', 'STT', 'CI', 'FLS', 'CWK', 'BK', 'BAC', 'TAK', 'IHG', 'BIRK', 'NWG', 'GSK',
    'LYG', 'BCS', 'BUD'
]

# S&P 500
# Taken at May 27, 2025 from: https://github.com/fja05680/sp500
sp500 = [
    'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN',
    'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME',
    'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T',
    'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH',
    'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP',
    'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR',
    'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
    'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY',
    'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE',
    'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK',
    'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX',
    'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO',
    'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX',
    'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS',
    'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM',
    'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP',
    'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K',
    'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS',
    'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR',
    'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU',
    'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ',
    'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE',
    'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW',
    'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL',
    'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG',
    'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE',
    'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK',
    'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA',
    'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR',
    'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS',
    'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL',
    'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS'
]

# S&P 100
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/S%26P_100
sp100 = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG',
    'BLK', 'BMY', 'BRK.B', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DE',
    'DHR', 'DIS', 'DUK', 'EMR', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC',
    'INTU', 'ISRG', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM',
    'MO', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NOW', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PLTR', 'PM', 'PYPL',
    'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB',
    'V', 'VZ', 'WFC', 'WMT', 'XOM'
]

# NASDAQ 100
# Taken at May 27, 2025 from: https://www.slickcharts.com/nasdaq100
nasdaq100 = [
    'MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'TSLA', 'AVGO', 'NFLX', 'COST', 'PLTR', 'ASML', 'TMUS',
    'CSCO', 'AZN', 'LIN', 'INTU', 'ISRG', 'AMD', 'PEP', 'ADBE', 'BKNG', 'PDD', 'QCOM', 'TXN', 'AMGN', 'HON', 'ARM',
    'GILD', 'ADP', 'CMCSA', 'AMAT', 'MELI', 'PANW', 'APP', 'CRWD', 'VRTX', 'MU', 'ADI', 'LRCX', 'MSTR', 'KLAC', 'SBUX',
    'CEG', 'CTAS', 'INTC', 'DASH', 'CDNS', 'MDLZ', 'FTNT', 'ORLY', 'ABNB', 'SNPS', 'MAR', 'PYPL', 'WDAY', 'REGN',
    'ADSK', 'MNST', 'ROP', 'CSX', 'AXON', 'TEAM', 'PAYX', 'CHTR', 'AEP', 'MRVL', 'CPRT', 'NXPI', 'PCAR', 'FAST', 'ROST',
    'KDP', 'EXC', 'VRSK', 'CCEP', 'TTWO', 'FANG', 'IDXX', 'XEL', 'DDOG', 'CTSH', 'ZS', 'LULU', 'EA', 'TTD', 'BKR',
    'ODFL', 'DXCM', 'GEHC', 'CSGP', 'KHC', 'MCHP', 'ANSS', 'CDW', 'WBD', 'GFS', 'BIIB', 'ON', 'MDB'
]

# HUI Gold Index
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/HUI_Gold_Index
hui_gold_index = [
    'NEM', 'ABX', 'GG', 'AEM', 'GOLD', 'KGC', 'AU', 'GFI', 'BVN', 'SBGL', 'IAG', 'KL', 'AUY', 'NGD' 'AGI', 'OR', 'HL',
    'EGO'
]

# Russell 1000
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/Russell_1000_Index
russell_1000 = [
    'TXG', 'MMM', 'AOS', 'AAON', 'ABT', 'ABBV', 'ACHC', 'ACN', 'AYI', 'ADBE', 'ADT', 'AAP', 'WMS', 'AMD', 'ACM', 'AES',
    'AMG', 'AFRM', 'AFL', 'AGCO', 'A', 'ADC', 'AGNC', 'AL', 'APD', 'ABNB', 'AKAM', 'ALK', 'ALB', 'ACI', 'AA', 'ARE',
    'ALGN', 'ALLE', 'ALGM', 'LNT', 'ALSN', 'ALL', 'ALLY', 'ALNY', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'DOX', 'AMED',
    'AMTM', 'AS', 'AEE', 'AAL', 'AEP', 'AXP', 'AFG', 'AMH', 'AIG', 'AMT', 'AWK', 'COLD', 'AMP', 'AME', 'AMGN', 'AMKR',
    'APH', 'ADI', 'ANGI', 'NLY', 'ANSS', 'AM', 'AR', 'AON', 'APA', 'APG', 'APLS', 'APO', 'APPF', 'AAPL', 'AMAT', 'APP',
    'ATR', 'APTV', 'ARMK', 'ACGL', 'ADM', 'ARES', 'ANET', 'AWI', 'ARW', 'AJG', 'ASH', 'AIZ', 'AGO', 'ALAB', 'T', 'ATI',
    'TEAM', 'ATO', 'ADSK', 'ADP', 'AN', 'AZO', 'AVB', 'AVTR', 'AVY', 'CAR', 'AVT', 'AXTA', 'AXS', 'AXON', 'AZEK',
    'AZTA', 'BKR', 'BALL', 'BAC', 'OZK', 'BBWI', 'BAX', 'BDX', 'BRBR', 'BSY', 'BRK.B', 'BBY', 'BILL', 'BIO', 'TECH',
    'BIIB', 'BMRN', 'BIRK', 'BJ', 'BLK', 'BX', 'HRB', 'XYZ', 'OWL', 'BK', 'BA', 'BOKF', 'BKNG', 'BAH', 'BWA', 'SAM',
    'BSX', 'BYD', 'BFAM', 'BHF', 'BMY', 'BRX', 'AVGO', 'BR', 'BEPC', 'BRO', 'BF.A', 'BF.B', 'BRKR', 'BC', 'BLDR', 'BG',
    'BURL', 'BWXT', 'BXP', 'CHRW', 'CACI', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CPRI', 'CAH', 'CSL', 'CG', 'KMX', 'CCL',
    'CARR', 'CRI', 'CVNA', 'CASY', 'CAT', 'CAVA', 'CBOE', 'CBRE', 'CCCS', 'CDW', 'CE', 'CELH', 'COR', 'CNC', 'CNP',
    'CERT', 'CF', 'CRL', 'SCHW', 'CHTR', 'CHE', 'CC', 'LNG', 'CVX', 'CMG', 'CHH', 'CHRD', 'CB', 'CHD', 'CHDN', 'CIEN',
    'CI', 'CINF', 'CTAS', 'CRUS', 'CSCO', 'C', 'CFG', 'CIVI', 'CLVT', 'CLH', 'CWEN.A', 'CWEN', 'CLF', 'CLX', 'NET',
    'CME', 'CMS', 'CNA', 'CNH', 'KO', 'COKE', 'CGNX', 'CTSH', 'COHR', 'COIN', 'CL', 'COLB', 'COLM', 'CMCSA', 'CMA',
    'FIX', 'CBSH', 'CAG', 'CNXC', 'CFLT', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'CNM', 'GLW', 'CPAY', 'CTVA',
    'CSGP', 'COST', 'CTRA', 'COTY', 'CPNG', 'CUZ', 'CR', 'CXT', 'CACC', 'CRH', 'CROX', 'CRWD', 'CCI', 'CCK', 'CSX',
    'CUBE', 'CMI', 'CW', 'CVS', 'DHI', 'DHR', 'DRI', 'DAR', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DAL', 'DELL', 'XRAY',
    'DVN', 'DXCM', 'FANG', 'DKS', 'DLR', 'DDS', 'DOCU', 'DLB', 'DG', 'DLTR', 'D', 'DPZ', 'DCI', 'DASH', 'DV', 'DOV',
    'DOW', 'DOCS', 'DKNG', 'DBX', 'DTM', 'DTE', 'DUK', 'DNB', 'DUOL', 'DD', 'BROS', 'DXC', 'DT', 'ELF', 'EXP', 'EWBC',
    'EGP', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'ELAN', 'ESTC', 'EA', 'ESI', 'ELV', 'EME', 'EMR', 'EHC', 'ENOV',
    'ENPH', 'ENTG', 'ETR', 'NVST', 'EOG', 'EPAM', 'EPR', 'EQT', 'EFX', 'EQIX', 'EQH', 'ELS', 'EQR', 'ESAB', 'WTRG',
    'ESS', 'EL', 'ETSY', 'EEFT', 'EVR', 'EG', 'EVRG', 'ES', 'ECG', 'EXAS', 'EXEL', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR',
    'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FERG', 'FNF', 'FIS', 'FITB', 'FAF', 'FCNCA', 'FHB', 'FHN',
    'FR', 'FSLR', 'FE', 'FI', 'FIVE', 'FIVN', 'FND', 'FLO', 'FLS', 'FMC', 'FNB', 'F', 'FTNT', 'FTV', 'FTRE', 'FBIN',
    'FOXA', 'FOX', 'BEN', 'FCX', 'FRPT', 'FYBR', 'CFR', 'FCN', 'GME', 'GLPI', 'GAP', 'GRMN', 'IT', 'GTES', 'GE', 'GEHC',
    'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'G', 'GNTX', 'GPC', 'GILD', 'GTLB', 'GPN', 'GFS', 'GLOB', 'GL', 'GMED',
    'GDDY', 'GS', 'GGG', 'GRAL', 'LOPE', 'GPK', 'GO', 'GWRE', 'GXO', 'HAL', 'THG', 'HOG', 'HIG', 'HAS', 'HAYW', 'HCA',
    'HR', 'DOC', 'HEI.A', 'HEI', 'JKHY', 'HSY', 'HES', 'HPE', 'HXL', 'DINO', 'HIW', 'HLT', 'HOLX', 'HD', 'HON', 'HRL',
    'HST', 'HLI', 'HHH', 'HWM', 'HPQ', 'HUBB', 'HUBS', 'HUM', 'HBAN', 'HII', 'HUN', 'H', 'IAC', 'IBM', 'IDA', 'IEX',
    'IDXX', 'ITW', 'ILMN', 'INCY', 'INFA', 'IR', 'INGM', 'INGR', 'INSP', 'PODD', 'INTC', 'IBKR', 'ICE', 'IFF', 'IP',
    'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IONS', 'IPGP', 'IQV', 'IRDM', 'IRM', 'ITT', 'JBL', 'J', 'JHG', 'JAZZ',
    'JBHT', 'JEF', 'JNJ', 'JCI', 'JLL', 'JPM', 'JNPR', 'KBR', 'K', 'KMPR', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KRC', 'KMB',
    'KIM', 'KMI', 'KNSL', 'KEX', 'KKR', 'KLAC', 'KNX', 'KSS', 'KHC', 'KR', 'KD', 'LHX', 'LH', 'LRCX', 'LAMR', 'LW',
    'LSTR', 'LVS', 'LSCC', 'LAZ', 'LEA', 'LEG', 'LDOS', 'LEN', 'LEN.B', 'LII', 'LBRDA', 'LBRDK', 'LBTYA', 'LBTYK',
    'FWONA', 'FWONK', 'LLYVA', 'LLYVK', 'LNW', 'LLY', 'LECO', 'LNC', 'LIN', 'LINE', 'LAD', 'LFUS', 'LYV', 'LKQ', 'LOAR',
    'LMT', 'L', 'LPX', 'LOW', 'LPLA', 'LCID', 'LULU', 'LITE', 'LYFT', 'LYB', 'MTB', 'MTSI', 'M', 'MSGS', 'MANH', 'MAN',
    'CART', 'MPC', 'MKL', 'MKTX', 'MAR', 'VAC', 'MMC', 'MLM', 'MRVL', 'MAS', 'MASI', 'MTZ', 'MA', 'MTDR', 'MTCH', 'MAT',
    'MKC', 'MCD', 'MCK', 'MDU', 'MPW', 'MEDP', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MTG', 'MGM', 'MCHP', 'MU', 'MSFT',
    'MSTR', 'MAA', 'MIDD', 'MRP', 'MKSI', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MDB', 'MPWR', 'MNST', 'MCO', 'MS',
    'MORN', 'MOS', 'MSI', 'MP', 'MSA', 'MSM', 'MSCI', 'MUSA', 'NDAQ', 'NTRA', 'NFG', 'NSA', 'NCNO', 'NTAP', 'NFLX',
    'NBIX', 'NFE', 'NYT', 'NWL', 'NEU', 'NEM', 'NWSA', 'NWS', 'NXST', 'NEE', 'NKE', 'NI', 'NNN', 'NDSN', 'NSC', 'NTRS',
    'NOC', 'NCLH', 'NOV', 'NRG', 'NU', 'NUE', 'NTNX', 'NVT', 'NVDA', 'NVR', 'ORLY', 'OXY', 'OGE', 'OKTA', 'ODFL', 'ORI',
    'OLN', 'OLLI', 'OHI', 'OMC', 'ON', 'OMF', 'OKE', 'ONTO', 'ORCL', 'OGN', 'OSK', 'OTIS', 'OVV', 'OC', 'PCAR', 'PKG',
    'PLTR', 'PANW', 'PARAA', 'PARA', 'PK', 'PH', 'PSN', 'PAYX', 'PAYC', 'PCTY', 'PYPL', 'PEGA', 'PENN', 'PAG', 'PNR',
    'PEN', 'PEP', 'PFGC', 'PR', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PPC', 'PNFP', 'PNW', 'PINS', 'PLNT', 'PLTK', 'PNC',
    'PII', 'POOL', 'BPOP', 'POST', 'PPG', 'PPL', 'PINC', 'TROW', 'PRI', 'PFG', 'PCOR', 'PG', 'PGR', 'PLD', 'PB', 'PRU',
    'PTC', 'PSA', 'PEG', 'PHM', 'PSTG', 'PVH', 'QGEN', 'QRVO', 'QCOM', 'PWR', 'QS', 'DGX', 'QDEL', 'RL', 'RRC', 'RJF',
    'RYN', 'RBA', 'RBC', 'O', 'RRX', 'REG', 'REGN', 'RF', 'RGA', 'RS', 'RNR', 'RGEN', 'RSG', 'RMD', 'RVTY', 'REXR',
    'REYN', 'RH', 'RNG', 'RITM', 'RIVN', 'RLI', 'RHI', 'HOOD', 'RBLX', 'RKT', 'ROK', 'ROIV', 'ROKU', 'ROL', 'ROP',
    'ROST', 'RCL', 'RGLD', 'RPRX', 'RPM', 'RTX', 'RYAN', 'R', 'SPGI', 'SAIA', 'SAIC', 'CRM', 'SLM', 'SNDK', 'SRPT',
    'SBAC', 'HSIC', 'SLB', 'SNDR', 'SMG', 'SEB', 'SEE', 'SEG', 'SEIC', 'SRE', 'ST', 'S', 'SCI', 'NOW', 'SN', 'SHW',
    'FOUR', 'SLGN', 'SPG', 'SSD', 'SIRI', 'SITE', 'SKX', 'SWKS', 'SFD', 'SJM', 'SW', 'SNA', 'SNOW', 'SOFI', 'SOLV',
    'SGI', 'SON', 'SHC', 'SO', 'SCCO', 'LUV', 'SPB', 'SPR', 'SPOT', 'SSNC', 'STAG', 'SARO', 'SWK', 'SBUX', 'STWD',
    'STT', 'STLD', 'STE', 'SF', 'SYK', 'SUI', 'SMCI', 'SYF', 'SNPS', 'SNV', 'SYY', 'TMUS', 'TTWO', 'TPR', 'TRGP', 'TGT',
    'SNX', 'FTI', 'TDY', 'TFX', 'THC', 'TDC', 'TER', 'TSLA', 'TTEK', 'TXN', 'TPL', 'TXRH', 'TXT', 'TMO', 'TFSL', 'THO',
    'TKR', 'TJX', 'TKO', 'TOST', 'TOL', 'BLD', 'TTC', 'TPG', 'TSCO', 'TTD', 'TW', 'TT', 'TDG', 'TRU', 'TNL', 'TRV',
    'TREX', 'TRMB', 'TRIP', 'TFC', 'DJT', 'TWLO', 'TYL', 'TSN', 'UHAL', 'UHAL.B', 'USB', 'X', 'UBER', 'UI', 'UDR',
    'UGI', 'PATH', 'ULTA', 'RARE', 'UAA', 'UA', 'UNP', 'UAL', 'UPS', 'URI', 'UTHR', 'UWMC', 'UNH', 'U', 'OLED', 'UHS',
    'UNM', 'USFD', 'MTN', 'VLO', 'VMI', 'VVV', 'VEEV', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VRT', 'VSTS',
    'VFC', 'VTRS', 'VICI', 'VKTX', 'VNOM', 'VIRT', 'V', 'VST', 'VNT', 'VNO', 'VOYA', 'VMC', 'WPC', 'WRB', 'GWW', 'WAB',
    'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WSO', 'W', 'WFRD', 'WBS', 'WEC', 'WFC', 'WELL', 'WEN', 'WCC', 'WST',
    'WAL', 'WDC', 'WU', 'WLK', 'WEX', 'WY', 'WHR', 'WTM', 'WMB', 'WSM', 'WTW', 'WSC', 'WING', 'WTFC', 'WOLF', 'WWD',
    'WDAY', 'WH', 'WYNN', 'XEL', 'XP', 'XPO', 'XYL', 'YETI', 'YUM', 'ZBRA', 'ZG', 'Z', 'ZBH', 'ZION', 'ZTS', 'ZM', 'ZI',
    'ZS'
]

# S&P 600
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/List_of_S%26P_600_companies
sp600 = [
    'AAP', 'AAT', 'ABCB', 'ABG', 'ABM', 'ABR', 'ACA', 'ACAD', 'ACIW', 'ACLS', 'ACT', 'ADEA', 'ADMA', 'ADNT', 'ADUS',
    'AEIS', 'AEO', 'AESI', 'AGO', 'AGYS', 'AHCO', 'AHH', 'AIN', 'AIR', 'AKR', 'AL', 'ALEX', 'ALG', 'ALGT', 'ALKS',
    'ALRM', 'AMN', 'AMPH', 'AMR', 'AMSF', 'AMTM', 'AMWD', 'ANDE', 'ANGI', 'ANIP', 'AORT', 'AOSL', 'APAM', 'APLE',
    'APOG', 'ARCB', 'ARI', 'ARLO', 'AROC', 'ARR', 'ARWR', 'ASIX', 'ASO', 'ASTE', 'ASTH', 'ATEN', 'ATGE', 'AUB', 'AVA',
    'AVAV', 'AVNS', 'AWI', 'AWR', 'AX', 'AXL', 'AZTA', 'AZZ', 'BANC', 'BANF', 'BANR', 'BCC', 'BCPC', 'BDN', 'BFH',
    'BFS', 'BGC', 'BGS', 'BHE', 'BHLB', 'BJRI', 'BKE', 'BKU', 'BL', 'BLFS', 'BLMN', 'BMI', 'BOH', 'BOOT', 'BOX', 'BRC',
    'BRKL', 'BSIG', 'BTU', 'BWA', 'BXMT', 'CABO', 'CAKE', 'CAL', 'CALM', 'CALX', 'CARG', 'CARS', 'CASH', 'CATY', 'CBRL',
    'CBU', 'CC', 'CCOI', 'CCS', 'CE', 'CENT', 'CENTA', 'CENX', 'CERT', 'CEVA', 'CFFN', 'CHCO', 'CHEF', 'CLB', 'CLSK',
    'CNK', 'CNMD', 'CNR', 'CNS', 'CNXN', 'COHU', 'COLL', 'CON', 'COOP', 'CORT', 'CPF', 'CPK', 'CPRX', 'CRC', 'CRGY',
    'CRI', 'CRK', 'CRSR', 'CRVL', 'CSGS', 'CSR', 'CSWI', 'CTKB', 'CTRE', 'CTS', 'CUBI', 'CURB', 'CVBF', 'CVCO', 'CVI',
    'CWEN', 'CWEN.A', 'CWK', 'CWT', 'CXM', 'CXW', 'DAN', 'DCOM', 'DEA', 'DEI', 'DFH', 'DFIN', 'DGII', 'DIOD', 'DLX',
    'DNOW', 'DOCN', 'DORM', 'DRH', 'DRQ', 'DV', 'DVAX', 'DXC', 'DXPE', 'DY', 'EAT', 'ECG', 'ECPG', 'EFC', 'EGBN', 'EIG',
    'ELME', 'EMBC', 'ENOV', 'ENR', 'ENVA', 'EPAC', 'EPC', 'EPRT', 'ESE', 'ESI', 'ETD', 'ETSY', 'EVTC', 'EXPI', 'EXTR',
    'EYE', 'EZPW', 'FBK', 'FBNC', 'FBP', 'FBRT', 'FCF', 'FCPT', 'FDP', 'FELE', 'FFBC', 'FHB', 'FIZZ', 'FL', 'FMC',
    'FORM', 'FOXF', 'FRPT', 'FSS', 'FTDR', 'FTRE', 'FUL', 'FULT', 'FUN', 'FWRD', 'GBX', 'GDEN', 'GDYN', 'GEO', 'GES',
    'GFF', 'GIII', 'GKOS', 'GMS', 'GNL', 'GNW', 'GO', 'GOGO', 'GOLF', 'GPI', 'GRBK', 'GSHD', 'GTES', 'GTY', 'GVA',
    'HAFC', 'HASI', 'HAYW', 'HBI', 'HCC', 'HCI', 'HCSG', 'HELE', 'HFWA', 'HI', 'HIW', 'HLIT', 'HLX', 'HMN', 'HNI',
    'HOPE', 'HP', 'HRMY', 'HSII', 'HSTM', 'HTH', 'HTLD', 'HTZ', 'HUBG', 'HWKN', 'HZO', 'IAC', 'IART', 'IBP', 'ICHR',
    'ICUI', 'IDCC', 'IIIN', 'IIPR', 'INDB', 'INN', 'INSP', 'INSW', 'INVA', 'IOSP', 'IPAR', 'ITGR', 'ITRI', 'JACK',
    'JBGS', 'JBLU', 'JBSS', 'JBT', 'JJSF', 'JOE', 'JXN', 'KAI', 'KALU', 'KAR', 'KFY', 'KLG', 'KLIC', 'KMT', 'KN', 'KOP',
    'KREF', 'KRYS', 'KSS', 'KTB', 'KTOS', 'KW', 'KWR', 'LBRT', 'LCII', 'LEG', 'LGIH', 'LGND', 'LKFN', 'LMAT', 'LNC',
    'LNN', 'LPG', 'LQDT', 'LRN', 'LTC', 'LUMN', 'LXP', 'LZB', 'MAC', 'MARA', 'MATW', 'MATX', 'MBC', 'MC', 'MCRI', 'MCW',
    'MCY', 'MD', 'MDU', 'MGEE', 'MGPI', 'MGY', 'MHO', 'MLAB', 'MLKN', 'MMI', 'MMSI', 'MNRO', 'MODG', 'MOG.A', 'MP',
    'MPW', 'MRCY', 'MRP', 'MRTN', 'MSEX', 'MSGS', 'MTH', 'MTRN', 'MTUS', 'MTX', 'MWA', 'MXL', 'MYGN', 'MYRG', 'NABL',
    'NATL', 'NAVI', 'NBHC', 'NBTB', 'NEO', 'NEOG', 'NGVT', 'NHC', 'NMIH', 'NOG', 'NPK', 'NPO', 'NSIT', 'NTCT', 'NVEE',
    'NVRI', 'NWBI', 'NWL', 'NWN', 'NX', 'NXRT', 'NYMT', 'OFG', 'OGN', 'OI', 'OII', 'OMCL', 'OMI', 'OSIS', 'OTTR', 'OUT',
    'OXM', 'PAHC', 'PARR', 'PAYO', 'PATK', 'PBH', 'PBI', 'PCRX', 'PDFS', 'PEB', 'PECO', 'PENN', 'PFBC', 'PFS', 'PGNY',
    'PHIN', 'PI', 'PINC', 'PIPR', 'PJT', 'PLAB', 'PLAY', 'PLMR', 'PLUS', 'PLXS', 'PMT', 'POWL', 'PPBI', 'PRA', 'PRAA',
    'PRDO', 'PRG', 'PRGS', 'PRK', 'PRLB', 'PRVA', 'PSMT', 'PTEN', 'PTGX', 'PUMP', 'PZZA', 'QDEL', 'QNST', 'QRVO',
    'RAMP', 'RC', 'RCUS', 'RDN', 'RDNT', 'RES', 'REX', 'REZI', 'RGR', 'RHI', 'RHP', 'RNST', 'ROCK', 'ROG', 'RUN',
    'RUSHA', 'RWT', 'RXO', 'SAFE', 'SABR', 'SAFT', 'SAH', 'SANM', 'SBCF', 'SBH', 'SBSI', 'SCHL', 'SCL', 'SCSC', 'SCVL',
    'SDGR', 'SEDG', 'SEE', 'SEM', 'SFBS', 'SFNC', 'SGH', 'SHAK', 'SHEN', 'SHO', 'SHOO', 'SIG', 'SITC', 'SITM', 'SJW',
    'SKT', 'SKY', 'SKYW', 'SLG', 'SLP', 'SLVM', 'SM', 'SMP', 'SMPL', 'SMTC', 'SNCY', 'SNDK', 'SNDR', 'SNEX', 'SONO',
    'SPNT', 'SPSC', 'SPTN', 'SPXC', 'SSTK', 'STAA', 'STBA', 'STC', 'STEL', 'STEP', 'STRA', 'STRL', 'SUPN', 'SXC', 'SXI',
    'SXT', 'TALO', 'TBBK', 'TDC', 'TDS', 'TDW', 'TFIN', 'TFX', 'TGI', 'TGNA', 'TGTX', 'THRM', 'THRY', 'THS', 'TILE',
    'TMDX', 'TMP', 'TNC', 'TNDM', 'TPH', 'TR', 'TRIP', 'TRMK', 'TRN', 'TRNO', 'TRST', 'TRUP', 'TTGT', 'TTMI', 'TWI',
    'TWO', 'UCBI', 'UCTT', 'UE', 'UFCS', 'UFPT', 'UHT', 'UNF', 'UNFI', 'UNIT', 'UPBD', 'URBN', 'USNA', 'USPH', 'UTL',
    'UVV', 'VBTX', 'VCEL', 'VECO', 'VIAV', 'VICR', 'VIR', 'VIRT', 'VRE', 'VRRM', 'VRTS', 'VSAT', 'VSCO', 'VSH', 'VSTS',
    'VTOL', 'VTLE', 'VVI', 'VYX', 'WABC', 'WAFD', 'WD', 'WDFC', 'WERN', 'WGO', 'WHD', 'WKC', 'WLY', 'WOLF', 'WOR',
    'WRLD', 'WS', 'WSC', 'WSFS', 'WSR', 'WT', 'WWW', 'XHR', 'XNCR', 'XPEL', 'XRX', 'YELP', 'YOU', 'ZD', 'ZWS'
]

# Dow Jones Industial Average
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average
djia = [
    'MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ',
    'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT'
]

# Dow Jones Transportation Average
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/Dow_Jones_Transportation_Average
djta = [
    'ALK', 'AAL', 'CAR', 'CHRW', 'CSX', 'DAL', 'EXPD', 'FDX', 'JBHT', 'KEX', 'LSTR', 'MATX', 'NSC', 'ODFL', 'R', 'LUV',
    'UBER', 'UNP', 'UAL', 'UPS'
]

# Dow Jones Utility Average
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/Dow_Jones_Utility_Average
djua = [
    'AEP', 'AWK', 'ATO', 'ED', 'D', 'DUK', 'EIX', 'EXC', 'FE', 'NEE', 'PEG', 'SRE', 'SO', 'VST', 'XEL'
]

# Dow Jones Composite Average
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/Dow_Jones_Composite_Average
djca = [
    'MMM', 'AES', 'ALK', 'AAL', 'AEP', 'AXP', 'AWK', 'AMGN', 'AAPL', 'ATO', 'CAR', 'BA', 'CAT', 'CHRW', 'CVX', 'CSCO',
    'KO', 'ED', 'CSX', 'DAL', 'D', 'DOW', 'DUK', 'EIX', 'EXC', 'EXPD', 'FDX', 'FE', 'GS', 'HD', 'HON', 'INTC', 'IBM',
    'JBHT', 'JBLU', 'JPM', 'JNJ', 'KEX', 'LSTR', 'MATX', 'MCD', 'MRK', 'MSFT', 'NEE', 'NKE', 'NSC', 'ODFL', 'PG', 'PEG',
    'R', 'CRM', 'SRE', 'SO', 'LUV', 'TRV', 'UNP', 'UAL', 'UPS', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'XEL'
]

# S&P 500 Dividend Aristocrats
# Taken at May 27, 2025 from: https://en.wikipedia.org/wiki/S%26P_500_Dividend_Aristocrats
sp_dividend_aristocrats = [
    'AOS', 'ABT', 'ABBV', 'AFL', 'APD', 'ALB', 'AMCR', 'ADM', 'ATO', 'ADP', 'BDX', 'BRO', 'BF.B', 'CAH', 'CAT', 'CHRW',
    'CVX', 'CB', 'CHD', 'CINF', 'CTAS', 'CLX', 'KO', 'CL', 'ED', 'DOV', 'ECL', 'EMR', 'ERIE', 'ES', 'ESS', 'EXPD',
    'XOM', 'FDS', 'FAST', 'FRT', 'BEN', 'GD', 'GPC', 'HRL', 'ITW', 'IBM', 'SJM', 'JNJ', 'KVUE', 'KMB', 'LIN', 'LOW',
    'MKC', 'MCD', 'MDT', 'NEE', 'NDSN', 'NUE', 'PNR', 'PEP', 'PPG', 'PG', 'O', 'ROP', 'SPGI', 'SHW', 'SWK', 'SYY',
    'TROW', 'TGT', 'GWW', 'WMT', 'WST'
]
```

# portwine/loaders/__init__.py

```py
from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.data.providers.loader_adapters import EODHDMarketDataLoader
from portwine.loaders.polygon import PolygonMarketDataLoader
from portwine.data.providers.loader_adapters import NoisyMarketDataLoader
from portwine.loaders.fred import FREDMarketDataLoader
from portwine.loaders.barchartindices import BarchartIndicesMarketDataLoader
from portwine.loaders.alternative import AlternativeMarketDataLoader
from portwine.loaders.dailytoopenclose import DailyToOpenCloseLoader
from portwine.data.providers.loader_adapters import AlpacaMarketDataLoader

```

# portwine/loaders/alpaca.py

```py
"""
Alpaca market data loader for the portwine framework.

This module provides a MarketDataLoader implementation for fetching data
from the Alpaca Markets API, both for historical and real-time data.
Uses direct REST API calls instead of the Alpaca Python SDK.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests

import pandas as pd
import pytz

from portwine.data.providers.loader_adapters import MarketDataLoader

# Configure logging
logger = logging.getLogger(__name__)

# API URLs
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaMarketDataLoader(MarketDataLoader):
    """
    Market data loader for Alpaca Markets API.
    
    This loader fetches historical and real-time data from Alpaca Markets API
    using direct REST calls. It supports fetching OHLCV data for stocks and ETFs.
    
    Parameters
    ----------
    api_key : str, optional
        Alpaca API key. If not provided, attempts to read from ALPACA_API_KEY env var.
    api_secret : str, optional
        Alpaca API secret. If not provided, attempts to read from ALPACA_API_SECRET env var.
    start_date : Union[str, datetime], optional
        Start date for historical data, defaults to 2 years ago
    end_date : Union[str, datetime], optional
        End date for historical data, defaults to today
    cache_dir : str, optional
        Directory to cache data to. If not provided, data is not cached.
    paper_trading : bool, default True
        Whether to use paper trading mode (sandbox)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        cache_dir: Optional[str] = None,
        paper_trading: bool = True,
    ):
        """Initialize Alpaca market data loader."""
        super().__init__()
        
        # Use environment variables if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not provided. "
                "Either pass as parameters or set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
        
        # Set up API URLs based on paper trading flag
        self.base_url = ALPACA_PAPER_URL if paper_trading else ALPACA_LIVE_URL
        self.data_url = ALPACA_DATA_URL
        
        # Auth headers for API requests
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        
        # Create requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Cache directory
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Date range for historical data
        today = datetime.now(pytz.UTC)
        if start_date is None:
            # Default to 2 years ago
            self.start_date = today - timedelta(days=365 * 2)
        else:
            self.start_date = pd.Timestamp(start_date).to_pydatetime()
            if self.start_date.tzinfo is None:
                self.start_date = pytz.UTC.localize(self.start_date)
        
        if end_date is None:
            self.end_date = today
        else:
            self.end_date = pd.Timestamp(end_date).to_pydatetime()
            if self.end_date.tzinfo is None:
                self.end_date = pytz.UTC.localize(self.end_date)
        
        # Latest data cache to avoid frequent API calls
        self._latest_data_cache: Dict[str, Dict] = {}
        self._latest_data_timestamp = None
    
    def _api_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Helper method to make authenticated GET requests to Alpaca API
        
        Parameters
        ----------
        url : str
            API endpoint URL (starting with /)
        params : Dict[str, Any], optional
            Query parameters for the request
            
        Returns
        -------
        Any
            JSON response data
            
        Raises
        ------
        Exception
            If API request fails
        """
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            raise
    
    def _get_cache_path(self, ticker: str) -> str:
        """
        Get path to cached data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        str
            Path to cached data
        """
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{ticker}.parquet")
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if data is cached, None otherwise
        """
        if not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Error loading cached data for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pd.DataFrame
            Data to cache
        """
        if not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(ticker)
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Error caching data for {ticker}: {e}")
    
    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for API requests"""
        return dt.isoformat()
    
    def _fetch_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Alpaca API.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            # Format parameters for API request
            params = {
                "symbols": ticker,
                "timeframe": "1Day",
                "start": self._format_datetime(self.start_date),
                "end": self._format_datetime(self.end_date),
                "limit": 10000,  # Maximum allowed by the API
                "adjustment": "raw"
            }
            
            # Make API request for bars
            url = f"{self.data_url}/v2/stocks/bars"
            response = self._api_get(url, params)
            
            # Extract data
            if 'bars' in response and ticker in response['bars']:
                bars = response['bars'][ticker]
                
                # Convert to dataframe
                df = pd.DataFrame(bars)
                
                # Convert timestamp string to datetime index
                df['t'] = pd.to_datetime(df['t'])
                df = df.set_index('t')
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                # Drop timezone for compatibility
                df.index = df.index.tz_localize(None)
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
        
        return None
    
    def _fetch_latest_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch latest data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        dict or None
            Latest OHLCV data or None if fetch fails
        """
        # Check if we have recent data in memory
        now = datetime.now()
        if (
            self._latest_data_timestamp 
            and (now - self._latest_data_timestamp).total_seconds() < 60
            and ticker in self._latest_data_cache
        ):
            return self._latest_data_cache[ticker]
        
        try:
            # Format parameters for API request
            params = {
                "symbols": ticker
            }
            
            # Make API request for latest bar
            url = f"{self.data_url}/v2/stocks/bars/latest"
            response = self._api_get(url, params)
            
            if 'bars' in response and ticker in response['bars']:
                bar = response['bars'][ticker]
                
                # Update cache
                self._latest_data_cache[ticker] = {
                    "open": float(bar['o']),
                    "high": float(bar['h']),
                    "low": float(bar['l']),
                    "close": float(bar['c']),
                    "volume": float(bar['v']),
                }
                self._latest_data_timestamp = now
                
                return self._latest_data_cache[ticker]
        
        except Exception as e:
            logger.error(f"Error fetching latest data for {ticker}: {e}")
        
        return None
    
    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if load fails
        """
        # Check cache first
        df = self._load_from_cache(ticker)
        
        # Fetch from API if not cached or outdated
        if df is None or df.index.max() < self.end_date.replace(tzinfo=None):
            df_new = self._fetch_historical_data(ticker)
            
            if df_new is not None and not df_new.empty:
                if df is not None:
                    # Append new data
                    df = pd.concat([df[~df.index.isin(df_new.index)], df_new])
                    df = df.sort_index()
                else:
                    df = df_new
                
                # Save to cache
                self._save_to_cache(ticker, df)
        
        return df
    
    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        timestamp : pd.Timestamp
            Timestamp to get data for
            
        Returns
        -------
        Dict[str, dict]
            Dictionary mapping tickers to bar data
        """
        result = {}
        
        # If timestamp is close to now, get live data
        now = pd.Timestamp.now()
        if abs((now - timestamp).total_seconds()) < 86400:  # Within 24 hours
            for ticker in tickers:
                bar_data = self._fetch_latest_data(ticker)
                if bar_data:
                    result[ticker] = bar_data
                else:
                    result[ticker] = None
        else:
            # Otherwise use historical data
            for ticker in tickers:
                df = self.fetch_data([ticker]).get(ticker)
                if df is not None:
                    bar = self._get_bar_at_or_before(df, timestamp)
                    if bar is not None:
                        result[ticker] = {
                            "open": float(bar["open"]),
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "volume": float(bar["volume"]),
                        }
                    else:
                        result[ticker] = None
                else:
                    result[ticker] = None
        
        return result
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close() 
```

# portwine/loaders/alternative.py

```py
from portwine.data.providers.loader_adapters import MarketDataLoader


class AlternativeMarketDataLoader(MarketDataLoader):
    """
    A unified market data loader that routes requests to specialized data loaders
    based on source identifiers in ticker symbols.

    This loader expects tickers in the format "SOURCE:TICKER" where SOURCE matches
    a source identifier from one of the provided data loaders.

    Examples:
    - "FRED:SP500" would fetch SP500 data from the FRED data loader
    - "BARCHARTINDEX:ADDA" would fetch ADDA index from the Barchart Indices loader

    Parameters
    ----------
    loaders : list
        List of MarketDataLoader instances with SOURCE_IDENTIFIER class attributes
    """

    def __init__(self, loaders):
        """
        Initialize the alternative market data loader with a list of specialized loaders.
        """
        super().__init__()

        # Create a dictionary mapping source identifiers to loaders
        self.source_loaders = {}
        for loader in loaders:
            if hasattr(loader, 'SOURCE_IDENTIFIER'):
                self.source_loaders[loader.SOURCE_IDENTIFIER] = loader
            else:
                print(f"Warning: Loader {loader.__class__.__name__} does not have a SOURCE_IDENTIFIER")

    def load_ticker(self, ticker):
        """
        Load data for a specific ticker by routing to the appropriate data loader.

        Parameters
        ----------
        ticker : str
            Ticker in format "SOURCE:TICKER" (e.g., "FRED:SP500")

        Returns
        -------
        pd.DataFrame or None
            DataFrame with daily date index and appropriate columns, or None if load fails
        """
        # Check if it's already in the cache
        if ticker in self._data_cache:
            return self._data_cache[ticker].copy()  # Important: return a copy from cache

        # Parse the ticker to extract source and actual ticker
        if ":" not in ticker:
            print(f"Error: Invalid ticker format for {ticker}. Expected format is 'SOURCE:TICKER'.")
            return None

        source, actual_ticker = ticker.split(":", 1)

        # Find the appropriate loader
        if source in self.source_loaders:
            loader = self.source_loaders[source]
            data = loader.load_ticker(actual_ticker)

            # Cache the result if successful
            if data is not None:
                self._data_cache[ticker] = data.copy()  # Important: store a copy in cache

            return data
        else:
            print(f"Error: No loader found for source identifier '{source}'.")
            print(f"Available sources: {list(self.source_loaders.keys())}")
            return None

    def fetch_data(self, tickers):
        """
        Ensures we have data loaded for each ticker in 'tickers'.
        Routes each ticker to the appropriate specialized data loader.
        OPTIMIZATION: Creates numpy caches for fast access.

        Parameters
        ----------
        tickers : list
            Tickers to ensure data is loaded for, in format "SOURCE:TICKER"

        Returns
        -------
        data_dict : dict
            { ticker: DataFrame }
        """
        fetched_data = {}

        for ticker in tickers:
            if ticker not in self._data_cache:
                data = self.load_ticker(ticker)
                # OPTIMIZATION: Create numpy caches for fast access
                if data is not None:
                    self._create_numpy_cache(ticker, data)
            
            # Add to the returned dictionary if we have cached data
            if ticker in self._data_cache:
                fetched_data[ticker] = self._data_cache[ticker]

        return fetched_data

    def get_available_sources(self):
        """
        Returns a list of available source identifiers.

        Returns
        -------
        list
            List of source identifiers
        """
        return list(self.source_loaders.keys())
```

# portwine/loaders/barchartindices.py

```py
import os
import pandas as pd
from portwine.data.providers.loader_adapters import MarketDataLoader


class BarchartIndicesMarketDataLoader(MarketDataLoader):
    """
    Loads data from local CSV files containing Barchart Indices data.

    This loader is designed to handle CSV files with index data following the format
    of Barchart indices. It reads from a local directory and does not download from
    any online source.

    Parameters
    ----------
    data_path : str
        Directory where index CSV files are stored
    """

    # Source identifier for AlternativeMarketDataLoader
    SOURCE_IDENTIFIER = 'BARCHARTINDEX'

    def __init__(self, data_path):
        """
        Initialize the Barchart Indices market data loader.
        """
        super().__init__()
        self.data_path = data_path

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def load_ticker(self, ticker):
        """
        Load data for a specific Barchart index from a CSV file.

        Parameters
        ----------
        ticker : str
            Barchart index code (e.g., 'ADDA')

        Returns
        -------
        pd.DataFrame
            DataFrame with daily date index and appropriate columns for the portwine framework
        """
        file_path = os.path.join(self.data_path, f"{ticker}.csv")

        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for Barchart index {ticker}: {file_path}")
            return None

        try:
            # Load CSV file with automatic date parsing
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            return self._format_dataframe(df)
        except Exception as e:
            print(f"Error loading data for Barchart index {ticker}: {str(e)}")
            return None

    def _format_dataframe(self, df):
        """
        Format the dataframe to match the expected structure for portwine.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with appropriate columns
        """
        # Make sure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Make sure the index name is 'date'
        df.index.name = 'date'

        # Sort by date
        df = df.sort_index()

        # Ensure we have all required columns for portwine
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check if we have a close column, if not look for other possible names
        if 'close' not in df.columns:
            for alt_name in ['Close', 'CLOSE', 'price', 'Price', 'value', 'Value']:
                if alt_name in df.columns:
                    df['close'] = df[alt_name]
                    break
            # If still not found, use the first numeric column
            if 'close' not in df.columns and not df.empty:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df['close'] = df[numeric_cols[0]]

        # Fill missing required columns
        for col in required_columns:
            if col not in df.columns:
                if col in ['open', 'high', 'low'] and 'close' in df.columns:
                    df[col] = df['close']
                elif col == 'volume':
                    df[col] = 0
                else:
                    # If we can't determine a suitable close value, use NaN
                    df[col] = float('nan')

        return df
```

# portwine/loaders/base.py

```py
import pandas as pd
import numpy as np
from typing import Optional

class MarketDataLoader:
    """
    Base loader. Override load_ticker; fetch_data remains unchanged.
    Adds:
      - get_all_dates: union calendar for any tickers
      - next: returns the bar at or immediately before a given ts via searchsorted
    
    OPTIMIZATION: Uses numpy arrays for fast data access instead of pandas operations.
    """

    def __init__(self):
        self._data_cache = {}
        self._numpy_cache = {}  # Store numpy arrays for fast access
        self._date_cache = {}   # Store date arrays for fast searchsorted

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        """
        Must be overridden to load and return a DataFrame indexed by pd.Timestamp
        with columns ['open','high','low','close','volume'], or return None.
        """
        raise NotImplementedError

    def fetch_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Caches & returns all requested tickers.
        OPTIMIZATION: Also creates numpy caches for fast access.
        """
        fetched = {}
        for t in tickers:
            if t not in self._data_cache:
                df = self.load_ticker(t)
                if df is not None:
                    self._data_cache[t] = df
                    # OPTIMIZATION: Create numpy caches for fast access
                    self._create_numpy_cache(t, df)
            if t in self._data_cache:
                fetched[t] = self._data_cache[t]
        return fetched

    def _create_numpy_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Create numpy arrays for fast data access.
        Replaces pandas operations with numpy for 2-5x speedup.
        """
        # Convert dates to numpy array for fast searchsorted
        self._date_cache[ticker] = df.index.values.astype('datetime64[ns]')
        
        # Convert OHLCV data to numpy array for fast indexing
        self._numpy_cache[ticker] = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float64)

    def get_all_dates(self, tickers: list[str]) -> list[pd.Timestamp]:
        """
        Build the *union* of all timestamps across these tickers.
        This is your intraday/daily trading calendar.
        """
        data = self.fetch_data(tickers)
        all_ts = {ts for df in data.values() for ts in df.index}
        return sorted(all_ts)

    def _get_bar_at_or_before_numpy(self, ticker: str, ts: pd.Timestamp) -> Optional[np.ndarray]:
        """
        OPTIMIZED: Get the bar at or immediately before the given timestamp using numpy.
        This replaces pandas operations with numpy for 2-5x speedup.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol to get data for
        ts : pd.Timestamp
            Timestamp to get data for
            
        Returns
        -------
        np.ndarray or None
            Array with [open, high, low, close, volume] if found, None otherwise
        """
        if ticker not in self._numpy_cache:
            return None
            
        date_array = self._date_cache[ticker]
        if len(date_array) == 0:
            return None
            
        # Convert timestamp to numpy datetime64 for comparison
        ts_np = np.datetime64(ts)
        
        # OPTIMIZATION: Use numpy searchsorted instead of pandas (much faster)
        pos = np.searchsorted(date_array, ts_np, side="right") - 1
        if pos < 0:
            return None
            
        # OPTIMIZATION: Direct numpy array access instead of df.iloc (much faster)
        return self._numpy_cache[ticker][pos]

    def _get_bar_at_or_before(self, df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
        """
        LEGACY: Get the bar at or immediately before the given timestamp.
        This method is kept for backwards compatibility but is slower than the numpy version.
        """
        if df.empty:
            return None
            
        # Ensure both timestamp and index are timezone-aware and match
        if ts.tzinfo is None:
            ts = ts.tz_localize(df.index.tz)
        elif df.index.tz is None:
            df.index = df.index.tz_localize(ts.tz)
        elif str(ts.tz) != str(df.index.tz):
            ts = ts.tz_convert(df.index.tz)
            
        idx = df.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        return df.iloc[pos]

    def next(self,
             tickers: list[str],
             ts: pd.Timestamp
    ) -> dict[str, dict[str, float] | None]:
        """
        For a given timestamp ts, return a dict:
          { ticker: {'open','high','low','close','volume'} }
        where the values come from the bar at or immediately before ts.
        
        OPTIMIZATION: Uses numpy arrays for 2-5x speedup vs pandas operations.
        """
        data = self.fetch_data(tickers)
        bar_dict: dict[str, dict[str, float] | None] = {}

        for t in data.keys():
            # OPTIMIZATION: Use numpy-based method instead of pandas
            row = self._get_bar_at_or_before_numpy(t, ts)
            if row is None:
                bar_dict[t] = None
            else:
                # Direct access to numpy array elements (much faster than pandas)
                bar_dict[t] = {
                    'open':   float(row[0]),
                    'high':   float(row[1]),
                    'low':    float(row[2]),
                    'close':  float(row[3]),
                    'volume': float(row[4])
                }

        return bar_dict

```

# portwine/loaders/broker.py

```py
from typing import Optional, List, Dict, Any
import pandas as pd
from portwine.data.providers.loader_adapters import MarketDataLoader
from portwine.brokers.base import BrokerBase


class BrokerDataLoader(MarketDataLoader):
    """
    Alternative data loader providing broker account fields (e.g., 'equity')
    in live mode via BrokerBase and in offline/backtest via evolving initial_equity.
    """
    SOURCE_IDENTIFIER = "BROKER"

    def __init__(self,
                 broker: Optional[BrokerBase] = None,
                 initial_equity: Optional[float] = None):
        super().__init__()
        if broker is None and initial_equity is None:
            raise ValueError("Give either a broker or an initial_equity")
        self.broker = broker
        self.equity = initial_equity  # Only used in backtest/offline mode

    def next(self, tickers: List[str], ts: pd.Timestamp) -> Dict[str, Dict[str, float] | None]:
        """
        Return a dict for each ticker; if prefixed with 'BROKER', return {'equity': value}, else None.
        """
        out: Dict[str, Dict[str, float] | None] = {}
        for t in tickers:
            # Only handle tickers with a prefix; non-colon tickers are not for BROKER
            if ":" not in t:
                out[t] = None
                continue
            src, key = t.split(":", 1)
            if src != self.SOURCE_IDENTIFIER:
                out[t] = None
                continue

            # live vs. offline
            if self.broker is not None:
                account = self.broker.get_account()
                eq = account.equity
            else:
                eq = self.equity

            out[t] = {"equity": float(eq)}
        return out

    def update(self,
               ts: pd.Timestamp,
               raw_sigs: Dict[str, Any],
               raw_rets: Dict[str, float],
               strat_ret: float) -> None:
        """
        Backtest-only hook: evolve self.equity by applying strategy return.
        """
        if self.broker is None and strat_ret is not None:
            self.equity *= (1 + strat_ret) 
```

# portwine/loaders/dailytoopenclose.py

```py
import pandas as pd
import numpy as np
from portwine.data.providers.loader_adapters import MarketDataLoader


class DailyToOpenCloseLoader(MarketDataLoader):
    """
    Wraps a daily OHLCV loader and emits two intraday bars per day:
      - 09:30 bar: OHLC all set to the daily open, volume = 0
      - 16:00 bar: OHLC all set to the daily close, volume = daily volume
    """
    def __init__(self, base_loader: MarketDataLoader):
        self.base_loader = base_loader
        super().__init__()

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        # 1) Load the daily OHLCV from the base loader
        df_daily = self.base_loader.load_ticker(ticker)
        if df_daily is None or df_daily.empty:
            return df_daily

        # 2) Ensure the index is datetime
        df_daily = df_daily.copy()
        df_daily.index = pd.to_datetime(df_daily.index)

        # 3) Build intraday records
        records = []
        for date, row in zip(df_daily.index, df_daily.itertuples()):
            # 09:30 bar using the daily open
            ts_open = date.replace(hour=9, minute=30)
            records.append({
                'timestamp': ts_open,
                'open':  row.open,
                'high':  row.open,
                'low':   row.open,
                'close': row.open,
                'volume': 0
            })
            # 16:00 bar using the daily close
            ts_close = date.replace(hour=16, minute=0)
            records.append({
                'timestamp': ts_close,
                'open':  row.close,
                'high':  row.close,
                'low':   row.close,
                'close': row.close,
                'volume': getattr(row, 'volume', np.nan)
            })

        # 4) Assemble into a DataFrame with a DatetimeIndex
        df_intraday = (
            pd.DataFrame.from_records(records)
              .set_index('timestamp')
              .sort_index()
        )

        return df_intraday

```

# portwine/loaders/eodhd.py

```py
import os
import pandas as pd
from typing import Dict, List
from portwine.data.providers.loader_adapters import MarketDataLoader


class EODHDMarketDataLoader(MarketDataLoader):
    """
    Loads historical market data for a list of tickers from CSV files.
    Each CSV must be located in data_path and named as TICKER.US.csv for each ticker.
    The CSV is assumed to have at least these columns:
        date, open, high, low, close, adjusted_close, volume
    The loaded data will be stored in a dictionary keyed by ticker symbol.

    Once loaded, data is cached in memory. Subsequent calls for the same ticker
    will not re-read from disk.
    """

    def __init__(self, data_path, exchange_code='US'):
        """
        Parameters
        ----------
        data_path : str
            The directory path where CSV files are located.
        """
        self.data_path = data_path
        self.exchange_code = exchange_code
        super().__init__()

    def load_ticker(self, ticker):
        file_path = os.path.join(self.data_path, f"{ticker}.{self.exchange_code}.csv")
        if not os.path.isfile(file_path):
            print(f"Warning: CSV file not found for {ticker}: {file_path}")
            return None

        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        # Calculate adjusted prices
        adj_ratio = df['adjusted_close'] / df['close']

        df['open'] = df['open'] * adj_ratio
        df['high'] = df['high'] * adj_ratio
        df['low'] = df['low'] * adj_ratio
        df['close'] = df['adjusted_close']

        # Optional: reorder columns if needed
        df = df[[
            'open', 'high', 'low', 'close', 'volume',
        ]]
        df.sort_index(inplace=True)

        return df

    def next(self, tickers: List[str], timestamp: pd.Timestamp) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.

        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        timestamp : pd.Timestamp or datetime
            Timestamp to get data for

        Returns
        -------
        Dict[str, dict]
            Dictionary mapping tickers to bar data
        """
        # Convert datetime to pandas Timestamp if needed
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
            
        result = {}
        for ticker in tickers:
            # Get data from cache or load it
            df = self.fetch_data([ticker]).get(ticker)
            if df is not None:
                # Find the bar at or before the timestamp
                bar = self._get_bar_at_or_before(df, timestamp)
                if bar is not None:
                    result[ticker] = {
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(bar["volume"]),
                    }
                else:
                    result[ticker] = None
            else:
                result[ticker] = None
        return result

```

# portwine/loaders/fred.py

```py
import os
import pandas as pd
from fredapi import Fred
from portwine.data.providers.loader_adapters import MarketDataLoader


class FREDMarketDataLoader(MarketDataLoader):
    """
    Loads data from the FRED (Federal Reserve Economic Data) system.

    This loader functions as a standard MarketDataLoader but with added capabilities:
    1. It can load existing parquet files from a specified directory
    2. If a file is not found and save_missing=True, it will attempt to download
       the data from FRED using the provided API key
    3. Downloaded data is saved as parquet for future use

    This is particularly useful for accessing economic indicators like interest rates,
    GDP, inflation metrics, and other macroeconomic data needed for advanced strategies.

    Parameters
    ----------
    data_path : str
        Directory where parquet files are stored
    api_key : str, optional
        FRED API key for downloading missing data
    save_missing : bool, default=False
        Whether to download and save missing data from FRED
    transform_to_daily : bool, default=True
        Convert non-daily data to daily frequency using forward-fill
    """

    # Source identifier for AlternativeMarketDataLoader
    SOURCE_IDENTIFIER = 'FRED'

    def __init__(self, data_path, api_key=None, save_missing=False, transform_to_daily=True):
        """
        Initialize the FRED market data loader.
        """
        super().__init__()
        self.data_path = data_path
        self.api_key = api_key
        self.save_missing = save_missing
        self.transform_to_daily = transform_to_daily

        self._fred_client = None

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    @property
    def fred_client(self):
        """
        Lazy initialization of the FRED client.
        """
        if self._fred_client is None and self.api_key:
            self._fred_client = Fred(api_key=self.api_key)
        return self._fred_client

    def load_ticker(self, ticker):
        """
        Load data for a specific ticker from parquet file or download from FRED.

        Parameters
        ----------
        ticker : str
            FRED series identifier (e.g., 'FEDFUNDS', 'DTB3', 'CPIAUCSL')

        Returns
        -------
        pd.DataFrame
            DataFrame with daily date index and appropriate columns for the portwine framework
        """
        file_path = os.path.join(self.data_path, f"{ticker}.parquet")

        # Check if file exists and load it
        if os.path.isfile(file_path):
            try:
                df = pd.read_parquet(file_path)
                return self._format_dataframe(df, ticker)
            except Exception as e:
                print(f"Error loading data for {ticker}: {str(e)}")

        # If file doesn't exist and save_missing is enabled, download from FRED
        if self.save_missing and self.fred_client:
            try:
                print(f"Downloading data for {ticker} from FRED...")
                # Get data from FRED
                series = self.fred_client.get_series(ticker)

                if series is not None and not series.empty:
                    # Convert to DataFrame with correct column name
                    df = pd.DataFrame(series, columns=['close'])

                    # Save to parquet for future use
                    df.to_parquet(file_path)

                    return self._format_dataframe(df, ticker)
                else:
                    print(f"No data found on FRED for ticker: {ticker}")
            except Exception as e:
                print(f"Error downloading data for {ticker} from FRED: {str(e)}")

        return None

    def _format_dataframe(self, df, ticker):
        """
        Format the dataframe to match the expected structure for portwine.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with appropriate columns
        """
        # Make sure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Make sure the index name is 'date'
        df.index.name = 'date'

        # Convert index to date (not datetime)
        if hasattr(df.index, 'normalize'):
            df.index = df.index.normalize().to_period('D').to_timestamp()

        # If we have only a Series, convert to DataFrame with 'close' column
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df, columns=['close'])

        # Handle the case where the first column might be the values
        if 'close' not in df.columns and len(df.columns) >= 1:
            # Rename the first column to 'close'
            df = df.rename(columns={df.columns[0]: 'close'})

        # If we have frequency that's not daily and transform_to_daily is True
        if self.transform_to_daily:
            # Reindex to daily frequency with forward fill
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(date_range, method='ffill')

        # Ensure we have all required columns for portwine
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df

    def get_fred_info(self, ticker):
        """
        Get information about a FRED series.

        Parameters
        ----------
        ticker : str
            FRED series identifier

        Returns
        -------
        pd.Series
            Series containing information about the FRED series
        """
        if self.fred_client:
            try:
                return self.fred_client.get_series_info(ticker)
            except Exception as e:
                print(f"Error getting info for {ticker}: {str(e)}")
        return None

    def search_fred(self, text, limit=10):
        """
        Search for FRED series by text.

        Parameters
        ----------
        text : str
            Text to search for
        limit : int, default=10
            Maximum number of results to return

        Returns
        -------
        pd.DataFrame
            DataFrame with search results
        """
        if self.fred_client:
            try:
                return self.fred_client.search(text, limit=limit)
            except Exception as e:
                print(f"Error searching FRED for '{text}': {str(e)}")
        return None

```

# portwine/loaders/noisy.py

```py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple


class NoisyMarketDataLoader:
    """
    A market data loader that injects noise scaled by rolling volatility.

    This implementation adds noise to price returns, with the magnitude of noise
    proportional to the local volatility (measured as rolling standard deviation
    of returns). This ensures the noise adapts to different market regimes.

    Parameters
    ----------
    base_loader : object
        A base loader with load_ticker(ticker) and fetch_data(tickers) methods
    noise_multiplier : float, optional
        Base multiplier for the noise magnitude (default: 1.0)
    volatility_window : int, optional
        Window size in days for rolling volatility calculation (default: 21)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
            self,
            base_loader: Any,
            noise_multiplier: float = 1.0,
            volatility_window: int = 21,
            seed: Optional[int] = None
    ):
        self.base_loader = base_loader
        self.noise_multiplier = noise_multiplier
        self.volatility_window = volatility_window
        self._original_data: Dict[str, pd.DataFrame] = {}  # Cache for original data

        if seed is not None:
            np.random.seed(seed)

    def get_original_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get original data for a ticker from cache or load it.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with OHLCV data or None if not available
        """
        if ticker in self._original_data:
            return self._original_data[ticker]

        df = self.base_loader.load_ticker(ticker)
        if df is not None:
            self._original_data[ticker] = df.sort_index()
        return df

    def calculate_rolling_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate rolling standard deviation of returns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with OHLCV data

        Returns
        -------
        numpy.ndarray
            Array of rolling volatility values
        """
        # Calculate returns
        returns = df['close'].pct_change().values

        # Initialize volatility array
        n = len(returns)
        volatility = np.ones(n)  # Default to 1 for scaling

        # Calculate overall volatility for initial values
        overall_std = np.std(returns[1:])  # Skip the first NaN value
        if not np.isfinite(overall_std) or overall_std <= 0:
            overall_std = 0.01  # Fallback if volatility is zero or invalid

        # Fill initial window with overall volatility to avoid NaNs
        volatility[:self.volatility_window] = overall_std

        # Calculate rolling standard deviation
        for i in range(self.volatility_window, n):
            window_returns = returns[i - self.volatility_window + 1:i + 1]
            # Remove any NaN values
            window_returns = window_returns[~np.isnan(window_returns)]
            if len(window_returns) > 0:
                vol = np.std(window_returns)
                # Ensure we have positive volatility
                volatility[i] = vol if np.isfinite(vol) and vol > 0 else overall_std
            else:
                volatility[i] = overall_std

        return volatility

    def generate_noise(self, size: int, volatility_scaling: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate volatility-scaled noise for OHLC data.

        Parameters
        ----------
        size : int
            Number of days to generate noise for
        volatility_scaling : numpy.ndarray
            Array of volatility scaling factors

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing noise for open, high, low, and close
        """

        # Generate base zero-mean noise
        def generate_zero_mean_noise(size):
            raw_noise = np.random.normal(0, 1, size=size)
            if size > 1:
                # Ensure perfect zero mean to avoid drift
                raw_noise = raw_noise - np.mean(raw_noise)
            return raw_noise

        # Generate noise for each price component and scale by volatility
        noise_open = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_high = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_low = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier
        noise_close = generate_zero_mean_noise(size) * volatility_scaling * self.noise_multiplier

        return noise_open, noise_high, noise_low, noise_close

    def inject_noise(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-scaled noise to OHLC data while preserving high/low consistency.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pandas.DataFrame
            DataFrame with OHLCV data

        Returns
        -------
        pandas.DataFrame
            Copy of input DataFrame with adaptive noise added to OHLC values
        """
        if df is None or len(df) < 2:
            return df.copy() if df is not None else None

        df_result = df.copy()

        # Compute volatility scaling factors
        volatility_scaling = self.calculate_rolling_volatility(df)

        # Extract original values for vectorized operations
        orig_open = df['open'].values
        orig_high = df['high'].values
        orig_low = df['low'].values
        orig_close = df['close'].values

        # Create arrays for new values
        new_open = np.empty_like(orig_open)
        new_high = np.empty_like(orig_high)
        new_low = np.empty_like(orig_low)
        new_close = np.empty_like(orig_close)

        # First day remains unchanged
        new_open[0] = orig_open[0]
        new_high[0] = orig_high[0]
        new_low[0] = orig_low[0]
        new_close[0] = orig_close[0]

        # Pre-generate all noise at once with volatility scaling
        n_days = len(df) - 1
        noise_open, noise_high, noise_low, noise_close = self.generate_noise(n_days, volatility_scaling[1:])

        # Process each day
        for i in range(1, len(df)):
            prev_orig_close = orig_close[i - 1]
            prev_new_close = new_close[i - 1]

            # Calculate returns relative to previous close
            r_open = (orig_open[i] / prev_orig_close) - 1
            r_high = (orig_high[i] / prev_orig_close) - 1
            r_low = (orig_low[i] / prev_orig_close) - 1
            r_close = (orig_close[i] / prev_orig_close) - 1

            # Add noise to returns and calculate new prices
            tentative_open = prev_new_close * (1 + r_open + noise_open[i - 1])
            tentative_high = prev_new_close * (1 + r_high + noise_high[i - 1])
            tentative_low = prev_new_close * (1 + r_low + noise_low[i - 1])
            tentative_close = prev_new_close * (1 + r_close + noise_close[i - 1])

            # Ensure high is the maximum and low is the minimum
            new_open[i] = tentative_open
            new_close[i] = tentative_close
            new_high[i] = max(tentative_open, tentative_high, tentative_low, tentative_close)
            new_low[i] = min(tentative_open, tentative_high, tentative_low, tentative_close)

        # Update the result DataFrame
        df_result['open'] = new_open
        df_result['high'] = new_high
        df_result['low'] = new_low
        df_result['close'] = new_close

        return df_result

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load ticker data and inject volatility-scaled noise.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with noisy OHLCV data or None if not available
        """
        df = self.get_original_ticker_data(ticker)
        if df is None:
            return None
        return self.inject_noise(ticker, df)

    def fetch_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers and inject volatility-scaled noise.

        Parameters
        ----------
        tickers : list of str
            List of ticker symbols

        Returns
        -------
        dict
            Dictionary mapping ticker symbols to DataFrames with noisy data
        """
        result = {}
        for ticker in tickers:
            df_noisy = self.load_ticker(ticker)
            if df_noisy is not None:
                result[ticker] = df_noisy
        return result

```

# portwine/loaders/polygon.py

```py
"""
Polygon market data loader for the portwine framework.

This module provides a MarketDataLoader implementation for fetching data
from the Polygon.io API, supporting both historical daily data and current
partial day data.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests
import pandas as pd
import pytz

from portwine.data.providers.loader_adapters import MarketDataLoader

# Configure logging
logger = logging.getLogger(__name__)

# API URLs
POLYGON_BASE_URL = "https://api.polygon.io"


class PolygonMarketDataLoader(MarketDataLoader):
    """
    Market data loader for Polygon.io API.
    
    This loader fetches historical daily data and current partial day data
    from Polygon.io API using direct REST calls.
    
    Parameters
    ----------
    api_key : str, optional
        Polygon API key. If not provided, attempts to read from POLYGON_API_KEY env var.
    data_dir : str
        Directory where historical data files are stored.
    timezone : str
        Timezone for the data. Default is "America/New_York".
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = "data",
        timezone: str = "America/New_York"
    ):
        """Initialize Polygon market data loader."""
        super().__init__()
        
        # Use environment variable if not provided
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            logger.warning("Polygon API key not provided. Will raise error if fetching historical data.")
            raise ValueError(
                "Polygon API key must be provided either as argument or POLYGON_API_KEY env var.")
        
        # Base URL for API requests
        self.base_url = POLYGON_BASE_URL
        
        # Create requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Data directory
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # In-memory cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Latest data cache for partial day data
        self._latest_data_cache: Dict[str, Dict] = {}
        self._latest_data_timestamp = None
        
        # Cache for last valid data used in ffill
        self._last_valid_data: Optional[Dict] = None
        
        # Timezone
        self.timezone = timezone

    def _api_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Helper method to make authenticated GET requests to Polygon API
        
        Parameters
        ----------
        url : str
            API endpoint URL (starting with /)
        params : Dict[str, Any], optional
            Query parameters for the request
            
        Returns
        -------
        Any
            JSON response data
            
        Raises
        ------
        Exception
            If API request fails
        """
        if url is None:
            raise ValueError("URL cannot be None")
            
        response = None
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            raise

    def _get_data_path(self, ticker: str) -> str:
        """
        Get path to data file for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        str
            Path to data file
            
        Raises
        ------
        ValueError
            If ticker contains invalid characters for a filename
        """
        # Check for invalid characters in ticker
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in ticker for char in invalid_chars):
            raise ValueError(f"Ticker {ticker} contains invalid characters for a filename")
            
        return os.path.join(self.data_dir, f"{ticker}.parquet")
    
    def _load_from_disk(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from disk if available.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame if data exists on disk, None otherwise
        """
        data_path = self._get_data_path(ticker)
        if os.path.exists(data_path):
            try:
                return pd.read_parquet(data_path)
            except Exception as e:
                logger.warning(f"Error loading data for {ticker}: {e}")
        else:
            logger.warning(f"Error loading data for {ticker}: File not found")
        
        return None
    
    def _save_to_disk(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save data to disk.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
        df : pd.DataFrame
            Data to save
        """
        data_path = self._get_data_path(ticker)
        try:
            df.to_parquet(data_path)
        except Exception as e:
            logger.warning(f"Error saving data for {ticker}: {e}")

    def _validate_and_convert_dates(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Validate and convert date inputs to proper format.
        
        Parameters
        ----------
        from_date : str, optional
            Start date in YYYY-MM-DD format or millisecond timestamp
        to_date : str, optional
            End date in YYYY-MM-DD format or millisecond timestamp
            
        Returns
        -------
        tuple[str, str]
            Tuple of (from_date, to_date) in proper format for API
            
        Raises
        ------
        ValueError
            If dates are malformed or if to_date is before from_date
        """
        # Set default dates if not provided
        today = datetime.now(pytz.UTC)
        if from_date is None:
            from_date = str(int(datetime.timestamp(today - timedelta(days=365 * 2)) * 1000))
        if to_date is None:
            to_date = str(int(datetime.timestamp(today) * 1000))
            
        # If date is in YYYY-MM-DD format, convert to milliseconds
        if "-" in str(from_date):
            try:
                dt = datetime.strptime(from_date, "%Y-%m-%d")
                dt = dt.replace(tzinfo=pytz.UTC)  # Make timezone-aware as UTC
                from_date = str(int(dt.timestamp() * 1000))
            except ValueError:
                raise ValueError(f"from_date must be in YYYY-MM-DD format or millisecond timestamp. Got {from_date}")
                
        if "-" in str(to_date):
            try:
                dt = datetime.strptime(to_date, "%Y-%m-%d")
                dt = dt.replace(tzinfo=pytz.UTC)  # Make timezone-aware as UTC
                to_date = str(int(dt.timestamp() * 1000))
            except ValueError:
                raise ValueError(f"to_date must be in YYYY-MM-DD format or millisecond timestamp. Got {to_date}")
        
        # Validate millisecond timestamps
        try:
            from_ms = int(from_date)
            to_ms = int(to_date)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid millisecond timestamps. Got from_date={from_date}, to_date={to_date}")
            
        # Check order
        if to_ms < from_ms:
            raise ValueError(f"to_date ({to_date}) must be after from_date ({from_date})")
            
        return from_date, to_date

    def fetch_historical_data(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a given ticker and date range from Polygon.io API.
        
        Parameters
        ----------
        ticker : str
            The stock ticker symbol to fetch data for
        from_date : str, optional
            Start date in YYYY-MM-DD format or millisecond timestamp. If None, defaults to 2 years ago.
        to_date : str, optional
            End date in YYYY-MM-DD format or millisecond timestamp. If None, defaults to today.
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data if successful, None if error occurs
            
        Raises
        ------
        ValueError
            If dates are malformed or if to_date is before from_date
            If API key is not provided
        """
        try:
            # Validate and convert dates
            from_date, to_date = self._validate_and_convert_dates(from_date, to_date)
            
            # Initialize list to store all results
            all_results = []
            
            # Construct initial API endpoint
            endpoint = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
            
            # Fetch data with pagination
            while endpoint:
                # Make API request
                response_data = self._api_get(endpoint, params={"adjusted": "true", "sort": "asc"})
                
                # Process response
                if response_data and response_data.get("results"):
                    all_results.extend(response_data["results"])
                    
                    # Get next URL for pagination
                    endpoint = response_data.get("next_url")
                else:
                    break
            
            if not all_results:
                logger.warning(f"No data returned for {ticker} from {from_date} to {to_date}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(all_results)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                "v": "volume",
                "o": "open",
                "c": "close",
                "h": "high",
                "l": "low",
                "t": "timestamp"
            })
            
            # Convert timestamp from milliseconds to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(self.timezone)
            df.set_index("timestamp", inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Cache the data
            self._data_cache[ticker] = df
            
            # Save to disk
            self._save_to_disk(ticker, df)
            
            logger.info(f"Successfully fetched historical data for {ticker} from {from_date} to {to_date}")
            return df

        except ValueError:
            # Re-raise ValueError exceptions (invalid dates)
            raise
        except Exception as e:
            # Log and return None for all other exceptions
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def _fetch_partial_day_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch current day's partial data from Polygon API.
        """
        try:
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            now_ms = int(now.timestamp() * 1000)
            from_ms = now_ms - (24 * 60 * 60 * 1000)  # 24 hours ago
            # Use path parameters for from and to
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute/{from_ms}/{now_ms}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000
            }
            response = self._api_get(url, params)
            if 'results' in response:
                today_bars = [
                    bar for bar in response['results']
                    if est.localize(datetime.fromtimestamp(bar['t'] / 1000)).hour >= 9
                    and est.localize(datetime.fromtimestamp(bar['t'] / 1000)).hour < 16
                ]
                if not today_bars:
                    return None
                first_bar = today_bars[0]
                today_open = first_bar['o']
                high = max(bar['h'] for bar in today_bars)
                low = min(bar['l'] for bar in today_bars)
                close = today_bars[-1]['c']
                volume = sum(bar['v'] for bar in today_bars)
                return {
                    "open": float(today_open),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume)
                }
        except Exception as e:
            logger.error(f"Error fetching partial day data for {ticker}: {e}")
            if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 404:
                return None
            return None
        return None

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data for a ticker from memory cache or disk.
        This method only reads from cache/disk and does not fetch new data.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if data is not found
        """
        # Check in-memory cache first
        if ticker in self._data_cache:
            df = self._data_cache[ticker]
        else:
            df = self._load_from_disk(ticker)
            if df is not None:
                self._data_cache[ticker] = df
        if df is not None:
            # Ensure index is timezone-aware and matches self.timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone)
            elif str(df.index.tz) != str(pytz.timezone(self.timezone)):
                df.index = df.index.tz_convert(self.timezone)
        return df

    def next(self, tickers: List[str], timestamp: pd.Timestamp, ffill: bool = False) -> Dict[str, Dict]:
        """
        Get data for tickers at or immediately before timestamp.
        For current day, returns partial day data.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to get data for
        timestamp : pd.Timestamp or datetime
            Timestamp to get data for
        ffill : bool, optional
            If True, when a ticker has no data, use the last non-None ticker's data.
            If False, return None for tickers with no data.
            Default is False.
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping ticker symbols to their OHLCV data or None
        """
        # Convert timestamp to pandas Timestamp if needed
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
            
        # Ensure timestamp is timezone-aware and in correct timezone
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(self.timezone)
        else:
            timestamp = timestamp.tz_convert(self.timezone)
            
        result = {}
        now = pd.Timestamp.now(tz=self.timezone)
        
        if timestamp.date() == now.date():
            for ticker in tickers:
                bar_data = self._fetch_partial_day_data(ticker)
                if bar_data:
                    result[ticker] = bar_data
                    if ffill:
                        self._last_valid_data = bar_data
                else:
                    if ffill and self._last_valid_data is not None:
                        result[ticker] = self._last_valid_data
                    else:
                        result[ticker] = None
        else:
            for ticker in tickers:
                df = self.load_ticker(ticker)
                if df is None:
                    result[ticker] = self._last_valid_data if ffill else None
                    continue
                    
                # Ensure DataFrame index is timezone-aware and matches timestamp timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize(self.timezone)
                elif str(df.index.tz) != str(timestamp.tz):
                    df.index = df.index.tz_convert(self.timezone)
                    
                bar = self._get_bar_at_or_before(df, timestamp)
                if bar is None:
                    result[ticker] = self._last_valid_data if ffill else None
                    continue
                    
                result[ticker] = {
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar["volume"]),
                }
                if ffill:
                    self._last_valid_data = result[ticker]
        return result

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
```

# portwine/logging.py

```py
# portwine/logging.py
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich import print as rprint
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from typing import Dict, List
from portwine.brokers.base import Order

class Logger:
    """
    Custom logger that outputs styled logs to the console using Rich
    and optionally writes to a rotating file handler.
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Path = None,
        rotate: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        propagate: bool = False,
    ):
        """
        Initialize and configure the logger.

        :param name: Name of the logger (usually __name__).
        :param level: Logging level.
        :param log_file: Path to the log file; if provided, file handler is added.
        :param rotate: Whether to use a rotating file handler.
        :param max_bytes: Maximum size of a log file before rotation (in bytes).
        :param backup_count: Number of rotated backup files to keep.
        :param propagate: Whether to propagate logs to parent loggers (default False).
        """
        # Create or get the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # Allow control over log propagation
        self.logger.propagate = propagate

        # Console handler with Rich
        console_handler = RichHandler(
            level=level,
            show_time=True,
            markup=True,
            rich_tracebacks=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            if rotate:
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)

            file_formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get(self) -> logging.Logger:
        """
        Return the configured standard logger instance.
        """
        return self.logger

    @classmethod
    def create(
        cls,
        name: str,
        level: int = logging.INFO,
        log_file: Path = None,
        rotate: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        propagate: bool = False,
    ) -> logging.Logger:
        """
        Convenience method to configure and return a logger in one step.
        """
        return cls(name, level, log_file, rotate, max_bytes, backup_count, propagate).get()

# Top-level rich-logging helpers
def log_position_table(logger: logging.Logger, current_positions: Dict[str, float], target_positions: Dict[str, float]) -> None:
    """Pretty-print position changes as a Rich table"""
    table = Table(title="Position Changes")
    table.add_column("Ticker")
    table.add_column("Current", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Change", justify="right")
    for t in sorted(set(current_positions) | set(target_positions)):
        curr = current_positions.get(t, 0)
        tgt = target_positions.get(t, 0)
        table.add_row(t, f"{curr:.4f}", f"{tgt:.4f}", f"{tgt-curr:.4f}")
    logger.info("Position changes:")
    rprint(table)

def log_weight_table(logger: logging.Logger, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> None:
    """Pretty-print weight changes as a Rich table"""
    table = Table(title="Weight Changes")
    table.add_column("Ticker")
    table.add_column("Current Wt", justify="right")
    table.add_column("Target Wt", justify="right")
    table.add_column("Delta Wt", justify="right")
    for t in sorted(set(current_weights) | set(target_weights)):
        cw = current_weights.get(t, 0)
        tw = target_weights.get(t, 0)
        table.add_row(t, f"{cw:.2%}", f"{tw:.2%}", f"{(tw-cw):.2%}")
    logger.info("Weight changes:")
    rprint(table)

def log_order_table(logger: logging.Logger, orders: List[Order]) -> None:
    """Pretty-print orders to execute as a Rich table"""
    table = Table(title="Orders to Execute")
    table.add_column("Ticker")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Type")
    table.add_column("TIF")
    table.add_column("Price", justify="right")
    for o in orders:
        table.add_row(o.ticker, o.side, str(int(o.quantity)), o.order_type, o.time_in_force, f"{o.average_price:.2f}")
    logger.info("Orders to execute:")
    rprint(table)

```

# portwine/optimizer.py

```py
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

class Optimizer:
    """
    Base/abstract optimizer class to allow different optimization approaches.
    """

    def __init__(self, backtester):
        """
        Parameters
        ----------
        backtester : Backtester
            A pre-initialized Backtester object with run_backtest(...).
        """
        self.backtester = backtester

    def run_optimization(self, strategy_class, param_grid, **kwargs):
        """
        Abstract method: run an optimization approach (grid search, train/test, etc.)
        Must be overridden.
        """
        raise NotImplementedError("Subclasses must implement run_optimization.")

# ---------------------------------------------------------------------

class TrainTestSplitOptimizer(Optimizer):
    """
    An optimizer that implements Approach A (two separate backtests):
      1) For each param combo, run a backtest on [train_start, train_end].
         Evaluate performance (score_fn).
      2) Choose best combo by training score.
      3) Optionally run a second backtest on [test_start, test_end] for
         the best combo, storing final out-of-sample performance.
    """

    def __init__(self, backtester):
        super().__init__(backtester)

    def run_optimization(self,
                         strategy_class,
                         param_grid,
                         split_frac=0.7,
                         score_fn=None,
                         benchmark=None):
        """
        Runs a grid search over 'param_grid' for the given 'strategy_class'.

        This approach does two separate backtests:
          - one on the train portion
          - one on the test portion for the best param set

        Parameters
        ----------
        strategy_class : type
            A strategy class (e.g. SomeStrategy) that can be constructed via
            the keys in param_grid.
        param_grid : dict
            param name -> list of possible values
        split_frac : float
            fraction of the data for training (0<split_frac<1)
        score_fn : callable
            function that takes { 'strategy_returns': Series, ... } -> float
            If None, defaults to annualized Sharpe on the daily returns.
        benchmark : str or None
            optional benchmark passed to run_backtest.

        Returns
        -------
        dict with:
          'best_params' : dict of chosen param set
          'best_score'  : float
          'results_df'  : DataFrame of all combos with train_score (and maybe test_score)
          'best_test_performance' : float or None
        """
        # Default scoring => annualized Sharpe
        if score_fn is None:
            def default_sharpe(res):
                # res = {'strategy_returns': Series of daily returns, ...}
                dr = res.get('strategy_returns', pd.Series(dtype=float))
                if len(dr) < 2:
                    return np.nan
                ann = 252.0
                mu = dr.mean()*ann
                sigma = dr.std()*np.sqrt(ann)
                return mu/sigma if sigma>1e-9 else 0.0
            score_fn = default_sharpe

        # The first step is to fetch the union of all relevant data to determine the date range.
        # We can do that by a "fake" strategy with "all combos" or just pick the first param set's tickers.
        # But it's safer to gather the earliest/largest date range among all combos. We'll do a simplified approach:
        all_dates = []
        # We'll keep a dictionary mapping from each combo -> (mindate, maxdate).
        # 1) If param_grid has a 'tickers' entry that is a list of lists, convert them to tuples:
        if "tickers" in param_grid:
            converted = []
            for item in param_grid["tickers"]:
                if isinstance(item, list):
                    converted.append(tuple(item))  # cast from list -> tuple
                else:
                    converted.append(item)
            param_grid["tickers"] = converted

        combos_date_range = {}

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))

        # We do a quick pass just to gather date range
        for combo in all_combos:
            combo_params = dict(zip(param_names, combo))
            # Make a strategy
            strat = strategy_class(**combo_params)
            if not hasattr(strat, 'tickers'):
                combos_date_range[combo] = (None, None)
                continue
            # fetch data
            data_dict = self.backtester.market_data_loader.fetch_data(strat.tickers)
            if not data_dict:
                combos_date_range[combo] = (None, None)
                continue
            # get min and max date across all relevant tickers
            min_d, max_d = None, None
            for tkr, df in data_dict.items():
                if df.empty:
                    continue
                dmin = df.index.min()
                dmax = df.index.max()
                if min_d is None or dmin>min_d:
                    min_d = dmin
                if max_d is None or dmax<max_d:
                    max_d = dmax
                # actually we want min_d = max of earliest
                # and max_d = min of latest
                # so let's invert:
                # if min_d is None => set it to dmin
                # else => min_d = max(min_d, dmin)
                # Similarly for max_d => min(max_d, dmax)
            combos_date_range[combo] = (min_d, max_d)

        # We'll pick the "largest common feasible range" among combos.
        # This is a design choice. Alternatively, we can do a combo-specific approach.
        # For simplicity, let's do a single union range for all combos.
        global_start = None
        global_end = None
        for c, (md, xd) in combos_date_range.items():
            if md is not None and xd is not None:
                if global_start is None or md>global_start:
                    global_start = md
                if global_end is None or xd<global_end:
                    global_end = xd
        if global_start is None or global_end is None or global_start>=global_end:
            print("No valid date range found for any combo.")
            return None

        # Now we have a global [global_start, global_end].
        # We'll pick a date range from that.
        # We'll use the naive approach: we convert them to sorted list of dates, pick split index, etc.
        # It's simpler to run two separate backtests with "start_date=..., end_date=..." for train, then for test.

        # Build a list of all daily timestamps from global_start..global_end
        # We can fetch from the underlying data in the backtester, or do a direct approach.
        # For simplicity, let's just do an approximate approach:
        # We'll gather the entire union in the same manner we do in the backtester:
        # But let's do it quickly:
        # We'll define a function get_all_dates_for_range:
        # Actually let's skip it, and do a simpler approach: we'll pass them to the backtester with end_date in train
        # and start_date in test.

        # We'll do the naive approach:
        # We want the full set of daily trading dates from global_start..global_end
        # Then we pick split. Then train is [global_start..some boundary], test is [boundary+1..global_end].
        # We'll do it with the backtester, but let's first gather a big combined df.

        # (In practice you might do a step to confirm we have a consistent set of daily trading dates. We'll do minimal approach.)
        date_range = pd.date_range(global_start, global_end, freq='B')  # business days
        n = len(date_range)
        split_idx = int(n*split_frac)
        if split_idx<1:
            print("Split fraction leaves no training days.")
            return None
        if split_idx>=n:
            print("Split fraction leaves no testing days.")
            return None
        train_end_date = date_range[split_idx-1]
        test_start_date = date_range[split_idx]
        # Summaries
        # We'll store combos results
        results_list = []

        for combo in tqdm(all_combos):
            combo_params = dict(zip(param_names, combo))
            # First => run training backtest
            strat_train = strategy_class(**combo_params)
            # train backtest
            train_res = self.backtester.run_backtest(
                strategy=strat_train,
                shift_signals=True,
                benchmark=benchmark,
                start_date=global_start,
                end_date=train_end_date
            )
            if not train_res or 'strategy_returns' not in train_res:
                results_list.append({**combo_params, "train_score": np.nan, "test_score": np.nan})
                continue
            train_dr = train_res['strategy_returns']
            if train_dr is None or len(train_dr)<2:
                results_list.append({**combo_params, "train_score": np.nan, "test_score": np.nan})
                continue

            # compute train score
            train_score = score_fn({"strategy_returns": train_dr})

            # We'll not do test backtest for each combo => to save time, do test only for best param after the loop
            # or if you want, you can do it here as well. We'll do it here so we can see how big the difference is for each combo.
            # But that can be expensive for big param grids. We'll do it anyway for completeness.

            strat_test = strategy_class(**combo_params)
            test_res = self.backtester.run_backtest(
                strategy=strat_test,
                shift_signals=True,
                benchmark=benchmark,
                start_date=test_start_date,
                end_date=global_end
            )
            if not test_res or 'strategy_returns' not in test_res:
                results_list.append({**combo_params, "train_score": train_score, "test_score": np.nan})
                continue
            test_dr = test_res['strategy_returns']
            if test_dr is None or len(test_dr)<2:
                results_list.append({**combo_params, "train_score": train_score, "test_score": np.nan})
                continue

            test_score = score_fn({"strategy_returns": test_dr})

            # store
            combo_result = {**combo_params,
                            "train_score": train_score,
                            "test_score": test_score}
            results_list.append(combo_result)

        df_results = pd.DataFrame(results_list)
        if df_results.empty:
            print("No results produced.")
            return None

        # pick best by train_score
        df_results.sort_values('train_score', ascending=False, inplace=True)
        best_row = df_results.iloc[0].to_dict()
        best_train_score = best_row['train_score']
        best_test_score = best_row['test_score']
        best_params = {k: v for k, v in best_row.items() if k not in ['train_score','test_score']}

        print("Best params:", best_params, f"train_score={best_train_score:.4f}, test_score={best_test_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_train_score,
            "results_df": df_results,
            "best_test_score": best_test_score
        }

```

# portwine/scheduler.py

```py
import pandas as pd
import pandas_market_calendars as mcal
import time
from datetime import datetime, date, timedelta, timezone
from typing import Iterator, Optional, List, Union


class DailySchedule(Iterator[int]):
    """
    Iterator of UNIX-ms timestamps for market events.

    Modes:
      - Finite: if end_date is set, yields events between start_date and end_date inclusive.
      - Live: if end_date is None, yields all future events from start_date (or today) onward, indefinitely.
    """

    def __init__(
        self,
        *,
        after_open_minutes: Optional[int] = None,
        before_close_minutes: Optional[int] = None,
        calendar_name: str = "NYSE",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        inclusive: bool = False,
    ):
        # Validate offsets
        if after_open_minutes is None and before_close_minutes is None:
            raise ValueError("Must specify after_open_minutes or before_close_minutes")
        if before_close_minutes is not None and interval_seconds is not None and after_open_minutes is None:
            raise ValueError("Cannot specify interval_seconds with close-only schedule")

        self.after_open = after_open_minutes
        self.before_close = before_close_minutes
        self.interval = interval_seconds
        self.inclusive = inclusive
        self.calendar = mcal.get_calendar(calendar_name)
        self.start_date = start_date
        self.end_date = end_date
        self._gen = None

    def __iter__(self):
        if self.end_date is not None:
            self._gen = self._finite_generator()
        else:
            self._gen = self._live_generator(self.start_date)
        return self

    def __next__(self) -> int:
        return next(self._gen)

    def _to_ms(self, ev: Union[datetime, pd.Timestamp]) -> int:
        # Normalize naive → UTC, leave tz-aware alone
        if isinstance(ev, pd.Timestamp):
            if ev.tzinfo is None:
                ev = ev.tz_localize("UTC")
        else:
            if ev.tzinfo is None:
                ev = ev.replace(tzinfo=timezone.utc)
        return int(ev.timestamp() * 1000)

    def _build_events(
        self,
        open_dt: Union[datetime, pd.Timestamp],
        close_dt: Union[datetime, pd.Timestamp],
    ) -> List[Union[datetime, pd.Timestamp]]:
        # Localize bounds
        if isinstance(open_dt, pd.Timestamp):
            if open_dt.tzinfo is None:
                open_dt = open_dt.tz_localize("UTC")
        else:
            if open_dt.tzinfo is None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)

        if isinstance(close_dt, pd.Timestamp):
            if close_dt.tzinfo is None:
                close_dt = close_dt.tz_localize("UTC")
        else:
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)

        # Compute window
        start_dt = open_dt if self.after_open is None else open_dt + timedelta(minutes=self.after_open)
        end_dt = close_dt if self.before_close is None else close_dt - timedelta(minutes=self.before_close)

        # Interval validation
        if self.interval is not None:
            window_secs = (end_dt - start_dt).total_seconds()
            if self.interval > window_secs:
                raise ValueError(f"interval_seconds={self.interval} exceeds session window of {window_secs:.0f}s")

        # Build events
        if self.after_open is None:
            # close-only
            return [end_dt]

        if self.before_close is None:
            # open-only or open+interval
            if self.interval is None:
                return [start_dt]
            events: List[Union[datetime, pd.Timestamp]] = []
            t = start_dt
            while t <= close_dt:
                events.append(t)
                t += timedelta(seconds=self.interval)
            return events

        # both open+close, with optional interval
        if self.interval is None:
            return [start_dt, end_dt]
        events = []
        t = start_dt
        last = None
        while t <= end_dt:
            events.append(t)
            last = t
            t += timedelta(seconds=self.interval)
        if self.inclusive and last and last < end_dt:
            events.append(end_dt)
        return events

    def _finite_generator(self):
        sched = self.calendar.schedule(start_date=self.start_date, end_date=self.end_date)
        for _, row in sched.iterrows():
            for ev in self._build_events(row["market_open"], row["market_close"]):
                yield self._to_ms(ev)

    def _live_generator(self, start_date=None):
        # If start_date is None, use today
        if start_date is not None:
            current_date = pd.Timestamp(start_date).date()
        else:
            current_date = date.today()
        # 2) determine tz from calendar
        today_str = date.today().isoformat()
        try:
            today_sched = self.calendar.schedule(start_date=today_str, end_date=today_str)
        except StopIteration:
            return
        tz = getattr(today_sched.index, "tz", None)
        # 3) current time from time.time()
        now_sec = time.time()
        now_ts = pd.Timestamp(now_sec, unit="s", tz=tz) if tz else pd.Timestamp(now_sec, unit="s")
        now_ms = int(now_ts.timestamp() * 1000)
        # 4) start looping from current_date
        while True:
            day_str = current_date.isoformat()
            try:
                day_sched = self.calendar.schedule(start_date=day_str, end_date=day_str)
            except StopIteration:
                return
            if not day_sched.empty:
                row = day_sched.iloc[0]
                for ev in self._build_events(row["market_open"], row["market_close"]):
                    ms = self._to_ms(ev)
                    if ms >= now_ms:
                        yield ms
                now_ms = -1
            current_date += timedelta(days=1)


def daily_schedule(
    after_open_minutes: Optional[int] = None,
    before_close_minutes: Optional[int] = None,
    calendar_name: str = "NYSE",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    inclusive: bool = False,
) -> Iterator[int]:
    """
    Backward‐compatible wrapper around DailySchedule.
    """
    return iter(
        DailySchedule(
            after_open_minutes=after_open_minutes,
            before_close_minutes=before_close_minutes,
            calendar_name=calendar_name,
            start_date=start_date,
            end_date=end_date,
            interval_seconds=interval_seconds,
            inclusive=inclusive,
        )
    )

```

# portwine/strategies/__init__.py

```py
from portwine.strategies.base import StrategyBase
from portwine.strategies.simple_moving_average import SimpleMovingAverageStrategy

```

# portwine/strategies/base.py

```py
from typing import Union, List, Set
from portwine.universe import Universe
from datetime import date
import abc

class StrategyBase(abc.ABC):
    """
    Base class for a trading strategy. Subclass this to implement a custom strategy.

    A 'step' method is called each day with that day's data. The method should return
    a dictionary of signals/weights for each ticker on that day.

    The strategy always uses a universe object internally. If you pass a list of tickers,
    it creates a static universe with those tickers from 1970-01-01 onwards.
    """

    def __init__(self, tickers: Union[List[str], Universe]):
        """
        Parameters
        ----------
        tickers : Union[List[str], Universe]
            Either a list of ticker symbols or a Universe object.
            If a list is provided, it creates a static universe with those tickers.
        """
        if isinstance(tickers, Universe):
            self.universe = tickers
            # Store all possible tickers for reference
            self.tickers = self.universe.all_tickers
        else:
            # Convert list to static universe
            self.universe = self._create_static_universe(tickers)
            # Store original tickers as set
            self.tickers = set(tickers)

    def _create_static_universe(self, tickers: List[str]) -> Universe:
        """
        Create a static universe from a list of tickers.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
            
        Returns
        -------
        Universe
            Static universe with tickers from 1970-01-01 onwards
        """
        # Remove duplicates and convert to set
        unique_tickers = set(tickers)
        
        # Create static universe mapping
        constituents = {date(1970, 1, 1): unique_tickers}
        
        return Universe(constituents)

    def step(self, current_date, daily_data):
        """
        Called each day with that day's data for each ticker.

        Parameters
        ----------
        current_date : pd.Timestamp
        daily_data : dict
            daily_data[ticker] = {
                'open': ..., 'high': ..., 'low': ...,
                'close': ..., 'volume': ...
            }
            or None if no data for that ticker on this date.

            The backtester ensures that daily_data only contains tickers
            that are currently in the universe.

        Returns
        -------
        signals : dict
            { ticker -> float weight }, where the weights are the fraction
            of capital allocated to each ticker (long/short).
        """
        ...

```

# portwine/strategies/rprtstrategy.py

```py

import numpy as np
from portwine.strategies.base import StrategyBase

class RPRTStrategy(StrategyBase):
    """
    Reweighted Price Relative Tracking (RPRT) Strategy

    This strategy implements the RPRT algorithm from the paper:
    "Reweighted Price Relative Tracking System for Automatic Portfolio Optimization"

    The strategy:
    1. Calculates price relatives for each asset
    2. Uses a reweighted price relative prediction system that adapts to each asset's performance
    3. Optimizes portfolio weights using a tracking system with a generalized increasing factor
    """

    def __init__(self, tickers, window_size=5, theta=0.8, epsilon=1.01):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to consider for investment
        window_size : int
            Window size for SMA calculations
        theta : float
            Mixing parameter for reweighted price relative prediction
        epsilon : float
            Expected profiting level for portfolio optimization
        """
        super().__init__(tickers)
        self.window_size = window_size
        self.theta = theta
        self.epsilon = epsilon

        # Initialize internal state for RPRT
        self.price_relatives_history = []
        self.phi_hat = None  # Reweighted price relative prediction
        self.current_portfolio = None
        self.price_history = {ticker: [] for ticker in tickers}
        self.has_initialized = False

    def step(self, current_date, daily_data):
        """
        Process daily data and determine allocations using RPRT

        Parameters
        ----------
        current_date : datetime
            Current backtesting date
        daily_data : dict
            Dictionary with price data for each ticker

        Returns
        -------
        dict
            Portfolio weights for each ticker
        """
        # Get today's close prices
        today_prices = {}
        for ticker in self.tickers:
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close')
                if price is not None:
                    today_prices[ticker] = price
                    # Update price history
                    self.price_history[ticker].append(price)
                elif len(self.price_history[ticker]) > 0:
                    # Forward fill missing prices
                    price = self.price_history[ticker][-1]
                    today_prices[ticker] = price
                    self.price_history[ticker].append(price)
            elif len(self.price_history[ticker]) > 0:
                # Forward fill missing prices
                price = self.price_history[ticker][-1]
                today_prices[ticker] = price
                self.price_history[ticker].append(price)

        # Need at least two days of data to calculate price relatives
        if not self.has_initialized:
            # On first day, initialize with equal weights
            if len(today_prices) > 0:
                weights = {ticker: 1.0 / len(today_prices) if ticker in today_prices else 0.0
                           for ticker in self.tickers}
                self.current_portfolio = np.array([weights.get(ticker, 0.0) for ticker in self.tickers])
                self.has_initialized = True
                return weights
            else:
                return {ticker: 0.0 for ticker in self.tickers}

        # Calculate price relatives (today's price / yesterday's price)
        yesterday_prices = {}
        for ticker in self.tickers:
            if len(self.price_history[ticker]) >= 2:
                yesterday_prices[ticker] = self.price_history[ticker][-2]

        price_relatives = []
        for ticker in self.tickers:
            if ticker in today_prices and ticker in yesterday_prices and yesterday_prices[ticker] > 0:
                price_relative = today_prices[ticker] / yesterday_prices[ticker]
            else:
                price_relative = 1.0  # No change for missing data
            price_relatives.append(price_relative)

        # Convert to numpy array
        price_relatives = np.array(price_relatives)

        # Update portfolio using RPRT algorithm
        new_portfolio = self.update_rprt(price_relatives)

        # Convert portfolio weights to dictionary
        weights = {ticker: weight for ticker, weight in zip(self.tickers, new_portfolio)}

        return weights

    def update_rprt(self, price_relatives):
        """
        Core RPRT algorithm for portfolio optimization

        Parameters
        ----------
        price_relatives : numpy.ndarray
            Array of price relatives for each ticker

        Returns
        -------
        numpy.ndarray
            Updated portfolio weights
        """
        # Store price relatives history
        self.price_relatives_history.append(price_relatives)

        # Only keep the recent window_size price relatives
        if len(self.price_relatives_history) > self.window_size:
            self.price_relatives_history.pop(0)

        # Step 1: Calculate SMA price relative prediction and diagonal matrix D
        xhat_sma = self._calculate_sma_prediction()
        D = np.diag(xhat_sma)

        # Step 2: Calculate the reweighted price relative prediction
        self._update_price_relative_prediction(price_relatives)

        # Step 3 & 4: Calculate lambda (step size)
        phi_hat_mean = np.mean(self.phi_hat)
        phi_hat_normalized = self.phi_hat - phi_hat_mean

        norm_squared = np.sum(phi_hat_normalized ** 2)

        if norm_squared == 0:
            lambda_hat = 0
        else:
            expected_profit = self.current_portfolio.dot(self.phi_hat)
            if expected_profit < self.epsilon:
                lambda_hat = (self.epsilon - expected_profit) / norm_squared
            else:
                lambda_hat = 0

        # Step 5: Optimization step
        b_next = self.current_portfolio + lambda_hat * D.dot(phi_hat_normalized)

        # Step 6: Projection onto simplex
        b_next = self._project_to_simplex(b_next)

        # Update the current portfolio
        self.current_portfolio = b_next

        return self.current_portfolio

    def _calculate_sma_prediction(self):
        """Calculate the SMA price relative prediction."""
        if len(self.price_relatives_history) < self.window_size:
            # If we don't have enough history, use the latest price relative
            return self.price_relatives_history[-1]

        # Calculate the SMA prediction using Equation (7) from the paper
        recent_prices = np.array(self.price_relatives_history)

        # Calculate the cumulative product of price relatives to get relative prices
        # This is equivalent to p_{t-k} / p_t in the paper
        relative_prices = np.cumprod(1.0 / recent_prices[::-1], axis=0)

        # Calculate SMA prediction according to Equation (7)
        xhat_sma = (1.0 / self.window_size) * (1 + np.sum(relative_prices[:-1], axis=0))

        return xhat_sma

    def _update_price_relative_prediction(self, price_relatives):
        """Update the reweighted price relative prediction."""
        if self.phi_hat is None:
            # Initialize phi_hat with the first price relative
            self.phi_hat = price_relatives
            return

        # Calculate gamma using Equation (13)
        gamma = (self.theta * price_relatives) / (self.theta * price_relatives + self.phi_hat)

        # Update phi_hat using Equation (12)
        self.phi_hat = gamma + (1 - gamma) * (self.phi_hat / price_relatives)

    def _project_to_simplex(self, b):
        """
        Project b onto the simplex.
        Ensures that all values in b are non-negative and sum to 1.
        """
        # Handle the case where b is already a valid distribution
        if np.all(b >= 0) and np.isclose(np.sum(b), 1.0):
            return b

        # Sort b in descending order
        b_sorted = np.sort(b)[::-1]

        # Calculate the cumulative sum
        cum_sum = np.cumsum(b_sorted)

        # Find the index where the projection condition is met
        indices = np.arange(1, len(b) + 1)
        is_greater = (b_sorted * indices) > (cum_sum - 1)

        if not np.any(is_greater):
            # If no element satisfies the condition, set rho to the last element
            rho = len(b) - 1
        else:
            rho = np.max(np.where(is_greater)[0])

        # Calculate the threshold value
        theta = (cum_sum[rho] - 1) / (rho + 1)

        # Project b onto the simplex
        return np.maximum(b - theta, 0)
```

# portwine/strategies/simple_moving_average.py

```py
"""
Simple Moving Average Strategy Implementation

This module provides a simple moving average crossover strategy that:
1. Calculates short and long moving averages for each ticker
2. Generates buy signals when short MA crosses above long MA
3. Generates sell signals when short MA crosses below long MA
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from portwine.strategies.base import StrategyBase

# Configure logging
logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy(StrategyBase):
    """
    Simple Moving Average Crossover Strategy
    
    This strategy:
    1. Calculates short and long moving averages for each ticker
    2. Generates buy signals when short MA crosses above long MA
    3. Generates sell signals when short MA crosses below long MA
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to trade
    short_window : int, default 20
        Short moving average window in days
    long_window : int, default 50
        Long moving average window in days
    position_size : float, default 0.1
        Position size as a fraction of portfolio (e.g., 0.1 = 10%)
    """
    
    def __init__(
        self, 
        tickers: List[str], 
        short_window: int = 20, 
        long_window: int = 50,
        position_size: float = 0.1,
        **kwargs
    ):
        """Initialize the strategy with parameters."""
        super().__init__(tickers)
        
        # Store parameters
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        
        # Validate parameters
        if short_window >= long_window:
            logger.warning("Short window should be smaller than long window")
        
        # Initialize price history
        self.price_history = {ticker: [] for ticker in tickers}
        self.dates = []
        
        # Current signals (allocations)
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        
        logger.info(f"Initialized SimpleMovingAverageStrategy with {len(tickers)} tickers")
        logger.info(f"Parameters: short_window={short_window}, long_window={long_window}, position_size={position_size}")
    
    def calculate_moving_averages(self, prices: List[float]) -> Dict[str, Optional[float]]:
        """
        Calculate short and long moving averages from price history.
        
        Parameters
        ----------
        prices : List[float]
            List of historical prices
            
        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary with short_ma and long_ma values, or None if not enough data
        """
        if len(prices) < self.long_window:
            return {"short_ma": None, "long_ma": None}
        
        # Calculate moving averages
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices[-self.long_window:]) / self.long_window
        
        return {"short_ma": short_ma, "long_ma": long_ma}
    
    def step(self, current_date: pd.Timestamp, daily_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Process daily data and generate trading signals.
        
        Parameters
        ----------
        current_date : pd.Timestamp
            Current trading date
        daily_data : Dict[str, Dict[str, Any]]
            Dictionary of ticker data for the current date
            
        Returns
        -------
        Dict[str, float]
            Dictionary of target weights for each ticker
        """
        # Track dates
        self.dates.append(current_date)
        
        # Update price history
        for ticker in self.tickers:
            price = None
            if ticker in daily_data and daily_data[ticker] is not None:
                price = daily_data[ticker].get('close')
            
            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]
            
            self.price_history[ticker].append(price)
        
        # Calculate signals for each ticker
        signals = {}
        for ticker in self.tickers:
            prices = self.price_history[ticker]
            
            # Skip tickers with None values
            if None in prices:
                signals[ticker] = 0.0
                continue
            
            # Calculate moving averages
            mas = self.calculate_moving_averages(prices)
            
            # Not enough data yet
            if mas["short_ma"] is None or mas["long_ma"] is None:
                signals[ticker] = 0.0
                continue
            
            # Get previous signal
            prev_signal = self.current_signals.get(ticker, 0.0)
            
            # Generate signal based on moving average crossover
            if mas["short_ma"] > mas["long_ma"]:
                # Bullish signal - short MA above long MA
                signals[ticker] = self.position_size
            else:
                # Bearish signal - short MA below long MA
                signals[ticker] = 0.0
            
            # Log signal changes
            if signals[ticker] != prev_signal:
                direction = "BUY" if signals[ticker] > 0 else "SELL"
                logger.info(f"{current_date}: {direction} signal for {ticker} - Short MA: {mas['short_ma']:.2f}, Long MA: {mas['long_ma']:.2f}")
        
        # Update current signals
        self.current_signals = signals.copy()
        
        return signals
    
    def generate_signals(self) -> Dict[str, float]:
        """
        Generate current trading signals.
        
        This method is used by the DailyExecutor to get the current signals.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of target weights for each ticker
        """
        return self.current_signals.copy()
    
    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down SimpleMovingAverageStrategy") 
```

# portwine/universe.py

```py
"""
Simple universe management for historical constituents.
"""

from typing import List, Dict, Set
from datetime import date


class Universe:
    """
    Base universe class for managing historical constituents.
    
    This class provides efficient lookup of constituents at any given date
    using binary search on pre-sorted dates.
    """

    def __init__(self, constituents: Dict[date, Set[str]]):
        """
        Initialize universe with constituent mapping.
        
        Parameters
        ----------
        constituents : Dict[date, Set[str]]
            Dictionary mapping dates to sets of ticker symbols
        """
        self.constituents = constituents
        
        # Pre-sort dates for binary search
        self.sorted_dates = sorted(self.constituents.keys())
        
        # Pre-compute all tickers
        self._all_tickers = self._compute_all_tickers()
    
    def get_constituents(self, dt) -> Set[str]:
        """
        Get the basket for a given date.
        
        Parameters
        ----------
        dt : datetime-like
            Date to get constituents for
            
        Returns
        -------
        Set[str]
            Set of tickers in the basket at the given date
        """
        # Convert to date object
        if hasattr(dt, 'date'):
            target_date = dt.date()
        else:
            target_date = date.fromisoformat(str(dt).split()[0])
        
        # Binary search to find the most recent date <= target_date
        left, right = 0, len(self.sorted_dates) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if self.sorted_dates[mid] <= target_date:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if result == -1:
            return set()
            
        return self.constituents[self.sorted_dates[result]]
    
    def _compute_all_tickers(self) -> set:
        """
        Compute all unique tickers that have ever been in the universe.
        
        Returns
        -------
        set
            Set of all ticker symbols
        """
        all_tickers = set()
        for tickers in self.constituents.values():
            all_tickers.update(tickers)
        return all_tickers
    
    @property
    def all_tickers(self) -> set:
        """
        Get all unique tickers that have ever been in the universe.
        
        Returns
        -------
        set
            Set of all ticker symbols
        """
        return self._all_tickers


class CSVUniverse(Universe):
    """
    Universe class that loads constituent data from CSV files.
    
    Expected CSV format:
    date,ticker1,ticker2,ticker3,...
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize universe from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with format: date,ticker1,ticker2,ticker3,...
        """
        constituents = self._load_from_csv(csv_path)
        super().__init__(constituents)
    
    def _load_from_csv(self, csv_path: str) -> Dict[date, Set[str]]:
        """
        Load constituent data from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file
            
        Returns
        -------
        Dict[date, Set[str]]
            Dictionary mapping dates to sets of tickers
        """
        constituents = {}
        
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                    
                # Parse date
                date_str = parts[0].strip()
                try:
                    year, month, day = map(int, date_str.split('-'))
                    current_date = date(year, month, day)
                except ValueError:
                    continue  # Skip invalid dates
                
                # Parse tickers (skip empty ones) and convert to set
                tickers = {ticker.strip() for ticker in parts[1:] if ticker.strip()}
                constituents[current_date] = tickers
        
        return constituents
```

# portwine/vectorized.py

```py
"""
Vectorized strategy base class and updated backtester implementation.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from portwine.strategies import StrategyBase
from portwine.backtester import Backtester, STANDARD_BENCHMARKS
from typing import Dict, List, Optional, Tuple, Set, Union
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def calculate_returns(price_array):
    """Calculate returns with Numba acceleration."""
    n_rows, n_cols = price_array.shape
    returns = np.zeros((n_rows-1, n_cols), dtype=np.float32)
    for i in range(1, n_rows):
        for j in range(n_cols):
            returns[i-1, j] = price_array[i, j] / price_array[i-1, j] - 1.0
    return returns

@nb.njit(parallel=True, fastmath=True)
def apply_weights(returns, weights):
    """Calculate weighted returns with Numba acceleration."""
    n_rows, n_cols = returns.shape
    result = np.zeros(n_rows)
    for i in range(n_rows):
        for j in range(n_cols):
            result[i] += returns[i, j] * weights[i, j]
    return result


def create_price_dataframe(market_data_loader, tickers, start_date=None, end_date=None):
    """
    Create a DataFrame of prices from a market data loader.

    Parameters:
    -----------
    market_data_loader : MarketDataLoader
        Market data loader from portwine
    tickers : list[str]
        List of ticker symbols to include
    start_date : datetime or str, optional
        Start date for data extraction
    end_date : datetime or str, optional
        End date for data extraction

    Returns:
    --------
    DataFrame
        DataFrame with dates as index and tickers as float columns
    """
    # 1) fetch raw data
    data_dict = market_data_loader.fetch_data(tickers)

    # 2) collect all dates
    all_dates = set()
    for df in data_dict.values():
        if df is not None and not df.empty:
            all_dates.update(df.index)
    all_dates = sorted(all_dates)

    # 3) apply optional date filters
    # all_dates_array = np.array(all_dates)
    all_dates_array = pd.to_datetime(np.array(all_dates))

    mask = np.ones(len(all_dates_array), dtype=bool)  # Start with all True
    

    # if start_date:
    #     sd = pd.to_datetime(start_date).to_numpy()
    #     mask &= (all_dates_array >= sd)
        
    # if end_date:
    #     ed = pd.to_datetime(end_date).to_numpy()
    #     mask &= (all_dates_array <= ed)

    if start_date:
        sd = pd.to_datetime(start_date)
        mask &= (all_dates_array >= sd)
    if end_date:
        ed = pd.to_datetime(end_date)
        mask &= (all_dates_array <= ed)

    # Apply the combined mask
    all_dates = all_dates_array[mask].tolist()

    # 4) build empty float DataFrame
    df_prices = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)

    # 5) fill in the close prices (alignment by index)
    for ticker, df in data_dict.items():
        if df is not None and not df.empty and ticker in tickers:
            df_prices[ticker] = df['close']

    # 6) forward‐fill across a float‐typed DataFrame → no downcasting warning
    df_prices = df_prices.ffill()

    # 7) drop any dates where we have no data at all
    df_prices = df_prices.dropna(how='all')

    return df_prices


class VectorizedStrategyBase(StrategyBase):
    """
    Base class for vectorized strategies that process the entire dataset at once.
    Subclasses must implement batch() to return a float‐typed weights DataFrame.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.prices_df = None
        self.weights_df = None

    def batch(self, prices_df):
        raise NotImplementedError("Subclasses must implement batch()")

    def step(self, current_date, daily_data):
        if self.weights_df is None or current_date not in self.weights_df.index:
            # fallback to equal weight
            return {t: 1.0 / len(self.tickers) for t in self.tickers}
        row = self.weights_df.loc[current_date]
        return {t: float(w) for t, w in row.items()}


class VectorizedBacktester:
    """
    A vectorized backtester that processes the entire dataset at once.
    """
    def __init__(self, market_data_loader=None):
        self.market_data_loader = market_data_loader


    def run_backtest(
        self,
        strategy: VectorizedStrategyBase,
        benchmark="equal_weight",
        start_date=None,
        end_date=None,
        shift_signals=True,
        require_all_history: bool = False,
        verbose=False
    ):
        # 0) type check
        if not isinstance(strategy, VectorizedStrategyBase):
            raise TypeError("Strategy must be a VectorizedStrategyBase")

        # 1) load full history of prices (float dtype)
        full_prices = create_price_dataframe(
            self.market_data_loader,
            tickers=strategy.tickers,
            start_date=start_date,
            end_date=end_date
        )

        # 2) compute all weights in one shot
        if verbose:
            print("Computing strategy weights…")
        all_weights = strategy.batch(full_prices)

        # 3) require that all tickers have data from a common start date?
        if require_all_history:
            # find first valid (non-NaN) date for each ticker
            first_dates = [full_prices[t].first_valid_index() for t in strategy.tickers]
            if any(d is None for d in first_dates):
                raise ValueError("Not all tickers have any data in the supplied range")
            common_start = max(first_dates)
            # truncate both prices and weights
            full_prices = full_prices.loc[full_prices.index >= common_start]
            all_weights = all_weights.loc[all_weights.index >= common_start]

        # 4) align dates between prices and weights
        common_idx = full_prices.index.intersection(all_weights.index)
        price_df = full_prices.loc[common_idx]
        weights_df = all_weights.loc[common_idx]

        # 5) shift signals if requested
        if shift_signals:
            weights_df = weights_df.shift(1).fillna(0.0)

        # 6) compute per‐ticker returns
        returns_df = price_df.pct_change(fill_method=None).fillna(0.0)

        # 7) strategy P&L
        strategy_rets = (returns_df * weights_df).sum(axis=1)

        # 8) benchmark
        benchmark_rets = None
        if benchmark is not None:
            if isinstance(benchmark, str) and benchmark in STANDARD_BENCHMARKS:
                benchmark_rets = STANDARD_BENCHMARKS[benchmark](returns_df)
            elif isinstance(benchmark, str) and self.market_data_loader:
                raw = self.market_data_loader.fetch_data([benchmark])
                series = raw.get(benchmark)
                if series is not None:
                    bm = series['close'].reindex(common_idx).ffill()
                    benchmark_rets = bm.pct_change(fill_method=None).fillna(0)
                    benchmark_rets.name = None  # Reset the name
            elif callable(benchmark):
                benchmark_rets = benchmark(returns_df)

        return {
            'signals_df': weights_df,
            'tickers_returns': returns_df,
            'strategy_returns': strategy_rets,
            'benchmark_returns': benchmark_rets,
        }


def benchmark_equal_weight(returns_df: pd.DataFrame) -> pd.Series:
    return returns_df.mean(axis=1)



def load_price_matrix(loader, tickers, start_date=None, end_date=None):
    """
    Optimized price matrix loader that minimizes pandas-numpy conversions.
    """
    # 1) Fetch raw data
    data_dict = loader.fetch_data(tickers)

    # 2) Collect all dates directly as numpy array
    all_dates_set = set()
    for df in data_dict.values():
        if df is not None and not df.empty:
            all_dates_set.update(df.index.values)

    all_dates_array = np.array(sorted(all_dates_set))

    # 3) Apply date filters in numpy
    mask = np.ones(len(all_dates_array), dtype=bool)
    if start_date:
        sd = pd.to_datetime(start_date)
        if len(all_dates_array) > 0:
            mask &= (all_dates_array >= sd)
    if end_date:
        ed = pd.to_datetime(end_date)
        if len(all_dates_array) > 0:
            mask &= (all_dates_array <= ed)

    all_dates_array = all_dates_array[mask]

    # Create a date-to-index mapping for fast lookups
    date_to_idx = {d: i for i, d in enumerate(all_dates_array)}

    # 4) Pre-allocate price matrix directly
    n_dates = len(all_dates_array)
    n_tickers = len(tickers)
    price_matrix = np.full((n_dates, n_tickers), np.nan, dtype=np.float32)

    # 5) Fill matrix directly without pandas intermediates
    for t_idx, ticker in enumerate(tickers):
        df = data_dict.get(ticker)
        if df is not None and not df.empty:
            # Get close prices as numpy array
            prices = df['close'].values

            # Get dates as numpy array
            dates = df.index.values

            # For each date in this ticker's data, find its position in our matrix
            for date_idx, date in enumerate(dates):
                if date in date_to_idx:
                    price_matrix[date_to_idx[date], t_idx] = prices[date_idx]

    # 6) Forward fill using numpy operations
    for col in range(n_tickers):
        mask = np.isnan(price_matrix[:, col])
        # Find first valid index
        valid_indices = np.where(~mask)[0]
        if len(valid_indices) > 0:
            # Forward fill
            for i in range(valid_indices[0], n_dates):
                if mask[i]:
                    # Find the last valid value
                    last_valid = np.where(~mask[:i])[0]
                    if len(last_valid) > 0:
                        price_matrix[i, col] = price_matrix[last_valid[-1], col]

    # 7) Compute returns directly in numpy
    returns_matrix = np.zeros_like(price_matrix[1:])
    returns_matrix = (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1]

    # 8) Create a minimal pandas DataFrame only for reference
    # This is just for API compatibility and doesn't get used in computations
    price_df = pd.DataFrame(price_matrix, index=all_dates_array, columns=tickers)

    return price_matrix, returns_matrix, all_dates_array[1:], price_df
class NumPyVectorizedStrategyBase(StrategyBase):
    """
    Base class for vectorized strategies that process the entire dataset at once
    using NumPy arrays for optimal performance.
    """
    def __init__(self, tickers: List[str]):
        """
        Initialize with the tickers this strategy uses.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols this strategy will use
        """
        self.tickers = tickers
        
    def batch(self, price_matrix: np.ndarray, dates: List[pd.Timestamp], 
              column_indices: List[int]) -> np.ndarray:
        """
        Compute weights for all dates based on price history.
        Must be implemented by subclasses.
        
        Parameters:
        -----------
        price_matrix : np.ndarray
            Price matrix with shape (n_dates, n_tickers)
        dates : List[pd.Timestamp]
            List of dates corresponding to rows in price_matrix
        column_indices : List[int]
            List of column indices in price_matrix that correspond to this strategy's tickers
            
        Returns:
        --------
        np.ndarray
            Weight matrix with shape (n_dates, n_strategy_tickers)
        """
        raise NotImplementedError("Subclasses must implement batch()")
    
    def step(self, current_date, daily_data):
        """
        Compatibility method for use with traditional backtester.
        This should generally not be used directly - prefer batch processing.
        """
        raise NotImplementedError("NumPyVectorizedStrategyBase is designed for batch processing")


class NumpyVectorizedBacktester:
    """
    A highly optimized NumPy-based vectorized backtester that supports
    strategies using subsets of tickers.
    """
    def __init__(self, loader, universe_tickers: List[str], start_date: str, end_date: str):
        """
        Initialize with minimal pandas-numpy conversions.
        """
        price_matrix, returns_matrix, dates_ret, price_df = load_price_matrix(
            loader, universe_tickers, start_date, end_date
        )
        
        # Store everything as numpy arrays
        self.price_matrix = price_matrix
        self.returns_matrix = returns_matrix
        self.dates_array = dates_ret  # store as numpy array
        
        # Create mappings for lookups
        self.universe_tickers = universe_tickers
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(universe_tickers)}
        
        # Keep minimal pandas objects
        self.date_to_i = None  # Don't create this dictionary unless needed
        self.price_df = None   # Don't store pandas objects
        
        # Keep reference to loader for benchmark calculations
        self.loader = loader

    def get_indices_for_tickers(self, tickers: List[str]) -> List[int]:
        """
        Get the column indices in the price/returns matrices for the given tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to get indices for
            
        Returns:
        --------
        List[int]
            List of column indices
        """
        return [self.ticker_to_idx.get(ticker) for ticker in tickers 
                if ticker in self.ticker_to_idx]

    def run_backtest(self, strategy, benchmark="equal_weight", shift_signals=True, verbose=False, **kwargs):
        """
        Run backtest with minimal pandas-numpy conversions.
        """
        # Get strategy info
        strategy_indices = np.array([
            self.ticker_to_idx.get(t, -1) for t in strategy.tickers
        ])
        strategy_indices = strategy_indices[strategy_indices >= 0]
        
        if len(strategy_indices) == 0:
            raise ValueError(f"None of the strategy tickers {strategy.tickers} are in the universe")
        
        # Extract price submatrix - avoid pandas operations
        strategy_price_matrix = self.price_matrix[:, strategy_indices]
        
        # Get weights - working entirely in numpy
        if verbose:
            print(f"Computing weights for {len(strategy_indices)} tickers...")
        
        # Get strategy weights as numpy array
        weights_matrix = strategy.batch(
            strategy_price_matrix, 
            self.dates_array, 
            strategy_indices
        )
        
        # Prepare benchmark weights - all in numpy
        benchmark_weights = None
        if isinstance(benchmark, str) and benchmark == "equal_weight":
            benchmark_weights = np.ones(len(strategy_indices), dtype=np.float32) / len(strategy_indices)
        elif isinstance(benchmark, list):
            # Convert benchmark tickers to indices
            benchmark_indices = np.array([
                self.ticker_to_idx.get(t, -1) for t in benchmark
            ])
            benchmark_indices = benchmark_indices[benchmark_indices >= 0]
            
            if len(benchmark_indices) == 0:
                raise ValueError(f"None of the benchmark tickers {benchmark} are in the universe")
            
            # Create benchmark weights array
            benchmark_weights = np.zeros(len(strategy_indices), dtype=np.float32)
            
            # Map universe indices to strategy indices
            for b_idx in benchmark_indices:
                if b_idx in strategy_indices:
                    s_idx = np.where(strategy_indices == b_idx)[0][0]
                    benchmark_weights[s_idx] = 1.0
            
            # Normalize
            if np.sum(benchmark_weights) > 0:
                benchmark_weights /= np.sum(benchmark_weights)
        elif isinstance(benchmark, np.ndarray):
            if len(benchmark) != len(strategy_indices):
                raise ValueError(
                    f"Benchmark weights has {len(benchmark)} elements, "
                    f"but strategy has {len(strategy_indices)} tickers"
                )
            benchmark_weights = benchmark.astype(np.float32)
        
        # Extract returns submatrix - avoid pandas operations
        strategy_returns_matrix = self.returns_matrix[:, strategy_indices]
        
        # Calculate returns using numpy operations
        result_dict = self.run_backtest_npy(
            returns_matrix=strategy_returns_matrix,
            weights_matrix=weights_matrix,
            benchmark_weights=benchmark_weights,
            shift_signals=shift_signals
        )
        
        # Only convert to pandas at the very end
        strategy_ticker_list = [self.universe_tickers[i] for i in strategy_indices]
        
        # Create minimal pandas output - only at the end
        return {
            'signals_df': pd.DataFrame(
                weights_matrix, 
                index=pd.DatetimeIndex(self.dates_array), 
                columns=strategy_ticker_list
            ),
            'tickers_returns': pd.DataFrame(
                strategy_returns_matrix,
                index=pd.DatetimeIndex(self.dates_array),
                columns=strategy_ticker_list
            ),
            'strategy_returns': pd.Series(
                result_dict["strategy_returns"],
                index=pd.DatetimeIndex(self.dates_array)
            ),
            'benchmark_returns': pd.Series(
                result_dict["benchmark_returns"],
                index=pd.DatetimeIndex(self.dates_array)
            ) if result_dict["benchmark_returns"] is not None else None,
        }
    
    def run_backtest_npy(self, returns_matrix, weights_matrix, benchmark_weights=None, shift_signals=True):
        """
        Pure numpy implementation of backtest calculations.
        """
        # Shift signals if needed - all numpy operations
        if shift_signals:
            W = np.zeros_like(weights_matrix)
            if weights_matrix.shape[0] > 1:
                W[1:] = weights_matrix[:-1]
        else:
            W = weights_matrix
            
        # Calculate strategy returns - use optimized dot product
        if W.shape[1] > 0:
            # Fast multiplication along rows
            strat_rets = np.sum(returns_matrix * W, axis=1)
        else:
            strat_rets = np.zeros(returns_matrix.shape[0], dtype=np.float32)
        
        # Calculate benchmark returns if needed
        if benchmark_weights is not None:
            # Use matrix multiplication for benchmark
            bench_rets = returns_matrix @ benchmark_weights
        else:
            bench_rets = np.zeros_like(strat_rets)
            
        return {
            "strategy_returns": strat_rets,
            "benchmark_returns": bench_rets
        }

class SubsetStrategy(NumPyVectorizedStrategyBase):
    """Strategy that only uses a subset of tickers."""
    def __init__(self, tickers: List[str], weight_type='equal'):
        super().__init__(tickers)
        self.weight_type = weight_type
    
    def batch(self, price_matrix: np.ndarray, dates: List[pd.Timestamp], 
              column_indices: List[int]) -> np.ndarray:
        """Returns equal weights for all tickers in the subset."""
        n_dates, n_tickers = price_matrix.shape
        
        if self.weight_type == 'equal':
            # Equal weight for all tickers
            weights = np.ones((n_dates, n_tickers)) / n_tickers
            # Return weights for dates[1:] to match returns dimension
            return weights[1:]
        elif self.weight_type == 'momentum':
            # Simple momentum strategy
            returns = np.zeros((n_dates, n_tickers))
            lookback = 20  # 20-day momentum
            
            # Calculate returns over lookback period
            for i in range(lookback, n_dates):
                returns[i] = price_matrix[i] / price_matrix[i-lookback] - 1
            
            # Set weights based on positive momentum
            weights = np.zeros((n_dates, n_tickers))
            weights[returns > 0] = 1
            
            # Normalize weights
            row_sums = np.sum(weights, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            weights = weights / row_sums
            
            # Return weights for dates[1:] to match returns dimension
            return weights[1:]
```

# pyproject.toml

```toml
[tool.poetry]
name = "portwine"
version = "0.1.3"
description = "A clean, elegant portfolio backtester that simplifies strategy development with online processing and comprehensive analysis tools"
authors = ["Stuart Farmer <stuart@lamden.io>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.2.3"
matplotlib = "^3.10.1"
scipy = "^1.15.2"
statsmodels = "^0.14.4"
tqdm = "^4.67.1"
cvxpy = "^1.6.4"
fredapi = "^0.5.2"
pandas-market-calendars = "^5.0.0"
requests = "^2.31.0"
pytz = "^2024.1"
django-scheduler = "^0.10.1"
flask = "^3.1.0"
rich = "^14.0.0"
fastparquet = "^2024.2.0"
freezegun = "^1.5.1"
numba = "^0.61.2"
httpx = "^0.28.1"

[tool.poetry.group.dev.dependencies]
mkdocs = "^1.6.1"
mkdocstrings = "^0.29.1"
mkdocstrings-python = "^1.16.12"
coverage = "^7.8.0"
pytest-cov = "^6.1.1"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

```

# README.md

```md
# portwine - a clean, elegant portfolio backtester
![The Triumph of Bacchus](imgs/header.jpg)
\`\`\`commandline
pip install portwine
\`\`\`
https://stuartfarmer.github.io/portwine/

Portfolio construction, optimization, and backtesting can be a complicated web of data wrangling, signal generation, lookahead bias reduction, and parameter tuning.

But with `portwine`, strategies are clear and written in an 'online' fashion that removes most of the complexity that comes with backtesting, analyzing, and deploying your trading strategies.

---

### Simple Strategies
Strategies are only given the last day of prices to make their determinations and allocate weights. This allows them to be completely encapsulated and portable.

\`\`\`python
class SimpleMomentumStrategy(StrategyBase):
    """
    A simple momentum strategy that:
    1. Calculates N-day momentum for each ticker
    2. Invests in the top performing ticker
    3. Rebalances weekly (every Friday)
    
    This demonstrates a step-based strategy implementation in a concise, easy-to-understand way.
    """
    
    def __init__(self, tickers, lookback_days=10):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to consider for investment
        lookback_days : int
            Number of days to use for momentum calculation
        """
        super().__init__(tickers)
        self.lookback_days = lookback_days
        self.price_history = {ticker: [] for ticker in tickers}
        self.current_signals = {ticker: 0.0 for ticker in tickers}
        self.dates = []
    
    def is_friday(self, date):
        """Check if given date is a Friday (weekday 4)"""
        return date.weekday() == 4
    
    def calculate_momentum(self, ticker):
        """Calculate simple price momentum over lookback period"""
        prices = self.price_history[ticker]
        
        # Need at least lookback_days+1 data points
        if len(prices) <= self.lookback_days:
            return -999.0
        
        # Get starting and ending prices for momentum calculation
        start_price = prices[-self.lookback_days-1]
        end_price = prices[-1]
        
        # Check for valid prices
        if start_price is None or end_price is None or start_price <= 0:
            return -999.0
        
        # Return simple momentum (end/start - 1)
        return end_price / start_price - 1.0
    
    def step(self, current_date, daily_data):
        """
        Process daily data and determine allocations
        """
        # Track dates for rebalancing logic
        self.dates.append(current_date)
        
        # Update price history for each ticker
        for ticker in self.tickers:
            price = None
            if daily_data.get(ticker) is not None:
                price = daily_data[ticker].get('close', None)
            
            # Forward fill missing data
            if price is None and len(self.price_history[ticker]) > 0:
                price = self.price_history[ticker][-1]
                
            self.price_history[ticker].append(price)
        
        # Only rebalance on Fridays
        if self.is_friday(current_date):
            # Calculate momentum for each ticker
            momentum_scores = {}
            for ticker in self.tickers:
                momentum_scores[ticker] = self.calculate_momentum(ticker)
            
            # Find best performing ticker
            best_ticker = max(momentum_scores.items(), 
                             key=lambda x: x[1] if x[1] != -999.0 else -float('inf'))[0]
            
            # Reset all allocations to zero
            self.current_signals = {ticker: 0.0 for ticker in self.tickers}
            
            # Allocate 100% to best performer if we have valid momentum
            if momentum_scores[best_ticker] != -999.0:
                self.current_signals[best_ticker] = 1.0
        
        # Return current allocations
        return self.current_signals.copy()
\`\`\`

---

### Breezy Backtesting

Backtesting strategies is a breeze, as well. Simply tell the backtester where your data is located with a data loader manager and give it a strategy. You get results immediately.

\`\`\`python
universe = ['MTUM', 'VTV', 'VUG', 'IJR', 'MDY']
strategy = SimpleMomentumStrategy(tickers=universe, lookback_days=10)

data_loader = EODHDMarketDataLoader(data_path='../../../Developer/Data/EODHD/us_sorted/US/')
backtester = Backtester(market_data_loader=data_loader)
results = backtester.run_backtest(strategy, benchmark_ticker='SPY')
\`\`\`
---
### Streamlined Data
Managing data can be a massive pain. But as long as you have your daily flat files from EODHD or Polygon saved in a directory, the data loaders will manage the rest. You don't have to worry about anything except writing code.

---

### Effortless Analysis

After running a strategy through the backtester, put it through an array of analyzers that are simple, visual, and clear. You can easily add your own analyzers to discover anything you need to know about your portfolio's performance, risk management, volatility, etc.

Check out what comes out of the box:

##### Equity Drawdown Analysis
\`\`\`python
EquityDrawdownAnalyzer().plot(results)
\`\`\`
![Equity Drawdown](imgs/equitydrawdown.jpg)

---
##### Monte Carlo Analysis
\`\`\`python
MonteCarloAnalyzer().plot(results)
\`\`\`
![Equity Drawdown](imgs/montecarlo.jpg)

---
##### Seasonality Analysis
\`\`\`python
EquityDrawdownAnalyzer().plot(results)
\`\`\`
![Equity Drawdown](imgs/seasonality.jpg)

With more on the way!

---
### Docs
https://stuartfarmer.github.io/portwine/

```

