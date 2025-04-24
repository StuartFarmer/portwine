"""
Vectorized strategy base class and updated backtester implementation.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from portwine.strategies import StrategyBase
from portwine.backtester import Backtester, STANDARD_BENCHMARKS

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
    if start_date:
        sd = pd.to_datetime(start_date)
        all_dates = [d for d in all_dates if d >= sd]
    if end_date:
        ed = pd.to_datetime(end_date)
        all_dates = [d for d in all_dates if d <= ed]

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

    # def run_backtest(self, strategy, benchmark="equal_weight",
    #                  start_date=None, end_date=None,
    #                  shift_signals=True, verbose=False):
    #     if not isinstance(strategy, VectorizedStrategyBase):
    #         raise TypeError("Strategy must be a VectorizedStrategyBase")
    #
    #     # 1) load prices (float dtype)
    #     price_df = create_price_dataframe(
    #         self.market_data_loader,
    #         tickers=strategy.tickers,
    #         start_date=start_date,
    #         end_date=end_date
    #     )
    #
    #     # 2) compute weights in one shot
    #     if verbose:
    #         print("Computing strategy weights…")
    #     weights_df = strategy.batch(price_df)
    #
    #     # 3) align dates
    #     common = price_df.index.intersection(weights_df.index)
    #     price_df = price_df.loc[common]
    #     weights_df = weights_df.loc[common]
    #
    #     # 4) shift if needed (all float ops → no warning)
    #     if shift_signals:
    #         weights_df = weights_df.shift(1).fillna(0)
    #
    #     # 5) compute returns
    #     returns_df = price_df.pct_change(fill_method=None).fillna(0)
    #
    #     # 6) portfolio P&L
    #     strat_rets = (returns_df * weights_df).sum(axis=1)
    #
    #     # 7) benchmark
    #     bm_rets = None
    #     if benchmark is not None:
    #         if isinstance(benchmark, str) and benchmark in STANDARD_BENCHMARKS:
    #             bm_rets = STANDARD_BENCHMARKS[benchmark](returns_df)
    #         elif isinstance(benchmark, str) and self.market_data_loader:
    #             raw = self.market_data_loader.fetch_data([benchmark])
    #             series = raw.get(benchmark)
    #             if series is not None:
    #                 bm = series['close'].reindex(common).ffill()
    #                 bm_rets = bm.pct_change(fill_method=None).fillna(0)
    #         elif callable(benchmark):
    #             bm_rets = benchmark(returns_df)
    #
    #     return {
    #         'signals_df': weights_df,
    #         'tickers_returns': returns_df,
    #         'strategy_returns': strat_rets,
    #         'benchmark_returns': bm_rets,
    #     }

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

        # 8) benchmark returns
        bm_rets = None
        if benchmark is not None:
            if isinstance(benchmark, str) and benchmark in STANDARD_BENCHMARKS:
                bm_rets = STANDARD_BENCHMARKS[benchmark](returns_df)
            elif isinstance(benchmark, str) and self.market_data_loader:
                raw = self.market_data_loader.fetch_data([benchmark])
                series = raw.get(benchmark)
                if series is not None:
                    bm_series = series["close"].reindex(common_idx).ffill()
                    bm_rets = bm_series.pct_change(fill_method=None).fillna(0.0)
            elif callable(benchmark):
                bm_rets = benchmark(returns_df)

        return {
            "signals_df": weights_df,
            "tickers_returns": returns_df,
            "strategy_returns": strategy_rets,
            "benchmark_returns": bm_rets,
        }


def benchmark_equal_weight(returns_df: pd.DataFrame) -> pd.Series:
    return returns_df.mean(axis=1)
