import pandas as pd
import numpy as np

class Backtester:
    """
    A step-based backtester that:
      1) Uses a MarketDataLoader to fetch and store market data.
      2) Allows multiple strategies to be tested via the run_backtest function.
      3) Steps through each date in chronological order.
      4) Feeds each day's data into the strategy to generate signals/weights.
      5) Optionally shifts signals by 1 day to avoid lookahead bias.
      6) Computes daily returns and returns results in percentage terms
         (equity curve starts at 1.0).
    """

    def __init__(self, market_data_loader):
        """
        Parameters
        ----------
        market_data_loader : MarketDataLoader
            An object that can fetch ticker data from CSVs (or another source).
        """
        self.market_data_loader = market_data_loader

    def _get_union_of_dates(self, data_dict):
        """
        Build a sorted list of all unique dates from the data_dict.
        """
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df.index)
        return sorted(list(all_dates))

    def _get_daily_data_dict(self, date, data_dict):
        """
        For each ticker in data_dict, return a dict of that day's row if it exists; else None.
        """
        daily_data = {}
        for tkr, df in data_dict.items():
            if date in df.index:
                daily_data[tkr] = df.loc[date].to_dict()
            else:
                daily_data[tkr] = None
        return daily_data

    def run_backtest(self,
                     strategy,
                     shift_signals=True,
                     benchmark_ticker=None):
        """
        Runs a daily step-based backtest for the given strategy.
        The equity curve starts at 1.0 (percentage terms).

        Parameters
        ----------
        strategy : StrategyBase
            A strategy object that has tickers, plus a step() method.
        shift_signals : bool
            If True, shift signals by 1 day to avoid lookahead bias.
        benchmark_ticker : str or None
            If provided, returns daily & equity curve for that benchmark.

        Returns
        -------
        results : dict
            {
                'signals_df'            : DataFrame of daily signals/weights,
                'strategy_daily_returns' : Series of daily strategy returns,
                'equity_curve'          : Series of cumulative returns starting at 1.0,
                'benchmark_daily_returns': Series of daily returns for benchmark (if any),
                'benchmark_equity_curve' : Series of benchmark cumulative returns (if any)
            }
        """

        # Fetch all needed data for the strategy
        strategy_data = self.market_data_loader.fetch_data(strategy.tickers)

        # Optionally fetch benchmark data
        benchmark_data = {}
        if benchmark_ticker:
            benchmark_data = self.market_data_loader.fetch_data([benchmark_ticker])

        # If no data fetched, bail
        if not strategy_data and not benchmark_data:
            print("No market data fetched. Check your tickers and file paths.")
            return None

        # Build the sorted union of all available dates (both strategy + benchmark)
        all_data = {**strategy_data, **benchmark_data}
        all_dates = self._get_union_of_dates(all_data)

        # Gather signals day by day
        signals_records = []
        for date in all_dates:
            daily_data = self._get_daily_data_dict(date, all_data)
            # Strategy step
            daily_signals = strategy.step(date, daily_data)
            # Build row for signals DataFrame
            row_dict = {'date': date}
            for tkr in strategy.tickers:
                row_dict[tkr] = daily_signals.get(tkr, 0.0)
            signals_records.append(row_dict)

        signals_df = pd.DataFrame(signals_records).set_index('date').sort_index()

        # Optionally shift signals by 1 day
        if shift_signals:
            signals_df = signals_df.shift(1).ffill().fillna(0.0)
        else:
            signals_df = signals_df.fillna(0.0)

        # Build a price DataFrame for the *primary* strategy tickers
        price_df = pd.DataFrame(index=signals_df.index)
        for tkr in strategy.tickers:
            if tkr in strategy_data:
                px = strategy_data[tkr]['close'].reindex(signals_df.index)
                px = px.ffill()
                price_df[tkr] = px
            else:
                price_df[tkr] = np.nan

        # Calculate daily returns (percentage changes)
        daily_ret_df = price_df.pct_change().fillna(0.0)

        # Strategy daily returns = sum of (weight * daily return of each ticker)
        strategy_daily_returns = (daily_ret_df * signals_df).sum(axis=1)

        # Handle benchmark if provided
        benchmark_daily_returns = None
        if benchmark_ticker and benchmark_data.get(benchmark_ticker) is not None:
            bm_px = benchmark_data[benchmark_ticker]['close']
            bm_px = bm_px.reindex(strategy_daily_returns.index).ffill()
            benchmark_daily_returns = bm_px.pct_change().fillna(0.0)

        # Return all relevant results
        return {
            'signals_df': signals_df,
            'tickers_returns': daily_ret_df,
            'strategy_returns': strategy_daily_returns,
            'benchmark_returns': benchmark_daily_returns,
        }


