import pandas as pd
import numpy as np
import cvxpy as cp
from tqdm import tqdm


def benchmark_equal_weight(daily_ret_df, *args, **kwargs):
    """
    Return daily returns of an equally weighted (rebalanced daily) portfolio
    of all columns in daily_ret_df.
    """
    eq_wt_ret = daily_ret_df.mean(axis=1)  # average across columns (tickers)
    return eq_wt_ret


def benchmark_markowitz(daily_ret_df, lookback=60, shift_signals=True, verbose=False):
    """
    Returns daily returns of a global minimum-variance portfolio (no short selling).
    Solves: min w^T Sigma w,  s.t. sum(w)=1, w >=0, each day using last 'lookback' days.
    If shift_signals=True, shift the weights by 1 day to avoid lookahead.
    """
    tickers = daily_ret_df.columns
    n = len(tickers)

    if verbose:
        ret_iter = tqdm(daily_ret_df.index, desc="Markowitz Benchmark")
    else:
        ret_iter = daily_ret_df.index

    weight_list = []
    for current_date in ret_iter:
        # Slice up to 'current_date' with length up to 'lookback'
        window_data = daily_ret_df.loc[:current_date].tail(lookback)

        if len(window_data) < 2:
            # Not enough data => fallback
            w = np.ones(n) / n
        else:
            cov = window_data.cov().values
            w_var = cp.Variable(n, nonneg=True)  # w >= 0
            objective = cp.quad_form(w_var, cov)
            constraints = [cp.sum(w_var) == 1.0]
            prob = cp.Problem(cp.Minimize(objective), constraints)

            try:
                prob.solve()
                if (w_var.value is None) or (prob.status not in ["optimal", "optimal_inaccurate"]):
                    w = np.ones(n) / n
                else:
                    w = w_var.value
            except:
                w = np.ones(n) / n

        weight_list.append(w)

    # Build a DataFrame of daily weights
    weights_df = pd.DataFrame(weight_list, index=daily_ret_df.index, columns=tickers)

    # Optionally shift weights
    if shift_signals:
        weights_df = weights_df.shift(1).ffill().fillna(1.0 / n)

    bm_ret = (weights_df * daily_ret_df).sum(axis=1)
    return bm_ret


STANDARD_BENCHMARKS = {
    "equal_weight": benchmark_equal_weight,
    "markowitz": benchmark_markowitz,
}


class Backtester:
    """
    A step-based backtester that can handle:
      - function-based or ticker-based benchmark
      - optional start_date
      - optional 'require_all_history' to ensure we only start once
        all tickers have data.
      - alternative data sources with SOURCE:TICKER format
    """

    def __init__(self, market_data_loader, alternative_data_loader=None):
        """
        Initialize the backtester with market data loader and optional alternative data loader.

        Parameters
        ----------
        market_data_loader : MarketDataLoader
            Primary loader for market data (prices, etc.)
        alternative_data_loader : AlternativeMarketDataLoader, optional
            Loader for alternative data sources that use SOURCE:TICKER format
        """
        self.market_data_loader = market_data_loader
        self.alternative_data_loader = alternative_data_loader

    def _parse_tickers(self, tickers):
        """
        Parse tickers into regular and alternative tickers.

        Parameters
        ----------
        tickers : list
            List of ticker symbols. Regular tickers have no prefix.
            Alternative tickers use SOURCE:TICKER format.

        Returns
        -------
        tuple
            (regular_tickers, alternative_tickers)
        """
        regular_tickers = []
        alternative_tickers = []

        for ticker in tickers:
            if ":" in ticker:
                alternative_tickers.append(ticker)
            else:
                regular_tickers.append(ticker)

        return regular_tickers, alternative_tickers

    def _get_union_of_dates(self, data_dict):
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df.index)
        return sorted(list(all_dates))

    def _get_daily_data_dict(self, date, data_dict):
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
                     benchmark=None,
                     start_date=None,
                     end_date=None,
                     require_all_history=False,
                     verbose=False):
        """
        Runs a step-based backtest.

        Parameters
        ----------
        strategy : StrategyBase
            The strategy to backtest (has .tickers and step() method).
        shift_signals : bool
            If True, shift signals by 1 day for no lookahead.
        benchmark : None, str, or callable
            - None -> no benchmark
            - str: if in STANDARD_BENCHMARKS -> calls that function
                   else interpret as a single ticker (may have SOURCE:TICKER format)
            - callable -> user-supplied function that (daily_ret_df) -> pd.Series
        start_date : None, str, or datetime
            The earliest date to start the backtest. If None, uses all data.
            If str (e.g., "2005-01-01"), will parse to datetime.
        end_date : None, str, or datetime
            The latest date to end the backtest. If None, uses all data up to the end.
            If str (e.g., "2010-12-31"), will parse to datetime.
        require_all_history : bool
            If True, the backtest will not start until *all* tickers in
            strategy.tickers have data. This is done by finding each ticker's
            earliest date, taking the maximum, and skipping any earlier dates.
            We also consider 'start_date' if provided, taking the maximum of
            that and the all-tickers earliest date.
        verbose : bool
            If True, shows tqdm progress bars.

        Returns
        -------
        dict with keys:
            'signals_df'
            'tickers_returns'
            'strategy_returns'
            'benchmark_returns'
        """
        # 1) Separate regular and alternative tickers from strategy
        regular_tickers, alternative_tickers = self._parse_tickers(strategy.tickers)

        # 2) fetch data for strategy tickers
        strategy_data = {}

        # Load regular market data
        if regular_tickers:
            regular_data = self.market_data_loader.fetch_data(regular_tickers)
            strategy_data.update(regular_data)

        # Load alternative data if available
        if alternative_tickers and self.alternative_data_loader:
            alt_data = self.alternative_data_loader.fetch_data(alternative_tickers)
            strategy_data.update(alt_data)

        if not strategy_data:
            print("No data found for strategy tickers!")
            return None

        # 3) parse benchmark argument
        single_bm_ticker = None
        benchmark_func = None

        if benchmark is None:
            pass
        elif isinstance(benchmark, str):
            if benchmark in STANDARD_BENCHMARKS:
                benchmark_func = STANDARD_BENCHMARKS[benchmark]
            else:
                single_bm_ticker = benchmark
        elif callable(benchmark):
            benchmark_func = benchmark
        else:
            print(f"Unrecognized benchmark: {benchmark}")
            return None

        # 4) fetch data for single benchmark ticker (if any)
        benchmark_data = {}
        if single_bm_ticker:
            # Check if benchmark is alternative data
            if ":" in single_bm_ticker and self.alternative_data_loader:
                benchmark_data = self.alternative_data_loader.fetch_data([single_bm_ticker])
            else:
                benchmark_data = self.market_data_loader.fetch_data([single_bm_ticker])

        # 5) union of data
        all_data = dict(strategy_data)
        if benchmark_data:
            all_data.update(benchmark_data)

        if not all_data:
            print("No data fetched. Check your tickers and file paths.")
            return None

        all_dates = self._get_union_of_dates(all_data)

        # 6) parse user-supplied start_date and end_date
        user_start_date = None
        if start_date is not None:
            user_start_date = pd.to_datetime(start_date)

        user_end_date = None
        if end_date is not None:
            user_end_date = pd.to_datetime(end_date)

        # Optional: check if both are set => assert start <= end
        if user_start_date is not None and user_end_date is not None:
            if user_start_date > user_end_date:
                raise ValueError(f"start_date {user_start_date} cannot be after end_date {user_end_date}.")

        # 7) If require_all_history == True, find the earliest date for each ticker
        #    then pick the maximum of those, plus user_start_date if any
        if require_all_history:
            earliest_per_ticker = []
            for tkr, df in strategy_data.items():
                if df.empty:
                    print(f"Warning: Ticker {tkr} has no data, can't start at all.")
                    return None
                earliest_date = df.index.min()
                earliest_per_ticker.append(earliest_date)
            # The date we can start once *all* tickers have data
            common_earliest = max(earliest_per_ticker)
            if user_start_date is not None:
                final_start = max(common_earliest, user_start_date)
            else:
                final_start = common_earliest
            # filter out earlier dates
            all_dates = [d for d in all_dates if d >= final_start]
        else:
            # if not requiring all, just do the user start_date filter if any
            if user_start_date is not None:
                all_dates = [d for d in all_dates if d >= user_start_date]

        # also filter out beyond end_date if set
        if user_end_date is not None:
            all_dates = [d for d in all_dates if d <= user_end_date]

        if not all_dates:
            print("No dates remain after filtering by start/end.")
            return None

        # 8) gather daily signals
        signals_records = []
        if verbose:
            from tqdm import tqdm
            date_iter = tqdm(all_dates, desc="Backtest")
        else:
            date_iter = all_dates

        for date in date_iter:
            daily_data = self._get_daily_data_dict(date, all_data)
            daily_signals = strategy.step(date, daily_data)
            row_dict = {'date': date}
            for tkr in strategy.tickers:
                row_dict[tkr] = daily_signals.get(tkr, 0.0)
            signals_records.append(row_dict)

        signals_df = pd.DataFrame(signals_records).set_index('date').sort_index()

        # 9) shift signals if requested
        if shift_signals:
            signals_df = signals_df.shift(1).ffill().fillna(0.0)
        else:
            signals_df = signals_df.fillna(0.0)

        # 10) build a price DataFrame for the strategy tickers
        price_df = pd.DataFrame(index=signals_df.index)
        for tkr in strategy.tickers:
            if tkr in strategy_data:  # Only include tickers that have data
                df = strategy_data[tkr]
                px = df['close'].reindex(signals_df.index).ffill()
                price_df[tkr] = px

        # 11) daily returns of each ticker
        daily_ret_df = price_df.pct_change(fill_method=None).fillna(0.0)

        # 12) strategy daily returns
        strategy_daily_returns = (daily_ret_df * signals_df[daily_ret_df.columns]).sum(axis=1)

        # 13) compute benchmark returns
        benchmark_daily_returns = None
        if single_bm_ticker and benchmark_data.get(single_bm_ticker) is not None:
            bm_price = benchmark_data[single_bm_ticker]['close'].reindex(signals_df.index).ffill()
            benchmark_daily_returns = bm_price.pct_change(fill_method=None).fillna(0.0)
        elif benchmark_func:
            # pass daily_ret_df to the benchmark function
            benchmark_daily_returns = benchmark_func(daily_ret_df, verbose=verbose)

        return {
            'signals_df': signals_df,
            'tickers_returns': daily_ret_df,
            'strategy_returns': strategy_daily_returns,
            'benchmark_returns': benchmark_daily_returns,
        }