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
            # Covariance
            cov = window_data.cov().values

            # Define w_var as a cvxpy variable
            w_var = cp.Variable(n, nonneg=True)  # w >= 0
            objective = cp.quad_form(w_var, cov)  # w^T Sigma w
            constraints = [cp.sum(w_var) == 1.0]
            prob = cp.Problem(cp.Minimize(objective), constraints)

            try:
                prob.solve()  # Solve the QP
                if (w_var.value is None) or (prob.status not in ["optimal", "optimal_inaccurate"]):
                    w = np.ones(n) / n
                else:
                    w = w_var.value
            except:
                w = np.ones(n) / n

        weight_list.append(w)

    # Build a DataFrame of daily weights
    weights_df = pd.DataFrame(weight_list, index=daily_ret_df.index, columns=tickers)

    # Optionally shift weights to avoid lookahead
    if shift_signals:
        weights_df = weights_df.shift(1).ffill().fillna(1.0 / n)

    # Portfolio daily returns = sum of ticker returns * weights
    bm_ret = (weights_df * daily_ret_df).sum(axis=1)
    return bm_ret


# ---------------------------
# 2) Dictionary of Standard Benchmarks
# ---------------------------
STANDARD_BENCHMARKS = {
    "equal_weight": benchmark_equal_weight,
    "markowitz": benchmark_markowitz,
}


# ---------------------------
# 3) Backtester Class
# ---------------------------
class Backtester:
    """
    A step-based backtester that can handle a function-based or ticker-based benchmark.
    """

    def __init__(self, market_data_loader):
        self.market_data_loader = market_data_loader

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
                     verbose=False):
        """
        Runs a step-based backtest.

        benchmark can be:
          - None -> no benchmark
          - a string -> if in STANDARD_BENCHMARKS, calls that function
                        else interpret as a single ticker
          - a callable -> a user-supplied function that (daily_ret_df)->pd.Series
        """

        # 1) fetch data for strategy tickers
        strategy_data = self.market_data_loader.fetch_data(strategy.tickers)
        if not strategy_data:
            print("No data found for strategy tickers!")
            return None

        # 2) parse benchmark argument
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

        # 3) fetch data for single benchmark ticker (if any)
        benchmark_data = {}
        if single_bm_ticker:
            benchmark_data = self.market_data_loader.fetch_data([single_bm_ticker])

        # 4) union of all dates
        all_data = dict(strategy_data)
        if benchmark_data:
            all_data.update(benchmark_data)

        if not all_data:
            print("No data fetched. Check your tickers and file paths.")
            return None

        all_dates = self._get_union_of_dates(all_data)

        # 5) gather daily signals
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

        # 6) shift signals if requested
        if shift_signals:
            signals_df = signals_df.shift(1).ffill().fillna(0.0)
        else:
            signals_df = signals_df.fillna(0.0)

        # 7) build a price DataFrame for the strategy tickers
        price_df = pd.DataFrame(index=signals_df.index)
        for tkr in strategy.tickers:
            df = strategy_data[tkr]
            px = df['close'].reindex(signals_df.index).ffill()
            price_df[tkr] = px

        # 8) daily returns of each ticker
        daily_ret_df = price_df.pct_change().fillna(0.0)

        # 9) strategy daily returns
        strategy_daily_returns = (daily_ret_df * signals_df).sum(axis=1)

        # 10) compute benchmark returns
        benchmark_daily_returns = None

        if single_bm_ticker and benchmark_data.get(single_bm_ticker) is not None:
            bm_price = benchmark_data[single_bm_ticker]['close'].reindex(signals_df.index).ffill()
            benchmark_daily_returns = bm_price.pct_change().fillna(0.0)
        elif benchmark_func:
            # pass daily_ret_df to the benchmark function
            # for Markowitz, we might pass lookback= etc. but let's keep it simple
            # The function signature is daily_ret_df -> pd.Series
            # We can do:
            benchmark_daily_returns = benchmark_func(daily_ret_df, verbose=verbose)

        return {
            'signals_df': signals_df,
            'tickers_returns': daily_ret_df,
            'strategy_returns': strategy_daily_returns,
            'benchmark_returns': benchmark_daily_returns,
        }

